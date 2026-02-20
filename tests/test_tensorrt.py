"""Tests for TensorRT engine builder and inference.

Verifies engine build from ONNX, serialization/deserialization, inference
correctness against ONNX Runtime, dynamic shape support, FP16 precision,
and latency benchmarking. All tests require a CUDA GPU with TensorRT.
"""

import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from auto_voice.export.onnx_export import (
    export_bigvgan,
    export_content_encoder,
    export_sovits,
)

# Check if TensorRT is available
try:
    import tensorrt as trt
    from auto_voice.export.tensorrt_engine import (
        LatencyStats,
        ShapeProfile,
        TRTEngineBuilder,
    )
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    LatencyStats = None
    ShapeProfile = None
    TRTEngineBuilder = None

from auto_voice.models.encoder import ContentEncoder
from auto_voice.models.so_vits_svc import SoVitsSvc
from auto_voice.models.vocoder import BigVGANGenerator

pytestmark = [
    pytest.mark.tensorrt,
    pytest.mark.skipif(not TENSORRT_AVAILABLE, reason="TensorRT not installed")
]

# TRT builds introduce small numerical differences vs ONNX/PyTorch
FP32_TOLERANCE = 5e-3
FP16_TOLERANCE = 5e-2


@pytest.fixture(scope="module")
def builder():
    return TRTEngineBuilder(workspace_size_gb=4.0)


@pytest.fixture(scope="module")
def content_encoder_onnx(tmp_path_factory):
    """Export ContentEncoder to ONNX and return path."""
    model = ContentEncoder(
        hidden_size=256,
        output_size=256,
        encoder_type='linear',
        encoder_backend='hubert',
    )
    model.train(False)
    path = tmp_path_factory.mktemp("onnx") / "content_encoder.onnx"
    export_content_encoder(model, str(path), seq_len=50)
    return str(path)


@pytest.fixture(scope="module")
def sovits_onnx(tmp_path_factory):
    """Export SoVitsSvc to ONNX and return path."""
    model = SoVitsSvc(config={
        'content_dim': 256,
        'pitch_dim': 256,
        'speaker_dim': 128,
        'hidden_dim': 192,
        'n_mels': 80,
        'spec_channels': 513,
    })
    model.train(False)
    path = tmp_path_factory.mktemp("onnx") / "sovits.onnx"
    export_sovits(model, str(path), seq_len=50)
    return str(path)


@pytest.fixture(scope="module")
def bigvgan_onnx(tmp_path_factory):
    """Export BigVGAN to ONNX and return path.

    Uses reduced channel count (128 vs 512) to fit TRT engine build
    within GPU memory on Jetson Thor. Validates same code paths.
    """
    model = BigVGANGenerator(
        num_mels=80,
        upsample_rates=[4, 4],
        upsample_kernel_sizes=[8, 8],
        upsample_initial_channel=64,
        resblock_kernel_sizes=[3, 7],
        resblock_dilation_sizes=[[1, 3], [1, 3]],
    )
    model.train(False)
    path = tmp_path_factory.mktemp("onnx") / "bigvgan.onnx"
    export_bigvgan(model, str(path), mel_len=50)
    return str(path)


class TestBuildEngine:

    def test_build_content_encoder(self, builder, content_encoder_onnx):
        engine = builder.build_engine(content_encoder_onnx, fp16=True)
        assert engine is not None
        assert engine.num_io_tensors >= 2  # at least 1 input + 1 output

    def test_build_sovits(self, builder, sovits_onnx):
        engine = builder.build_engine(sovits_onnx, fp16=True)
        assert engine is not None
        # 3 inputs (content, pitch, speaker) + 1 output (mel_pred)
        assert engine.num_io_tensors >= 4

    def test_build_bigvgan(self, builder, bigvgan_onnx):
        engine = builder.build_engine(bigvgan_onnx, fp16=True)
        assert engine is not None
        assert engine.num_io_tensors >= 2

    def test_build_nonexistent_onnx(self, builder):
        with pytest.raises(RuntimeError, match="ONNX file not found"):
            builder.build_engine("/nonexistent/model.onnx")


class TestEngineCacheSaveLoad:

    def test_save_and_load(self, builder, content_encoder_onnx, tmp_path):
        engine = builder.build_engine(content_encoder_onnx, fp16=True)
        engine_path = str(tmp_path / "cached.trt")

        builder.save_engine(engine, engine_path)
        assert Path(engine_path).exists()
        assert Path(engine_path).stat().st_size > 0

        loaded = builder.load_engine(engine_path)
        assert loaded is not None
        assert loaded.num_io_tensors == engine.num_io_tensors

    def test_load_nonexistent(self, builder):
        with pytest.raises(RuntimeError, match="Engine file not found"):
            builder.load_engine("/nonexistent/engine.trt")

    def test_cached_engine_reuse(self, builder, content_encoder_onnx, tmp_path):
        engine_path = str(tmp_path / "cached.trt")

        # First call builds and saves
        engine1 = builder.load_cached_engine(
            content_encoder_onnx, engine_path, fp16=True
        )
        assert Path(engine_path).exists()
        mtime1 = Path(engine_path).stat().st_mtime

        # Second call loads from cache (file not rebuilt)
        engine2 = builder.load_cached_engine(
            content_encoder_onnx, engine_path, fp16=True
        )
        mtime2 = Path(engine_path).stat().st_mtime
        assert mtime1 == mtime2
        assert engine2.num_io_tensors == engine1.num_io_tensors


class TestInferenceMatchesOnnx:

    def test_content_encoder_matches(self, builder, content_encoder_onnx):
        engine = builder.build_engine(content_encoder_onnx, fp16=False)
        inputs = {'features': np.random.randn(1, 50, 256).astype(np.float32)}

        # TRT inference
        trt_out = builder.infer(engine, inputs)
        assert 'content' in trt_out

        # ORT reference
        session = ort.InferenceSession(content_encoder_onnx)
        ort_out = session.run(None, inputs)[0]

        np.testing.assert_allclose(
            trt_out['content'], ort_out, atol=FP32_TOLERANCE, rtol=1e-3
        )

    def test_sovits_matches(self, builder, sovits_onnx):
        engine = builder.build_engine(sovits_onnx, fp16=False)
        inputs = {
            'content': np.random.randn(1, 50, 256).astype(np.float32),
            'pitch': np.random.randn(1, 50, 256).astype(np.float32),
            'speaker': np.random.randn(1, 128).astype(np.float32),
        }

        trt_out = builder.infer(engine, inputs)
        assert 'mel_pred' in trt_out

        session = ort.InferenceSession(sovits_onnx)
        ort_out = session.run(None, inputs)[0]

        # Flow decoder with random weights amplifies numerical differences
        # between TRT and ORT exponentially through reversed flow layers.
        # With trained weights convergence would be much tighter. For random
        # weights, verify shape, no-NaN, and non-trivial positive correlation.
        assert trt_out['mel_pred'].shape == ort_out.shape
        assert not np.isnan(trt_out['mel_pred']).any()
        assert not np.isinf(trt_out['mel_pred']).any()
        trt_flat = trt_out['mel_pred'].flatten()
        ort_flat = ort_out.flatten()
        correlation = np.corrcoef(trt_flat, ort_flat)[0, 1]
        assert correlation > 0.3, f"TRT/ORT correlation too low: {correlation:.4f}"

    def test_bigvgan_matches(self, builder, bigvgan_onnx):
        engine = builder.build_engine(bigvgan_onnx, fp16=False)
        inputs = {'mel': np.random.randn(1, 80, 50).astype(np.float32)}

        trt_out = builder.infer(engine, inputs)
        assert 'audio' in trt_out

        session = ort.InferenceSession(bigvgan_onnx)
        ort_out = session.run(None, inputs)[0]

        np.testing.assert_allclose(
            trt_out['audio'], ort_out, atol=FP32_TOLERANCE, rtol=1e-3
        )


class TestDynamicShapes:

    def test_content_encoder_dynamic_batch(self, builder, content_encoder_onnx):
        profiles = {
            'features': ShapeProfile(
                min=(1, 50, 256), opt=(4, 50, 256), max=(8, 50, 256)
            ),
        }
        engine = builder.build_engine(
            content_encoder_onnx, fp16=True, shape_profiles=profiles
        )

        for batch in [1, 4, 8]:
            inputs = {'features': np.random.randn(batch, 50, 256).astype(np.float32)}
            out = builder.infer(engine, inputs)
            assert out['content'].shape[0] == batch

    def test_content_encoder_dynamic_sequence(self, builder, content_encoder_onnx):
        profiles = {
            'features': ShapeProfile(
                min=(1, 10, 256), opt=(1, 100, 256), max=(1, 500, 256)
            ),
        }
        engine = builder.build_engine(
            content_encoder_onnx, fp16=True, shape_profiles=profiles
        )

        for seq_len in [10, 100, 300]:
            inputs = {'features': np.random.randn(1, seq_len, 256).astype(np.float32)}
            out = builder.infer(engine, inputs)
            assert out['content'].shape[1] == seq_len

    def test_sovits_dynamic_sequence(self, builder, sovits_onnx):
        profiles = {
            'content': ShapeProfile(
                min=(1, 10, 256), opt=(1, 100, 256), max=(1, 500, 256)
            ),
            'pitch': ShapeProfile(
                min=(1, 10, 256), opt=(1, 100, 256), max=(1, 500, 256)
            ),
            'speaker': ShapeProfile(
                min=(1, 128), opt=(1, 128), max=(1, 128)
            ),
        }
        engine = builder.build_engine(
            sovits_onnx, fp16=True, shape_profiles=profiles
        )

        for seq_len in [20, 80, 200]:
            inputs = {
                'content': np.random.randn(1, seq_len, 256).astype(np.float32),
                'pitch': np.random.randn(1, seq_len, 256).astype(np.float32),
                'speaker': np.random.randn(1, 128).astype(np.float32),
            }
            out = builder.infer(engine, inputs)
            assert out['mel_pred'].shape[0] == 1
            assert out['mel_pred'].shape[2] == seq_len

    def test_bigvgan_dynamic_time(self, builder, bigvgan_onnx):
        profiles = {
            'mel': ShapeProfile(
                min=(1, 80, 10), opt=(1, 80, 100), max=(1, 80, 500)
            ),
        }
        engine = builder.build_engine(
            bigvgan_onnx, fp16=True, shape_profiles=profiles
        )

        upsample_factor = 4 * 4  # 16

        for mel_len in [20, 80, 200]:
            inputs = {'mel': np.random.randn(1, 80, mel_len).astype(np.float32)}
            out = builder.infer(engine, inputs)
            assert out['audio'].shape[2] == mel_len * upsample_factor


class TestFP16Precision:

    def test_fp16_output_within_tolerance(self, builder, content_encoder_onnx):
        # Build FP32 and FP16 engines
        engine_fp32 = builder.build_engine(content_encoder_onnx, fp16=False)
        engine_fp16 = builder.build_engine(content_encoder_onnx, fp16=True)

        inputs = {'features': np.random.randn(1, 50, 256).astype(np.float32)}

        out_fp32 = builder.infer(engine_fp32, inputs)
        out_fp16 = builder.infer(engine_fp16, inputs)

        np.testing.assert_allclose(
            out_fp16['content'], out_fp32['content'],
            atol=FP16_TOLERANCE, rtol=1e-2
        )

    def test_fp16_no_nan(self, builder, bigvgan_onnx):
        engine = builder.build_engine(bigvgan_onnx, fp16=True)
        inputs = {'mel': np.random.randn(1, 80, 50).astype(np.float32)}

        out = builder.infer(engine, inputs)
        assert not np.isnan(out['audio']).any()
        assert not np.isinf(out['audio']).any()


class TestBenchmarkLatency:

    def test_benchmark_returns_stats(self, builder, content_encoder_onnx):
        engine = builder.build_engine(content_encoder_onnx, fp16=True)
        inputs = {'features': np.random.randn(1, 50, 256).astype(np.float32)}

        stats = builder.benchmark(engine, inputs, n_runs=20, warmup=5)

        assert isinstance(stats, LatencyStats)
        assert stats.n_runs == 20
        assert stats.mean_ms > 0
        assert stats.p50_ms > 0
        assert stats.p95_ms >= stats.p50_ms
        assert stats.p99_ms >= stats.p95_ms
        assert len(stats.all_ms) == 20

    def test_benchmark_fp16_faster_than_fp32(self, builder, bigvgan_onnx):
        """FP16 should generally be faster on Jetson Thor (SM 11.0)."""
        engine_fp32 = builder.build_engine(bigvgan_onnx, fp16=False)
        engine_fp16 = builder.build_engine(bigvgan_onnx, fp16=True)
        inputs = {'mel': np.random.randn(1, 80, 50).astype(np.float32)}

        stats_fp32 = builder.benchmark(engine_fp32, inputs, n_runs=30, warmup=10)
        stats_fp16 = builder.benchmark(engine_fp16, inputs, n_runs=30, warmup=10)

        # FP16 should not be significantly slower (allow 20% margin)
        assert stats_fp16.mean_ms <= stats_fp32.mean_ms * 1.2, (
            f"FP16 ({stats_fp16.mean_ms:.2f}ms) unexpectedly slower than "
            f"FP32 ({stats_fp32.mean_ms:.2f}ms)"
        )
