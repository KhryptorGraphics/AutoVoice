"""Tests for ONNX export of inference models.

Verifies that exported ONNX models produce outputs matching PyTorch
within tolerance, and that dynamic axes work correctly.
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
from auto_voice.models.encoder import ContentEncoder
from auto_voice.models.so_vits_svc import SoVitsSvc
from auto_voice.models.vocoder import BigVGANGenerator


TOLERANCE = 1e-4


@pytest.fixture
def content_encoder():
    model = ContentEncoder(
        hidden_size=256,
        output_size=256,
        encoder_type='linear',
        encoder_backend='hubert',
    )
    model.train(False)
    return model


@pytest.fixture
def sovits_model():
    model = SoVitsSvc(config={
        'content_dim': 256,
        'pitch_dim': 256,
        'speaker_dim': 128,
        'hidden_dim': 192,
        'n_mels': 80,
        'spec_channels': 513,
    })
    model.train(False)
    return model


@pytest.fixture
def bigvgan_model():
    model = BigVGANGenerator(
        num_mels=100,
        upsample_rates=[4, 4, 2, 2, 2],
        upsample_kernel_sizes=[8, 8, 4, 4, 4],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    )
    model.train(False)
    return model


class TestContentEncoderExport:

    def test_export_creates_file(self, content_encoder, tmp_path):
        out = tmp_path / "encoder.onnx"
        result = export_content_encoder(content_encoder, str(out))
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_output_matches_pytorch(self, content_encoder, tmp_path):
        out = tmp_path / "encoder.onnx"
        export_content_encoder(content_encoder, str(out), seq_len=50)

        # PyTorch reference
        features = torch.randn(1, 50, 256)
        with torch.no_grad():
            pt_output = content_encoder.projection(features).numpy()

        # ONNX inference
        session = ort.InferenceSession(str(out))
        onnx_output = session.run(None, {'features': features.numpy()})[0]

        np.testing.assert_allclose(onnx_output, pt_output, atol=TOLERANCE)

    def test_dynamic_batch(self, content_encoder, tmp_path):
        out = tmp_path / "encoder.onnx"
        export_content_encoder(content_encoder, str(out), seq_len=50)

        session = ort.InferenceSession(str(out))

        # Batch size 1
        inp1 = np.random.randn(1, 50, 256).astype(np.float32)
        out1 = session.run(None, {'features': inp1})[0]
        assert out1.shape[0] == 1

        # Batch size 4
        inp4 = np.random.randn(4, 50, 256).astype(np.float32)
        out4 = session.run(None, {'features': inp4})[0]
        assert out4.shape[0] == 4

    def test_dynamic_sequence(self, content_encoder, tmp_path):
        out = tmp_path / "encoder.onnx"
        export_content_encoder(content_encoder, str(out), seq_len=50)

        session = ort.InferenceSession(str(out))

        # Different sequence lengths
        for seq_len in [30, 50, 120]:
            inp = np.random.randn(1, seq_len, 256).astype(np.float32)
            result = session.run(None, {'features': inp})[0]
            assert result.shape == (1, seq_len, 256)


class TestSoVitsSvcExport:

    def test_export_creates_file(self, sovits_model, tmp_path):
        out = tmp_path / "sovits.onnx"
        result = export_sovits(sovits_model, str(out))
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_output_matches_pytorch(self, sovits_model, tmp_path):
        out = tmp_path / "sovits.onnx"
        export_sovits(sovits_model, str(out), seq_len=50)

        # PyTorch reference (inference path)
        content = torch.randn(1, 50, 256)
        pitch = torch.randn(1, 50, 256)
        speaker = torch.randn(1, 128)
        with torch.no_grad():
            pt_output = sovits_model.infer(content, pitch, speaker).numpy()

        # ONNX inference
        session = ort.InferenceSession(str(out))
        onnx_output = session.run(None, {
            'content': content.numpy(),
            'pitch': pitch.numpy(),
            'speaker': speaker.numpy(),
        })[0]

        np.testing.assert_allclose(onnx_output, pt_output, atol=TOLERANCE)

    def test_dynamic_batch(self, sovits_model, tmp_path):
        out = tmp_path / "sovits.onnx"
        export_sovits(sovits_model, str(out), seq_len=50)

        session = ort.InferenceSession(str(out))

        for batch in [1, 3]:
            result = session.run(None, {
                'content': np.random.randn(batch, 50, 256).astype(np.float32),
                'pitch': np.random.randn(batch, 50, 256).astype(np.float32),
                'speaker': np.random.randn(batch, 128).astype(np.float32),
            })[0]
            assert result.shape[0] == batch
            assert result.shape[1] == 80  # n_mels

    def test_dynamic_sequence(self, sovits_model, tmp_path):
        out = tmp_path / "sovits.onnx"
        export_sovits(sovits_model, str(out), seq_len=50)

        session = ort.InferenceSession(str(out))

        for seq_len in [30, 80]:
            result = session.run(None, {
                'content': np.random.randn(1, seq_len, 256).astype(np.float32),
                'pitch': np.random.randn(1, seq_len, 256).astype(np.float32),
                'speaker': np.random.randn(1, 128).astype(np.float32),
            })[0]
            assert result.shape == (1, 80, seq_len)


class TestBigVGANExport:

    def test_export_creates_file(self, bigvgan_model, tmp_path):
        out = tmp_path / "bigvgan.onnx"
        result = export_bigvgan(bigvgan_model, str(out))
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_output_matches_pytorch(self, bigvgan_model, tmp_path):
        out = tmp_path / "bigvgan.onnx"

        # Get PyTorch reference BEFORE export (deepcopy in export can
        # disturb parametrized module state in PyTorch 2.11+)
        mel = torch.randn(1, 100, 50)
        with torch.no_grad():
            pt_output = bigvgan_model(mel).numpy()

        export_bigvgan(bigvgan_model, str(out), mel_len=50)

        # ONNX inference
        session = ort.InferenceSession(str(out))
        onnx_output = session.run(None, {'mel': mel.numpy()})[0]

        np.testing.assert_allclose(onnx_output, pt_output, atol=TOLERANCE)

    def test_dynamic_time(self, bigvgan_model, tmp_path):
        out = tmp_path / "bigvgan.onnx"
        export_bigvgan(bigvgan_model, str(out), mel_len=50)

        session = ort.InferenceSession(str(out))

        # upsample_rates = [4,4,2,2,2] -> total factor = 128
        upsample_factor = 4 * 4 * 2 * 2 * 2  # 128

        for mel_len in [30, 50, 80]:
            mel = np.random.randn(1, 100, mel_len).astype(np.float32)
            result = session.run(None, {'mel': mel})[0]
            assert result.shape[0] == 1
            assert result.shape[1] == 1
            assert result.shape[2] == mel_len * upsample_factor

    def test_output_not_nan(self, bigvgan_model, tmp_path):
        out = tmp_path / "bigvgan.onnx"
        export_bigvgan(bigvgan_model, str(out), mel_len=50)

        session = ort.InferenceSession(str(out))
        mel = np.random.randn(1, 100, 50).astype(np.float32)
        result = session.run(None, {'mel': mel})[0]

        assert not np.isnan(result).any()
        assert not np.isinf(result).any()
        # Output should be in [-1, 1] due to tanh
        assert result.max() <= 1.0
        assert result.min() >= -1.0
