from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from auto_voice.web.offline_realtime import run_offline_realtime_conversion


class _FakeRealtimePipeline:
    def __init__(self):
        self.sample_rate = 16000
        self.output_sample_rate = 22050
        self.embedding = None
        self.clear_calls = 0
        self.chunk_lengths = []

    def set_speaker_embedding(self, embedding):
        self.embedding = np.asarray(embedding, dtype=np.float32)

    def process_chunk(self, audio):
        self.chunk_lengths.append(len(audio))
        return np.full(len(audio), 0.25, dtype=np.float32)

    def clear_speaker(self):
        self.clear_calls += 1

    def get_latency_metrics(self):
        return {"total_ms": 12.5}


def test_run_offline_realtime_conversion_downmixes_chunks_and_clears_speaker(tmp_path: Path):
    stereo = np.stack(
        [
            np.linspace(-0.2, 0.2, 22050, dtype=np.float32),
            np.linspace(0.2, -0.2, 22050, dtype=np.float32),
        ],
        axis=1,
    )
    input_path = tmp_path / "input.wav"
    sf.write(input_path, stereo, 22050)

    pipeline = _FakeRealtimePipeline()
    result = run_offline_realtime_conversion(
        str(input_path),
        np.ones(256, dtype=np.float32),
        chunk_duration_seconds=0.2,
        pipeline=pipeline,
    )

    assert result["sample_rate"] == 22050
    assert result["metadata"]["pipeline"] == "realtime"
    assert result["metadata"]["latency_metrics"]["total_ms"] == 12.5
    assert result["mixed_audio"].dtype == np.float32
    assert pipeline.embedding.shape == (256,)
    assert pipeline.clear_calls == 1
    assert len(pipeline.chunk_lengths) >= 2
