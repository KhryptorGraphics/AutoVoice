"""Regression coverage for brightness-only corpus generation."""

import numpy as np
import pytest
import soundfile as sf

build_corpus_v4 = pytest.importorskip("experiments.build_corpus_v4")


def _band_ratio_db(audio, sr, band, reference):
    spectrum = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(len(audio), 1.0 / sr)
    band_energy = np.linalg.norm(spectrum[(frequencies >= band[0]) &
                                          (frequencies < band[1])])
    ref_energy = np.linalg.norm(spectrum[(frequencies >= reference[0]) &
                                         (frequencies < reference[1])])
    return 20.0 * np.log10(
        (band_energy + 1e-12) / (ref_energy + 1e-12)
    )


def test_empty_shift_list_writes_only_brightened_plain_copy(tmp_path, monkeypatch):
    sr = build_corpus_v4.SR
    time = np.arange(sr, dtype=np.float32) / sr
    audio = (0.2 * np.sin(2 * np.pi * 440 * time) +
             0.002 * np.sin(2 * np.pi * 9000 * time)).astype(np.float32)
    sample_dir = tmp_path / "sample_001"
    sample_dir.mkdir()
    source = sample_dir / "vocals.wav"
    sf.write(source, audio, sr)

    monkeypatch.setattr(build_corpus_v4, "SHIFTS", [])
    monkeypatch.setattr(build_corpus_v4.glob, "glob",
                        lambda _pattern: [str(source)])
    monkeypatch.setattr(build_corpus_v4.os, "cpu_count", lambda: 3)
    out_dir = tmp_path / "brightness_only"

    build_corpus_v4.main(str(out_dir))

    generated = sorted(
        (out_dir / "dataset_raw" / "spk_fb17af66").glob("*.wav")
    )
    assert [path.name for path in generated] == ["sample_001_000.wav"]
    output, _ = sf.read(generated[0], dtype="float32")
    before = _band_ratio_db(audio, sr, (8000, 12000), (300, 3000))
    after = _band_ratio_db(output, sr, (8000, 12000), (300, 3000))
    assert after - before > 3.0
