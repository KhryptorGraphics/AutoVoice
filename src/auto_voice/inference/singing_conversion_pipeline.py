"""Singing voice conversion pipeline.

Orchestrates: audio separation -> content encoding -> voice conversion -> vocoder -> mixing.
"""
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class SeparationError(Exception):
    """Raised when vocal/instrumental separation fails."""
    pass


class ConversionError(Exception):
    """Raised when voice conversion fails."""
    pass


def _active_audio(vocals: np.ndarray, sample_rate: int, spans) -> np.ndarray:
    """Concatenate the audio inside spans (active regions only).

    Used for per-cluster signal measurement — zero-padded tracks would dilute
    the statistics, so this deliberately drops the silence between spans.
    """
    parts = [vocals[int(s * sample_rate):int(e * sample_rate)] for s, e in spans]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def _voiced_fraction(audio: np.ndarray, sample_rate: int, max_s: float = 20.0) -> float:
    """Fraction of pyin-voiced frames — the convertibility signal.

    Clean monophonic melody (what the fork engine converts well) scores high
    (calibrated leads: 0.76-0.87); harmony stacks / textures it would butcher
    score low (calibrated: 0.14-0.51). Measured on up to ``max_s`` of active
    audio with pyin fmin=80 fmax=1000, matching the calibration convention.
    """
    import librosa

    a = np.asarray(audio, dtype=np.float32)[: int(max_s * sample_rate)]
    _, voiced_flag, _ = librosa.pyin(a, fmin=80, fmax=1000, sr=sample_rate)
    if voiced_flag is None or len(voiced_flag) == 0:
        return 0.0
    return float(np.mean(voiced_flag))


def _group_notes_into_lines(notes, max_lines: int = 3):
    """Group polyphonic note events into monophonic voice lines.

    Voice-leading heuristic: notes (time-ordered) are assigned to the free
    line with the nearest last pitch; a new line opens while under
    ``max_lines``. Weak notes (short, or quiet relative to the loudest note)
    are dropped first — basic-pitch emits spurious high-harmonic blips on
    dense stacks.

    Returns a list of note-event lists, one per line, longest-duration first.
    """
    if not notes:
        return []
    max_amp = max(n.get('amplitude', 1.0) for n in notes) or 1.0
    kept = [n for n in notes
            if (n['end'] - n['start']) >= 0.1
            and n.get('amplitude', 1.0) >= 0.15 * max_amp]

    # Drop harmonic duplicates: basic-pitch often reports a strong harmonic of
    # a concurrent lower note as its own note; converting those would add
    # squeaky phantom voices. A note is a dup if a concurrent note sits ~1/k
    # of its frequency (k=2..5) and is not much quieter.
    def _hz(n):
        return 440.0 * 2.0 ** ((n['pitch_midi'] - 69.0) / 12.0)

    def _is_harmonic_dup(note):
        dur = note['end'] - note['start']
        for other in kept:
            if other is note:
                continue
            overlap = min(note['end'], other['end']) - max(note['start'], other['start'])
            if overlap < 0.5 * dur:
                continue
            ratio = _hz(note) / _hz(other)
            for k in (2, 3, 4, 5):
                if abs(ratio - k) < 0.06 * k and \
                        note.get('amplitude', 1.0) <= 1.2 * other.get('amplitude', 1.0):
                    return True
        return False

    kept = [n for n in kept if not _is_harmonic_dup(n)]

    lines = []  # {'notes': [...], 'end': float, 'pitch': float}
    for note in sorted(kept, key=lambda n: (n['start'], n['pitch_midi'])):
        free = [ln for ln in lines if ln['end'] <= note['start'] + 0.05]
        if free:
            best = min(free, key=lambda ln: abs(ln['pitch'] - note['pitch_midi']))
        elif len(lines) < max_lines:
            best = {'notes': [], 'end': 0.0, 'pitch': note['pitch_midi']}
            lines.append(best)
        else:
            # All lines busy (dense stack): merge into the nearest-pitch line.
            best = min(lines, key=lambda ln: abs(ln['pitch'] - note['pitch_midi']))
        best['notes'].append(note)
        best['end'] = max(best['end'], note['end'])
        best['pitch'] = note['pitch_midi']

    grouped = [ln['notes'] for ln in lines if ln['notes']]
    grouped.sort(key=lambda ns: -sum(n['end'] - n['start'] for n in ns))
    return grouped


# Fraction of each harmonic band held at full weight before the taper starts.
# 1.0 is a binary mask (hard edges ring); 0.0 a pure raised cosine. Measured on
# a vibrato-carrying harmonic series, the cosine kept only 25% of the energy a
# binary mask does and 0.5 only 39% - sung notes drift off the nominal harmonic,
# so a centre-weighted window throws away exactly the energy that matters, which
# both starved the converted harmony and pushed real lines under the
# concentration gate. 0.95 keeps ~100% while still softening the edge.
_BAND_FLAT_FRAC = 0.95

# Bounds on the per-line level match (+/- 12 dB). A converted harmony line is
# placed at the level of the extract it replaced; that ratio should sit near
# 1.0, so a wild value means the measurement is untrustworthy rather than that
# the line needs that much gain. Left unbounded it amplified a quietly-rendered
# line's artifacts into distortion, and the resulting stem-RMS inflation made
# _finish_backing duck every other line - heard as uneven background singers.
_LINE_GAIN_MIN = 0.25
_LINE_GAIN_MAX = 4.0


def _extract_line_audios(stack: np.ndarray, sample_rate: int, lines,
                         n_harmonics: int = 24, width_cents: float = 50.0,
                         onset_s: float = 0.03, onset_weight: float = 0.5,
                         onset_max_hz: float = 8000.0):
    """Isolate every harmony line from a stack at once, sharing one STFT.

    Replaces a per-line 10-harmonic ±40-cent BINARY comb. That comb produced
    both symptoms operators reported on converted harmonies — they landed too
    quiet, and their individual notes did not articulate:

    * **Bandwidth.** 10 harmonics discards everything above ~10·f0 — about
      2.2 kHz for a 220 Hz line — so the presence and air that make a voice
      audible in a mix were dropped, and (because the caller levels the
      converted line against this extract's RMS) the harmony was then placed
      at the level of that thin slice rather than the singer's true level.
    * **Attacks.** Note onsets are transient and broadband; a harmonic comb
      structurally cannot pass them. Without attacks a converted line reads as
      one legato smear instead of separate notes. A short full-band window at
      each onset restores them, weighted below 1.0 so a chord's shared attack
      is not counted at full strength in every line.
    * **Overlap.** The caller subtracts each extract from the stack in turn.
      Narrow combs rarely collided, but wider masks routinely claim the same
      bin in two lines, which would subtract that energy once per line and
      punch holes in the residual. The masks are scaled down wherever they
      overlap so they form a partition (``sum(masks) <= 1`` per bin), which
      keeps ``sum(extracts) <= stack``.

    Band edges are raised-cosine rather than binary; a hard mask edge rings.

    Returns ``[(mix_extract, gate_extract), ...]``, one pair per entry in
    ``lines`` (same order):

    * ``mix_extract`` uses the partitioned mask. This is what the caller
      subtracts from the stack and level-matches against, so shared energy
      leaves the stack exactly once.
    * ``gate_extract`` uses the line's own unpartitioned mask - its full claim.
      The caller's gates ask "is this a real harmonic line?", which is a
      property of the line itself, not of how many neighbours happen to
      contest its bins. Gating on the partitioned share instead makes a line's
      admissibility depend on its neighbours: measured on a real stack, two of
      three lines that previously converted fell to concentration 0.13 against
      a 0.15 threshold purely because partitioning had split their shared
      upper harmonics.
    """
    import librosa

    if not lines:
        return []

    n_fft, hop = 2048, 512
    S = librosa.stft(np.asarray(stack, dtype=np.float32), n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sample_rate, hop_length=hop)
    onset_band = freqs <= onset_max_hz

    masks = []
    for line_notes in lines:
        mask = np.zeros(S.shape, dtype=np.float32)
        for note in line_notes:
            f0 = 440.0 * 2.0 ** ((float(note['pitch_midi']) - 69.0) / 12.0)
            fsel = (times >= note['start']) & (times <= note['end'])
            if fsel.any():
                weights = np.zeros(freqs.shape, dtype=np.float32)
                for k in range(1, n_harmonics + 1):
                    fk = k * f0
                    if fk >= sample_rate / 2:
                        break
                    half = fk * (2.0 ** (width_cents / 1200.0) - 1.0)
                    if half <= 0.0:
                        continue
                    bsel = (freqs >= fk - half) & (freqs <= fk + half)
                    if not bsel.any():
                        continue
                    # Flat-top (Tukey), not a pure raised cosine: hold 1.0
                    # across the inner half of the band and taper only the
                    # outer half. A pure cosine peaks at the nominal harmonic
                    # and falls away either side, but sung notes carry vibrato
                    # and drift, so their energy sits OFF centre and was being
                    # attenuated - measured as lower captured energy than the
                    # binary mask this replaced, which pushed real harmony
                    # lines under the concentration gate. The taper still
                    # avoids the ringing a hard edge causes.
                    offset = np.abs(np.clip((freqs[bsel] - fk) / half, -1.0, 1.0))
                    taper = np.where(
                        offset <= _BAND_FLAT_FRAC,
                        1.0,
                        0.5 * (1.0 + np.cos(np.pi * np.clip(
                            (offset - _BAND_FLAT_FRAC)
                            / max(1.0 - _BAND_FLAT_FRAC, 1e-6),
                            0.0, 1.0))),
                    ).astype(np.float32)
                    weights[bsel] = np.maximum(weights[bsel], taper)
                mask[:, fsel] = np.maximum(mask[:, fsel], weights[:, None])
            if onset_weight > 0.0 and onset_s > 0.0:
                osel = (times >= note['start']) & (times < note['start'] + onset_s)
                if osel.any():
                    block = mask[np.ix_(onset_band, osel)]
                    mask[np.ix_(onset_band, osel)] = np.maximum(block, onset_weight)
        masks.append(mask)

    # Partition: where several lines claim a bin, share it rather than letting
    # each take the full magnitude (see "Overlap" above).
    overlap = np.sum(masks, axis=0)
    share = 1.0 / np.maximum(overlap, 1.0)

    def _istft(mask):
        return np.asarray(
            librosa.istft(S * mask, hop_length=hop, length=len(stack)),
            dtype=np.float32)

    def _occupancy(mask):
        """Fraction of the spectrogram this mask passes, weighted by its own
        gain — i.e. the share of a FLAT-spectrum input it would capture.

        The caller's "is this a real harmonic line?" test compares captured
        energy against the stack's. That raw fraction scales with how wide the
        comb is, so it says as much about the mask as about the audio: widening
        the comb from 10 to 24 harmonics raised what pure noise scores from
        0.168 to 0.512 against a 0.15 threshold, admitting noise as a harmony.
        Dividing by this makes the test an ENRICHMENT over chance, which is
        invariant to mask width: noise scores ~1 whatever the comb, a real line
        scores far higher because its energy sits exactly where the comb looks.
        """
        active = mask.sum(axis=0) > 0
        if not active.any():
            return 0.0
        return float(np.mean(mask[:, active] ** 2))

    return [(_istft(mask * share), _istft(mask), _occupancy(mask))
            for mask in masks]


# Preset configurations
PRESETS = {
    'draft': {'n_steps': 10, 'denoise': 0.3},
    'fast': {'n_steps': 20, 'denoise': 0.5},
    'balanced': {'n_steps': 50, 'denoise': 0.7},
    'high': {'n_steps': 100, 'denoise': 0.8},
    'studio': {'n_steps': 200, 'denoise': 0.9},
}


def _detect_bandwidth_hz(audio: np.ndarray, sr: int, floor_db: float = 40.0) -> float:
    """Highest frequency in ``audio`` that still carries real content.

    Separated stems are routinely sourced from lossy encodes and brick-wall
    well below Nyquist. The fork decoder is trained on full-band vocals and
    happily synthesises an octave above that wall: measured on this song the
    render carried +25 dB more 16-22 kHz energy than the source, which has
    essentially nothing up there. That is invented content no input evidence
    supports, and the checkpoint the operator rated best carried the least of
    it.

    Returns Nyquist for a genuinely full-band input, so matching against this
    is a no-op on sources that were never band-limited.
    """
    import librosa

    audio = np.asarray(audio, dtype=np.float32).ravel()
    n_fft = 4096
    nyquist = sr * 0.5
    if audio.size < n_fft * 2:
        return nyquist
    spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=n_fft // 2))
    psd_db = 10.0 * np.log10(np.square(spec, dtype=np.float64).mean(axis=1) + 1e-20)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # Reference against the band where a voice always has energy rather than
    # the peak bin, so one resonance cannot drag the threshold around.
    ref_band = (freqs >= 300.0) & (freqs <= 4000.0)
    if not ref_band.any():
        return nyquist
    threshold = float(np.median(psd_db[ref_band])) - float(floor_db)
    # Band the spectrum before thresholding. Testing raw bins finds the last
    # bin above the floor, and a single stray bin next to Nyquist then reports
    # a full-band signal for audio that visibly walls 6 kHz lower.
    band_hz = 500.0
    edges = np.arange(0.0, nyquist, band_hz)
    cutoff = 0.0
    for lo in edges:
        in_band = (freqs >= lo) & (freqs < lo + band_hz)
        if in_band.any() and float(np.median(psd_db[in_band])) > threshold:
            cutoff = min(lo + band_hz, nyquist)
    return float(cutoff) if cutoff > 0.0 else nyquist


def _lowpass_to(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Zero-phase lowpass at ``cutoff_hz``. Returns the input unchanged if the
    cutoff is at or above Nyquist, or if the filter cannot be built."""
    from scipy.signal import butter, sosfiltfilt

    audio = np.asarray(audio, dtype=np.float32)
    nyquist = sr * 0.5
    if not (0.0 < cutoff_hz < nyquist):
        return audio
    # sosfiltfilt needs more samples than the filter's padding length.
    if audio.shape[-1] < 64:
        return audio
    # Order 20: at order 10 the transition band still sat above the detector's
    # own floor, so a signal filtered at C re-measured as C+1000 Hz and 3 dB
    # more of the invented octave survived. Order 20 halves that to the 500 Hz
    # banding resolution and is numerically stable in SOS form.
    sos = butter(20, cutoff_hz / nyquist, btype="low", output="sos")
    return sosfiltfilt(sos, audio.astype(np.float64), axis=-1).astype(np.float32)


class SingingConversionPipeline:
    """Main voice conversion pipeline for singing audio."""

    def __init__(self, device=None, config: Optional[Dict] = None, voice_cloner=None):
        """Initialize the singing voice conversion pipeline.

        Args:
            device: PyTorch device (cpu/cuda). Auto-detects if None.
            config: Configuration dict with model paths and settings.
                Expected keys: hubert_path, vocoder_path, vocoder_type,
                encoder_backend, encoder_type, conformer_config,
                voice_model_path, speaker_id.
            voice_cloner: Optional VoiceCloner instance for profile loading.
        """
        import torch
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        self._separator = None
        self._diarizer = None
        self._voice_cloner = voice_cloner
        self._sample_rate = 22050
        # Separation and the final mix run at this rate (Demucs is natively
        # 44.1k); only the vocal-conversion chain drops to _sample_rate.
        self._output_sample_rate = int(self.config.get('output_sample_rate', 44100))
        self._model_manager = None
        self._nsf_enhancer = None
        self._pupu_vocoder = None
        self._hq_enhancer = None

        logger.info(f"SingingConversionPipeline initialized on {self.device}")

    def _get_separator(self):
        """Lazy-load vocal separator on first use.

        Returns:
            VocalSeparator instance configured for this device.
        """
        if self._separator is None:
            from ..audio.separation import VocalSeparator
            self._separator = VocalSeparator(
                device=self.device,
                model_name=self.config.get('separation_model', 'htdemucs'),
            )
            logger.info("Vocal separator loaded")
        return self._separator

    def _separate_vocals(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Separate vocals from instrumental using Demucs model.

        Args:
            audio: Input audio signal (mono or stereo)
            sr: Sample rate of the audio

        Returns:
            Dict with 'vocals' and 'instrumental' keys, each containing
            separated audio arrays at the same sample rate.

        Raises:
            SeparationError: If vocal separation fails
        """
        separator = self._get_separator()
        try:
            return separator.separate(audio, sr)
        except Exception as e:
            raise SeparationError(f"Vocal separation failed: {e}")

    def _get_diarizer(self):
        """Lazy-load and cache the speaker diarizer (mirrors _get_separator)."""
        if self._diarizer is None:
            from ..audio.speaker_diarization import SpeakerDiarizer
            self._diarizer = SpeakerDiarizer(device=str(self.device))
            logger.info("Speaker diarizer loaded")
        return self._diarizer

    def _select_speaker_spans(self, result, voc_mono, sr, preserve=None):
        """Partition diarization segments into lead vs backing spans.

        ``preserve`` lists diarization cluster ids whose voice is already the
        target (e.g. the target artist featuring on the song): they are
        excluded from primary selection, never merged into the converted lead,
        and their spans ride along as kept-original backing. Identity cannot
        be auto-detected on singing (SV-embedding negative result 2026-07-04),
        so this is an explicit per-request operator choice.

        The primary speaker (most total speaking time) is the lead. Each other
        cluster is routed by *convertibility*, not identity (calibration
        2026-07-04: speaker-verification embeddings cannot do identity on sung
        audio — different-person sims 0.855/0.900 exceeded most same-person
        pairs): a cluster whose active audio is clean monophonic melody
        (pyin voiced fraction >= ``multi_speaker_merge_voiced_min``) merges
        into the lead so it gets converted — leaving it original would poke
        through as the source voice; low-voiced clusters are harmony stacks /
        textures the fork engine would butcher (it needs clean F0), so they
        stay original as backing. Calibrated margin: leads 0.76-0.87 vs
        keep-cases 0.14-0.51, threshold 0.65.

        Non-primary segments shorter than ``multi_speaker_min_segment_s``
        (default 2s) are reassigned to the lead (blip policy). Lead segments
        are never dropped — that would punch holes in the converted vocal.

        Returns:
            ``(primary_spans, backing_spans, info)`` or ``None`` when the
            multi-speaker path should not run (<=1 speaker, or no backing
            segment survives the gates).
        """
        if result.num_speakers <= 1:
            logger.info("Multi-speaker path: only %d speaker(s), using single-stem",
                        result.num_speakers)
            return None

        # Sorted so duration ties break deterministically (get_all_speaker_ids
        # returns set order).
        speakers = sorted(result.get_all_speaker_ids())

        # Preserve tokens are cluster ids ("SPEAKER_02") or time ranges
        # ("1:23-1:40" / "83.5-100"). Ranges resolve against THIS run's
        # clustering (labels are not stable run-to-run), picking the cluster
        # with the most singing time inside the range.
        def _parse_ts(tok):
            parts = tok.strip().split(':')
            try:
                if len(parts) == 2:
                    return float(parts[0]) * 60.0 + float(parts[1])
                return float(tok)
            except ValueError:
                return None

        preserved = set()
        for item in (preserve or []):
            item = str(item).strip()
            if not item:
                continue
            if item in speakers:
                preserved.add(item)
                continue
            lo_tok, sep, hi_tok = item.partition('-')
            lo_s = _parse_ts(lo_tok) if sep else None
            hi_s = _parse_ts(hi_tok) if sep else None
            if lo_s is None or hi_s is None or hi_s <= lo_s:
                logger.warning("Preserve token %r matches no cluster and is not "
                               "a valid time range; ignored", item)
                continue
            best, best_ov = None, 0.0
            for spk in speakers:
                ov = sum(max(0.0, min(seg.end, hi_s) - max(seg.start, lo_s))
                         for seg in result.get_speaker_segments(spk))
                if ov > best_ov:
                    best, best_ov = spk, ov
            if best is not None:
                logger.info("Preserve range %s -> cluster %s (%.1fs overlap)",
                            item, best, best_ov)
                preserved.add(best)
            else:
                logger.warning("Preserve range %s overlaps no diarized singing; "
                               "ignored", item)
        candidates = [s for s in speakers if s not in preserved]
        if not candidates:
            logger.info(
                "Multi-speaker path: all %d cluster(s) preserved, nothing to "
                "convert; using single-stem", len(speakers))
            return None
        primary = max(candidates, key=result.get_speaker_total_duration)
        min_seg = float(self.config.get('multi_speaker_min_segment_s', 2.0))
        merge_voiced = float(self.config.get('multi_speaker_merge_voiced_min', 0.65))

        # Convertibility role per non-primary cluster. Clusters with <1.5s of
        # active audio skip measurement (pyin is unreliable there); the blip
        # policy below covers them.
        roles, voiced = {}, {}
        merged_speakers = set()
        for spk in speakers:
            if spk == primary:
                continue
            if spk in preserved:
                roles[spk] = 'preserved'
                continue
            spans = [(s.start, s.end) for s in result.get_speaker_segments(spk)]
            active = _active_audio(voc_mono, sr, spans)
            if len(active) < int(1.5 * sr):
                continue
            vf = _voiced_fraction(active, sr)
            voiced[spk] = round(vf, 3)
            if vf >= merge_voiced:
                roles[spk] = 'lead_merge'
                merged_speakers.add(spk)
            else:
                roles[spk] = 'backing'

        primary_spans, backing_spans, preserved_spans = [], [], []
        reassigned = 0
        for seg in result.segments:
            span = (seg.start, seg.end)
            if seg.speaker_id in preserved:
                # Preserved voices never blip-reassign into the converted lead.
                preserved_spans.append(span)
            elif seg.speaker_id == primary or seg.speaker_id in merged_speakers:
                primary_spans.append(span)
            elif seg.duration < min_seg:
                primary_spans.append(span)
                reassigned += 1
            else:
                backing_spans.append(span)

        # Clip backing spans against the lead's spans: overlapping regions must
        # not land in both tracks, or that audio plays twice in the output
        # (converted lead + original echo). The lead wins the overlap — it gets
        # converted. Post-clip slivers < min_seg follow the blip policy (their
        # remainder joins the lead; the region itself is already lead-covered).
        primary_union: list = []
        for start, end in sorted(primary_spans):
            if primary_union and start <= primary_union[-1][1]:
                primary_union[-1] = (primary_union[-1][0],
                                     max(primary_union[-1][1], end))
            else:
                primary_union.append((start, end))
        def _clip_against_lead(spans):
            clipped = []
            for start, end in spans:
                for u_start, u_end in primary_union:
                    if u_end <= start:
                        continue
                    if u_start >= end:
                        break
                    if u_start > start:
                        clipped.append((start, u_start))
                    start = max(start, u_end)
                    if start >= end:
                        break
                if start < end:
                    clipped.append((start, end))
            return clipped

        kept = []
        for start, end in _clip_against_lead(backing_spans):
            if end - start >= min_seg:
                kept.append((start, end))
            else:
                primary_spans.append((start, end))
                reassigned += 1
        # Preserved slivers stay original — reassigning them would convert an
        # already-target voice.
        backing_spans = kept + _clip_against_lead(preserved_spans)

        if not backing_spans:
            logger.info(
                "Multi-speaker path: no backing survives (%d cluster(s) merged "
                "into lead as convertible, %d blips reassigned), using single-stem",
                len(merged_speakers), reassigned)
            return None

        info = {
            'separator': 'diarization',
            'num_speakers': result.num_speakers,
            'primary_speaker': primary,
            'backing_s': round(sum(e - s for s, e in backing_spans), 1),
            'reassigned_blips': reassigned,
            'roles': roles,
            'voiced': voiced,
        }
        if preserved:
            info['preserved_speakers'] = sorted(preserved)
            info['preserved_s'] = round(sum(e - s for s, e in preserved_spans), 1)
        return primary_spans, backing_spans, info

    def _multi_speaker_enabled(self, override: Optional[bool] = None) -> bool:
        """Whether the per-speaker conversion path is enabled.

        An explicit ``override`` (from the request/job settings) wins over
        everything; ``None`` (the default) keeps the config-then-env resolution.
        Config ``enable_multi_speaker_conversion`` wins when set (testable);
        otherwise the ``ENABLE_MULTI_SPEAKER_CONVERSION`` env var flips it on for
        ops. Off by default: the single-stem path stays the production behaviour.
        """
        if override is not None:
            return bool(override)
        cfg = self.config.get('enable_multi_speaker_conversion')
        if cfg is not None:
            return bool(cfg)
        return os.environ.get('ENABLE_MULTI_SPEAKER_CONVERSION', '').strip().lower() in (
            '1', 'true', 'yes', 'on')

    def _split_lead_backing_karaoke(self, voc_mono, sr):
        """Split lead/backing with a karaoke separation model (Mel-RoFormer).

        Pre-stage to the span path: separates SIMULTANEOUS voices, so
        harmony doubles stacked on the lead land in the backing stem instead
        of contaminating the lead. Stacked echo-answer phrases stay in the
        lead stem (calibrated 2026-07-05) — the span machinery downstream
        handles those. Gates (all return ``None`` -> caller runs the span
        path on the raw vocal):
        - bridge/env unavailable or separation fails;
        - negligible backing energy (< ``multi_speaker_min_backing_ratio`` of
          the vocal energy) — effectively a solo;
        - the backing stem looks like clean lead (voiced fraction >= the merge
          threshold): the known failure mode of karaoke models is retaining
          lead in the backing stem, which would poke through unconverted.

        Returns ``(lead_track, backing, info)`` or ``None``.
        """
        import librosa
        from . import separation_bridge

        if not separation_bridge.is_available():
            logger.info("Karaoke separator configured but uvr bridge unavailable; "
                        "using diarization spans")
            return None
        try:
            lead, backing = separation_bridge.separate_lead_backing(
                voc_mono, sr,
                data_dir=str(self.config.get('data_dir', 'data')),
                model=self.config.get('multi_speaker_karaoke_model'),
            )
        except Exception as e:
            logger.warning("Karaoke separation failed (%s); using diarization spans", e)
            return None

        voc_energy = float(np.sum(np.square(voc_mono, dtype=np.float64))) + 1e-12
        backing_ratio = float(
            np.sum(np.square(backing, dtype=np.float64)) / voc_energy)
        min_ratio = float(self.config.get('multi_speaker_min_backing_ratio', 0.01))
        if backing_ratio < min_ratio:
            logger.info(
                "Karaoke separation: negligible backing energy (%.4f < %.3f); "
                "treating as solo", backing_ratio, min_ratio)
            return None

        intervals = librosa.effects.split(backing, top_db=40)
        active = (np.concatenate([backing[s:e] for s, e in intervals])
                  if len(intervals) else np.zeros(0, dtype=np.float32))
        backing_s = float(sum(int(e) - int(s) for s, e in intervals)) / sr
        vf = _voiced_fraction(active, sr) if len(active) >= int(1.5 * sr) else 0.0
        # Dedicated leak-guard knob: the merge threshold is the default, but
        # solo covers with strongly-voiced self-harmony doubles legitimately
        # measure in the clean-lead band — raising this accepts the karaoke
        # split anyway (the backing then goes through the normal line gates).
        leak_voiced = float(self.config.get(
            'multi_speaker_karaoke_leak_voiced_min',
            self.config.get('multi_speaker_merge_voiced_min', 0.65)))
        if vf >= leak_voiced:
            logger.info(
                "Karaoke separation: backing stem looks like clean lead "
                "(voiced %.2f >= %.2f); using diarization spans", vf, leak_voiced)
            return None

        info = {
            'separator': 'karaoke_model',
            'backing_s': round(backing_s, 1),
            'backing_energy_ratio': round(backing_ratio, 4),
            'roles': {'backing_stem': 'backing'},
            'voiced': {'backing_stem': round(vf, 3)},
        }
        return lead, backing, info

    def _suppress_lead_bleed(self, backing, lead, sr):
        """Cancel lead leakage from a karaoke backing stem, using coherence.

        The separator estimates only the lead; the backing stem the bridge
        reads is computed as ``orig_mix - lead`` (an exact arithmetic residual
        inside the separator). So every dB of lead the model under-estimates
        lands in the backing at 1:1, on the same sample grid and *phase
        coherent* with the lead stem we already hold.

        That is what makes this tractable. Lead and harmony sing consonant
        intervals, so their harmonic series genuinely collide and no
        frequency-domain rule can separate them - masking by pitch would gut
        the harmony along with the bleed. Coherence is a different axis: a real
        backing singer, even in exact unison, has independent vibrato, phase
        and micro-timing, so its cross-spectrum with the lead averages toward
        zero. Leakage is the same waveform, so its coherence approaches one.
        Discriminating on coherence rather than frequency makes consonant
        collisions harmless.

        Per bin: fit a complex least-squares leakage coefficient ``h`` from
        lead to backing, then subtract ``gamma^2 * h * L``. The fit uses only
        "calibration" frames - lead loud, backing quietest relative to it -
        because fitting over all frames would let least squares partially fit
        the *harmony* (at a consonant interval the colliding partials' phase
        difference rotates at the detuning beat rate, which does not average
        out over a few seconds). The coherence factor is the second defence: in
        bands where the backing is mostly an independent voice, gamma^2 is
        small and almost nothing is subtracted.

        Returns ``(cleaned_backing, info)``. Never raises - on any failure the
        input is returned unchanged with the reason in ``info``.
        """
        import librosa

        info = {'mode': 'off', 'cancelled_db': 0.0, 'preserved_db': 0.0,
                'mean_coherence': 0.0, 'cal_frames': 0}
        mode = str(self.config.get('multi_speaker_bleed_suppression', 'off')).lower()
        info['mode'] = mode
        if mode in ('off', 'none', 'false') or lead is None or not len(backing):
            return backing, info

        max_db = float(self.config.get('multi_speaker_bleed_max_db', 12.0))
        h_max = float(self.config.get('multi_speaker_bleed_h_max', 0.7))
        cal_pct = float(self.config.get('multi_speaker_bleed_cal_pct', 40.0))

        try:
            n_fft, hop = 2048, 512
            n = min(len(backing), len(lead))
            X = librosa.stft(np.asarray(backing[:n], dtype=np.float32),
                             n_fft=n_fft, hop_length=hop)
            L = librosa.stft(np.asarray(lead[:n], dtype=np.float32),
                             n_fft=n_fft, hop_length=hop)
            t = min(X.shape[1], L.shape[1])
            X, L = X[:, :t], L[:, :t]

            lead_pow = np.sum(np.abs(L) ** 2, axis=0)
            back_pow = np.sum(np.abs(X) ** 2, axis=0)
            if not np.any(lead_pow > 0):
                info['mode'] = 'skipped:silent_lead'
                return backing, info
            # Lead genuinely present, not just noise floor.
            active = lead_pow > 1e-4 * np.median(
                lead_pow[lead_pow >= np.percentile(lead_pow, 90)])
            if active.sum() < 8:
                info['mode'] = 'skipped:too_short'
                return backing, info
            ratio = back_pow[active] / (lead_pow[active] + 1e-20)
            thresh = np.percentile(ratio, cal_pct)
            cal = np.zeros(t, dtype=bool)
            cal[active] = ratio <= thresh
            if cal.sum() < 4:
                info['mode'] = 'skipped:no_calibration_frames'
                return backing, info
            info['cal_frames'] = int(cal.sum())

            Lc, Xc = L[:, cal], X[:, cal]
            Sxx = np.sum(np.abs(Lc) ** 2, axis=1)
            Syy = np.sum(np.abs(Xc) ** 2, axis=1)
            Sxy = np.sum(Xc * np.conj(Lc), axis=1)
            h = Sxy / (Sxx + 1e-20)
            coh = (np.abs(Sxy) ** 2) / (Sxx * Syy + 1e-20)
            coh = np.clip(coh, 0.0, 1.0)
            info['mean_coherence'] = float(np.mean(coh))

            mag = np.abs(h)
            h = np.where(mag > h_max, h * (h_max / (mag + 1e-20)), h)
            d = (coh * h)[:, None] * L
            # Never remove more than max_db from any single bin.
            cap = 1.0 - 10.0 ** (-max_db / 20.0)
            dmag = np.abs(d)
            scale = np.minimum(1.0, cap * np.abs(X) / (dmag + 1e-20))
            d = d * scale
            X_clean = X - d

            lead_silent = ~active
            def _e(spec, sel):
                return float(np.sum(np.abs(spec[:, sel]) ** 2)) if sel.any() else 0.0
            e_before, e_after = _e(X, active), _e(X_clean, active)
            info['cancelled_db'] = round(
                10.0 * np.log10((e_before + 1e-20) / (e_after + 1e-20)), 2)
            pb, pa = _e(X, lead_silent), _e(X_clean, lead_silent)
            info['preserved_db'] = round(
                10.0 * np.log10((pb + 1e-20) / (pa + 1e-20)), 2)

            cleaned = librosa.istft(X_clean, hop_length=hop, length=len(backing))
            logger.info(
                "Lead-bleed cancel: %.2f dB removed while the lead is active, "
                "%.2f dB touched while it is silent (want ~0), mean coherence "
                "%.2f over %d calibration frames",
                info['cancelled_db'], info['preserved_db'],
                info['mean_coherence'], info['cal_frames'])
            return np.asarray(cleaned, dtype=np.float32), info
        except Exception as e:
            logger.warning("Lead-bleed cancel failed (%s); backing unchanged", e)
            info['mode'] = f'failed:{type(e).__name__}'
            return backing, info

    def _split_lead_unison(self, backing, sr, lead):
        """Move voices singing in UNISON with the lead out of the backing stack.

        A karaoke separator's job is to pull simultaneous voices apart, and it
        does that faithfully — including for a lead that was double-tracked.
        The double is a real, energetic voice (measured at 27.5% of the backing
        stem on the reference song) sitting within a quarter-tone of the lead,
        so every downstream gate accepts it as a harmony line and converts it
        as its own singer.

        That is the bug behind "the lead and the background singers bleed into
        each other": the same phrase gets converted TWICE, independently, and
        the two results are summed. Conversion is stochastic and phase-blind,
        so the copies do not line up — the pair reads as a smeared, chorused
        lead rather than as one voice plus a harmony.

        Nothing here was ever wrong about the audio; the pipeline simply never
        compared the detected lines against the lead's own pitch, despite the
        lead being available in the same call. This routes unison lines back to
        the lead so they convert together, as one coherent signal.

        Returns ``(backing_without_unison, unison_audio, info)``. ``info`` is
        always populated so the decision is visible in conversion metadata
        rather than only in the log.
        """
        from . import separation_bridge

        info = {'unison_lines': 0, 'harmony_lines': 0}
        empty = np.zeros_like(backing)
        if lead is None or not len(backing):
            return backing, empty, info

        max_semitones = float(self.config.get(
            'multi_speaker_unison_semitones', 1.0))
        min_frac = float(self.config.get('multi_speaker_unison_note_frac', 0.5))
        if max_semitones <= 0.0:
            return backing, empty, info

        try:
            notes = separation_bridge.polyphonic_notes(backing, sr)
        except Exception as e:
            logger.warning("Unison split: basic-pitch unavailable (%s); "
                           "leaving the backing stack intact", e)
            return backing, empty, info

        lines = _group_notes_into_lines(notes)
        if not lines:
            return backing, empty, info

        lead_f0 = self._extract_pitch(lead, sr)
        if lead_f0 is None or not len(lead_f0):
            return backing, empty, info
        lead_t = np.arange(len(lead_f0)) * (len(lead) / sr) / max(len(lead_f0), 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            lead_midi = np.where(np.asarray(lead_f0) > 0,
                                 69.0 + 12.0 * np.log2(np.maximum(lead_f0, 1e-6) / 440.0),
                                 np.nan)

        unison_idx = []
        for i, line_notes in enumerate(lines):
            diffs = []
            for note in line_notes:
                sel = (lead_t >= note['start']) & (lead_t <= note['end'])
                seg = lead_midi[sel]
                seg = seg[np.isfinite(seg)]
                if len(seg):
                    diffs.append(abs(float(np.median(seg)) - float(note['pitch_midi'])))
            if not diffs:
                continue
            d = np.asarray(diffs)
            frac = float(np.mean(d <= max_semitones))
            median = float(np.median(d))
            # Two independent signals must agree, and the decision is
            # deliberately biased AGAINST folding.
            #
            # The errors are not symmetric. Folding a genuine harmony merges a
            # distinct voice into the lead and destroys it - audible damage.
            # Failing to fold a real double just converts it separately, which
            # is the behaviour that existed before any of this, so the cost is
            # only that the artifact remains. Ambiguity should therefore resolve
            # to "harmony".
            #
            # Requiring both also fixes a real fragility: on one song the SAME
            # content measured 49%, 51% and 53% across three runs against a 50%
            # cutoff, folding on some runs and not others. A note-fraction near
            # its threshold is a coin flip; a median pitch distance is a
            # different statistic, so both landing on "unison" is far steadier
            # than either alone.
            is_unison = frac >= min_frac and median <= max_semitones
            if is_unison:
                unison_idx.append(i)
                logger.info(
                    "Backing line %d sings in unison with the lead (%.0f%% of "
                    "notes within %.2f semitones, median %.2f); folding it into "
                    "the lead so the pair converts once instead of twice",
                    i, 100.0 * frac, max_semitones, median)
            else:
                logger.info(
                    "Backing line %d is a genuine harmony (%.0f%% of notes "
                    "within %.2f semitones of the lead, median %.2f)",
                    i, 100.0 * frac, max_semitones, median)

        info['harmony_lines'] = len(lines) - len(unison_idx)
        info['unison_lines'] = len(unison_idx)
        if not unison_idx:
            return backing, empty, info

        # Extract every line together so the masks share contested bins, then
        # take only the unison ones: what leaves the backing is exactly what
        # joins the lead, so no energy is duplicated or lost.
        extracts = _extract_line_audios(
            backing, sr, lines,
            n_harmonics=int(self.config.get('multi_speaker_line_harmonics', 24)),
            onset_s=float(self.config.get('multi_speaker_line_onset_ms', 30.0)) / 1000.0,
        )
        unison = np.zeros_like(backing)
        for i in unison_idx:
            mix = extracts[i][0]
            n = min(len(mix), len(unison))
            unison[:n] += mix[:n]
        reduced = backing.astype(np.float32).copy()
        n = min(len(unison), len(reduced))
        reduced[:n] -= unison[:n]
        return reduced, unison, info

    def _convert_backing_stack(self, backing, sr, target_profile_id, mm):
        """Convert a polyphonic backing stack to the target voice, per line.

        The mono-F0 fork engine butchers polyphony, so: basic-pitch note
        events (via the uvr bridge) -> group into <=3 monophonic voice lines
        -> isolate each line with a harmonic comb mask -> convert lines whose
        extracted audio is clean melody (voiced fraction >= the calibrated
        merge threshold and non-negligible energy) -> subtract the converted
        lines' extracts from the stack and add the converted versions
        (RMS-matched per line), so breaths/consonants/unmatched content stay
        original.

        Returns ``(new_backing, harmony_info)``; on any failure or when no
        line clears the gates, returns the original stack with mode 'kept'.
        """
        from . import separation_bridge

        # Backing lines get their own gate: comb extracts measure voicing
        # slightly below cluster audio (HB's real harmony lines straddled the
        # 0.65 merge threshold at 0.60-0.66), so the default sits a notch under
        # the merge gate while staying inside the calibrated 0.51-0.76 margin.
        merge_voiced = float(self.config.get(
            'multi_speaker_backing_voiced_min',
            self.config.get('multi_speaker_merge_voiced_min', 0.65)))
        kept = {'mode': 'kept', 'lines_detected': 0, 'lines_converted': 0}

        import librosa
        intervals = librosa.effects.split(backing, top_db=40)
        active = (np.concatenate([backing[s:e] for s, e in intervals])
                  if len(intervals) else np.zeros(0, dtype=np.float32))
        stem_vf = _voiced_fraction(active, sr) if len(active) >= int(1.5 * sr) else 0.0
        whole_min = float(self.config.get(
            'multi_speaker_backing_whole_voiced_min', 0.7))

        if stem_vf >= whole_min:
            # Near-monophonic backing — a single doubled voice rather than a
            # polyphonic stack — converts cleanly as one piece, and whole-stem
            # conversion leaves NO original residual between notes (per-line
            # comb swaps keep unmatched content original, which audibly
            # speckles the source voice through doubles-heavy tracks).
            converted = np.asarray(
                mm.infer(backing, target_profile_id,
                         np.zeros(256, dtype=np.float32), sr),
                dtype=np.float32)
            new_backing = backing.astype(np.float32).copy()
            n = min(len(converted), len(new_backing))
            new_backing[:n] = converted[:n]
            lines_detected = lines_converted = 1
            method = 'whole_stem'
            logger.info("Backing conversion: near-monophonic stem "
                        "(voiced %.2f >= %.2f); converted whole", stem_vf, whole_min)
            return self._finish_backing(backing, new_backing, intervals,
                                        lines_detected, lines_converted, method)

        try:
            notes = separation_bridge.polyphonic_notes(backing, sr)
        except Exception as e:
            logger.warning("Backing conversion: basic-pitch unavailable (%s); "
                           "keeping backing original", e)
            return backing, kept

        lines = _group_notes_into_lines(notes)
        kept['lines_detected'] = len(lines)
        if not lines:
            logger.info("Backing conversion: no harmony lines detected; keeping original")
            return backing, kept

        stack_rms = float(np.sqrt(np.mean(np.square(backing, dtype=np.float64))) + 1e-12)
        new_backing = backing.astype(np.float32).copy()
        converted_count = 0
        # Share of the stack's in-span energy a line must claim to count as a
        # real harmony rather than comb-filtered noise. Exposed because it is
        # sensitive to how the masks are built: it silently rejected genuine
        # lines twice while the extractor was being changed underneath it.
        # Enrichment over chance (see _extract_line_audios._occupancy), not a
        # raw fraction: 1.0 is "no better than noise", so the default demands a
        # line be meaningfully more concentrated than that. Width-invariant, so
        # changing the comb no longer silently re-tunes this gate.
        concentration_min = float(self.config.get(
            'multi_speaker_line_concentration_min', 1.2))
        # Extracted for every line in one pass: the masks have to see each
        # other to share the bins they both claim (see _extract_line_audios).
        extracts = _extract_line_audios(
            backing, sr, lines,
            n_harmonics=int(self.config.get('multi_speaker_line_harmonics', 24)),
            onset_s=float(self.config.get('multi_speaker_line_onset_ms', 30.0)) / 1000.0,
        )
        for i, line_notes in enumerate(lines):
            # mix_extract is this line's partitioned share (what actually
            # leaves the stack); gate_extract is its full unpartitioned claim,
            # which is what the gates below must judge - see
            # _extract_line_audios for why the two must not be conflated.
            extract, gate_extract, occupancy = extracts[i]
            line_rms = float(np.sqrt(np.mean(np.square(gate_extract, dtype=np.float64))))
            if line_rms < 0.05 * stack_rms:
                logger.info("Backing line %d: negligible energy, skipped", i)
                continue
            # Concentration gate: a real harmonic line captures a large share
            # of the stack's energy inside its note spans; comb-filtering
            # noise/texture captures little (and pyin can't tell — a comb
            # mask MANUFACTURES pitch, so the voiced gate alone is blind here).
            spans = [(n['start'], n['end']) for n in line_notes]
            stack_active = _active_audio(backing, sr, spans)
            gate_active = _active_audio(gate_extract, sr, spans)
            extract_active = _active_audio(extract, sr, spans)
            stack_e = float(np.sum(np.square(stack_active, dtype=np.float64))) + 1e-12
            captured = float(np.sum(np.square(gate_active, dtype=np.float64)) / stack_e)
            # Enrichment over chance, not raw captured fraction: see
            # _extract_line_audios._occupancy. Noise lands near 1.0 regardless
            # of how wide the comb is; a real line sits well above it.
            concentration = captured / max(occupancy, 1e-6)
            if concentration < concentration_min:
                logger.info("Backing line %d: harmonic enrichment %.2f < %.2f "
                            "(no better concentrated than noise), kept original",
                            i, concentration, concentration_min)
                continue
            # Measure voicing on the span-active audio (calibration convention);
            # the full-length extract is mostly silence, which starves pyin's
            # 20s measurement window and would zero the gate for any song
            # whose backing starts late.
            vf = (_voiced_fraction(gate_active, sr)
                  if len(gate_active) >= int(1.5 * sr) else 0.0)
            if vf < merge_voiced:
                logger.info("Backing line %d: voiced %.2f < %.2f, kept original", i, vf, merge_voiced)
                continue
            # Convert the full claim, not the partitioned share: the engine
            # sings better from a complete line than from one with its shared
            # harmonics scooped out. Only the SUBTRACTION has to be partitioned.
            converted = np.asarray(
                mm.infer(gate_extract, target_profile_id, np.zeros(256, dtype=np.float32), sr),
                dtype=np.float32,
            )
            n = min(len(converted), len(new_backing))
            # Level-match over the line's OWN note spans, not the whole track.
            # extract is silent between notes but the converted signal is not
            # (the engine emits low-level output there), so a full-track
            # conv_rms is inflated by however much silence that line happens to
            # contain - and by a different amount per line. That made lines
            # with sparser notes come back quieter than dense ones, which is
            # audible as background singers at uneven volumes. Comparing both
            # sides over the same active spans removes the silence term.
            conv_active = _active_audio(converted[:n], sr, spans)
            if len(conv_active) and len(extract_active):
                ref_rms = float(np.sqrt(np.mean(np.square(extract_active, dtype=np.float64))))
                conv_rms = float(np.sqrt(np.mean(np.square(conv_active, dtype=np.float64))) + 1e-12)
            else:
                # ponytail: fall back to the old whole-track ratio when a line
                # has no measurable active audio; better a rough level than none.
                ref_rms = line_rms
                conv_rms = float(np.sqrt(np.mean(np.square(converted[:n], dtype=np.float64))) + 1e-12)
            gain = ref_rms / conv_rms
            # Bound the correction. This is a level MATCH, not an amplifier: the
            # engine's output level should already be in the right region, and a
            # ratio far from 1.0 means the measurement was unreliable, not that
            # the line genuinely needs 20 dB. Unbounded it did real damage in
            # both directions - a line the engine rendered quietly got its
            # artifacts amplified into audible distortion, and because that one
            # loud line inflated the whole stem's RMS, _finish_backing's
            # restore then scaled every OTHER line down, which is heard as
            # background singers at inconsistent volumes.
            clamped = float(np.clip(gain, _LINE_GAIN_MIN, _LINE_GAIN_MAX))
            if abs(clamped - gain) > 1e-9:
                logger.info(
                    "Backing line %d: level match wanted %.1fx (%.1f dB), "
                    "clamped to %.1fx - the extract and the converted line "
                    "disagree too much to trust", i, gain,
                    20.0 * np.log10(max(gain, 1e-9)), clamped)
            else:
                logger.info("Backing line %d: level match %.2fx", i, clamped)
            # Swap the line: remove the original extract, add the converted one.
            new_backing[:n] = new_backing[:n] - extract[:n] + converted[:n] * clamped
            converted_count += 1
            logger.info("Backing line %d: converted (voiced %.2f, rms %.4f)", i, vf, line_rms)

        if converted_count == 0:
            return backing, kept
        return self._finish_backing(backing, new_backing, intervals,
                                    len(lines), converted_count, 'lines')

    def _finish_backing(self, backing, new_backing, intervals,
                        lines_detected, lines_converted, method):
        """Level-match a converted backing stem and build its harmony info.

        Per-line RMS matching targets the comb-mask extract, which captures
        only part of each line's true level — converted harmonies landed
        audibly weak. Restore the stem's active loudness, then apply the
        operator taste knob.

        The peak guard runs BEFORE the gain, not after. Applied afterwards it
        silently cancelled the knob outright: normalising to a fixed ceiling
        makes ``X·g·(0.99/max|X·g|)`` collapse to ``X·(0.99/max|X|)``, with the
        gain dividing out exactly, so every value above the point where the
        stem hit the ceiling produced byte-identical audio (measured: 1.6, 2.2
        and 3.0 were indistinguishable). Guarding first keeps the runaway
        protection and lets the gain mean what it says; the caller's own mix
        guard is what ultimately prevents clipping.
        """
        if len(intervals):
            orig_act = np.concatenate([backing[s:e] for s, e in intervals])
            new_act = np.concatenate([new_backing[s:e] for s, e in intervals])
            orig_rms = float(np.sqrt(np.mean(np.square(orig_act, dtype=np.float64))))
            new_rms = float(np.sqrt(np.mean(np.square(new_act, dtype=np.float64))) + 1e-12)
            if orig_rms > 0.0:
                new_backing = new_backing * (orig_rms / new_rms)
        peak = float(np.abs(new_backing).max())
        if peak > 0.99:
            new_backing = new_backing * (0.99 / peak)
        gain = float(self.config.get('multi_speaker_backing_gain', 1.0))
        new_backing = new_backing * gain
        boosted = float(np.abs(new_backing).max())
        if boosted > 1.0:
            # Not silently swallowed: say so, since the mix guard downstream
            # will pull the whole mix down to accommodate this.
            logger.info(
                "Backing gain %.2f puts the backing stem at peak %.2f; the mix "
                "peak guard will scale the final mix to fit", gain, boosted)

        mode = 'converted' if lines_converted == lines_detected else 'partial'
        return new_backing, {'mode': mode, 'lines_detected': lines_detected,
                             'lines_converted': lines_converted, 'method': method}

    def _convert_multi_speaker(self, voc_mono, sr, target_profile_id, mm, pitch_shift,
                               convert_backing=None, preserve_speakers=None):
        """Convert the lead vocal per-speaker; keep backing vocals as original.

        Diarizes the mono vocal stem, converts the primary speaker (most total
        speaking time — the lead) through the same model as the single-stem path,
        and sums the untouched backing speakers back in. This avoids the muddiness
        of converting a mixed vocal as one voice (validated on Hotline Bling / Cry
        Me A River: 5-6x high-frequency energy).

        Gates (all route back to single-stem): <=1 speaker detected; no backing
        segment >= ``multi_speaker_min_segment_s``; lead+backing spans capture
        less than ``multi_speaker_min_coverage`` of the vocal energy (the
        diarizer's VAD missed content that would otherwise be silently dropped
        from the mix). The coverage gate runs before the expensive fork call.

        Returns ``(combined_vocal, info_dict)`` for the result metadata, or
        ``None`` to tell the caller to fall back to single-stem conversion
        (gate tripped, or any failure).
        """
        import librosa
        import soundfile as sf
        from ..audio.multi_artist_separator import build_speaker_track

        tmp = None
        try:
            # HYBRID pre-stage (opt-in): the karaoke model and diarization
            # spans catch ORTHOGONAL backing phenomena (calibrated 2026-07-05
            # on Hotline Bling): the model separates SIMULTANEOUS harmony
            # doubles from under the lead (spans structurally cannot), while
            # stacked echo-answer phrases land in its lead stem — exactly what
            # the span path already handles. So: strip the doubles first, run
            # the proven span machinery on the de-doubled lead, and carry the
            # doubles as extra backing.
            simul_backing = None
            karaoke_info = {}
            voc_for_spans = voc_mono
            if str(self.config.get('multi_speaker_separator', 'diarization')) == 'karaoke_model':
                split = self._split_lead_backing_karaoke(voc_mono, sr)
                if split is not None:
                    voc_for_spans, simul_backing, karaoke_info = split
                    # Cancel lead leakage here, before anything else touches
                    # the stem. This placement is load-bearing: the pitch
                    # shift below rewrites phase (destroying the coherence the
                    # cancellation depends on), basic-pitch would otherwise
                    # detect the leaked lead as its own harmony line, the
                    # whole-stem voiced gate reads HIGH on a bleedy stem
                    # (leakage is clean monophonic melody) and would convert
                    # the stack as one voice, and _finish_backing's RMS
                    # restore would re-inflate whatever bleed survived.
                    simul_backing, bleed_info = self._suppress_lead_bleed(
                        simul_backing, voc_for_spans, sr)
                    karaoke_info['bleed'] = bleed_info

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                tmp = f.name
                sf.write(tmp, voc_for_spans, sr)

            result = self._get_diarizer().diarize(tmp)
            selection = self._select_speaker_spans(
                result, voc_for_spans, sr, preserve=preserve_speakers)
            if selection is None:
                if simul_backing is None:
                    return None
                # De-doubled lead is a single voice: convert it whole and
                # keep the separated doubles as the only backing.
                info = {'num_speakers': 1, 'backing_s': 0.0}
                primary_track = voc_for_spans
                backing = np.zeros_like(voc_for_spans)
            else:
                primary_spans, backing_spans, info = selection
                primary_track = build_speaker_track(voc_for_spans, sr, primary_spans)
                backing = build_speaker_track(voc_for_spans, sr, backing_spans)

            # Coverage gate: energy the span tracks capture vs the (de-
            # doubled) vocal — computed BEFORE the doubles are merged back,
            # since they are not part of this reference. Anything the
            # diarizer's VAD missed is in neither track and would vanish from
            # the mix — below threshold, single-stem is the safer output.
            voc_energy = float(np.sum(np.square(voc_for_spans, dtype=np.float64)))
            min_cov = float(self.config.get('multi_speaker_min_coverage', 0.9))
            coverage = 1.0 if voc_energy <= 0 else float(
                np.sum(np.square(primary_track + backing, dtype=np.float64)) / voc_energy)
            info['coverage'] = round(coverage, 4)
            if coverage < min_cov:
                logger.info(
                    "Multi-speaker path: span coverage %.3f < %.2f "
                    "(VAD missed vocal content), using single-stem", coverage, min_cov)
                return None

            # Transpose backing/simul_backing before any conversion attempt,
            # matching primary_track's later shift-then-convert treatment
            # (below) - shifting after conversion would smear already-
            # synthesized audio through a second phase-vocoder pass, and
            # leaving kept backing unshifted would mismatch a transposed lead.
            if pitch_shift:
                backing = librosa.effects.pitch_shift(
                    backing, sr=sr, n_steps=float(pitch_shift))
                if simul_backing is not None:
                    simul_backing = librosa.effects.pitch_shift(
                        simul_backing, sr=sr, n_steps=float(pitch_shift))

            # Experimental: convert the backing stack to the target voice too
            # (per-line decomposition; falls back to keeping it original).
            do_convert_backing = (bool(convert_backing) if convert_backing is not None
                                  else bool(self.config.get('multi_speaker_convert_backing')))
            preserved_active = bool(info.get('preserved_speakers'))
            info['backing_mode'] = 'kept'
            harmony = None

            # Fold any unison double back into the lead BEFORE either is
            # converted. Left in the backing it would be converted as its own
            # singer and summed against the separately-converted lead, and two
            # stochastic conversions of the same phrase do not line up - which
            # is what makes the lead and the backing singers smear into each
            # other. Skipped when a preserved speaker is present: that backing
            # is kept verbatim by explicit operator choice, so nothing may be
            # moved out of it.
            if do_convert_backing and not preserved_active:
                target = simul_backing if simul_backing is not None else backing
                if target is not None and float(np.abs(target).max()) > 1e-4:
                    reduced, unison, unison_info = self._split_lead_unison(
                        target, sr, primary_track)
                    info['unison_folded_into_lead'] = unison_info['unison_lines']
                    info['harmony_lines_detected'] = unison_info['harmony_lines']
                    if unison_info['unison_lines']:
                        nu = min(len(unison), len(primary_track))
                        primary_track = primary_track.astype(np.float32).copy()
                        primary_track[:nu] += unison[:nu]
                        if simul_backing is not None:
                            simul_backing = reduced
                        else:
                            backing = reduced

            if (do_convert_backing and preserved_active and simul_backing is not None
                    and float(np.abs(simul_backing).max()) > 1e-4):
                # The span backing now carries preserved (already-target)
                # voices verbatim — only the karaoke-separated simultaneous
                # doubles are safe to convert. Do it before the merge.
                simul_backing, harmony = self._convert_backing_stack(
                    simul_backing, sr, target_profile_id, mm)
                if harmony.get('mode') == 'kept':
                    # Gates declined every line: raw separation residue with
                    # no level-matching (unlike 'partial', which already went
                    # through _finish_backing's RMS-restore).
                    simul_backing = simul_backing * float(
                        self.config.get('multi_speaker_kept_backing_gain', 1.0))

            if simul_backing is not None:
                n2 = min(len(backing), len(simul_backing))
                backing = (backing[:n2] + simul_backing[:n2]).astype(np.float32)
                primary_track = primary_track[:n2]
                info['separator'] = 'karaoke_model+diarization'
                info['backing_s'] = round(
                    float(info.get('backing_s', 0.0)) + karaoke_info.get('backing_s', 0.0), 1)
                info['simul_backing_s'] = karaoke_info.get('backing_s', 0.0)
                info['backing_energy_ratio'] = karaoke_info.get('backing_energy_ratio')

            if (do_convert_backing and not preserved_active
                    and float(np.abs(backing).max()) > 1e-4):
                backing, harmony = self._convert_backing_stack(
                    backing, sr, target_profile_id, mm)
                if harmony.get('mode') == 'kept':
                    backing = backing * float(
                        self.config.get('multi_speaker_kept_backing_gain', 1.0))
            if harmony is not None:
                info['backing_mode'] = harmony['mode']
                info['harmony_lines'] = {
                    'detected': harmony['lines_detected'],
                    'converted': harmony['lines_converted'],
                }

            if pitch_shift:
                primary_track = librosa.effects.pitch_shift(
                    primary_track, sr=sr, n_steps=float(pitch_shift))
            converted_primary = np.asarray(
                mm.infer(primary_track, target_profile_id, np.zeros(256, dtype=np.float32), sr),
                dtype=np.float32,
            )

            n = min(len(converted_primary), len(backing))
            if n == 0:
                return None
            combined = (converted_primary[:n] + backing[:n]).astype(np.float32)
            logger.info(
                "Multi-speaker conversion (%s): lead converted, %.1fs backing "
                "%s, coverage=%.3f",
                info.get('separator', 'diarization'),
                info.get('backing_s', -1.0),
                info.get('backing_mode', 'kept'), coverage)
            return combined, info
        except Exception as e:
            logger.warning("Multi-speaker conversion failed, falling back to single-stem: %s", e)
            return None
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _convert_song_fork_hq(self, song_path, target_profile_id, vocal_volume,
                              instrumental_volume, return_stems, preset, pitch_shift,
                              enable_multi_speaker=None, convert_backing=None,
                              preserve_speakers=None):
        """Fork-backed conversion: stereo, native output rate.

        Bypasses the two quality losses of the legacy lane for so-vits-svc-fork
        profiles: (1) resampling the vocal through 22.05kHz (band-limits it to
        ~11kHz vs the full-band instrumental), and (2) mono-summing. Keeps the
        instrumental in stereo and runs the fork vocal at the native output rate
        (the bridge returns that rate without downsampling). Mirrors
        convert_song's result contract.
        """
        import librosa
        import soundfile as sf
        start_time = time.time()
        target_sr = int(self._output_sample_rate)

        try:
            audio, sr = sf.read(song_path, dtype='float32', always_2d=True)  # (N, C)
        except Exception as e:
            raise ConversionError(f"Failed to load audio: {e}")
        if audio.size == 0:
            raise ConversionError("Empty audio file")
        audio = audio.T  # (C, N)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)   # mono source -> dual for separation
        elif audio.shape[0] > 2:
            audio = audio[:2]
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        mm = self._get_model_manager()
        try:
            stems = self._get_separator().separate(audio, sr, mono=False)  # stereo stems
        except Exception as e:
            raise SeparationError(f"Vocal separation failed: {e}")
        voc_st = np.atleast_2d(np.asarray(stems['vocals'], dtype=np.float32))
        inst_st = np.atleast_2d(np.asarray(stems['instrumental'], dtype=np.float32))
        voc_mono = voc_st.mean(axis=0).astype(np.float32)

        # Per-speaker path (opt-in): convert only the lead vocal and keep backing
        # vocals as-is. Returns None -> fall through to single-stem below. voc_mono
        # stays unshifted here; the helper pitch-shifts the lead track it builds.
        converted = None
        multi_speaker = False
        multi_speaker_info = None
        if self._multi_speaker_enabled(override=enable_multi_speaker):
            ms = self._convert_multi_speaker(
                voc_mono, sr, target_profile_id, mm, pitch_shift,
                convert_backing=convert_backing,
                preserve_speakers=preserve_speakers)
            if ms is not None:
                converted, multi_speaker_info = ms
                multi_speaker = True

        if converted is None:
            voc_single = voc_mono
            if pitch_shift:
                voc_single = librosa.effects.pitch_shift(
                    voc_single, sr=sr, n_steps=float(pitch_shift))
            # Fork inference at the native rate (svc_fork_bridge returns target_sr,
            # so no downsample); embedding is unused by the fork engine.
            converted = np.asarray(
                mm.infer(voc_single, target_profile_id, np.zeros(256, dtype=np.float32), sr),
                dtype=np.float32,
            )

        n = min(inst_st.shape[-1], len(converted))
        if n == 0:
            raise ConversionError("Fork conversion produced empty audio")
        conv = converted[:n] * float(vocal_volume)
        inst = inst_st[:, :n] * float(instrumental_volume)

        # Stereo image of the converted vocal.
        #
        # The fork returns mono, and adding it identically to both channels put
        # the lead - the most prominent element in the mix - at a hard centre
        # with zero width, while the instrumental kept its own. Measured on this
        # song the source vocal is WIDER than the instrumental (0.245 vs 0.183),
        # so that is a large part of the image thrown away, and listeners
        # describe the result as "mono sounding".
        #
        # Two obvious repairs were measured and rejected. Re-panning the mono
        # vocal by the original's per-frame L/R ratio recovers +1.8% - the width
        # is decorrelation (doubles, stereo reverb), not panning. Converting L
        # and R as independent streams overshoots to 4x the original width at
        # correlation -0.04, i.e. two different-sounding takes.
        #
        # What works: keep the mono conversion as the mid, and take the side
        # from the difference between an L-converted and R-converted pass of the
        # same voice, scaled so the result matches the source vocal's measured
        # width. The side carries only the target voice's own channel
        # divergence, so nothing of the original singer leaks back in, and
        # ``L + R == 2 * mid`` exactly, so mono playback is bit-identical to the
        # centred behaviour at any width.
        #
        # Defaults to 0.0 (unchanged behaviour): this costs two extra fork
        # passes, and no default should change for every conversion on one
        # song's evidence.
        stereo_width = float(self.config.get('fork_hq_stereo_width', 0.0) or 0.0)
        side = None
        if stereo_width > 0.0 and voc_st.shape[0] >= 2:
            try:
                pair = []
                for ch in (0, 1):
                    src_ch = np.ascontiguousarray(voc_st[ch])
                    if pitch_shift:
                        src_ch = librosa.effects.pitch_shift(
                            src_ch, sr=sr, n_steps=float(pitch_shift))
                    pair.append(np.asarray(
                        mm.infer(src_ch, target_profile_id,
                                 np.zeros(256, dtype=np.float32), sr),
                        dtype=np.float32,
                    ))
                m = min(len(pair[0]), len(pair[1]), n)
                raw_side = (pair[0][:m] - pair[1][:m]) * 0.5
                side_rms = float(np.sqrt(np.mean(np.square(raw_side, dtype=np.float64))))
                mid_rms = float(np.sqrt(np.mean(np.square(conv[:m], dtype=np.float64))))
                if side_rms > 1e-9 and mid_rms > 1e-9:
                    side = np.zeros(n, dtype=np.float32)
                    side[:m] = raw_side * (stereo_width * mid_rms / side_rms)
                else:
                    side = None
            except Exception as exc:
                # A stereo treatment must never cost the conversion itself.
                logger.warning("Fork HQ stereo widening failed, keeping centred vocal: %s", exc)
                side = None

        # Do not invent bandwidth the source never had. The decoder is
        # full-band; a separated stem off a lossy encode usually is not, and
        # the octave the model extrapolates above the source's wall is where
        # the "electronic" character was measured. Filtering mid and side
        # separately is equivalent to filtering L and R, and preserves
        # ``L + R == 2 * mid`` exactly.
        match_bandwidth = bool(self.config.get('fork_hq_match_source_bandwidth', True))
        source_bandwidth_hz = None
        applied_bandwidth_hz = None
        if match_bandwidth:
            try:
                # Measure the ORIGINAL mix, not the separated stem. The
                # separator emits its own energy above the source's wall, so a
                # stem off a 16 kHz-limited song reads as full-band and the
                # match silently no-ops. Caught end-to-end: "One Last Time"
                # walls at 16 kHz, its vocal stem measured 22050, and the
                # render kept +40 dB of invented top octave. `audio` is the
                # original at `sr` here (resampled at most upward, which cannot
                # add content above the source Nyquist).
                source_bandwidth_hz = _detect_bandwidth_hz(
                    audio.mean(axis=0) if audio.ndim > 1 else audio, sr)
                if source_bandwidth_hz < sr * 0.5 * 0.95:
                    conv = _lowpass_to(conv, sr, source_bandwidth_hz)
                    if side is not None:
                        side = _lowpass_to(side, sr, source_bandwidth_hz)
                    applied_bandwidth_hz = source_bandwidth_hz
                    logger.info(
                        "Fork HQ: matched converted vocal to source bandwidth "
                        "%.0f Hz", source_bandwidth_hz
                    )
            except Exception as exc:
                # Band matching must never cost the conversion itself.
                logger.warning(
                    "Fork HQ bandwidth match failed, keeping full-band vocal: %s", exc)
                applied_bandwidth_hz = None

        if side is not None:
            mixed = np.stack([inst[0] + conv + side, inst[1] + conv - side], axis=-1)
        else:
            mixed = np.stack([inst[0] + conv, inst[1] + conv], axis=-1)  # (n, 2) stereo
        peak = float(np.abs(mixed).max())
        if peak > 0.95:
            mixed = mixed * (0.95 / peak)

        duration = n / sr
        f0_original = self._extract_pitch(voc_mono, sr)
        f0_contour = self._extract_pitch(converted[:n], sr)
        result = {
            'mixed_audio': mixed,
            'sample_rate': sr,
            'duration': duration,
            'metadata': {
                'preset': preset,
                'pitch_shift': pitch_shift,
                'vocal_volume': vocal_volume,
                'instrumental_volume': instrumental_volume,
                'processing_time': time.time() - start_time,
                'target_profile_id': target_profile_id,
                'active_model_type': 'svc_fork',
                'speaker_id': target_profile_id,
                # Report what actually ran. This key previously always read
                # 'svc_fork_hq_stereo' while no such processing existed
                # anywhere in the codebase.
                'quality_post_processing': (
                    (['svc_fork_hq_stereo'] if side is not None
                     else ['svc_fork_hq_mono_vocal'])
                    + (['svc_fork_hq_bandwidth_matched']
                       if applied_bandwidth_hz else [])
                ),
                'vocal_stereo_width': stereo_width if side is not None else 0.0,
                'source_bandwidth_hz': source_bandwidth_hz,
                'vocal_bandwidth_hz': applied_bandwidth_hz or (sr * 0.5),
                'stereo': True,
                'multi_speaker': multi_speaker,
            },
            'f0_contour': f0_contour,
            'f0_original': f0_original,
            'f0_sample_rate': sr,
        }
        if multi_speaker_info:
            result['metadata']['multi_speaker_info'] = multi_speaker_info
        if return_stems:
            result['stems'] = {
                'vocals': conv.astype(np.float32),
                'instrumental': inst.mean(axis=0).astype(np.float32),
            }
        logger.info(
            f"Fork HQ conversion complete: {duration:.1f}s stereo/{sr}Hz "
            f"(profile={target_profile_id})"
        )
        return result

    def _get_model_manager(self):
        """Get or create ModelManager and load models from config.

        Lazy-loads ModelManager on first call, then loads models using paths
        from self.config (hubert_path, vocoder_path, voice_model_path, etc).

        Returns:
            ModelManager instance with loaded models ready for inference.

        Raises:
            RuntimeError: If model loading fails or required paths not in config.
        """
        if self._model_manager is None:
            from .model_manager import ModelManager
            self._model_manager = ModelManager(device=self.device, config=self.config)

            # Load models from config paths.
            # CRITICAL: these must match how the decoder was TRAINED, or the
            # trained decoder receives out-of-distribution content and the
            # vocoder renders random weights — both produce a tonal whine
            # instead of vocals. Training uses ContentVec content features
            # (trainer.py builds ContentEncoder(encoder_backend='contentvec'))
            # and the 80-mel universal HiFiGAN. Default serving to the same
            # instead of the old 'hubert' (random) + no-vocoder (random).
            hubert_path = self.config.get('hubert_path')
            encoder_backend = self.config.get('encoder_backend', 'contentvec')
            encoder_type = self.config.get('encoder_type', 'linear')
            vocoder_type = self.config.get('vocoder_type', 'hifigan')
            vocoder_path = self.config.get('vocoder_path')
            if not vocoder_path and vocoder_type == 'hifigan':
                default_vocoder = os.path.join(
                    'models', 'pretrained', 'generator_universal.pth.tar')
                if os.path.exists(default_vocoder):
                    vocoder_path = default_vocoder
                else:
                    raise RuntimeError(
                        "No HiFiGAN vocoder checkpoint configured and the "
                        f"default {default_vocoder} is missing; conversion "
                        "would render a random vocoder (silent whine).")
            conformer_config = self.config.get('conformer_config')
            self._model_manager.load(
                hubert_path=hubert_path,
                vocoder_path=vocoder_path,
                vocoder_type=vocoder_type,
                encoder_backend=encoder_backend,
                encoder_type=encoder_type,
                conformer_config=conformer_config,
            )

            # Load voice model if configured
            voice_model_path = self.config.get('voice_model_path')
            speaker_id = self.config.get('speaker_id', 'default')
            if voice_model_path:
                self._model_manager.load_voice_model(voice_model_path, speaker_id)
        return self._model_manager

    def _resolve_target_speaker(
        self,
        target_profile_id: str,
        target_embedding: np.ndarray,
        active_model_type: Optional[str] = None,
    ) -> tuple[str, str]:
        """Resolve which target model should drive conversion for a profile.

        The profile's ``active_model_type`` (set by training and by the
        adapter-select endpoint) decides the artifact preference; the other
        artifact family is the fallback when the preferred file is absent.

        Returns:
            Tuple of (speaker_id, model_type), where model_type is one of
            ``full_model`` or ``adapter``.
        """
        model_manager = self._get_model_manager()

        store = getattr(self._voice_cloner, 'store', None)
        trained_models_dir = getattr(store, 'trained_models_dir', None)
        if trained_models_dir:
            candidates = [
                ('full_model', Path(trained_models_dir) / f"{target_profile_id}_full_model.pt"),
                ('adapter', Path(trained_models_dir) / f"{target_profile_id}_adapter_model.pt"),
            ]
            if active_model_type == 'adapter':
                candidates.reverse()
            for model_type, artifact_path in candidates:
                if artifact_path.exists():
                    model_manager.load_voice_model(
                        str(artifact_path),
                        target_profile_id,
                        speaker_embedding=target_embedding,
                    )
                    logger.info(
                        "Using trained %s artifact for target profile %s from %s (active_model_type=%s)",
                        model_type,
                        target_profile_id,
                        artifact_path,
                        active_model_type,
                    )
                    return target_profile_id, model_type

        return self.config.get('speaker_id', 'default'), 'adapter'

    def _convert_voice(self, vocals: np.ndarray, target_embedding: np.ndarray,
                       sr: int, speaker_id: str, preset: str = 'balanced') -> np.ndarray:
        """Convert vocals to target voice using trained So-VITS-SVC model.

        Args:
            vocals: Input vocal audio signal (mono)
            target_embedding: Target speaker embedding (256-dim L2-normalized)
            sr: Sample rate of vocal audio
            preset: Quality preset name (unused, reserved for future use)

        Returns:
            Converted vocal audio at the same sample rate.

        Raises:
            ConversionError: If voice conversion fails or models not loaded.
        """
        try:
            model_manager = self._get_model_manager()
            return model_manager.infer(vocals, speaker_id, target_embedding, sr)
        except RuntimeError as e:
            raise ConversionError(f"Voice conversion failed: {e}")

    def _extract_pitch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract pitch contour (F0) from audio using librosa pyin.

        Args:
            audio: Input audio signal (mono)
            sr: Sample rate of the audio

        Returns:
            F0 contour array with NaN replaced by 0.0.
            Falls back to zero array if extraction fails.
        """
        try:
            import librosa
            f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=1100, sr=sr)
            f0 = np.nan_to_num(f0, nan=0.0)
            return f0
        except Exception:
            hop_length = 512
            n_frames = len(audio) // hop_length
            return np.zeros(n_frames)

    def _detect_techniques(self, audio: np.ndarray, sr: int) -> Optional[Dict[str, Any]]:
        """Detect vocal techniques (vibrato, melisma) in audio.

        Args:
            audio: Vocal audio signal
            sr: Sample rate

        Returns:
            Dict with technique information or None if detection fails
        """
        try:
            from ..audio.technique_detector import TechniqueAwarePitchExtractor

            extractor = TechniqueAwarePitchExtractor(sample_rate=sr)
            f0, flags = extractor.extract_with_flags(audio)

            return {
                'f0': f0,
                'technique_flags': flags,
                'has_vibrato': flags.has_vibrato,
                'has_melisma': flags.has_melisma,
                'vibrato_rate': flags.vibrato_rate,
                'vibrato_depth_cents': flags.vibrato_depth_cents,
            }
        except Exception as e:
            logger.warning(f"Technique detection failed: {e}")
            return None

    def _resample_audio(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        if from_sr == to_sr:
            return np.asarray(audio, dtype=np.float32)

        import librosa

        return librosa.resample(
            np.asarray(audio, dtype=np.float32),
            orig_sr=from_sr,
            target_sr=to_sr,
        ).astype(np.float32)

    def _apply_quality_post_processing(
        self,
        converted_vocals: np.ndarray,
        instrumental: np.ndarray,
        sample_rate: int,
        f0_contour: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        """Apply optional quality upgrade stages in a portable order."""
        metadata: Dict[str, Any] = {'post_processing': []}
        vocals = np.asarray(converted_vocals, dtype=np.float32)
        backing = np.asarray(instrumental, dtype=np.float32)
        output_sr = sample_rate

        if self.config.get('enable_nsf_harmonic_enhancement'):
            if self._nsf_enhancer is None:
                from ..models.nsf_module import NSFHarmonicEnhancer

                self._nsf_enhancer = NSFHarmonicEnhancer(
                    harmonic_strength=self.config.get('nsf_harmonic_strength', 0.12),
                    max_harmonics=self.config.get('nsf_max_harmonics', 6),
                    blend=self.config.get('nsf_blend', 0.2),
                )
            vocals = self._nsf_enhancer.enhance(vocals, f0_contour, sample_rate)
            metadata['post_processing'].append('nsf_harmonic_enhancement')

        if self.config.get('enable_pupu_vocoder_refinement'):
            if self._pupu_vocoder is None:
                from ..models.pupu_vocoder import PupuVocoderEnhancer

                self._pupu_vocoder = PupuVocoderEnhancer(
                    brightness=self.config.get('pupu_brightness', 0.08),
                    transient_boost=self.config.get('pupu_transient_boost', 0.1),
                )
            vocals = self._pupu_vocoder.refine(vocals, sample_rate)
            metadata['post_processing'].append('pupu_vocoder_refinement')

        if self.config.get('enable_hq_super_resolution'):
            try:
                import torch

                if self._hq_enhancer is None:
                    from .hq_svc_wrapper import HQSVCWrapper

                    self._hq_enhancer = HQSVCWrapper(
                        device=self.device,
                        require_gpu=self.config.get('hq_require_gpu', False),
                    )

                super_res = self._hq_enhancer.super_resolve(
                    torch.tensor(vocals, dtype=torch.float32),
                    sample_rate=sample_rate,
                )
                vocals = np.asarray(super_res['audio'], dtype=np.float32)
                output_sr = int(super_res['sample_rate'])
                if output_sr != sample_rate:
                    backing = self._resample_audio(backing, sample_rate, output_sr)
                metadata['post_processing'].append('hq_super_resolution')
            except Exception as exc:
                logger.warning("HQ-SVC enhancement unavailable, skipping: %s", exc)
                metadata['hq_super_resolution_skipped'] = str(exc)

        return vocals, backing, output_sr, metadata

    def convert_song(self, song_path: str, target_profile_id: str,
                     vocal_volume: float = 1.0, instrumental_volume: float = 0.9,
                     pitch_shift: float = 0.0, return_stems: bool = False,
                     preset: str = 'balanced',
                     preserve_techniques: bool = True,
                     enable_multi_speaker: Optional[bool] = None,
                     convert_backing: Optional[bool] = None,
                     preserve_speakers: Optional[list] = None) -> Dict[str, Any]:
        """Convert a song to target voice.

        Args:
            song_path: Path to input audio file
            target_profile_id: Target voice profile ID
            vocal_volume: Vocal volume multiplier [0.0-2.0]
            instrumental_volume: Instrumental volume [0.0-2.0]
            pitch_shift: Pitch shift in semitones [-12 to 12]
            return_stems: Whether to return separate stems
            preset: Quality preset (draft/fast/balanced/high/studio)
            preserve_techniques: Detect and preserve vocal techniques (vibrato, melisma)

        Returns:
            Dict with: mixed_audio, sample_rate, duration, metadata,
                       f0_contour, f0_original, stems (optional),
                       techniques (if preserve_techniques=True)
        """
        import librosa

        start_time = time.time()

        # Load audio
        if not os.path.exists(song_path):
            raise ConversionError(f"Song file not found: {song_path}")

        # Fork-backed profiles use a native-44.1kHz STEREO lane (no 22.05k vocal
        # bottleneck, no mono-summing). The legacy mono path below is unchanged
        # for every other profile.
        from . import svc_fork_bridge
        if svc_fork_bridge.is_available(
                target_profile_id, self.config.get('data_dir', 'data')):
            # ponytail: pass enable_multi_speaker only when set so existing
            # 7-arg stubs of _convert_song_fork_hq keep working (default None
            # already resolves via config/env inside the fork lane).
            fork_kwargs = {}
            if enable_multi_speaker is not None:
                fork_kwargs['enable_multi_speaker'] = enable_multi_speaker
            if convert_backing is not None:
                fork_kwargs['convert_backing'] = convert_backing
            if preserve_speakers:
                fork_kwargs['preserve_speakers'] = list(preserve_speakers)
            return self._convert_song_fork_hq(
                song_path, target_profile_id, vocal_volume,
                instrumental_volume, return_stems, preset, pitch_shift,
                **fork_kwargs)

        try:
            audio, sr = librosa.load(song_path, sr=self._output_sample_rate, mono=True)
        except Exception as e:
            raise ConversionError(f"Failed to load audio: {e}")

        if len(audio) == 0:
            raise ConversionError("Empty audio file")

        # Load target profile
        cloner = self._voice_cloner
        if cloner is None:
            from .voice_cloner import VoiceCloner
            cloner = VoiceCloner(device=self.device)
            # keep it: _resolve_target_speaker needs cloner.store to find
            # the profile's trained artifacts (full model / adapter model)
            self._voice_cloner = cloner
        try:
            profile = cloner.load_voice_profile(target_profile_id)
        except Exception as e:
            from ..storage.voice_profiles import ProfileNotFoundError
            raise ProfileNotFoundError(f"Profile {target_profile_id} not found: {e}")

        target_embedding = profile.get('embedding')
        if target_embedding is None:
            raise ConversionError("Profile missing embedding data")
        if isinstance(target_embedding, list):
            target_embedding = np.array(target_embedding)

        speaker_id, model_type = self._resolve_target_speaker(
            target_profile_id,
            target_embedding,
            active_model_type=profile.get('active_model_type'),
        )

        # Separate at the mix rate so the instrumental keeps full bandwidth;
        # only the vocals drop to the model's processing rate.
        stems = self._separate_vocals(audio, sr)
        instrumental = stems['instrumental']
        proc_sr = self._sample_rate
        vocals = self._resample_audio(stems['vocals'], sr, proc_sr)

        # Extract original pitch
        f0_original = self._extract_pitch(vocals, proc_sr)

        # Detect vocal techniques (vibrato, melisma) if requested
        techniques = None
        if preserve_techniques:
            techniques = self._detect_techniques(vocals, proc_sr)
            if techniques:
                logger.info(
                    f"Techniques detected - vibrato: {techniques['has_vibrato']}, "
                    f"melisma: {techniques['has_melisma']}"
                )

        # Apply pitch shift if requested
        if abs(pitch_shift) > 0.01:
            try:
                vocals = librosa.effects.pitch_shift(
                    vocals, sr=proc_sr, n_steps=pitch_shift
                )
            except Exception as e:
                logger.warning(f"Pitch shift failed: {e}")

        # Convert vocals to target voice
        converted_vocals = self._convert_voice(
            vocals,
            target_embedding,
            proc_sr,
            speaker_id=speaker_id,
            preset=preset,
        )

        # Extract converted pitch
        f0_contour = self._extract_pitch(converted_vocals, proc_sr)
        f0_rate = proc_sr

        # Bring converted vocals up to the mix rate before post-processing
        # and mixing with the full-bandwidth instrumental
        converted_vocals = self._resample_audio(converted_vocals, proc_sr, sr)

        converted_vocals, instrumental, sr, quality_metadata = self._apply_quality_post_processing(
            converted_vocals,
            instrumental,
            sr,
            f0_contour,
        )
        if quality_metadata.get('post_processing'):
            f0_contour = self._extract_pitch(converted_vocals, sr)
            f0_rate = sr

        # Mix with volume adjustments
        converted_vocals = converted_vocals * vocal_volume
        instrumental = instrumental * instrumental_volume

        # Ensure same length
        min_len = min(len(converted_vocals), len(instrumental))
        mixed_audio = converted_vocals[:min_len] + instrumental[:min_len]

        # Normalize to prevent clipping
        peak = np.abs(mixed_audio).max()
        if peak > 0.95:
            mixed_audio = mixed_audio * (0.95 / peak)

        duration = len(mixed_audio) / sr
        elapsed = time.time() - start_time

        result = {
            'mixed_audio': mixed_audio,
            'sample_rate': sr,
            'duration': duration,
            'metadata': {
                'preset': preset,
                'pitch_shift': pitch_shift,
                'vocal_volume': vocal_volume,
                'instrumental_volume': instrumental_volume,
                'processing_time': elapsed,
                'target_profile_id': target_profile_id,
                'active_model_type': model_type,
                'speaker_id': speaker_id,
                'quality_post_processing': quality_metadata.get('post_processing', []),
            },
            'f0_contour': f0_contour,
            'f0_original': f0_original,
            # F0 contours are extracted at the vocal processing rate, not the
            # mix rate — consumers computing frame times must use this.
            'f0_sample_rate': f0_rate,
        }

        if quality_metadata.get('hq_super_resolution_skipped'):
            result['metadata']['hq_super_resolution_skipped'] = quality_metadata['hq_super_resolution_skipped']

        if return_stems:
            result['stems'] = {
                'vocals': converted_vocals[:min_len],
                'instrumental': instrumental[:min_len],
            }

        if techniques:
            # Include technique info (excluding non-serializable flags)
            result['techniques'] = {
                'has_vibrato': techniques['has_vibrato'],
                'has_melisma': techniques['has_melisma'],
                'vibrato_rate': techniques['vibrato_rate'],
                'vibrato_depth_cents': techniques['vibrato_depth_cents'],
            }

        logger.info(
            f"Song conversion complete: {duration:.1f}s audio in {elapsed:.1f}s "
            f"(preset={preset}, profile={target_profile_id}, model_type={model_type})"
        )

        return result
