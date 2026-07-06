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


def _extract_line_audio(stack: np.ndarray, sample_rate: int, line_notes,
                        n_harmonics: int = 10, width_cents: float = 40.0) -> np.ndarray:
    """Isolate one harmony line from a stack via an STFT harmonic comb mask.

    For each note, passes ±``width_cents`` bands around the first
    ``n_harmonics`` multiples of the note's F0 during the note's time span.
    Binary mask + ISTFT; good enough for stacked harmonies whose lines sit at
    different pitches most of the time.
    """
    import librosa

    n_fft, hop = 2048, 512
    S = librosa.stft(np.asarray(stack, dtype=np.float32), n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sample_rate, hop_length=hop)
    mask = np.zeros(S.shape, dtype=np.float32)
    lo_ratio = 2.0 ** (-width_cents / 1200.0)
    hi_ratio = 2.0 ** (width_cents / 1200.0)

    for note in line_notes:
        f0 = 440.0 * 2.0 ** ((float(note['pitch_midi']) - 69.0) / 12.0)
        fsel = (times >= note['start']) & (times <= note['end'])
        if not fsel.any():
            continue
        for k in range(1, n_harmonics + 1):
            fk = k * f0
            if fk >= sample_rate / 2:
                break
            bsel = (freqs >= fk * lo_ratio) & (freqs <= fk * hi_ratio)
            if bsel.any():
                mask[np.ix_(bsel, fsel)] = 1.0

    line = librosa.istft(S * mask, hop_length=hop, length=len(stack))
    return np.asarray(line, dtype=np.float32)


# Preset configurations
PRESETS = {
    'draft': {'n_steps': 10, 'denoise': 0.3},
    'fast': {'n_steps': 20, 'denoise': 0.5},
    'balanced': {'n_steps': 50, 'denoise': 0.7},
    'high': {'n_steps': 100, 'denoise': 0.8},
    'studio': {'n_steps': 200, 'denoise': 0.9},
}


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
        for i, line_notes in enumerate(lines):
            extract = _extract_line_audio(backing, sr, line_notes)
            line_rms = float(np.sqrt(np.mean(np.square(extract, dtype=np.float64))))
            if line_rms < 0.05 * stack_rms:
                logger.info("Backing line %d: negligible energy, skipped", i)
                continue
            # Concentration gate: a real harmonic line captures a large share
            # of the stack's energy inside its note spans; comb-filtering
            # noise/texture captures little (and pyin can't tell — a comb
            # mask MANUFACTURES pitch, so the voiced gate alone is blind here).
            spans = [(n['start'], n['end']) for n in line_notes]
            stack_active = _active_audio(backing, sr, spans)
            extract_active = _active_audio(extract, sr, spans)
            stack_e = float(np.sum(np.square(stack_active, dtype=np.float64))) + 1e-12
            concentration = float(
                np.sum(np.square(extract_active, dtype=np.float64)) / stack_e)
            if concentration < 0.15:
                logger.info("Backing line %d: energy concentration %.2f < 0.15 "
                            "(not a real harmonic line), kept original", i, concentration)
                continue
            # Measure voicing on the span-active audio (calibration convention);
            # the full-length extract is mostly silence, which starves pyin's
            # 20s measurement window and would zero the gate for any song
            # whose backing starts late.
            vf = (_voiced_fraction(extract_active, sr)
                  if len(extract_active) >= int(1.5 * sr) else 0.0)
            if vf < merge_voiced:
                logger.info("Backing line %d: voiced %.2f < %.2f, kept original", i, vf, merge_voiced)
                continue
            converted = np.asarray(
                mm.infer(extract, target_profile_id, np.zeros(256, dtype=np.float32), sr),
                dtype=np.float32,
            )
            n = min(len(converted), len(new_backing))
            conv_rms = float(np.sqrt(np.mean(np.square(converted[:n], dtype=np.float64))) + 1e-12)
            gain = line_rms / conv_rms
            # Swap the line: remove the original extract, add the converted one.
            new_backing[:n] = new_backing[:n] - extract[:n] + converted[:n] * gain
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
        operator taste knob and a peak guard.
        """
        if len(intervals):
            orig_act = np.concatenate([backing[s:e] for s, e in intervals])
            new_act = np.concatenate([new_backing[s:e] for s, e in intervals])
            orig_rms = float(np.sqrt(np.mean(np.square(orig_act, dtype=np.float64))))
            new_rms = float(np.sqrt(np.mean(np.square(new_act, dtype=np.float64))) + 1e-12)
            if orig_rms > 0.0:
                new_backing = new_backing * (orig_rms / new_rms)
        new_backing = new_backing * float(
            self.config.get('multi_speaker_backing_gain', 1.0))
        peak = float(np.abs(new_backing).max())
        if peak > 0.99:
            new_backing = new_backing * (0.99 / peak)

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

            # Experimental: convert the backing stack to the target voice too
            # (per-line decomposition; falls back to keeping it original).
            do_convert_backing = (bool(convert_backing) if convert_backing is not None
                                  else bool(self.config.get('multi_speaker_convert_backing')))
            preserved_active = bool(info.get('preserved_speakers'))
            info['backing_mode'] = 'kept'
            harmony = None

            if (do_convert_backing and preserved_active and simul_backing is not None
                    and float(np.abs(simul_backing).max()) > 1e-4):
                # The span backing now carries preserved (already-target)
                # voices verbatim — only the karaoke-separated simultaneous
                # doubles are safe to convert. Do it before the merge.
                simul_backing, harmony = self._convert_backing_stack(
                    simul_backing, sr, target_profile_id, mm)

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
                              preserve_speakers=None, quality_overrides=None):
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

        # Per-request quality settings (config defaults + sanitized overrides).
        q = self._resolve_quality_settings(quality_overrides)
        pre_stages = []
        if q['enable_dereverb']:
            try:
                from ..audio.dereverb import dereverb_vocals
                voc_mono = dereverb_vocals(voc_mono, sr, strength=q['dereverb_strength'])
                pre_stages.append('dereverb')
            except Exception as e:
                logger.warning(f"De-reverb failed, using raw vocals: {e}")

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
        converted = np.asarray(converted[:n], dtype=np.float32)

        stages = ['svc_fork_hq_stereo'] + pre_stages
        f0_original = self._extract_pitch(voc_mono, sr, method=q['f0_method'])
        f0_contour = self._extract_pitch(converted, sr, method=q['f0_method'])
        if q['enable_f0_postprocess']:
            try:
                from .f0_utils import postprocess_f0
                f0_original = postprocess_f0(f0_original)
                f0_contour = postprocess_f0(f0_contour)
                stages.append('f0_postprocess')
            except Exception as e:
                logger.warning(f"F0 post-processing failed: {e}")

        # Vocal-only enhancement stages at the native rate. HQ super-resolution
        # is intentionally skipped: the fork lane already runs at output rate.
        if q['enable_nsf_harmonic_enhancement']:
            try:
                if self._nsf_enhancer is None:
                    from ..models.nsf_module import NSFHarmonicEnhancer
                    self._nsf_enhancer = NSFHarmonicEnhancer(
                        harmonic_strength=self.config.get('nsf_harmonic_strength', 0.12),
                        max_harmonics=self.config.get('nsf_max_harmonics', 6),
                        blend=self.config.get('nsf_blend', 0.2),
                    )
                converted = self._nsf_enhancer.enhance(converted, f0_contour, sr)
                stages.append('nsf_harmonic_enhancement')
            except Exception as e:
                logger.warning(f"NSF enhancement failed: {e}")
        if q['enable_pupu_vocoder_refinement']:
            try:
                if self._pupu_vocoder is None:
                    from ..models.pupu_vocoder import PupuVocoderEnhancer
                    self._pupu_vocoder = PupuVocoderEnhancer(
                        brightness=self.config.get('pupu_brightness', 0.08),
                        transient_boost=self.config.get('pupu_transient_boost', 0.1),
                    )
                converted = self._pupu_vocoder.refine(converted, sr)
                stages.append('pupu_vocoder_refinement')
            except Exception as e:
                logger.warning(f"Pupu refinement failed: {e}")

        # Post-mix repairs against the (dereverbed) source vocal.
        try:
            if q['enable_consonant_passthrough'] and f0_original.size:
                from ..audio.post_mix import voicing_gated_passthrough
                converted = voicing_gated_passthrough(
                    voc_mono[:n], converted, sr, f0_original, 512,
                    mix=float(q['consonant_passthrough_mix']),
                )
                stages.append('consonant_passthrough')
            if q['enable_loudness_transfer']:
                from ..audio.post_mix import transfer_loudness_envelope
                converted = transfer_loudness_envelope(
                    voc_mono[:n], converted, sr,
                    strength=float(self.config.get('loudness_transfer_strength', 0.85)),
                )
                stages.append('loudness_transfer')
        except Exception as e:
            logger.warning(f"Post-mix repair failed, keeping converted vocals: {e}")

        if len(stages) > 1:
            f0_contour = self._extract_pitch(converted, sr)

        conv = converted * float(vocal_volume)
        inst = inst_st[:, :n] * float(instrumental_volume)
        mixed = np.stack([inst[0] + conv, inst[1] + conv], axis=-1)  # (n, 2) stereo
        peak = float(np.abs(mixed).max())
        if peak > 0.95:
            mixed = mixed * (0.95 / peak)

        duration = n / sr
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
                'quality_post_processing': stages,
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

    def _extract_pitch(self, audio: np.ndarray, sr: int,
                       method: Optional[str] = None) -> np.ndarray:
        """Extract pitch contour (F0) from audio.

        Args:
            audio: Input audio signal (mono)
            sr: Sample rate of the audio
            method: Optional per-request F0 backend ('rmvpe'/'pyin'). When set,
                the quality-ordered f0_extractor chain is used (with automatic
                fallback); when None the legacy librosa pyin path is kept so
                default behavior is unchanged.

        Returns:
            F0 contour array with NaN replaced by 0.0 (hop 512 frame grid).
            Falls back to zero array if extraction fails.
        """
        if method:
            try:
                from .f0_extractor import extract_f0
                f0, _used = extract_f0(
                    np.asarray(audio, dtype=np.float32), sr=sr, hop_length=512,
                    method=method, device=self.device,
                    rmvpe_model_path=self.config.get('rmvpe_model_path'),
                    rmvpe_is_half=bool(self.config.get('rmvpe_is_half', False)),
                )
                return np.asarray(f0, dtype=np.float32)
            except Exception as e:
                logger.warning(f"F0 extraction via '{method}' failed, "
                               f"falling back to pyin: {e}")
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

    def _resolve_quality_settings(
        self, overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge config-level quality toggles with per-request overrides.

        Only whitelisted keys (mirroring the web layer's
        ``_parse_quality_overrides`` contract) are honored; everything else in
        ``overrides`` is ignored so a stale/forged settings dict can't reach
        arbitrary config. Defaults keep every new stage OFF so behavior is
        unchanged unless configured or explicitly requested.
        """
        q: Dict[str, Any] = {
            'enable_nsf_harmonic_enhancement': bool(self.config.get('enable_nsf_harmonic_enhancement')),
            'enable_pupu_vocoder_refinement': bool(self.config.get('enable_pupu_vocoder_refinement')),
            'enable_hq_super_resolution': bool(self.config.get('enable_hq_super_resolution')),
            'enable_dereverb': bool(self.config.get('enable_dereverb')),
            'enable_consonant_passthrough': bool(self.config.get('enable_consonant_passthrough')),
            'enable_loudness_transfer': bool(self.config.get('enable_loudness_transfer')),
            'enable_f0_postprocess': bool(self.config.get('enable_f0_postprocess')),
            'dereverb_strength': float(self.config.get('dereverb_strength', 0.5)),
            'consonant_passthrough_mix': float(self.config.get('consonant_passthrough_mix', 0.6)),
            'f0_method': self.config.get('f0_method'),
        }
        if overrides:
            for key, value in overrides.items():
                if key in q:
                    q[key] = value
        return q

    def _apply_quality_post_processing(
        self,
        converted_vocals: np.ndarray,
        instrumental: np.ndarray,
        sample_rate: int,
        f0_contour: np.ndarray,
        quality_settings: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        """Apply optional quality upgrade stages in a portable order."""
        q = quality_settings if quality_settings is not None else self.config
        metadata: Dict[str, Any] = {'post_processing': []}
        vocals = np.asarray(converted_vocals, dtype=np.float32)
        backing = np.asarray(instrumental, dtype=np.float32)
        output_sr = sample_rate

        if q.get('enable_nsf_harmonic_enhancement'):
            if self._nsf_enhancer is None:
                from ..models.nsf_module import NSFHarmonicEnhancer

                self._nsf_enhancer = NSFHarmonicEnhancer(
                    harmonic_strength=self.config.get('nsf_harmonic_strength', 0.12),
                    max_harmonics=self.config.get('nsf_max_harmonics', 6),
                    blend=self.config.get('nsf_blend', 0.2),
                )
            vocals = self._nsf_enhancer.enhance(vocals, f0_contour, sample_rate)
            metadata['post_processing'].append('nsf_harmonic_enhancement')

        if q.get('enable_pupu_vocoder_refinement'):
            if self._pupu_vocoder is None:
                from ..models.pupu_vocoder import PupuVocoderEnhancer

                self._pupu_vocoder = PupuVocoderEnhancer(
                    brightness=self.config.get('pupu_brightness', 0.08),
                    transient_boost=self.config.get('pupu_transient_boost', 0.1),
                )
            vocals = self._pupu_vocoder.refine(vocals, sample_rate)
            metadata['post_processing'].append('pupu_vocoder_refinement')

        if q.get('enable_hq_super_resolution'):
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
                     preserve_speakers: Optional[list] = None,
                     quality_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            quality_overrides: Optional per-request quality toggles (already
                sanitized by the web layer's whitelist); shadow config for
                this call only — never mutate shared config.

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
            if quality_overrides:
                fork_kwargs['quality_overrides'] = dict(quality_overrides)
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

        # Per-request quality settings (config defaults + sanitized overrides).
        q = self._resolve_quality_settings(quality_overrides)
        pre_stages: list = []

        # Separate at the mix rate so the instrumental keeps full bandwidth;
        # only the vocals drop to the model's processing rate.
        stems = self._separate_vocals(audio, sr)
        instrumental = stems['instrumental']
        source_vocals = np.asarray(stems['vocals'], dtype=np.float32)

        if q['enable_dereverb']:
            try:
                from ..audio.dereverb import dereverb_vocals
                source_vocals = dereverb_vocals(
                    source_vocals, sr, strength=q['dereverb_strength'])
                pre_stages.append('dereverb')
            except Exception as e:
                logger.warning(f"De-reverb failed, using raw vocals: {e}")

        proc_sr = self._sample_rate
        vocals = self._resample_audio(source_vocals, sr, proc_sr)

        # Extract original pitch
        f0_original = self._extract_pitch(vocals, proc_sr, method=q['f0_method'])
        if q['enable_f0_postprocess']:
            try:
                from .f0_utils import postprocess_f0
                f0_original = postprocess_f0(f0_original)
                pre_stages.append('f0_postprocess')
            except Exception as e:
                logger.warning(f"F0 post-processing failed: {e}")

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
        f0_contour = self._extract_pitch(converted_vocals, proc_sr, method=q['f0_method'])
        if q['enable_f0_postprocess']:
            try:
                from .f0_utils import postprocess_f0
                f0_contour = postprocess_f0(f0_contour)
            except Exception as e:
                logger.warning(f"F0 post-processing failed: {e}")
        f0_rate = proc_sr

        # Bring converted vocals up to the mix rate before post-processing
        # and mixing with the full-bandwidth instrumental
        converted_vocals = self._resample_audio(converted_vocals, proc_sr, sr)

        mix_sr = sr  # rate of source_vocals; helper may change sr (HQ super-res)
        converted_vocals, instrumental, sr, quality_metadata = self._apply_quality_post_processing(
            converted_vocals,
            instrumental,
            sr,
            f0_contour,
            quality_settings=q,
        )

        # Post-mix repairs against the (dereverbed) source vocals. `sr` already
        # tracks the helper's output rate; the reference was captured at the
        # original mix rate, so resample it if HQ super-resolution changed sr.
        ref_vocals = source_vocals
        if quality_metadata.get('post_processing') and 'hq_super_resolution' in quality_metadata['post_processing']:
            ref_vocals = self._resample_audio(ref_vocals, mix_sr, sr)
        try:
            if q['enable_consonant_passthrough'] and f0_original.size:
                from ..audio.post_mix import voicing_gated_passthrough
                if ref_vocals.shape[0] != converted_vocals.shape[0]:
                    ref_vocals = ref_vocals[:converted_vocals.shape[0]]
                hop = max(1, int(round(512 * sr / float(proc_sr))))
                converted_vocals = voicing_gated_passthrough(
                    ref_vocals, converted_vocals, sr, f0_original, hop,
                    mix=float(q['consonant_passthrough_mix']),
                )
                quality_metadata['post_processing'].append('consonant_passthrough')
            if q['enable_loudness_transfer']:
                from ..audio.post_mix import transfer_loudness_envelope
                converted_vocals = transfer_loudness_envelope(
                    ref_vocals, converted_vocals, sr,
                    strength=float(self.config.get('loudness_transfer_strength', 0.85)),
                )
                quality_metadata['post_processing'].append('loudness_transfer')
        except Exception as e:
            logger.warning(f"Post-mix repair failed, keeping converted vocals: {e}")

        quality_metadata['post_processing'] = pre_stages + quality_metadata['post_processing']
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
