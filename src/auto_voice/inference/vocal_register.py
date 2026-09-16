"""Vocal register (male/female) classification and duet routing decisions.


REGISTER IS NOT IDENTITY, and must never become it
====================================================
This module answers exactly one question about a group of spans: is it sung in
the lower (male-typical) or the upper (female-typical) register? It answers it
from ONE scalar — the median pyin f0 of the voiced frames — and it returns
``None`` whenever that scalar is not trustworthy.

It does not say WHO is singing, and it must not be grown into something that
tries. This codebase carries a measured negative result on exactly that
(2026-07-04 calibration, recorded in ``_select_speaker_spans``'s docstring in
``singing_conversion_pipeline.py``): speaker-verification embeddings cannot do
identity on sung audio — different-person similarities of 0.855 and 0.900
exceeded most same-person pairs. A related result from 2026-08 is in the
project memory: Resemblyzer could not separate two different singers at all
(0.899 vs 0.907). Identity on singing is an ear call, not a measurement.

So the split of labour is deliberate and should stay this way:

* ``singing_conversion_pipeline._select_speaker_spans`` decides WHICH spans are
  a foreign voice — a relative spectral-centroid gate plus WavLM embedding
  reassignment, both calibrated on two real duets. Untouched by this module.
* This module decides only what REGISTER those already-flagged spans sit in,
  which is one bit derived from pitch, and which model the operator asked for.

Where the register bit comes from: the operator, not a measurement
------------------------------------------------------------------
``other_register`` derives it by ELIMINATION from the operator's own model
choices. The slot holding the request's main profile is the lead's register, so
the other filled slot is the co-vocal's. No threshold, no measurement, and it
covers every case the user described: one model chosen, the other voice stays
original; both chosen; neither.

An absolute median-f0 gate was built, measured and DISPROVED first
(``scratchpad/registercal/FINDINGS.md``, 2026-09-15, 60 profile samples plus
both calibration duets). It is disabled, not deleted:

* At the proposed 165-175 Hz split, **36 of 40** male training samples classify
  as female. The speech male/female boundary does not transfer to sung pop.
* No threshold has a positive margin. The optimum over the 60 samples is 271 Hz
  and still gets 6/40 male samples wrong. Per-sample medians: the two male
  singers span 128-392 Hz, the female target 274-315 Hz — the male upper tail
  sits inside and above the female's entire range. The worst-case margin is
  **-118 Hz**, not thin.
* On both duets the question dies one level upstream: the diarizer does not
  produce singer-pure clusters, with or without a forced speaker count, so no
  per-cluster statistic has clean spans to read. Per-cluster medians came out
  5.9 Hz apart (All of Me) and 33.2 Hz apart (Beauty and the Beast), each
  cluster holding both singers. WavLM timbre clustering was at chance (0.598)
  on the harder one, so finer segmentation would not rescue it either.

That result and the 2026-07-04 one point the same way: **on singing, neither
pitch nor timbre reliably recovers who is who.** Operator choice is the honest
mechanism, and it is what this module uses.

``classify_register`` survives as opt-in diagnostics. ``REGISTER_SPLIT_F0_HZ``
is ``None`` — disabled — so it reports the median f0 and no register. Set the
config key ``register_split_f0_hz`` to arm it if you have a threshold the
measurement above does not already rule out; its verdict lands in the ledger as
``measured_register`` and does NOT drive routing. ``REGISTER_MARGIN_HZ`` is a
symmetric dead zone around whatever split is armed, so a borderline group
answers "unknown" rather than guessing.

Singing spreads f0 far wider than speech does, in both directions, so a
*median* over voiced frames is used rather than a mean, and a low-voiced group
(a harmony stack, a texture) returns ``None`` instead of a number computed from
a handful of frames.
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# DISABLED, and disproved rather than merely uncalibrated — see the module
# docstring for the numbers. ``None`` means classify_register reports a median
# and no register. Config key ``register_split_f0_hz`` arms it; its verdict is
# diagnostics only and never drives routing.
REGISTER_SPLIT_F0_HZ = None

# Symmetric dead zone in Hz around whatever split is armed; inside it the
# register is ``None``. Config key: ``register_margin_hz``.
REGISTER_MARGIN_HZ = 15.0

# Below these the f0 median is not trustworthy and the register is ``None``.
REGISTER_MIN_AUDIO_S = 1.5
REGISTER_MIN_VOICED = 0.15

# pyin bounds. fmin is 110, NOT the 80 that ``_voiced_fraction`` uses: below
# ~110 pyin octave-halves on these stems and manufactures a spurious low mode
# (measured 2026-09-15 on both duets — median energy at the claimed f0 was
# 0.19x and 0.51x the spectral floor, i.e. no partial there at all), which
# corrupts a median. That gate cares whether a frame is voiced, this one cares
# what the pitch VALUE is, so the two parameters are deliberately decoupled.
PYIN_FMIN = 110
PYIN_FMAX = 1000

# Preserve reasons whose spans may be re-routed to an operator-chosen model.
#
# ``voice_category`` and ``embedding`` are the two AUTO-DETECTED foreign-voice
# verdicts: the software decided this is somebody else, so the operator gets to
# say which model sings it.
#
# Everything else stays original, and for two different reasons:
#   ``low_voiced``      harmony stacks and textures. The fork engine needs a
#                       clean monophonic f0 and butchers these — no model
#                       renders them well, so the register is irrelevant.
#   ``operator``,       the operator explicitly named this cluster or these
#   ``operator_range``  seconds as already-target / keep-verbatim. An explicit
#                       instruction outranks a detector.
ROUTABLE_REASONS = frozenset({'voice_category', 'embedding'})


def classify_register(
    audio: np.ndarray,
    sample_rate: int,
    split_hz: Optional[float] = REGISTER_SPLIT_F0_HZ,
    margin_hz: float = REGISTER_MARGIN_HZ,
    max_s: float = 20.0,
) -> Tuple[Optional[str], Optional[float]]:
    """Measure median f0 and, if a split is armed, call the register from it.

    DIAGNOSTICS ONLY by default. ``split_hz`` is ``None`` unless configured, in
    which case this returns ``(None, median)`` — the median is genuinely useful
    when judging a duet, the register verdict from it is not (see the module
    docstring's disproof). Routing takes its register from ``other_register``.

    Args:
        audio: Active (silence-dropped) mono audio for one span group, taken
            from the SEPARATED VOCAL STEM. Not the mix — the centroid gate in
            the pipeline measures the mix on purpose, and crossing the two
            would invalidate both calibrations.
        sample_rate: Sample rate of ``audio``.
        split_hz: Median-f0 boundary between the registers, or ``None`` to
            measure without classifying (the default).
        margin_hz: Symmetric dead zone around ``split_hz``; inside it the
            answer is unknown.
        max_s: Cap on how much audio is measured.

    Returns:
        ``(register, median_f0_hz)``. ``register`` is ``'male'``, ``'female'``
        or ``None``; ``median_f0_hz`` is ``None`` only when no f0 could be
        measured at all. An unknown register with a real median means either no
        split is armed or the median landed inside the dead zone.
    """
    import librosa

    a = np.asarray(audio, dtype=np.float32)[: int(max_s * sample_rate)]
    if len(a) < int(REGISTER_MIN_AUDIO_S * sample_rate):
        return None, None

    # pyin, not yin: yin never returns NaN and reports a confident ~1000 Hz on
    # silence (measured 2026-09-07, it cut voiced output -10.8 dB in the
    # subharmonic stage before it was caught). pyin's voiced_flag is the gate.
    f0, voiced_flag, _ = librosa.pyin(
        a, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=sample_rate)
    if f0 is None or voiced_flag is None or len(f0) == 0:
        return None, None
    voiced = np.asarray(voiced_flag, dtype=bool)
    if float(np.mean(voiced)) < REGISTER_MIN_VOICED:
        # Harmony stack, texture, or near-silence: too few voiced frames for a
        # median to mean anything.
        return None, None
    vals = np.asarray(f0, dtype=np.float64)[voiced]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None, None

    median = float(np.median(vals))
    if split_hz is None:
        return None, median
    if abs(median - float(split_hz)) < float(margin_hz):
        return None, median
    return ('female' if median >= float(split_hz) else 'male'), median


def other_register(
    target_profile_id: Optional[str],
    register_profiles: Optional[Dict[str, Optional[str]]],
) -> Optional[str]:
    """Which register the CO-VOCAL sings in, by elimination. Pure function.

    The operator already knows which singer is which — they are listening to
    the song — and they express it by putting a model in the 'female' or the
    'male' slot. The slot holding the request's main profile is the LEAD's
    register, so the remaining filled slot is the co-vocal's. That is the whole
    rule, and it needs no measurement, which matters because the measurement
    was disproved (see the module docstring).

    Args:
        target_profile_id: The lead's model — the request's main profile.
        register_profiles: ``{'female': id|None, 'male': id|None}``.

    Returns:
        The co-vocal's register, or ``None`` when no model was chosen for a
        voice other than the lead's (the user's "leaving the original voice of
        whatever voice a model has not been chosen for").

    Raises:
        ValueError: Both slots hold models and neither is the lead's, so which
            one the co-vocal should get is genuinely ambiguous. Refusing is the
            only honest answer — there is no register signal to break the tie.
    """
    lead = None if target_profile_id is None else str(target_profile_id)
    others = sorted(
        register for register, profile_id in (register_profiles or {}).items()
        if profile_id and str(profile_id) != lead
    )
    if not others:
        return None
    if len(others) > 1:
        raise ValueError(
            "register_profiles assigns a model to both the female and the male "
            "slot and neither is the lead's own profile (%s), so there is no "
            "way to tell which one the co-vocal should be sung with. Put the "
            "lead's own model in its register's slot, or choose a model for "
            "one register only." % (lead,))
    return others[0]


def route_action(
    reason: str,
    register: Optional[str],
    register_profiles: Optional[Dict[str, Optional[str]]],
) -> Tuple[str, Optional[str]]:
    """Decide what happens to one group of preserved spans. Pure function.

    Args:
        reason: Why the spans were preserved — see ``ROUTABLE_REASONS``.
        register: ``'male'``, ``'female'`` or ``None`` (unknown).
        register_profiles: ``{'female': id|None, 'male': id|None}`` as chosen
            by the operator. A missing key and ``None`` both mean "no model
            chosen for that register", which is the user's "leaving the
            original voice of whatever voice a model has not been chosen for".

    Returns:
        ``('convert', profile_id)`` or ``('keep', None)``. Keeping is the
        default for every uncertainty: a non-routable reason, an unknown
        register, or no model assigned to that register.
    """
    if reason not in ROUTABLE_REASONS:
        return ('keep', None)
    if not register:
        return ('keep', None)
    profile_id = (register_profiles or {}).get(register)
    if not profile_id:
        return ('keep', None)
    return ('convert', str(profile_id))
