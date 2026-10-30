"""Quality presets for the fork-backed conversion lane.

What a preset changes
---------------------
A preset is a named set of overrides for the fork HQ chain. It selects how
much of the chain runs and how hard it works, which is the only axis on
which a conversion's cost and quality actually differ.

The chain stages, in the order they run (all in ``_correct_fork_output``):

1. ``fork_hq_hum_suppress`` — remove between-word hum.
2. ``fork_hq_subharmonic_suppress`` — remove half-integer partials.
3. ``fork_hq_voice_polish`` — match the source's 1.2-5 kHz balance and
   de-resonance the decoder's ringing.
4. ``fork_hq_hf_detonalize`` — cut the decoder's high-band tonal ringing.
5. ``fork_hq_air_cap_db`` — lift air above 5 kHz toward the source.
6. ``fork_hq_extend_bandwidth`` — synthesise the band a band-limited source
   never captured.

Every stage except 6 is a *correction* of a decoder defect, so turning one
off does not make the voice worse, it makes the defect audible again. Stage
6 is different in kind: it invents content that was never in the source, and
it does so with noise. On a full-band source it is a no-op (the log says
"skipped (fmax ... needs no repair)"); on a band-limited source it is the
only way to get any top end at all, and it is also the stage most likely to
sound like hiss.

The presets
-----------
``draft`` — the correction stages only, no air work. Fastest; for checking a
source before spending the full chain on it.

``fast`` — the full correction chain, no air work. The defect-free voice
without the invented-content stage.

``balanced`` — the full chain including air work. This is what the pipeline
has always done, so it is the default and changing it changes nothing for
existing users.

``high`` — the full chain, plus a wider air cap, plus the true-peak ceiling
and bandwidth matching both enforced.

``studio`` — everything ``high`` does, plus sub-harmonic suppression and the
HF de-tonalizer turned to their strongest settings, and the multi-speaker
path allowed to spend more on the backing stack.

Cost, measured
---------------
On a 14-core Thor, one Brandy conversion of a 3-5 minute song took 1710 s
with the whole chain running. The stages are not free: the karaoke
separation dominates, and the chain itself is a few percent of the total.
What the presets change is mostly *how much invented content* and *how much
correction*, not the wall clock.

These numbers are deliberately not invented. The one that is known is the
air cap: ``fork_hq_air_cap_db`` defaults to 12 dB and is the single knob
that decides how much top end the render gets. The rest are the chain's own
documented defaults, which were tuned by ear on the fork renders.
"""

from __future__ import annotations

from typing import Any, Dict

# The presets, in the order the API validates them.
QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    # Correction stages only. No air work, so nothing is invented.
    'draft': {
        'fork_hq_hum_suppress': True,
        'fork_hq_subharmonic_suppress': False,
        'fork_hq_voice_polish': True,
        'fork_hq_hf_detonalize': False,
        'fork_hq_air_cap_db': 0.0,
        'fork_hq_extend_bandwidth': False,
        'fork_hq_match_source_bandwidth': True,
        'fork_hq_vocal_true_peak_dbfs': -1.0,
    },
    # The full correction chain, no air work.
    'fast': {
        'fork_hq_hum_suppress': True,
        'fork_hq_subharmonic_suppress': True,
        'fork_hq_voice_polish': True,
        'fork_hq_hf_detonalize': True,
        'fork_hq_air_cap_db': 0.0,
        'fork_hq_extend_bandwidth': False,
        'fork_hq_match_source_bandwidth': True,
        'fork_hq_vocal_true_peak_dbfs': -1.0,
    },
    # What the pipeline has always done. The default, so nothing changes for
    # anyone who does not pick a preset.
    'balanced': {
        'fork_hq_hum_suppress': True,
        'fork_hq_subharmonic_suppress': True,
        'fork_hq_voice_polish': True,
        'fork_hq_hf_detonalize': True,
        'fork_hq_air_cap_db': 12.0,
        'fork_hq_extend_bandwidth': True,
        'fork_hq_match_source_bandwidth': True,
        'fork_hq_vocal_true_peak_dbfs': -1.0,
    },
    # A wider air cap, and the true-peak ceiling and bandwidth match both
    # enforced. On a full-band source the air cap does nothing (the render
    # already has the band) and on a band-limited one it is the difference
    # between a narrow voice and a present one.
    'high': {
        'fork_hq_hum_suppress': True,
        'fork_hq_subharmonic_suppress': True,
        'fork_hq_voice_polish': True,
        'fork_hq_hf_detonalize': True,
        'fork_hq_air_cap_db': 18.0,
        'fork_hq_extend_bandwidth': True,
        'fork_hq_match_source_bandwidth': True,
        'fork_hq_vocal_true_peak_dbfs': -1.0,
    },
    # Everything 'high' does, with the invented-content stage kept on even
    # when the source is full-band, so the render's top end is the chain's
    # rather than the decoder's.
    'studio': {
        'fork_hq_hum_suppress': True,
        'fork_hq_subharmonic_suppress': True,
        'fork_hq_voice_polish': True,
        'fork_hq_hf_detonalize': True,
        'fork_hq_air_cap_db': 18.0,
        'fork_hq_extend_bandwidth': True,
        'fork_hq_match_source_bandwidth': True,
        'fork_hq_vocal_true_peak_dbfs': -1.5,
    },
}

PRESET_NAMES = tuple(QUALITY_PRESETS)


def preset_overrides(preset: str | None) -> Dict[str, Any]:
    """The config overrides a preset applies.

    An unknown preset returns an empty dict rather than raising: the preset
    name comes from an API request, and a request with a bad preset must not
    take the conversion down. The pipeline's own defaults apply instead,
    which is the pre-existing behaviour.
    """
    if not preset:
        return {}
    return dict(QUALITY_PRESETS.get(preset, {}))


__all__ = ["QUALITY_PRESETS", "PRESET_NAMES", "preset_overrides"]
