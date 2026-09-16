"""Register classification and the routing decision table.

Register is ONE BIT FROM PITCH, not identity — see ``vocal_register``'s module
docstring for the measured negative result that keeps those apart. These tests
pin that bit and the decision table built on it; end-to-end routing through the
pipeline is integration-level and validated by a real render.
"""
import numpy as np
import pytest

from auto_voice.inference.vocal_register import (
    REGISTER_MARGIN_HZ,
    REGISTER_SPLIT_F0_HZ,
    ROUTABLE_REASONS,
    classify_register,
    other_register,
    route_action,
)

SR = 16000

# The disproved absolute gate, armed explicitly. Tests below that pass this are
# exercising the OPT-IN mechanism, not a default: 175 Hz misclassified 36 of 40
# male training samples (scratchpad/registercal/FINDINGS.md, 2026-09-15).
ARMED_SPLIT = 175.0


def harmonic_voice(f0, seconds=2.5, sr=SR, n_harmonics=12, seed=0):
    """A synthetic sung vowel: harmonic stack with a falling spectral slope.

    Real enough for pyin (it tracks the fundamental of a harmonic series);
    deliberately not real enough to stand in for a voice anywhere else.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    # Slight vibrato so every frame is not bit-identical.
    phase = 2 * np.pi * f0 * t + 0.25 * np.sin(2 * np.pi * 5.0 * t)
    y = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        if f0 * k >= sr / 2:
            break
        y += np.sin(k * phase) / k
    y += 0.001 * rng.standard_normal(len(t))
    return (0.5 * y / np.max(np.abs(y))).astype(np.float32)


class TestClassifyRegister:
    """Two synthetic voices an octave apart, either side of an ARMED split."""

    def test_disabled_by_default_but_still_measures(self):
        """The absolute gate is disproved, so the default reports the median
        and no register — nothing routes on it."""
        assert REGISTER_SPLIT_F0_HZ is None
        register, median = classify_register(harmonic_voice(120.0), SR)
        assert register is None
        assert median == pytest.approx(120.0, abs=6.0)

    def test_low_voice_is_male(self):
        register, median = classify_register(harmonic_voice(120.0), SR,
                                             split_hz=ARMED_SPLIT)
        assert register == 'male'
        assert median == pytest.approx(120.0, abs=6.0)

    def test_high_voice_is_female(self):
        register, median = classify_register(harmonic_voice(260.0), SR,
                                             split_hz=ARMED_SPLIT)
        assert register == 'female'
        assert median == pytest.approx(260.0, abs=12.0)

    def test_split_is_configurable(self):
        """Moving the armed split moves the verdict."""
        register, _ = classify_register(harmonic_voice(260.0), SR,
                                        split_hz=400.0, margin_hz=5.0)
        assert register == 'male', "260 Hz sits below a 400 Hz split"

    def test_margin_dead_zone_returns_unknown(self):
        """Inside the dead zone the answer is unknown, and a real median."""
        register, median = classify_register(
            harmonic_voice(ARMED_SPLIT), SR,
            split_hz=ARMED_SPLIT, margin_hz=40.0)
        assert register is None
        assert median is not None, "unknown-by-margin still reports its median"
        assert abs(median - ARMED_SPLIT) < 40.0

    def test_margin_zero_still_classifies(self):
        register, _ = classify_register(harmonic_voice(120.0), SR,
                                        split_hz=ARMED_SPLIT, margin_hz=0.0)
        assert register == 'male'

    def test_noise_is_unknown(self):
        """A harmony stack / texture has too few voiced frames to measure."""
        rng = np.random.default_rng(1)
        noise = (0.2 * rng.standard_normal(int(2.5 * SR))).astype(np.float32)
        register, _ = classify_register(noise, SR)
        assert register is None

    def test_too_short_is_unknown(self):
        register, median = classify_register(harmonic_voice(120.0, 0.5), SR)
        assert register is None and median is None

    def test_empty_is_unknown(self):
        register, median = classify_register(np.zeros(0, dtype=np.float32), SR)
        assert register is None and median is None

    def test_silence_is_unknown(self):
        """pyin's voiced_flag, not yin: yin reports a confident ~1000 Hz on
        silence (the 2026-09-07 bug that cut voiced output -10.8 dB)."""
        register, _ = classify_register(np.zeros(int(3 * SR), np.float32), SR)
        assert register is None

    def test_margin_default_is_symmetric(self):
        """Both registers survive the default dead zone at an octave apart."""
        low, _ = classify_register(harmonic_voice(120.0), SR,
                                   split_hz=ARMED_SPLIT,
                                   margin_hz=REGISTER_MARGIN_HZ)
        high, _ = classify_register(harmonic_voice(260.0), SR,
                                    split_hz=ARMED_SPLIT,
                                    margin_hz=REGISTER_MARGIN_HZ)
        assert (low, high) == ('male', 'female')


# target_profile_id, register_profiles -> the CO-VOCAL's register
#
# The rule: the slot holding the lead's own model is the lead's register, so
# the other filled slot is the co-vocal's. No measurement — the absolute f0
# gate was disproved (scratchpad/registercal/FINDINGS.md, 2026-09-15).
ELIMINATION_TABLE = [
    # lead is the female slot -> the co-vocal is the male one
    ('brandy', {'female': 'brandy', 'male': 'conor'}, 'male'),
    # lead is the male slot -> the co-vocal is the female one
    ('conor', {'female': 'brandy', 'male': 'conor'}, 'female'),
    # only one model chosen and it is not the lead's: that is the co-vocal's
    ('brandy', {'male': 'conor'}, 'male'),
    ('conor', {'female': 'brandy'}, 'female'),
    # only the lead's own model is named: nothing to route to
    ('brandy', {'female': 'brandy'}, None),
    ('brandy', {'female': 'brandy', 'male': None}, None),
    # nothing chosen at all
    ('brandy', {}, None),
    ('brandy', None, None),
    ('brandy', {'female': None, 'male': None}, None),
    # non-string ids compare by value, not identity
    (7, {'female': 7, 'male': 'conor'}, 'male'),
]


@pytest.mark.parametrize("lead,profiles,expected", ELIMINATION_TABLE)
def test_other_register(lead, profiles, expected):
    assert other_register(lead, profiles) == expected


def test_other_register_refuses_a_genuine_tie():
    """Both slots filled, neither is the lead's: nothing can break the tie, so
    refuse rather than guess. Caught in validation, before any work."""
    with pytest.raises(ValueError) as exc:
        other_register('brandy', {'female': 'someone', 'male': 'conor'})
    assert 'brandy' in str(exc.value)


# reason, register, register_profiles -> expected (action, profile_id)
#
# The two things this table exists to pin:
#   * only the AUTO-DETECTED foreign-voice reasons are routable; a harmony
#     stack and the operator's own explicit preserve are never re-sung;
#   * every uncertainty keeps the original voice, which is the user's
#     "leaving the original voice of whatever voice a model has not been
#     chosen for".
BOTH = {'female': 'fem-id', 'male': 'male-id'}
FEM_ONLY = {'female': 'fem-id', 'male': None}
MALE_ONLY = {'male': 'male-id'}

ROUTING_TABLE = [
    # routable reason + register with a model -> convert
    ('voice_category', 'male', BOTH, ('convert', 'male-id')),
    ('voice_category', 'female', BOTH, ('convert', 'fem-id')),
    ('embedding', 'male', BOTH, ('convert', 'male-id')),
    ('embedding', 'female', BOTH, ('convert', 'fem-id')),
    # one register chosen, the other keeps its original voice
    ('voice_category', 'male', FEM_ONLY, ('keep', None)),
    ('voice_category', 'female', FEM_ONLY, ('convert', 'fem-id')),
    ('voice_category', 'male', MALE_ONLY, ('convert', 'male-id')),
    ('voice_category', 'female', MALE_ONLY, ('keep', None)),
    # harmony stacks stay original whatever the register or the assignment
    ('low_voiced', 'male', BOTH, ('keep', None)),
    ('low_voiced', 'female', BOTH, ('keep', None)),
    # the operator's explicit choices outrank the detector
    ('operator', 'male', BOTH, ('keep', None)),
    ('operator', 'female', BOTH, ('keep', None)),
    ('operator_range', 'male', BOTH, ('keep', None)),
    ('operator_range', 'female', BOTH, ('keep', None)),
    # unknown register never guesses
    ('voice_category', None, BOTH, ('keep', None)),
    ('embedding', None, BOTH, ('keep', None)),
    # nothing chosen at all is today's behaviour
    ('voice_category', 'male', {}, ('keep', None)),
    ('voice_category', 'male', None, ('keep', None)),
    ('voice_category', 'male', {'male': None}, ('keep', None)),
    ('voice_category', 'male', {'female': 'fem-id'}, ('keep', None)),
]


@pytest.mark.parametrize("reason,register,profiles,expected", ROUTING_TABLE)
def test_route_action(reason, register, profiles, expected):
    assert route_action(reason, register, profiles) == expected


def test_routable_reasons_are_only_the_detected_ones():
    """A new preserve reason must opt IN to routing, never inherit it."""
    assert ROUTABLE_REASONS == {'voice_category', 'embedding'}
    for reason in ('low_voiced', 'operator', 'operator_range', 'something_new'):
        assert route_action(reason, 'male', BOTH) == ('keep', None)


def test_profile_id_is_stringified():
    assert route_action('voice_category', 'male', {'male': 7}) == ('convert', '7')
