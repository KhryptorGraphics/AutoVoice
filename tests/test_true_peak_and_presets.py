"""True-peak limiting and the fork HQ quality presets.

Two defects these cover, both measured on real Brandy conversions:

1. A converted vocal stem came out at 0.00 dBFS sample peak and +0.16 dBTP
   once the inter-sample peaks were reconstructed, with 5 samples at or above
   0.999. The existing guards compare a SAMPLE peak against 0.95, so they
   cannot see that.

2. The quality presets were a dict of 'n_steps'/'denoise' that no stage ever
   read, so choosing 'studio' changed nothing at all. The presets now set
   the chain's real knobs, and they are per-request overrides rather than
   mutations of the shared pipeline config.
"""
import numpy as np
import pytest

from auto_voice.audio.true_peak import (
    limit_true_peak,
    true_peak_dbfs,
)
from auto_voice.inference.quality_presets import (
    QUALITY_PRESETS,
    preset_overrides,
)

SR = 44100


def _tone(freq_hz=1000.0, seconds=0.5, amp=0.99, sr=SR):
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


class TestTruePeak:
    def test_sample_peak_below_zero_can_still_exceed_it(self):
        """The whole reason this module exists: sample peak != true peak."""
        # A 0.99-amplitude sine overshoots between its samples.
        x = _tone(amp=0.99)
        sample_peak_dbfs = 20 * np.log10(float(np.abs(x).max()))
        tp = true_peak_dbfs(x)
        assert sample_peak_dbfs < 0.0
        assert tp > sample_peak_dbfs

    def test_oversample_factor_one_is_the_sample_peak(self):
        x = _tone(amp=0.5)
        sample_peak_dbfs = 20 * np.log10(float(np.abs(x).max()))
        assert true_peak_dbfs(x, oversample=1) == pytest.approx(
            sample_peak_dbfs, abs=0.01)

    def test_silence_is_reported_as_very_low(self):
        assert true_peak_dbfs(np.zeros(1000, dtype=np.float32)) == -999.0

    def test_empty_input_is_reported_as_very_low(self):
        assert true_peak_dbfs(np.array([], dtype=np.float32)) == -999.0

    def test_multichannel_peak_is_the_worst_channel(self):
        left = _tone(amp=0.3)
        right = _tone(amp=0.9)
        stereo = np.stack([left, right], axis=1)
        mono = true_peak_dbfs(right)
        assert true_peak_dbfs(stereo) == pytest.approx(mono, abs=0.05)

    def test_limit_trims_to_the_ceiling(self):
        x = _tone(amp=0.99)
        trimmed, gain_db = limit_true_peak(x, target_dbfs=-1.0)
        assert gain_db < 0.0
        assert true_peak_dbfs(trimmed) == pytest.approx(-1.0, abs=0.1)

    def test_limit_leaves_a_quiet_signal_alone(self):
        x = _tone(amp=0.1)
        trimmed, gain_db = limit_true_peak(x, target_dbfs=-1.0)
        assert gain_db == 0.0
        assert np.array_equal(trimmed, x)

    def test_limit_preserves_the_waveform_shape(self):
        """A constant trim, not a compressor: only the gain changes."""
        x = _tone(amp=0.99)
        trimmed, _ = limit_true_peak(x, target_dbfs=-3.0)
        ratio = trimmed[trimmed != 0] / x[trimmed != 0]
        assert np.allclose(ratio, ratio[0], atol=1e-5)

    def test_limit_is_idempotent(self):
        x = _tone(amp=0.99)
        once, _ = limit_true_peak(x, target_dbfs=-1.0)
        twice, gain_db = limit_true_peak(once, target_dbfs=-1.0)
        assert gain_db == 0.0
        assert np.array_equal(once, twice)


class TestQualityPresets:
    def test_every_preset_names_the_chain_knobs(self):
        required = {
            'fork_hq_hum_suppress',
            'fork_hq_subharmonic_suppress',
            'fork_hq_voice_polish',
            'fork_hq_hf_detonalize',
            'fork_hq_air_cap_db',
            'fork_hq_extend_bandwidth',
            'fork_hq_match_source_bandwidth',
            'fork_hq_vocal_true_peak_dbfs',
        }
        for name, cfg in QUALITY_PRESETS.items():
            assert required <= set(cfg), name

    def test_balanced_is_what_the_pipeline_always_did(self):
        """The default must not change existing conversions."""
        cfg = QUALITY_PRESETS['balanced']
        assert cfg['fork_hq_hum_suppress'] is True
        assert cfg['fork_hq_subharmonic_suppress'] is True
        assert cfg['fork_hq_voice_polish'] is True
        assert cfg['fork_hq_hf_detonalize'] is True
        assert cfg['fork_hq_air_cap_db'] == 12.0
        assert cfg['fork_hq_extend_bandwidth'] is True
        assert cfg['fork_hq_match_source_bandwidth'] is True

    def test_cheaper_presets_do_no_air_work(self):
        """Air is the only stage that invents content, so skip it."""
        assert QUALITY_PRESETS['draft']['fork_hq_air_cap_db'] == 0.0
        assert QUALITY_PRESETS['fast']['fork_hq_air_cap_db'] == 0.0
        assert QUALITY_PRESETS['draft']['fork_hq_extend_bandwidth'] is False
        assert QUALITY_PRESETS['fast']['fork_hq_extend_bandwidth'] is False

    def test_presets_never_exceed_full_scale_on_the_stem(self):
        for name, cfg in QUALITY_PRESETS.items():
            assert cfg['fork_hq_vocal_true_peak_dbfs'] <= 0.0, name

    def test_unknown_preset_yields_no_overrides(self):
        """A bad preset name must not take the conversion down."""
        assert preset_overrides('nonexistent') == {}
        assert preset_overrides(None) == {}
        assert preset_overrides('') == {}

    def test_overrides_are_a_copy(self):
        """A caller mutating the result must not corrupt the table."""
        cfg = preset_overrides('high')
        cfg['fork_hq_air_cap_db'] = 999.0
        assert QUALITY_PRESETS['high']['fork_hq_air_cap_db'] == 18.0


class TestSeparatorThreadCap:
    """The separation subprocess must not try to own every core.

    Three concurrent conversions ran three audio-separator processes, each
    letting torch spawn 14 threads on a 14-core box: load average 36, and
    the separator at 39 s per iteration. One conversion alone took 1710 s;
    the same three together took over an hour, and the separation step was
    the whole difference.
    """

    def test_thread_budget_is_applied_to_the_subprocess_env(self):
        from auto_voice.inference.separation_bridge import (
            _SEPARATOR_THREADS,
            _clean_env,
        )
        env = _clean_env()
        threads = str(_SEPARATOR_THREADS)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            assert env[key] == threads, key

    def test_thread_budget_leaves_room_for_concurrent_jobs(self):
        """N concurrent jobs must fit in the machine, not each own it."""
        from auto_voice.inference.separation_bridge import _SEPARATOR_THREADS
        assert _SEPARATOR_THREADS >= 1
        # 3 concurrent conversions x the budget must stay inside 14 cores.
        assert 3 * _SEPARATOR_THREADS <= 14

    def test_serving_env_leakage_is_still_stripped(self):
        from auto_voice.inference.separation_bridge import _clean_env
        env = _clean_env()
        for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONNOUSERSITE'):
            assert key not in env


class TestChainConfigPrecedence:
    """Operator-set settings must win over a quality preset.

    A preset's overrides contain all 8 chain keys, so if the preset wins
    outright the stored config is never consulted and a PATCH'd air cap has
    no effect. The precedence is: code defaults < preset < explicit operator
    setting.
    """

    def _pipeline(self):
        from auto_voice.inference.singing_conversion_pipeline import (
            SingingConversionPipeline,
        )
        return SingingConversionPipeline(config={'fork_hq_air_cap_db': 12.0})

    def test_preset_wins_over_code_default(self):
        """With no explicit setting, the preset's value applies."""
        from auto_voice.inference.quality_presets import preset_overrides
        p = self._pipeline()
        cfg = p._chain_config(preset_overrides('high'))
        assert cfg['fork_hq_air_cap_db'] == 18.0

    def test_explicit_setting_wins_over_preset(self):
        """An operator's PATCH beats the preset's profile."""
        from auto_voice.inference.quality_presets import preset_overrides
        p = self._pipeline()
        p.set_explicit_settings({'fork_hq_air_cap_db': 20.0})
        cfg = p._chain_config(preset_overrides('high'))
        assert cfg['fork_hq_air_cap_db'] == 20.0

    def test_explicit_setting_persists_across_requests(self):
        """The explicit set is not consumed by one conversion."""
        from auto_voice.inference.quality_presets import preset_overrides
        p = self._pipeline()
        p.set_explicit_settings({'fork_hq_air_cap_db': 20.0})
        p._chain_config(preset_overrides('high'))
        p._chain_config(preset_overrides('balanced'))
        assert p._chain_config(preset_overrides('high'))['fork_hq_air_cap_db'] == 20.0

    def test_preset_fills_keys_the_operator_never_set(self):
        """An explicit setting on one key must not silence the preset's others."""
        from auto_voice.inference.quality_presets import preset_overrides
        p = self._pipeline()
        p.set_explicit_settings({'fork_hq_air_cap_db': 20.0})
        cfg = p._chain_config(preset_overrides('band_limited'))
        assert cfg['fork_hq_air_cap_db'] == 20.0          # explicit wins
        assert cfg['fork_hq_match_source_bandwidth'] is False  # preset still applies

    def test_no_preset_uses_config(self):
        """With no preset and no explicit setting, the config is unchanged."""
        p = self._pipeline()
        cfg = p._chain_config(None)
        assert cfg['fork_hq_air_cap_db'] == 12.0

    def test_set_explicit_settings_merges(self):
        """Recording settings twice does not drop the first batch."""
        p = self._pipeline()
        p.set_explicit_settings({'fork_hq_air_cap_db': 20.0})
        p.set_explicit_settings({'fork_hq_vocal_true_peak_dbfs': -2.0})
        cfg = p._chain_config(None)
        assert cfg['fork_hq_air_cap_db'] == 20.0
        assert cfg['fork_hq_vocal_true_peak_dbfs'] == -2.0


class TestUnsetExplicitSettings:
    """An override must be removable, or a preset can never govern again.

    set_explicit_settings only ever merged, so a key the operator set once
    stayed in _explicit_settings for the life of the process - and
    _chain_config ranks that dict above any preset. The only way back to
    preset authority was a restart with the key edited out of the store by
    hand.
    """

    def _pipeline(self):
        from auto_voice.inference.singing_conversion_pipeline import (
            SingingConversionPipeline,
        )
        return SingingConversionPipeline(config={'fork_hq_air_cap_db': 12.0})

    def test_unset_restores_preset_authority(self):
        """After an unset, the preset's value governs again."""
        from auto_voice.inference.quality_presets import preset_overrides
        p = self._pipeline()
        p.set_explicit_settings({'fork_hq_air_cap_db': 20.0})
        assert p._chain_config(preset_overrides('band_limited'))['fork_hq_air_cap_db'] == 20.0

        p.unset_explicit_settings(['fork_hq_air_cap_db'])

        cfg = p._chain_config(preset_overrides('band_limited'))
        assert cfg['fork_hq_air_cap_db'] == 18.0
        assert cfg['fork_hq_match_source_bandwidth'] is False

    def test_unset_removes_the_key_from_the_explicit_set(self):
        """The override is gone from the dict, not merely outranked."""
        p = self._pipeline()
        p.set_explicit_settings({'fork_hq_air_cap_db': 20.0})
        p.unset_explicit_settings(['fork_hq_air_cap_db'])
        assert 'fork_hq_air_cap_db' not in p._explicit_settings

    def test_unset_leaves_other_overrides_alone(self):
        """Unsetting one key must not drop the others."""
        p = self._pipeline()
        p.set_explicit_settings({
            'fork_hq_air_cap_db': 20.0,
            'fork_hq_vocal_true_peak_dbfs': -2.0,
        })
        p.unset_explicit_settings(['fork_hq_air_cap_db'])
        cfg = p._chain_config(None)
        # The unset key falls back to the baseline the pipeline was built
        # with, not to a stale override.
        assert cfg['fork_hq_air_cap_db'] == 12.0
        assert cfg['fork_hq_vocal_true_peak_dbfs'] == -2.0  # still overridden

    def test_unset_falls_through_to_the_preset(self):
        """With the override gone, the preset's value governs the key."""
        from auto_voice.inference.quality_presets import preset_overrides
        p = self._pipeline()
        p.set_explicit_settings({'fork_hq_air_cap_db': 20.0})
        p.unset_explicit_settings(['fork_hq_air_cap_db'])
        for preset, expected in (('balanced', 12.0), ('high', 18.0),
                                 ('band_limited', 18.0), ('draft', 0.0)):
            cfg = p._chain_config(preset_overrides(preset))
            assert cfg['fork_hq_air_cap_db'] == expected, preset

    def test_unset_restores_the_baseline_in_the_pipeline_config(self):
        """The PATCH handler writes into pipeline.config; the override there
        must go too, or it survives in the lowest-precedence layer."""
        p = self._pipeline()
        p.config['fork_hq_air_cap_db'] = 20.0
        p.set_explicit_settings({'fork_hq_air_cap_db': 20.0})
        p.unset_explicit_settings(['fork_hq_air_cap_db'])
        assert p.config['fork_hq_air_cap_db'] == 12.0

    def test_unset_removes_a_key_that_has_no_baseline(self):
        """A key with no baseline is removed, so the stage reads its own
        code default rather than a value the pipeline never had."""
        p = self._pipeline()
        p.config['fork_hq_never_shipped'] = 7.0
        p.set_explicit_settings({'fork_hq_never_shipped': 7.0})
        p.unset_explicit_settings(['fork_hq_never_shipped'])
        assert 'fork_hq_never_shipped' not in p.config

    def test_unset_preserves_a_yaml_supplied_value(self):
        """A value the config shipped with must survive an override+unset."""
        from auto_voice.inference.singing_conversion_pipeline import (
            SingingConversionPipeline,
        )
        p = SingingConversionPipeline(config={'fork_hq_air_cap_db': 9.5})
        p.set_explicit_settings({'fork_hq_air_cap_db': 20.0})
        p.unset_explicit_settings(['fork_hq_air_cap_db'])
        assert p.config['fork_hq_air_cap_db'] == 9.5

    def test_unset_of_an_unknown_key_is_a_no_op(self):
        """Unsetting a key that was never set must not raise."""
        p = self._pipeline()
        p.set_explicit_settings({'fork_hq_air_cap_db': 20.0})
        p.unset_explicit_settings(['fork_hq_not_a_real_key'])
        assert p._chain_config(None)['fork_hq_air_cap_db'] == 20.0

    def test_unset_survives_a_partially_constructed_instance(self):
        """A test double built with __new__ has no _explicit_settings."""
        from auto_voice.inference.singing_conversion_pipeline import (
            SingingConversionPipeline,
        )
        p = object.__new__(SingingConversionPipeline)
        p.config = {}
        p.unset_explicit_settings(['fork_hq_air_cap_db'])
        assert p._explicit_settings == {}
