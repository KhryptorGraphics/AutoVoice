"""Test VoiceCloner extensions: SNR validation, multi-sample support, and versioning"""

import pytest
import numpy as np
from pathlib import Path

try:
    from src.auto_voice.inference.voice_cloner import VoiceCloner, InvalidAudioError, VoiceCloningError
except ImportError:
    pytest.skip("VoiceCloner not available", allow_module_level=True)


class TestSNRValidation:
    """Test SNR computation and validation"""

    def test_compute_snr_basic(self):
        """Test basic SNR computation"""
        cloner = VoiceCloner(config={'min_snr_db': None})  # Disable validation

        # Create audio with known SNR characteristics
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Signal: 440 Hz sine wave
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Add noise
        noise = 0.05 * np.random.randn(len(signal))
        audio = signal + noise

        # Compute SNR
        snr = cloner._compute_snr(audio.astype(np.float32))

        assert snr is not None
        assert snr > 10.0  # Should have decent SNR
        assert snr < 50.0  # Reasonable upper bound
        print(f"Computed SNR: {snr:.1f} dB")

    def test_compute_snr_too_short(self):
        """Test SNR computation with too-short audio"""
        cloner = VoiceCloner(config={'min_snr_db': None})

        # Very short audio
        audio = np.random.randn(1000).astype(np.float32)
        snr = cloner._compute_snr(audio)

        # Should return None for too-short audio
        assert snr is None

    def test_validate_audio_with_snr(self):
        """Test audio validation with SNR threshold"""
        cloner = VoiceCloner(config={
            'min_snr_db': 15.0,
            'min_duration': 1.0,
            'max_duration': 120.0
        })

        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # High SNR audio (clean signal)
        clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        clean_noise = 0.01 * np.random.randn(len(clean_signal))
        clean_audio = (clean_signal + clean_noise).astype(np.float32)

        is_valid, error_msg, error_code = cloner.validate_audio(clean_audio, sample_rate)
        assert is_valid, f"Clean audio should be valid: {error_msg}"

        # Low SNR audio (noisy signal)
        noisy_signal = 0.1 * np.sin(2 * np.pi * 440 * t)
        noisy_noise = 0.5 * np.random.randn(len(noisy_signal))
        noisy_audio = (noisy_signal + noisy_noise).astype(np.float32)

        is_valid, error_msg, error_code = cloner.validate_audio(noisy_audio, sample_rate)
        assert not is_valid
        assert error_code == 'snr_too_low'
        print(f"Noisy audio rejected: {error_msg}")


class TestAudioQualityReport:
    """Test audio quality report generation"""

    def test_quality_report_basic(self):
        """Test basic quality report generation"""
        cloner = VoiceCloner(config={
            'min_snr_db': None,
            'min_duration': 1.0,
            'max_duration': 120.0
        })

        # Create test audio
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # Generate report
        report = cloner.get_audio_quality_report(audio, sample_rate)

        assert 'duration' in report
        assert 'sample_rate' in report
        assert 'rms' in report
        assert 'peak' in report
        assert 'snr_db' in report
        assert 'dynamic_range_db' in report
        assert 'is_valid' in report
        assert 'validation_errors' in report

        assert abs(report['duration'] - duration) < 0.01
        assert report['sample_rate'] == sample_rate
        assert 0 < report['rms'] < 1.0
        assert 0 < report['peak'] <= 1.0

        print(f"Quality report: SNR={report['snr_db']:.1f} dB, RMS={report['rms']:.3f}")


class TestMultiSampleSupport:
    """Test multi-sample profile creation and management"""

    def test_create_multi_sample_profile(self):
        """Test creating profile from multiple samples"""
        cloner = VoiceCloner(config={
            'min_duration': 1.0,
            'max_duration': 120.0,
            'min_snr_db': None,
            'multi_sample_min_samples': 2,
            'multi_sample_max_samples': 5
        })

        # Create multiple audio samples
        sample_rate = 22050
        duration = 1.0
        samples = []

        for i in range(3):
            t = np.linspace(0, duration, int(sample_rate * duration))
            # Slightly different frequencies to simulate different recordings
            audio = 0.5 * np.sin(2 * np.pi * (440 + i * 10) * t)
            audio = audio.astype(np.float32)
            samples.append(audio)

        # Create profile
        profile = cloner.create_voice_profile_from_multiple_samples(
            audio_samples=samples,
            user_id='test_user',
            sample_rate=sample_rate,
            metadata={'test': True}
        )

        assert 'profile_id' in profile
        assert profile['user_id'] == 'test_user'
        assert 'multi_sample_info' in profile
        assert profile['multi_sample_info']['num_samples'] == 3
        assert 'embedding' not in profile  # Not returned in response

        print(f"Created multi-sample profile with {profile['multi_sample_info']['num_samples']} samples")

        # Verify version history
        assert 'profile_version' in profile
        assert profile['profile_version'] == 1
        assert 'version_history' in profile
        assert len(profile['version_history']) == 1

    def test_insufficient_samples(self):
        """Test error when insufficient samples provided"""
        cloner = VoiceCloner(config={
            'multi_sample_min_samples': 3,
            'min_duration': 1.0,
            'max_duration': 120.0
        })

        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # Only provide 1 sample when 3 required
        with pytest.raises(InvalidAudioError) as exc_info:
            cloner.create_voice_profile_from_multiple_samples(
                audio_samples=[audio],
                sample_rate=sample_rate
            )

        assert exc_info.value.error_code == 'insufficient_samples'


class TestProfileVersioning:
    """Test profile versioning functionality"""

    def test_single_sample_versioning(self):
        """Test versioning with single-sample profile"""
        cloner = VoiceCloner(config={
            'min_duration': 1.0,
            'max_duration': 120.0,
            'min_snr_db': None,
            'versioning_enabled': True
        })

        # Create test audio
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # Create profile
        profile = cloner.create_voice_profile(
            audio=audio,
            user_id='test_user',
            sample_rate=sample_rate
        )

        # Verify versioning fields
        assert 'schema_version' in profile
        assert 'profile_version' in profile
        assert profile['profile_version'] == 1
        assert 'version_history' in profile
        assert len(profile['version_history']) == 1

        history_entry = profile['version_history'][0]
        assert history_entry['version'] == 1
        assert 'timestamp' in history_entry
        assert 'change_description' in history_entry

    def test_add_sample_versioning(self):
        """Test versioning when adding samples"""
        cloner = VoiceCloner(config={
            'min_duration': 1.0,
            'max_duration': 120.0,
            'min_snr_db': None,
            'versioning_enabled': True,
            'multi_sample_max_samples': 5
        })

        # Create initial profile
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio1 = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        profile = cloner.create_voice_profile(
            audio=audio1,
            user_id='test_user',
            sample_rate=sample_rate
        )

        profile_id = profile['profile_id']
        assert profile['profile_version'] == 1

        # Add another sample
        audio2 = (0.5 * np.sin(2 * np.pi * 450 * t)).astype(np.float32)
        updated_profile = cloner.add_sample_to_profile(
            profile_id=profile_id,
            audio=audio2,
            sample_rate=sample_rate
        )

        # Verify version incremented
        assert updated_profile['profile_version'] == 2
        assert len(updated_profile['version_history']) == 2
        assert updated_profile['multi_sample_info']['num_samples'] == 2

        # Check version history
        history = cloner.get_profile_version_history(profile_id)
        assert len(history) == 2
        assert history[0]['version'] == 1
        assert history[1]['version'] == 2
        assert 'Added sample' in history[1]['change_description']

        print(f"Profile updated to version {updated_profile['profile_version']}")


class TestIntegration:
    """Integration tests for all features"""

    def test_full_workflow(self):
        """Test complete workflow with SNR, multi-sample, and versioning"""
        cloner = VoiceCloner(config={
            'min_duration': 1.0,
            'max_duration': 120.0,
            'min_snr_db': 10.0,
            'multi_sample_quality_weighting': True,
            'versioning_enabled': True
        })

        sample_rate = 22050
        duration = 1.0

        # Create samples with different SNR
        samples = []
        for i in range(3):
            t = np.linspace(0, duration, int(sample_rate * duration))
            signal = 0.5 * np.sin(2 * np.pi * 440 * t)
            # Varying noise levels
            noise = (0.01 + i * 0.005) * np.random.randn(len(signal))
            audio = (signal + noise).astype(np.float32)
            samples.append(audio)

        # Generate quality reports
        print("\nQuality reports:")
        for i, sample in enumerate(samples):
            report = cloner.get_audio_quality_report(sample, sample_rate)
            print(f"  Sample {i}: SNR={report['snr_db']:.1f} dB, Valid={report['is_valid']}")

        # Create multi-sample profile
        profile = cloner.create_voice_profile_from_multiple_samples(
            audio_samples=samples,
            user_id='integration_test',
            sample_rate=sample_rate
        )

        assert profile['multi_sample_info']['num_samples'] == 3
        assert profile['profile_version'] == 1

        # Get version history
        history = cloner.get_profile_version_history(profile['profile_id'])
        assert len(history) == 1

        print(f"\nCreated profile {profile['profile_id']} with {len(samples)} samples")
        print(f"Average SNR: {profile['multi_sample_info']['average_snr_db']:.1f} dB")


if __name__ == '__main__':
    # Run basic tests
    print("Testing SNR validation...")
    test_snr = TestSNRValidation()
    test_snr.test_compute_snr_basic()

    print("\nTesting quality report...")
    test_quality = TestAudioQualityReport()
    test_quality.test_quality_report_basic()

    print("\nTesting multi-sample support...")
    test_multi = TestMultiSampleSupport()
    test_multi.test_create_multi_sample_profile()

    print("\nTesting versioning...")
    test_version = TestProfileVersioning()
    test_version.test_single_sample_versioning()

    print("\nRunning integration test...")
    test_integration = TestIntegration()
    test_integration.test_full_workflow()

    print("\n✓ All tests passed!")
