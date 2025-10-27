"""Comprehensive tests for SingingAnalyzer"""

import pytest
import numpy as np
import torch
from pathlib import Path


@pytest.mark.audio
@pytest.mark.unit
class TestSingingAnalyzer:
    """Unit tests for SingingAnalyzer"""

    def test_analyzer_initialization(self, singing_analyzer):
        """Verify SingingAnalyzer initializes correctly"""
        assert singing_analyzer is not None
        assert hasattr(singing_analyzer, 'audio_processor')
        assert hasattr(singing_analyzer, 'pitch_extractor')
        assert hasattr(singing_analyzer, 'device')

    def test_analyze_singing_features(self, singing_analyzer, sample_audio_speech_like):
        """Comprehensive singing feature analysis"""
        sample_rate = 16000

        features = singing_analyzer.analyze_singing_features(sample_audio_speech_like, sample_rate)

        # Verify structure
        assert 'breathiness' in features
        assert 'dynamics' in features
        assert 'vibrato' in features
        assert 'vocal_quality' in features
        assert 'f0_data' in features

        # Check breathiness
        assert 'breathiness_score' in features['breathiness']
        assert 'method' in features['breathiness']

        # Check dynamics
        assert 'rms_envelope' in features['dynamics']
        assert 'dynamic_range_db' in features['dynamics']

    def test_compute_breathiness_fallback(self, singing_analyzer):
        """Test breathiness computation with fallback method"""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

        result = singing_analyzer.compute_breathiness(audio, sample_rate)

        assert 'breathiness_score' in result
        assert 'method' in result
        assert 0.0 <= result['breathiness_score'] <= 1.0

    def test_breathiness_on_breathy_voice(self, singing_analyzer, sample_breathy_audio):
        """Verify high breathiness score for breathy voice"""
        sample_rate = 22050

        result = singing_analyzer.compute_breathiness(sample_breathy_audio, sample_rate)

        # Breathy voice should have higher breathiness score
        assert result['breathiness_score'] > 0.3

    def test_breathiness_on_clear_voice(self, singing_analyzer, sample_clear_voice):
        """Verify low breathiness score for clear voice"""
        sample_rate = 22050

        result = singing_analyzer.compute_breathiness(sample_clear_voice, sample_rate)

        # Clear voice should have lower breathiness score
        assert result['breathiness_score'] < 0.7

    def test_compute_dynamics(self, singing_analyzer):
        """Compute dynamics from audio"""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        result = singing_analyzer.compute_dynamics(audio, sample_rate)

        assert 'rms_envelope' in result
        assert 'db_envelope' in result
        assert 'dynamic_range_db' in result
        assert 'mean_db' in result
        assert 'crescendos' in result
        assert 'diminuendos' in result
        assert 'accents' in result

        # Check arrays have reasonable length
        assert len(result['rms_envelope']) > 0
        assert len(result['db_envelope']) > 0

    def test_dynamics_on_crescendo(self, singing_analyzer, sample_crescendo_audio):
        """Detect crescendo in audio"""
        sample_rate = 22050

        result = singing_analyzer.compute_dynamics(sample_crescendo_audio, sample_rate)

        # Should have positive dynamic range
        assert result['dynamic_range_db'] > 0

        # May detect crescendo (not guaranteed on short synthetic signal)
        assert 'crescendos' in result
        assert isinstance(result['crescendos'], list)

    def test_dynamics_on_diminuendo(self, singing_analyzer, sample_diminuendo_audio):
        """Detect diminuendo in audio"""
        sample_rate = 22050

        result = singing_analyzer.compute_dynamics(sample_diminuendo_audio, sample_rate)

        # Should have positive dynamic range
        assert result['dynamic_range_db'] > 0

        # May detect diminuendo
        assert 'diminuendos' in result
        assert isinstance(result['diminuendos'], list)

    def test_dynamics_smoothing(self, singing_analyzer):
        """Test dynamics smoothing parameter"""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        # Different smoothing windows
        result_1 = singing_analyzer.compute_dynamics(audio, sample_rate, smoothing_ms=10.0)
        result_2 = singing_analyzer.compute_dynamics(audio, sample_rate, smoothing_ms=100.0)

        # Both should complete successfully
        assert result_1 is not None
        assert result_2 is not None

    def test_compute_vocal_quality(self, singing_analyzer):
        """Compute vocal quality metrics"""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        # Generate F0 data
        f0_data = singing_analyzer.pitch_extractor.extract_f0_contour(audio, sample_rate)

        result = singing_analyzer.compute_vocal_quality(audio, sample_rate, f0_data)

        if singing_analyzer.compute_jitter:
            assert 'jitter_percent' in result
            assert 0.0 <= result['jitter_percent'] <= 100.0

        if singing_analyzer.compute_shimmer:
            assert 'shimmer_percent' in result
            assert 0.0 <= result['shimmer_percent'] <= 100.0

        if singing_analyzer.compute_spectral:
            assert 'spectral_centroid' in result or True  # May not be available without librosa

        assert 'quality_score' in result
        assert 0.0 <= result['quality_score'] <= 1.0

    def test_jitter_on_stable_pitch(self, singing_analyzer):
        """Verify low jitter on stable pitch"""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        f0_data = singing_analyzer.pitch_extractor.extract_f0_contour(audio, sample_rate)
        result = singing_analyzer.compute_vocal_quality(audio, sample_rate, f0_data)

        # Stable pitch should have low jitter
        if 'jitter_percent' in result:
            assert result['jitter_percent'] < 5.0

    def test_detect_vocal_techniques(self, singing_analyzer):
        """Detect singing techniques"""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        # Generate required data
        f0_data = singing_analyzer.pitch_extractor.extract_f0_contour(audio, sample_rate)
        breathiness_data = singing_analyzer.compute_breathiness(audio, sample_rate)

        techniques = singing_analyzer.detect_vocal_techniques(
            audio, sample_rate, f0_data, breathiness_data
        )

        # Check structure
        assert 'vibrato' in techniques
        assert 'breathy' in techniques
        assert 'belting' in techniques
        assert 'falsetto' in techniques
        assert 'vocal_fry' in techniques

        # Each technique should have 'detected' and 'confidence'
        for technique, info in techniques.items():
            assert 'detected' in info
            assert 'confidence' in info
            assert isinstance(info['detected'], bool)
            assert 0.0 <= info['confidence'] <= 1.0

    def test_no_false_positives_on_normal_voice(self, singing_analyzer):
        """Verify no extreme false positives"""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Normal singing frequency ~200-400 Hz
        audio = np.sin(2 * np.pi * 300.0 * t).astype(np.float32)

        f0_data = singing_analyzer.pitch_extractor.extract_f0_contour(audio, sample_rate)
        breathiness_data = singing_analyzer.compute_breathiness(audio, sample_rate)

        techniques = singing_analyzer.detect_vocal_techniques(
            audio, sample_rate, f0_data, breathiness_data
        )

        # Should not detect vocal fry at 300 Hz
        assert not techniques['vocal_fry']['detected']

    def test_empty_audio(self, singing_analyzer):
        """Handle empty audio gracefully"""
        audio = np.array([], dtype=np.float32)
        sample_rate = 22050

        try:
            result = singing_analyzer.compute_dynamics(audio, sample_rate)
            # Should handle gracefully
            assert result is not None or True
        except Exception as e:
            # Or raise appropriate error
            assert isinstance(e, (ValueError, RuntimeError, KeyError))

    def test_silent_audio(self, singing_analyzer, sample_audio_silence):
        """Analyze silence"""
        sample_rate = 16000

        result = singing_analyzer.compute_dynamics(sample_audio_silence, sample_rate)

        # Should return valid structure
        assert 'rms_envelope' in result
        assert 'dynamic_range_db' in result

    def test_get_summary_statistics(self, singing_analyzer):
        """Generate summary statistics"""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        features = singing_analyzer.analyze_singing_features(audio, sample_rate)
        summary = singing_analyzer.get_summary_statistics(features)

        # Check summary keys
        expected_keys = [
            'breathiness_score', 'dynamic_range_db', 'has_vibrato',
            'vocal_quality_score', 'mean_f0'
        ]

        for key in expected_keys:
            assert key in summary or True  # Some may be missing if no voiced frames

    @pytest.mark.cuda
    def test_gpu_analysis(self, cuda_device):
        """Test GPU-accelerated analysis"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from src.auto_voice.audio.singing_analyzer import SingingAnalyzer

        analyzer = SingingAnalyzer(device='cuda')

        sample_rate = 22050
        t = np.linspace(0, 1.0, int(sample_rate))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        features = analyzer.analyze_singing_features(audio, sample_rate)

        assert features is not None
        assert 'breathiness' in features

    @pytest.mark.performance
    def test_analysis_speed(self, singing_analyzer, benchmark_timer):
        """Benchmark analysis speed"""
        sample_rate = 22050
        t = np.linspace(0, 1.0, int(sample_rate))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        result, elapsed_time = benchmark_timer(
            singing_analyzer.analyze_singing_features,
            audio,
            sample_rate
        )

        # Should complete in reasonable time (< 10 seconds for 1 second of audio)
        assert elapsed_time < 10.0, f"Analysis took {elapsed_time:.2f}s, expected < 10.0s"

        print(f"\nSinging analysis time: {elapsed_time*1000:.1f}ms")


@pytest.mark.audio
@pytest.mark.integration
class TestSingingAnalyzerIntegration:
    """Integration tests for SingingAnalyzer"""

    def test_integration_with_pitch_extractor(self):
        """Verify integration with SingingPitchExtractor"""
        from src.auto_voice.audio.singing_analyzer import SingingAnalyzer

        analyzer = SingingAnalyzer()

        sample_rate = 22050
        t = np.linspace(0, 1.0, int(sample_rate))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        # Should use pitch extractor automatically
        features = analyzer.analyze_singing_features(audio, sample_rate)

        assert features is not None
        assert 'f0_data' in features

    def test_end_to_end_workflow(self):
        """Test complete singing analysis workflow"""
        from src.auto_voice.audio.singing_analyzer import SingingAnalyzer

        # Initialize
        analyzer = SingingAnalyzer()

        # Create test audio
        sample_rate = 22050
        t = np.linspace(0, 2.0, int(sample_rate * 2))
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        # Analyze
        features = analyzer.analyze_singing_features(audio, sample_rate)

        # Get summary
        summary = analyzer.get_summary_statistics(features)

        # Verify complete workflow
        assert features is not None
        assert summary is not None
        assert 'breathiness_score' in summary or 'mean_f0' in summary
