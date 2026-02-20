"""Comprehensive tests for speaker diarization module - Target 70% coverage.

Extends basic diarization tests with voice activity detection, clustering,
embedding extraction, and full diarization workflow tests.
"""
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import tempfile
import soundfile as sf


class TestVoiceActivityDetection:
    """Test voice activity detection functionality."""

    @pytest.fixture
    def diarizer(self):
        """Create SpeakerDiarizer instance."""
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer
        return SpeakerDiarizer(device='cpu', min_segment_duration=0.3)

    def test_detect_voice_activity_silence(self, diarizer):
        """Test VAD with silent audio."""
        # Silent audio
        waveform = torch.zeros(16000)  # 1 second of silence

        regions = diarizer._detect_voice_activity(
            waveform, sample_rate=16000, energy_threshold=0.02
        )

        # Should detect no speech in silence
        assert len(regions) == 0

    def test_detect_voice_activity_constant_signal(self, diarizer):
        """Test VAD with constant signal (speech-like energy)."""
        # Constant signal above threshold
        waveform = torch.ones(16000 * 5) * 0.1  # 5 seconds

        regions = diarizer._detect_voice_activity(
            waveform, sample_rate=16000, energy_threshold=0.02
        )

        # Should detect entire duration as speech
        assert len(regions) >= 1
        # Total covered should be close to 5 seconds
        total_duration = sum(end - start for start, end in regions)
        assert total_duration > 4.0

    def test_detect_voice_activity_intermittent(self, diarizer):
        """Test VAD with intermittent speech."""
        # Create intermittent speech pattern
        sr = 16000
        waveform = torch.zeros(sr * 10)  # 10 seconds

        # Add speech at 0-2s, 4-6s, 8-10s
        waveform[0:sr*2] = torch.randn(sr*2) * 0.1
        waveform[sr*4:sr*6] = torch.randn(sr*2) * 0.1
        waveform[sr*8:sr*10] = torch.randn(sr*2) * 0.1

        regions = diarizer._detect_voice_activity(
            waveform, sample_rate=sr, energy_threshold=0.02
        )

        # Should detect multiple regions
        assert len(regions) >= 2

    def test_detect_voice_activity_frame_parameters(self, diarizer):
        """Test VAD with different frame parameters."""
        waveform = torch.randn(16000 * 2) * 0.1  # 2 seconds

        # Default parameters
        regions_default = diarizer._detect_voice_activity(
            waveform, sample_rate=16000
        )

        # Larger frame size
        regions_large = diarizer._detect_voice_activity(
            waveform, sample_rate=16000, frame_duration=0.05
        )

        # Both should detect speech (specific regions may vary)
        # This tests that different parameters work without error
        assert isinstance(regions_default, list)
        assert isinstance(regions_large, list)

    def test_detect_voice_activity_min_duration(self, diarizer):
        """Test VAD filters out short segments."""
        sr = 16000
        waveform = torch.zeros(sr * 5)  # 5 seconds

        # Very short burst of sound (100ms)
        waveform[sr:sr+int(0.1*sr)] = torch.ones(int(0.1*sr)) * 0.5

        # Long speech segment (2s)
        waveform[sr*3:sr*5] = torch.randn(sr*2) * 0.1

        regions = diarizer._detect_voice_activity(
            waveform, sample_rate=sr,
            min_speech_duration=0.5  # Require at least 500ms
        )

        # Short burst should be filtered, only long segment detected
        assert len(regions) >= 1
        # All regions should be >= min duration
        for start, end in regions:
            assert end - start >= 0.5


class TestSegmentGeneration:
    """Test audio segmentation functionality."""

    @pytest.fixture
    def diarizer(self):
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer
        return SpeakerDiarizer(device='cpu', min_segment_duration=0.5)

    def test_segment_audio_fixed_basic(self, diarizer):
        """Test basic fixed-duration segmentation."""
        waveform = torch.zeros(16000 * 10)  # 10 seconds
        sr = 16000

        segments = diarizer._segment_audio_fixed(
            waveform, sr, segment_duration=2.0, overlap=0.0
        )

        # Should have 5 segments of 2s each
        assert len(segments) == 5
        assert segments[0] == (0.0, 2.0)
        assert segments[-1] == (8.0, 10.0)

    def test_segment_audio_fixed_with_overlap(self, diarizer):
        """Test segmentation with overlap."""
        waveform = torch.zeros(16000 * 10)  # 10 seconds
        sr = 16000

        segments = diarizer._segment_audio_fixed(
            waveform, sr, segment_duration=2.0, overlap=0.5
        )

        # With 50% overlap, hop is 1s, so more segments
        assert len(segments) > 5

        # Check overlap pattern
        for i in range(1, len(segments)):
            # Each segment should start 1s after previous
            assert segments[i][0] == pytest.approx(segments[i-1][0] + 1.0, abs=0.1)

    def test_segment_audio_fixed_short_audio(self, diarizer):
        """Test segmentation with audio shorter than segment duration."""
        waveform = torch.zeros(16000 * 1)  # 1 second
        sr = 16000

        segments = diarizer._segment_audio_fixed(
            waveform, sr, segment_duration=2.0, overlap=0.0
        )

        # Should have 1 segment covering entire audio
        assert len(segments) == 1
        assert segments[0][0] == 0.0
        assert segments[0][1] <= 1.0


class TestEmbeddingClustering:
    """Test embedding clustering functionality."""

    @pytest.fixture
    def diarizer(self):
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer
        return SpeakerDiarizer(device='cpu', max_speakers=5)

    def test_cluster_embeddings_single_speaker(self, diarizer):
        """Test clustering with single speaker embeddings."""
        # Similar embeddings (same speaker)
        base_embedding = np.random.randn(512)
        embeddings = [
            base_embedding + np.random.randn(512) * 0.01
            for _ in range(5)
        ]

        labels = diarizer._cluster_embeddings(embeddings)

        # All should be same cluster
        unique_labels = set(labels)
        # Should have 1 or 2 labels (0 for None, 1 for speaker)
        assert len(unique_labels) <= 2

    def test_cluster_embeddings_two_speakers(self, diarizer):
        """Test clustering with two distinct speakers."""
        # Two very different embeddings
        speaker1_base = np.random.randn(512)
        speaker2_base = np.random.randn(512) + 10  # Very different

        embeddings = []
        # Add samples from each speaker
        for _ in range(3):
            embeddings.append(speaker1_base + np.random.randn(512) * 0.01)
        for _ in range(3):
            embeddings.append(speaker2_base + np.random.randn(512) * 0.01)

        labels = diarizer._cluster_embeddings(embeddings, num_speakers=2)

        # Should detect 2 clusters
        unique_labels = set(labels)
        non_zero_labels = [l for l in unique_labels if l > 0]
        assert len(non_zero_labels) == 2

    def test_cluster_embeddings_with_none(self, diarizer):
        """Test clustering handles None embeddings."""
        embeddings = [
            np.random.randn(512),
            None,  # Failed extraction
            np.random.randn(512),
            None,
            np.random.randn(512),
        ]

        labels = diarizer._cluster_embeddings(embeddings)

        # Should produce labels for all, None entries get label 0
        assert len(labels) == 5

    def test_cluster_embeddings_empty(self, diarizer):
        """Test clustering with empty list."""
        labels = diarizer._cluster_embeddings([])

        assert len(labels) == 0

    def test_cluster_embeddings_single(self, diarizer):
        """Test clustering with single embedding."""
        embeddings = [np.random.randn(512)]

        labels = diarizer._cluster_embeddings(embeddings)

        assert len(labels) == 1

    def test_cluster_embeddings_max_speakers(self, diarizer):
        """Test that max_speakers limit is enforced."""
        # Create many distinct speakers
        embeddings = []
        for i in range(20):
            base = np.zeros(512)
            base[i*10:(i+1)*10] = 1.0  # Distinct pattern
            embeddings.append(base)

        labels = diarizer._cluster_embeddings(embeddings)

        # Should be limited to max_speakers
        unique_labels = set(labels)
        non_zero_labels = [l for l in unique_labels if l > 0]
        assert len(non_zero_labels) <= diarizer.max_speakers


class TestSegmentMerging:
    """Test adjacent segment merging."""

    @pytest.fixture
    def diarizer(self):
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer
        return SpeakerDiarizer(device='cpu')

    def test_merge_adjacent_same_speaker(self, diarizer):
        """Test merging adjacent segments from same speaker."""
        from auto_voice.audio.speaker_diarization import SpeakerSegment

        segments = [
            SpeakerSegment(0.0, 2.0, 'SPEAKER_00'),
            SpeakerSegment(2.1, 4.0, 'SPEAKER_00'),  # Small gap
            SpeakerSegment(4.2, 6.0, 'SPEAKER_00'),
        ]

        merged = diarizer._merge_adjacent_segments(segments, max_gap=0.3)

        # Should merge into one segment
        assert len(merged) == 1
        assert merged[0].start == 0.0
        assert merged[0].end == 6.0

    def test_merge_different_speakers(self, diarizer):
        """Test segments from different speakers are not merged."""
        from auto_voice.audio.speaker_diarization import SpeakerSegment

        segments = [
            SpeakerSegment(0.0, 2.0, 'SPEAKER_00'),
            SpeakerSegment(2.1, 4.0, 'SPEAKER_01'),  # Different speaker
            SpeakerSegment(4.1, 6.0, 'SPEAKER_00'),
        ]

        merged = diarizer._merge_adjacent_segments(segments, max_gap=0.3)

        # Should not merge different speakers
        assert len(merged) == 3

    def test_merge_large_gap(self, diarizer):
        """Test segments with large gap are not merged."""
        from auto_voice.audio.speaker_diarization import SpeakerSegment

        segments = [
            SpeakerSegment(0.0, 2.0, 'SPEAKER_00'),
            SpeakerSegment(5.0, 7.0, 'SPEAKER_00'),  # 3 second gap
        ]

        merged = diarizer._merge_adjacent_segments(segments, max_gap=0.5)

        # Should not merge due to large gap
        assert len(merged) == 2

    def test_merge_empty_list(self, diarizer):
        """Test merging empty segment list."""
        merged = diarizer._merge_adjacent_segments([], max_gap=0.3)

        assert merged == []


class TestAudioLoading:
    """Test audio file loading functionality."""

    @pytest.fixture
    def diarizer(self):
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer
        return SpeakerDiarizer(device='cpu')

    def test_load_audio_wav_file(self, diarizer, tmp_path):
        """Test loading WAV file."""
        # Create test WAV file
        audio_path = tmp_path / 'test.wav'
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(str(audio_path), audio, 16000)

        waveform, sr = diarizer._load_audio(audio_path)

        assert sr == 16000
        assert isinstance(waveform, torch.Tensor)
        assert len(waveform) == 16000

    def test_load_audio_stereo_to_mono(self, diarizer, tmp_path):
        """Test stereo to mono conversion."""
        audio_path = tmp_path / 'stereo.wav'
        audio = np.random.randn(2, 16000).astype(np.float32) * 0.1
        sf.write(str(audio_path), audio.T, 16000)  # soundfile expects (samples, channels)

        waveform, sr = diarizer._load_audio(audio_path)

        # Should be mono
        assert waveform.dim() == 1

    def test_load_audio_resample_to_16k(self, diarizer, tmp_path):
        """Test resampling to 16kHz."""
        audio_path = tmp_path / 'highrate.wav'
        audio = np.random.randn(44100).astype(np.float32) * 0.1  # 1s at 44.1kHz
        sf.write(str(audio_path), audio, 44100)

        waveform, sr = diarizer._load_audio(audio_path)

        # Should be resampled to 16kHz
        assert sr == 16000
        assert len(waveform) == 16000

    def test_load_audio_nonexistent(self, diarizer):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            diarizer._load_audio(Path('/nonexistent/audio.wav'))

    def test_load_audio_int16_conversion(self, diarizer, tmp_path):
        """Test int16 audio conversion to float."""
        audio_path = tmp_path / 'int16.wav'
        # Create int16 audio
        audio = (np.random.randn(16000) * 16384).astype(np.int16)
        sf.write(str(audio_path), audio, 16000, subtype='PCM_16')

        waveform, sr = diarizer._load_audio(audio_path)

        # Should be converted to float
        assert waveform.dtype == torch.float32
        # Values should be normalized
        assert waveform.abs().max() <= 1.0 + 0.01  # Small tolerance


class TestExtractSpeakerEmbedding:
    """Test speaker embedding extraction."""

    @pytest.fixture
    def diarizer(self):
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer
        return SpeakerDiarizer(device='cpu')

    @patch.object(__import__('auto_voice.audio.speaker_diarization', fromlist=['SpeakerDiarizer']).SpeakerDiarizer, '_load_model')
    def test_extract_embedding_basic(self, mock_load, diarizer, tmp_path):
        """Test basic embedding extraction."""
        # Create test audio file
        audio_path = tmp_path / 'test.wav'
        audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1
        sf.write(str(audio_path), audio, 16000)

        # Mock model
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(embeddings=torch.randn(1, 512))
        diarizer._model = mock_model

        mock_extractor = MagicMock()
        mock_extractor.return_value = {'input_values': torch.randn(1, 16000)}
        diarizer._feature_extractor = mock_extractor

        embedding = diarizer.extract_speaker_embedding(audio_path)

        # Should return 512-dim embedding
        assert embedding.shape == (512,)

    @patch.object(__import__('auto_voice.audio.speaker_diarization', fromlist=['SpeakerDiarizer']).SpeakerDiarizer, '_load_model')
    def test_extract_embedding_with_segment(self, mock_load, diarizer, tmp_path):
        """Test embedding extraction from audio segment."""
        audio_path = tmp_path / 'test.wav'
        audio = np.random.randn(16000 * 10).astype(np.float32) * 0.1  # 10s
        sf.write(str(audio_path), audio, 16000)

        mock_model = MagicMock()
        mock_model.return_value = MagicMock(embeddings=torch.randn(1, 512))
        diarizer._model = mock_model

        mock_extractor = MagicMock()
        mock_extractor.return_value = {'input_values': torch.randn(1, 16000)}
        diarizer._feature_extractor = mock_extractor

        # Extract from segment
        embedding = diarizer.extract_speaker_embedding(audio_path, start=2.0, end=5.0)

        assert embedding.shape == (512,)

    @patch.object(__import__('auto_voice.audio.speaker_diarization', fromlist=['SpeakerDiarizer']).SpeakerDiarizer, '_load_model')
    def test_embedding_l2_normalized(self, mock_load, diarizer, tmp_path):
        """Test that embedding is L2 normalized."""
        audio_path = tmp_path / 'test.wav'
        audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1
        sf.write(str(audio_path), audio, 16000)

        # Mock with non-normalized embedding
        mock_model = MagicMock()
        raw_embedding = torch.randn(1, 512) * 10  # Large values
        mock_model.return_value = MagicMock(embeddings=raw_embedding)
        diarizer._model = mock_model

        mock_extractor = MagicMock()
        mock_extractor.return_value = {'input_values': torch.randn(1, 16000)}
        diarizer._feature_extractor = mock_extractor

        embedding = diarizer.extract_speaker_embedding(audio_path)

        # Should be L2 normalized (magnitude ~= 1)
        magnitude = np.linalg.norm(embedding)
        assert abs(magnitude - 1.0) < 0.01


class TestDiarizeMethod:
    """Test main diarize method."""

    @pytest.fixture
    def diarizer(self):
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer
        return SpeakerDiarizer(device='cpu', min_segment_duration=0.3)

    @patch.object(__import__('auto_voice.audio.speaker_diarization', fromlist=['SpeakerDiarizer']).SpeakerDiarizer, '_extract_segment_embeddings')
    @patch.object(__import__('auto_voice.audio.speaker_diarization', fromlist=['SpeakerDiarizer']).SpeakerDiarizer, '_load_model')
    def test_diarize_no_speech(self, mock_load, mock_extract, diarizer, tmp_path):
        """Test diarization with silent audio."""
        audio_path = tmp_path / 'silence.wav'
        audio = np.zeros(16000 * 5).astype(np.float32)  # 5s silence
        sf.write(str(audio_path), audio, 16000)

        result = diarizer.diarize(audio_path)

        # Should return empty result
        assert result.num_speakers == 0
        assert len(result.segments) == 0

    @patch.object(__import__('auto_voice.audio.speaker_diarization', fromlist=['SpeakerDiarizer']).SpeakerDiarizer, '_extract_segment_embeddings')
    @patch.object(__import__('auto_voice.audio.speaker_diarization', fromlist=['SpeakerDiarizer']).SpeakerDiarizer, '_load_model')
    def test_diarize_with_embeddings(self, mock_load, mock_extract, diarizer, tmp_path):
        """Test diarization with mocked embeddings."""
        audio_path = tmp_path / 'speech.wav'
        audio = np.random.randn(16000 * 10).astype(np.float32) * 0.1  # 10s
        sf.write(str(audio_path), audio, 16000)

        # Mock embeddings for 2 speakers
        mock_extract.return_value = [
            np.random.randn(512),
            np.random.randn(512) + 5,  # Different speaker
            np.random.randn(512),
            np.random.randn(512) + 5,
        ]

        result = diarizer.diarize(audio_path, num_speakers=2)

        # Should detect segments
        assert result.audio_duration > 0

    @patch.object(__import__('auto_voice.audio.speaker_diarization', fromlist=['SpeakerDiarizer']).SpeakerDiarizer, '_load_model')
    def test_diarize_chunked_long_audio(self, mock_load, diarizer, tmp_path):
        """Test chunked diarization for long audio."""
        audio_path = tmp_path / 'long.wav'
        # Create 3-minute audio
        audio = np.random.randn(16000 * 180).astype(np.float32) * 0.1
        sf.write(str(audio_path), audio, 16000)

        # Force chunked processing
        result = diarizer.diarize(
            audio_path,
            use_chunked_processing=True
        )

        # Should process without error
        assert result.audio_duration > 0


class TestExtractSpeakerAudio:
    """Test speaker audio extraction."""

    @pytest.fixture
    def diarizer(self):
        from auto_voice.audio.speaker_diarization import SpeakerDiarizer
        return SpeakerDiarizer(device='cpu')

    def test_extract_speaker_audio_basic(self, diarizer, tmp_path):
        """Test extracting audio for a speaker."""
        from auto_voice.audio.speaker_diarization import SpeakerSegment, DiarizationResult

        # Create test audio
        audio_path = tmp_path / 'multi_speaker.wav'
        audio = np.random.randn(16000 * 10).astype(np.float32) * 0.1  # 10s
        sf.write(str(audio_path), audio, 16000)

        # Create mock diarization result
        result = DiarizationResult(
            segments=[
                SpeakerSegment(0.0, 3.0, 'SPEAKER_00'),
                SpeakerSegment(3.5, 6.0, 'SPEAKER_01'),
                SpeakerSegment(6.5, 10.0, 'SPEAKER_00'),
            ],
            num_speakers=2,
            audio_duration=10.0
        )

        output_path = diarizer.extract_speaker_audio(
            audio_path, result, 'SPEAKER_00'
        )

        assert output_path.exists()
        # Load and verify
        extracted, sr = sf.read(str(output_path))
        assert sr == 16000
        assert len(extracted) > 0

    def test_extract_speaker_audio_not_found(self, diarizer, tmp_path):
        """Test extracting audio for non-existent speaker."""
        from auto_voice.audio.speaker_diarization import SpeakerSegment, DiarizationResult

        audio_path = tmp_path / 'test.wav'
        audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1
        sf.write(str(audio_path), audio, 16000)

        result = DiarizationResult(
            segments=[SpeakerSegment(0.0, 5.0, 'SPEAKER_00')],
            num_speakers=1,
            audio_duration=5.0
        )

        with pytest.raises(ValueError, match="No segments found"):
            diarizer.extract_speaker_audio(audio_path, result, 'SPEAKER_99')


class TestSpeakerMatching:
    """Test speaker matching utilities."""

    def test_match_speaker_to_profile(self):
        """Test matching speaker embedding to profiles."""
        from auto_voice.audio.speaker_diarization import match_speaker_to_profile

        # Create test embeddings
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)

        # Create similar profile
        similar_profile = embedding + np.random.randn(512) * 0.01
        similar_profile = similar_profile / np.linalg.norm(similar_profile)

        # Create different profile
        different_profile = np.random.randn(512)
        different_profile = different_profile / np.linalg.norm(different_profile)

        profiles = {
            'profile_similar': similar_profile,
            'profile_different': different_profile,
        }

        result = match_speaker_to_profile(embedding, profiles, threshold=0.7)

        # Should match similar profile
        assert result == 'profile_similar'

    def test_match_speaker_no_match(self):
        """Test when no profile matches."""
        from auto_voice.audio.speaker_diarization import match_speaker_to_profile

        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)

        # Very different profiles
        profiles = {
            'profile_1': np.random.randn(512),
            'profile_2': np.random.randn(512),
        }
        # Normalize
        for k in profiles:
            profiles[k] = profiles[k] / np.linalg.norm(profiles[k])

        result = match_speaker_to_profile(embedding, profiles, threshold=0.99)

        # Should not match with high threshold
        assert result is None

    def test_match_speaker_empty_profiles(self):
        """Test matching with no profiles."""
        from auto_voice.audio.speaker_diarization import match_speaker_to_profile

        embedding = np.random.randn(512)
        result = match_speaker_to_profile(embedding, {}, threshold=0.7)

        assert result is None

    def test_compute_speaker_similarity(self):
        """Test computing speaker similarity."""
        from auto_voice.audio.speaker_diarization import compute_speaker_similarity

        # Same embedding should have similarity 1.0
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)

        similarity = compute_speaker_similarity(embedding, embedding)
        assert abs(similarity - 1.0) < 0.01

        # Orthogonal embeddings should have similarity ~0
        embedding1 = np.zeros(512)
        embedding1[0] = 1.0
        embedding2 = np.zeros(512)
        embedding2[1] = 1.0

        similarity = compute_speaker_similarity(embedding1, embedding2)
        assert abs(similarity) < 0.01


class TestMemoryUtilities:
    """Test memory management utilities."""

    def test_get_available_memory_gb(self):
        """Test getting available system memory."""
        from auto_voice.audio.speaker_diarization import get_available_memory_gb

        memory = get_available_memory_gb()

        assert memory > 0
        assert isinstance(memory, float)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=2 * 1024**3)  # 2GB
    @patch('torch.cuda.get_device_properties')
    def test_get_gpu_memory_with_cuda(self, mock_props, mock_alloc, mock_avail):
        """Test getting GPU memory when CUDA available."""
        from auto_voice.audio.speaker_diarization import get_gpu_memory_gb

        mock_device = Mock()
        mock_device.total_memory = 8 * 1024**3  # 8GB
        mock_props.return_value = mock_device

        used, total = get_gpu_memory_gb()

        assert used == pytest.approx(2.0, abs=0.1)
        assert total == pytest.approx(8.0, abs=0.1)

    @patch('torch.cuda.is_available', return_value=False)
    def test_get_gpu_memory_no_cuda(self, mock_avail):
        """Test getting GPU memory when CUDA not available."""
        from auto_voice.audio.speaker_diarization import get_gpu_memory_gb

        used, total = get_gpu_memory_gb()

        assert used == 0.0
        assert total == 0.0
