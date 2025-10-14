"""Comprehensive tests for AutoVoice utility modules."""

import pytest
import numpy as np
import torch
import tempfile
import json
import os
import logging
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.auto_voice.utils import (
    # Config utilities
    load_config, load_config_from_file, merge_configs, load_config_from_env, validate_config,
    # Data utilities
    AudioCollator, DataBatcher, DataSampler, DataPreprocessor,
    # Metrics utilities
    AudioMetrics, ModelMetrics, PerformanceMetrics, MetricsAggregator,
    # Helper utilities
    StringUtils, MathUtils, ValidationUtils, RetryUtils, CacheUtils,
    ensure_dir, safe_divide, safe_log, flatten_dict
)


class TestConfigLoader:
    """Test configuration loading and validation."""
    
    def test_load_config_from_json(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "audio": {"sample_rate": 44100},
            "model": {"device": "cpu"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            loaded_config = load_config_from_file(config_path)
            assert loaded_config["audio"]["sample_rate"] == 44100
            assert loaded_config["model"]["device"] == "cpu"
        finally:
            os.unlink(config_path)
    
    def test_merge_configs(self):
        """Test configuration merging."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 20}, "e": 4}
        
        result = merge_configs(base, override)
        
        assert result["a"] == 1
        assert result["b"]["c"] == 20
        assert result["b"]["d"] == 3
        assert result["e"] == 4
    
    def test_load_config_from_env(self):
        """Test loading configuration from environment variables."""
        config = {"audio": {"sample_rate": 22050}, "web": {"port": 5000}}
        
        with patch.dict(os.environ, {
            'AUTOVOICE_AUDIO__SAMPLE_RATE': '44100',
            'AUTOVOICE_WEB__PORT': '8080'
        }):
            updated_config = load_config_from_env(config)
            
            assert updated_config["audio"]["sample_rate"] == 44100
            assert updated_config["web"]["port"] == 8080
    
    def test_validate_config(self):
        """Test configuration validation."""
        valid_config = {
            "audio": {"sample_rate": 22050, "channels": 1},
            "model": {"device": "cpu"},
            "gpu": {"device_id": 0, "memory_fraction": 0.9},
            "web": {"host": "0.0.0.0", "port": 5000},
            "logging": {"level": "INFO"}
        }
        
        # Should not raise
        validate_config(valid_config)
        
        # Test invalid config
        invalid_config = {
            "audio": {"sample_rate": -1},
            "model": {},
            "gpu": {},
            "web": {},
            "logging": {}
        }
        
        with pytest.raises(ValueError):
            validate_config(invalid_config)


class TestAudioCollator:
    """Test audio data collation."""
    
    def test_collate_audio_sequences(self):
        """Test collating audio sequences with padding."""
        collator = AudioCollator(padding="longest", return_tensors="pt")
        
        batch = [
            {"audio": torch.randn(100), "length": 100},
            {"audio": torch.randn(150), "length": 150},
            {"audio": torch.randn(80), "length": 80}
        ]
        
        result = collator(batch)
        
        assert result["audio"].shape == (3, 150)  # Batch size 3, max length 150
        assert len(result["length"]) == 3
    
    def test_collate_with_max_length(self):
        """Test collating with maximum length constraint."""
        collator = AudioCollator(padding="max_length", max_length=120, return_tensors="pt")
        
        batch = [
            {"audio": torch.randn(100)},
            {"audio": torch.randn(150)},  # Will be truncated
            {"audio": torch.randn(80)}
        ]
        
        result = collator(batch)
        
        assert result["audio"].shape == (3, 120)


class TestDataBatcher:
    """Test data batching utilities."""
    
    def test_batch_data(self):
        """Test creating batches from data."""
        data = list(range(10))
        batcher = DataBatcher(batch_size=3, shuffle=False, drop_last=False)
        
        batches = batcher.batch_data(data)
        
        assert len(batches) == 4  # [0,1,2], [3,4,5], [6,7,8], [9]
        assert batches[0] == [0, 1, 2]
        assert batches[-1] == [9]
    
    def test_batch_data_drop_last(self):
        """Test batching with drop_last=True."""
        data = list(range(10))
        batcher = DataBatcher(batch_size=3, shuffle=False, drop_last=True)
        
        batches = batcher.batch_data(data)
        
        assert len(batches) == 3  # Last incomplete batch dropped
        assert batches[-1] == [6, 7, 8]


class TestDataSampler:
    """Test data sampling utilities."""
    
    def test_random_sample(self):
        """Test random sampling."""
        data = list(range(100))
        sample = DataSampler.random_sample(data, 10, seed=42)
        
        assert len(sample) == 10
        assert all(item in data for item in sample)
    
    def test_stratified_sample(self):
        """Test stratified sampling."""
        data = list(range(100))
        labels = [i % 4 for i in data]  # 4 classes
        
        sampled_data, sampled_labels = DataSampler.stratified_sample(
            data, labels, 20, seed=42
        )
        
        assert len(sampled_data) == 20
        assert len(sampled_labels) == 20
        
        # Check class distribution is preserved
        unique_labels = set(sampled_labels)
        assert len(unique_labels) == 4
    
    def test_weighted_sample(self):
        """Test weighted sampling."""
        data = [1, 2, 3, 4, 5]
        weights = [0.5, 0.3, 0.1, 0.05, 0.05]  # Heavily weighted towards first elements
        
        sample = DataSampler.weighted_sample(data, weights, 100, seed=42)
        
        assert len(sample) == 100
        # First element should appear more frequently
        assert sample.count(1) > sample.count(5)


class TestDataPreprocessor:
    """Test data preprocessing utilities."""
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        audio = np.array([0.1, -0.5, 0.8, -0.2])
        
        # Peak normalization
        normalized = DataPreprocessor.normalize_audio(audio, method="peak", target_level=0.95)
        assert np.abs(normalized).max() == pytest.approx(0.95, abs=1e-6)
        
        # RMS normalization
        normalized_rms = DataPreprocessor.normalize_audio(audio, method="rms", target_level=0.5)
        rms = np.sqrt(np.mean(normalized_rms ** 2))
        assert rms == pytest.approx(0.5, abs=1e-6)
    
    def test_apply_gain(self):
        """Test gain application."""
        audio = np.array([0.1, -0.1])
        gained = DataPreprocessor.apply_gain(audio, 20.0)  # +20dB
        
        expected_gain = 10 ** (20.0 / 20)  # 10x gain
        assert np.allclose(gained, audio * expected_gain)
    
    def test_split_into_chunks(self):
        """Test splitting data into chunks."""
        data = np.arange(100)
        chunks = DataPreprocessor.split_into_chunks(data, chunk_size=30, hop_size=20)
        
        assert len(chunks) == 5  # Chunks at positions 0, 20, 40, 60, 80
        assert chunks[0].shape == (30,)
        assert chunks[-1].shape == (30,)  # Last chunk padded


class TestAudioMetrics:
    """Test audio quality metrics."""
    
    def test_snr_calculation(self):
        """Test SNR calculation."""
        signal = np.array([1.0, 0.5, -0.5, -1.0])
        noise = np.array([0.1, -0.1, 0.05, -0.05])
        
        snr = AudioMetrics.snr(signal, noise)
        
        # SNR should be positive (signal power > noise power)
        assert snr > 0
    
    def test_dynamic_range(self):
        """Test dynamic range calculation."""
        # Signal with known dynamic range
        signal = np.concatenate([
            np.full(90, 0.01),  # Low level (1st percentile)
            np.full(10, 1.0)    # High level (99th percentile)
        ])
        
        dr = AudioMetrics.dynamic_range(signal, percentiles=(1, 99))
        expected_dr = 20 * np.log10(1.0 / 0.01)  # 40 dB
        
        assert dr == pytest.approx(expected_dr, abs=1.0)
    
    def test_zero_crossing_rate(self):
        """Test zero crossing rate calculation."""
        # Signal that crosses zero every sample
        signal = np.array([1, -1, 1, -1, 1, -1])
        zcr = AudioMetrics.zero_crossing_rate(signal)
        
        assert zcr == 1.0  # Maximum ZCR
        
        # Constant signal
        constant_signal = np.ones(10)
        zcr_constant = AudioMetrics.zero_crossing_rate(constant_signal)
        
        assert zcr_constant == 0.0  # No zero crossings
    
    def test_spectral_centroid(self):
        """Test spectral centroid calculation."""
        # Simple test signal
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000))  # 440 Hz sine wave
        
        centroid = AudioMetrics.spectral_centroid(signal, 1000)
        
        # Centroid should be around the fundamental frequency
        assert 400 < centroid < 480  # Allow some tolerance


class TestModelMetrics:
    """Test machine learning model metrics."""
    
    def test_accuracy(self):
        """Test accuracy calculation."""
        predictions = np.array([1, 2, 3, 2, 1])
        targets = np.array([1, 2, 3, 3, 1])
        
        acc = ModelMetrics.accuracy(predictions, targets)
        assert acc == 0.8  # 4/5 correct
    
    def test_precision_recall_f1(self):
        """Test precision, recall, F1 calculation."""
        predictions = np.array([1, 1, 2, 2, 3])
        targets = np.array([1, 2, 2, 2, 3])
        
        metrics = ModelMetrics.precision_recall_f1(predictions, targets)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        predictions = np.array([1, 2, 1, 2])
        targets = np.array([1, 2, 2, 1])
        
        cm = ModelMetrics.confusion_matrix(predictions, targets)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == 4  # Total number of samples
    
    def test_regression_metrics(self):
        """Test regression metrics."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.1, 2.1, 2.9, 3.8])
        
        mse = ModelMetrics.mse(predictions, targets)
        mae = ModelMetrics.mae(predictions, targets)
        r2 = ModelMetrics.r2_score(predictions, targets)
        
        assert mse > 0
        assert mae > 0
        assert r2 < 1.0


class TestPerformanceMetrics:
    """Test performance monitoring metrics."""
    
    def test_timer_functionality(self):
        """Test timing functionality."""
        metrics = PerformanceMetrics()
        
        metrics.start_timer("test_operation")
        time.sleep(0.01)  # Sleep for 10ms
        duration = metrics.end_timer("test_operation")
        
        assert duration >= 0.01
        
        stats = metrics.get_stats()
        assert "test_operation" in stats["timings"]
        assert stats["timings"]["test_operation"]["count"] == 1
    
    def test_counter_functionality(self):
        """Test counter functionality."""
        metrics = PerformanceMetrics()
        
        metrics.increment_counter("events", 5)
        metrics.increment_counter("events", 3)
        
        stats = metrics.get_stats()
        assert stats["counters"]["events"] == 8
    
    def test_timer_context_manager(self):
        """Test timer context manager."""
        metrics = PerformanceMetrics()
        
        with metrics.timer_context("context_test"):
            time.sleep(0.01)
        
        stats = metrics.get_stats()
        assert "context_test" in stats["timings"]


class TestStringUtils:
    """Test string manipulation utilities."""
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        unsafe = "file<>name:with/bad|chars?.txt"
        safe = StringUtils.sanitize_filename(unsafe)
        
        assert "<" not in safe
        assert ">" not in safe
        assert ":" not in safe
        assert "|" not in safe
        assert "?" not in safe
    
    def test_camel_snake_conversion(self):
        """Test camelCase to snake_case conversion."""
        camel = "camelCaseString"
        snake = StringUtils.camel_to_snake(camel)
        
        assert snake == "camel_case_string"
        
        # Test reverse conversion
        back_to_camel = StringUtils.snake_to_camel(snake)
        assert back_to_camel == "camelCaseString"
    
    def test_extract_numbers(self):
        """Test number extraction from strings."""
        text = "Temperature is 23.5°C and pressure is 1013 hPa"
        numbers = StringUtils.extract_numbers(text)
        
        assert 23.5 in numbers
        assert 1013.0 in numbers
    
    def test_format_bytes(self):
        """Test byte formatting."""
        assert StringUtils.format_bytes(1024) == "1.0 KB"
        assert StringUtils.format_bytes(1024 * 1024) == "1.0 MB"
        assert StringUtils.format_bytes(0) == "0B"
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert "3.50s" in StringUtils.format_duration(3.5)
        assert "1m" in StringUtils.format_duration(65)
        assert "1h" in StringUtils.format_duration(3665)
    
    def test_hash_string(self):
        """Test string hashing."""
        text = "test string"
        md5_hash = StringUtils.hash_string(text, "md5")
        sha256_hash = StringUtils.hash_string(text, "sha256")
        
        assert len(md5_hash) == 32  # MD5 hex length
        assert len(sha256_hash) == 64  # SHA256 hex length
        assert md5_hash != sha256_hash


class TestMathUtils:
    """Test mathematical utilities."""
    
    def test_clamp(self):
        """Test value clamping."""
        assert MathUtils.clamp(5, 0, 10) == 5
        assert MathUtils.clamp(-5, 0, 10) == 0
        assert MathUtils.clamp(15, 0, 10) == 10
    
    def test_lerp(self):
        """Test linear interpolation."""
        assert MathUtils.lerp(0, 10, 0.5) == 5
        assert MathUtils.lerp(0, 10, 0) == 0
        assert MathUtils.lerp(0, 10, 1) == 10
    
    def test_db_conversions(self):
        """Test dB to linear conversions."""
        # 0 dB should be 1.0 linear
        assert MathUtils.db_to_linear(0) == pytest.approx(1.0)
        
        # 20 dB should be 10x linear
        assert MathUtils.db_to_linear(20) == pytest.approx(10.0)
        
        # Test reverse conversion
        assert MathUtils.linear_to_db(1.0) == pytest.approx(0.0)
        assert MathUtils.linear_to_db(10.0) == pytest.approx(20.0)
    
    def test_rms(self):
        """Test RMS calculation."""
        values = [3, 4]  # 3-4-5 triangle
        rms = MathUtils.rms(values)
        expected = math.sqrt((9 + 16) / 2)  # sqrt(25/2)
        
        assert rms == pytest.approx(expected)
    
    def test_power_of_2_functions(self):
        """Test power of 2 utilities."""
        assert MathUtils.is_power_of_2(8) == True
        assert MathUtils.is_power_of_2(7) == False
        
        assert MathUtils.next_power_of_2(7) == 8
        assert MathUtils.next_power_of_2(8) == 8
        assert MathUtils.next_power_of_2(9) == 16


class TestValidationUtils:
    """Test validation utilities."""
    
    def test_email_validation(self):
        """Test email validation."""
        assert ValidationUtils.is_valid_email("test@example.com") == True
        assert ValidationUtils.is_valid_email("invalid.email") == False
        assert ValidationUtils.is_valid_email("test@") == False
    
    def test_url_validation(self):
        """Test URL validation."""
        assert ValidationUtils.is_valid_url("https://example.com") == True
        assert ValidationUtils.is_valid_url("http://localhost:8080/path") == True
        assert ValidationUtils.is_valid_url("not_a_url") == False
    
    def test_range_validation(self):
        """Test range validation."""
        assert ValidationUtils.validate_range(5, 0, 10) == True
        assert ValidationUtils.validate_range(15, 0, 10) == False
        assert ValidationUtils.validate_range(0, 0, 10, inclusive=True) == True
        assert ValidationUtils.validate_range(0, 0, 10, inclusive=False) == False
    
    def test_dict_validation(self):
        """Test dictionary validation."""
        data = {"name": "test", "age": 25, "extra": "value"}
        required = ["name", "age"]
        optional = ["email"]
        
        is_valid, errors = ValidationUtils.validate_dict_keys(data, required, optional)
        
        assert is_valid == False  # "extra" is not allowed
        assert "Unexpected key: extra" in errors


class TestRetryUtils:
    """Test retry utilities."""
    
    def test_exponential_backoff_success(self):
        """Test successful retry with exponential backoff."""
        call_count = 0
        
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = RetryUtils.exponential_backoff(
            flaky_function,
            max_retries=3,
            base_delay=0.001  # Small delay for testing
        )
        
        assert result == "success"
        assert call_count == 3
    
    def test_exponential_backoff_failure(self):
        """Test retry failure after max attempts."""
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            RetryUtils.exponential_backoff(
                always_fails,
                max_retries=2,
                base_delay=0.001
            )


class TestCacheUtils:
    """Test caching utilities."""
    
    def test_basic_cache_operations(self):
        """Test basic cache operations."""
        cache = CacheUtils(max_size=3)
        
        # Set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Non-existent key
        assert cache.get("nonexistent") is None
        
        # Cache size
        assert cache.size() == 1
    
    def test_cache_size_limit(self):
        """Test cache size limiting."""
        cache = CacheUtils(max_size=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict oldest
        
        assert cache.size() == 2
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key3") == "value3"
    
    def test_cache_ttl(self):
        """Test cache TTL functionality."""
        cache = CacheUtils(ttl=0.01)  # 10ms TTL
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        time.sleep(0.02)  # Wait for expiration
        assert cache.get("key1") is None


class TestHelperFunctions:
    """Test standalone helper functions."""
    
    def test_ensure_dir(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "new" / "nested" / "dir"
            result_path = ensure_dir(test_path)
            
            assert result_path.exists()
            assert result_path.is_dir()
    
    def test_safe_divide(self):
        """Test safe division."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=99) == 99
    
    def test_safe_log(self):
        """Test safe logarithm."""
        assert safe_log(10, 10) == pytest.approx(1.0)
        assert safe_log(0) == float('-inf')
        assert safe_log(-5, default=0) == 0
    
    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        
        flattened = flatten_dict(nested)
        
        assert flattened["a"] == 1
        assert flattened["b.c"] == 2
        assert flattened["b.d.e"] == 3


if __name__ == "__main__":
    pytest.main([__file__])