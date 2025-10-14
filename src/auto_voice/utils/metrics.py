"""Metrics calculation utilities for audio processing and model evaluation."""

import numpy as np
import torch
from typing import Dict, List, Union, Optional, Tuple, Any
import logging
from collections import defaultdict
import time
import psutil
import threading

logger = logging.getLogger(__name__)


class AudioMetrics:
    """Audio quality and processing metrics."""
    
    @staticmethod
    def snr(signal: np.ndarray, noise: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR) in dB.
        
        Args:
            signal: Clean signal
            noise: Noise signal
            
        Returns:
            SNR in decibels
        """
        if signal.size == 0 or noise.size == 0:
            return float('inf')
        
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr_linear = signal_power / noise_power
        return 10 * np.log10(snr_linear)
    
    @staticmethod
    def thd(signal: np.ndarray, fundamental_freq: float, sample_rate: int) -> float:
        """
        Calculate Total Harmonic Distortion (THD).
        
        Args:
            signal: Audio signal
            fundamental_freq: Fundamental frequency in Hz
            sample_rate: Sample rate in Hz
            
        Returns:
            THD as a percentage
        """
        # Compute FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # Find fundamental frequency peak
        fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
        fund_magnitude = magnitude[fund_idx]
        
        # Find harmonics (2f, 3f, 4f, etc.)
        harmonic_power = 0
        for n in range(2, 6):  # Up to 5th harmonic
            harmonic_freq = n * fundamental_freq
            if harmonic_freq < sample_rate / 2:  # Below Nyquist
                harm_idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_power += magnitude[harm_idx] ** 2
        
        if fund_magnitude == 0:
            return 100.0
        
        thd = np.sqrt(harmonic_power) / fund_magnitude
        return thd * 100
    
    @staticmethod
    def dynamic_range(signal: np.ndarray, percentiles: Tuple[float, float] = (1, 99)) -> float:
        """
        Calculate dynamic range using percentiles.
        
        Args:
            signal: Audio signal
            percentiles: Lower and upper percentiles for calculation
            
        Returns:
            Dynamic range in dB
        """
        if signal.size == 0:
            return 0.0
        
        signal_abs = np.abs(signal)
        low_perc, high_perc = np.percentile(signal_abs, percentiles)
        
        if low_perc == 0:
            return float('inf')
        
        return 20 * np.log10(high_perc / low_perc)
    
    @staticmethod
    def spectral_centroid(signal: np.ndarray, sample_rate: int) -> float:
        """
        Calculate spectral centroid (brightness measure).
        
        Args:
            signal: Audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Spectral centroid in Hz
        """
        fft = np.fft.fft(signal)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)[:len(fft)//2]
        
        if np.sum(magnitude) == 0:
            return 0.0
        
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        return centroid
    
    @staticmethod
    def zero_crossing_rate(signal: np.ndarray) -> float:
        """
        Calculate zero crossing rate.
        
        Args:
            signal: Audio signal
            
        Returns:
            Zero crossing rate (0-1)
        """
        if len(signal) <= 1:
            return 0.0
        
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal)))) / 2
        return zero_crossings / (len(signal) - 1)


class ModelMetrics:
    """Machine learning model evaluation metrics."""
    
    @staticmethod
    def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate classification accuracy."""
        if len(predictions) == 0:
            return 0.0
        return np.mean(predictions == targets)
    
    @staticmethod
    def precision_recall_f1(
        predictions: np.ndarray,
        targets: np.ndarray,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            predictions: Predicted labels
            targets: True labels
            average: Averaging method ('weighted', 'macro', 'micro')
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        unique_labels = np.unique(np.concatenate([predictions, targets]))
        
        precisions = []
        recalls = []
        f1s = []
        supports = []
        
        for label in unique_labels:
            tp = np.sum((predictions == label) & (targets == label))
            fp = np.sum((predictions == label) & (targets != label))
            fn = np.sum((predictions != label) & (targets == label))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            support = np.sum(targets == label)
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)
        
        if average == 'weighted':
            total_support = sum(supports)
            if total_support == 0:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
            precision = sum(p * s for p, s in zip(precisions, supports)) / total_support
            recall = sum(r * s for r, s in zip(recalls, supports)) / total_support
            f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support
        elif average == 'macro':
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)
        elif average == 'micro':
            # For micro averaging, calculate global TP, FP, FN
            tp_total = np.sum(predictions == targets)
            fp_total = np.sum(predictions != targets)
            fn_total = fp_total  # In multiclass, FP of one class contributes to FN of others
            
            precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            raise ValueError(f"Unknown average method: {average}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        unique_labels = np.unique(np.concatenate([predictions, targets]))
        n_labels = len(unique_labels)
        
        # Create label to index mapping
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        matrix = np.zeros((n_labels, n_labels), dtype=int)
        
        for pred, target in zip(predictions, targets):
            pred_idx = label_to_idx[pred]
            target_idx = label_to_idx[target]
            matrix[target_idx, pred_idx] += 1
        
        return matrix
    
    @staticmethod
    def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        if len(predictions) == 0:
            return 0.0
        return np.mean((predictions - targets) ** 2)
    
    @staticmethod
    def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        if len(predictions) == 0:
            return 0.0
        return np.mean(np.abs(predictions - targets))
    
    @staticmethod
    def r2_score(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate R-squared coefficient of determination."""
        if len(predictions) == 0:
            return 0.0
        
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)


class PerformanceMetrics:
    """System and processing performance metrics."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.counters = defaultdict(int)
        self.memory_usage = []
        self.cpu_usage = []
        self._lock = threading.Lock()
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        with self._lock:
            if not hasattr(self, '_timers'):
                self._timers = {}
            self._timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a named timer and record the duration."""
        with self._lock:
            if not hasattr(self, '_timers') or name not in self._timers:
                logger.warning(f"Timer '{name}' was not started")
                return 0.0
            
            duration = time.time() - self._timers[name]
            self.timings[name].append(duration)
            del self._timers[name]
            return duration
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        with self._lock:
            self.counters[name] += value
    
    def record_memory_usage(self) -> None:
        """Record current memory usage."""
        try:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            with self._lock:
                self.memory_usage.append(memory_mb)
        except Exception as e:
            logger.warning(f"Could not record memory usage: {e}")
    
    def record_cpu_usage(self) -> None:
        """Record current CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent()
            with self._lock:
                self.cpu_usage.append(cpu_percent)
        except Exception as e:
            logger.warning(f"Could not record CPU usage: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._lock:
            stats = {
                'timings': {},
                'counters': dict(self.counters),
                'memory': {},
                'cpu': {}
            }
            
            # Timing statistics
            for name, times in self.timings.items():
                if times:
                    stats['timings'][name] = {
                        'count': len(times),
                        'total': sum(times),
                        'mean': np.mean(times),
                        'std': np.std(times),
                        'min': min(times),
                        'max': max(times),
                        'median': np.median(times)
                    }
            
            # Memory statistics
            if self.memory_usage:
                stats['memory'] = {
                    'current_mb': self.memory_usage[-1] if self.memory_usage else 0,
                    'mean_mb': np.mean(self.memory_usage),
                    'max_mb': max(self.memory_usage),
                    'min_mb': min(self.memory_usage)
                }
            
            # CPU statistics
            if self.cpu_usage:
                stats['cpu'] = {
                    'current_percent': self.cpu_usage[-1] if self.cpu_usage else 0,
                    'mean_percent': np.mean(self.cpu_usage),
                    'max_percent': max(self.cpu_usage),
                    'min_percent': min(self.cpu_usage)
                }
            
            return stats
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.timings.clear()
            self.counters.clear()
            self.memory_usage.clear()
            self.cpu_usage.clear()
            if hasattr(self, '_timers'):
                self._timers.clear()
    
    def timer_context(self, name: str):
        """Context manager for timing operations."""
        class TimerContext:
            def __init__(self, metrics, timer_name):
                self.metrics = metrics
                self.name = timer_name
            
            def __enter__(self):
                self.metrics.start_timer(self.name)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.metrics.end_timer(self.name)
        
        return TimerContext(self, name)


class MetricsAggregator:
    """Aggregates metrics from multiple sources."""
    
    def __init__(self):
        self.metrics = {}
        self.audio_metrics = AudioMetrics()
        self.model_metrics = ModelMetrics()
        self.performance_metrics = PerformanceMetrics()
    
    def add_metric(self, name: str, value: Union[float, int, Dict], category: str = 'custom') -> None:
        """Add a custom metric."""
        if category not in self.metrics:
            self.metrics[category] = {}
        self.metrics[category][name] = value
    
    def calculate_audio_metrics(
        self,
        signal: np.ndarray,
        sample_rate: int,
        reference: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive audio metrics."""
        metrics = {}
        
        try:
            metrics['dynamic_range'] = self.audio_metrics.dynamic_range(signal)
            metrics['zero_crossing_rate'] = self.audio_metrics.zero_crossing_rate(signal)
            metrics['spectral_centroid'] = self.audio_metrics.spectral_centroid(signal, sample_rate)
            
            if reference is not None:
                # Calculate relative metrics
                noise = signal - reference
                metrics['snr'] = self.audio_metrics.snr(reference, noise)
        
        except Exception as e:
            logger.error(f"Error calculating audio metrics: {e}")
            
        return metrics
    
    def calculate_model_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        task_type: str = 'classification'
    ) -> Dict[str, Any]:
        """Calculate comprehensive model metrics."""
        metrics = {}
        
        try:
            if task_type == 'classification':
                metrics['accuracy'] = self.model_metrics.accuracy(predictions, targets)
                prf = self.model_metrics.precision_recall_f1(predictions, targets)
                metrics.update(prf)
                metrics['confusion_matrix'] = self.model_metrics.confusion_matrix(predictions, targets).tolist()
            
            elif task_type == 'regression':
                metrics['mse'] = self.model_metrics.mse(predictions, targets)
                metrics['mae'] = self.model_metrics.mae(predictions, targets)
                metrics['r2'] = self.model_metrics.r2_score(predictions, targets)
        
        except Exception as e:
            logger.error(f"Error calculating model metrics: {e}")
            
        return metrics
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        all_metrics = {
            'performance': self.performance_metrics.get_stats(),
            'custom': self.metrics
        }
        
        return all_metrics
    
    def export_metrics(self, format: str = 'dict') -> Union[Dict, str]:
        """Export metrics in specified format."""
        metrics = self.get_all_metrics()
        
        if format == 'dict':
            return metrics
        elif format == 'json':
            import json
            return json.dumps(metrics, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global metrics instance for convenience
_global_metrics = MetricsAggregator()

def get_global_metrics() -> MetricsAggregator:
    """Get the global metrics aggregator."""
    return _global_metrics

def reset_global_metrics() -> None:
    """Reset the global metrics aggregator."""
    global _global_metrics
    _global_metrics = MetricsAggregator()


# Export all classes and functions
__all__ = [
    'AudioMetrics',
    'ModelMetrics', 
    'PerformanceMetrics',
    'MetricsAggregator',
    'get_global_metrics',
    'reset_global_metrics'
]