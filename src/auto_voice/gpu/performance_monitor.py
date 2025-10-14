"""Advanced GPU performance monitoring with metrics tracking and alerting"""
import torch
import time
import threading
import statistics
from typing import Dict, Any, Optional, List, Callable, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import contextmanager
import json
from datetime import datetime, timedelta

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics to monitor"""
    GPU_UTILIZATION = "gpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    TEMPERATURE = "temperature"
    POWER_USAGE = "power_usage"
    MEMORY_ALLOCATED = "memory_allocated"
    COMPUTE_TIME = "compute_time"
    THROUGHPUT = "throughput"
    LATENCY = "latency"

@dataclass
class MetricSample:
    """A single metric sample"""
    timestamp: float
    value: float
    device_id: int
    metric_type: MetricType

@dataclass
class Alert:
    """Performance alert"""
    level: AlertLevel
    message: str
    timestamp: float
    device_id: int
    metric_type: MetricType
    value: float
    threshold: float

@dataclass
class PerformanceProfile:
    """Performance profiling result"""
    operation_name: str
    device_id: int
    duration_ms: float
    memory_used_mb: float
    memory_peak_mb: float
    gpu_utilization: Optional[float]
    efficiency_score: float
    bottlenecks: List[str]
    recommendations: List[str]

class MetricHistory:
    """Maintains rolling history of metrics"""
    def __init__(self, max_samples: int = 1000):
        self.samples: deque = deque(maxlen=max_samples)
        self.max_samples = max_samples
    
    def add_sample(self, sample: MetricSample):
        self.samples.append(sample)
    
    def get_recent(self, seconds: float) -> List[MetricSample]:
        cutoff = time.time() - seconds
        return [s for s in self.samples if s.timestamp >= cutoff]
    
    def get_statistics(self, seconds: float = None) -> Dict[str, float]:
        samples = self.get_recent(seconds) if seconds else list(self.samples)
        if not samples:
            return {}
        
        values = [s.value for s in samples]
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'count': len(values)
        }

class PerformanceMonitor:
    """Comprehensive GPU performance monitoring with alerting and profiling"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize NVML
        self.nvml_available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_available = True
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.logger.info("NVML initialized for performance monitoring")
            except Exception as e:
                self.logger.warning(f"NVML initialization failed: {e}")
                self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        else:
            self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            self.logger.warning("NVML not available, limited monitoring capabilities")
        
        # Monitoring configuration
        self.sampling_interval = config.get('sampling_interval', 1.0)  # seconds
        self.history_size = config.get('history_size', 1000)
        self.enable_alerting = config.get('enable_alerting', True)
        self.enable_profiling = config.get('enable_profiling', True)
        self.enable_continuous_monitoring = config.get('enable_continuous_monitoring', True)
        
        # Metric histories per device
        self.metric_histories: Dict[int, Dict[MetricType, MetricHistory]] = defaultdict(
            lambda: defaultdict(lambda: MetricHistory(self.history_size))
        )
        
        # Alert system
        self.alert_thresholds = self._setup_default_thresholds()
        self.alert_thresholds.update(config.get('alert_thresholds', {}))
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.alert_history: deque = deque(maxlen=100)
        
        # Performance profiling
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self.profile_history: deque = deque(maxlen=100)
        
        # Synchronization
        self.lock = threading.RLock()
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Benchmarking
        self.benchmark_results: Dict[str, List[float]] = defaultdict(list)
        
        # Start continuous monitoring if enabled
        if self.enable_continuous_monitoring:
            self.start_monitoring()

    def _setup_default_thresholds(self) -> Dict[MetricType, Dict[AlertLevel, float]]:
        """Setup default alert thresholds"""
        return {
            MetricType.GPU_UTILIZATION: {
                AlertLevel.WARNING: 85.0,
                AlertLevel.ERROR: 95.0,
                AlertLevel.CRITICAL: 98.0
            },
            MetricType.MEMORY_UTILIZATION: {
                AlertLevel.WARNING: 80.0,
                AlertLevel.ERROR: 90.0,
                AlertLevel.CRITICAL: 95.0
            },
            MetricType.TEMPERATURE: {
                AlertLevel.WARNING: 75.0,
                AlertLevel.ERROR: 85.0,
                AlertLevel.CRITICAL: 90.0
            },
            MetricType.POWER_USAGE: {
                AlertLevel.WARNING: 250.0,  # Watts
                AlertLevel.ERROR: 300.0,
                AlertLevel.CRITICAL: 350.0
            }
        }

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)

    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                for device_id in range(self.device_count):
                    self._collect_metrics(device_id)
                time.sleep(self.sampling_interval)
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.sampling_interval)

    def _collect_metrics(self, device_id: int):
        """Collect metrics for a device"""
        timestamp = time.time()
        
        try:
            # Get comprehensive GPU stats
            stats = self.get_gpu_stats(device_id)
            
            # Record metrics
            metrics_to_record = [
                (MetricType.GPU_UTILIZATION, stats.get('gpu_utilization')),
                (MetricType.MEMORY_UTILIZATION, stats.get('memory_utilization')),
                (MetricType.TEMPERATURE, stats.get('temperature')),
                (MetricType.POWER_USAGE, stats.get('power_watts')),
                (MetricType.MEMORY_ALLOCATED, stats.get('memory_used_gb', 0) * 1024)  # Convert to MB
            ]
            
            for metric_type, value in metrics_to_record:
                if value is not None:
                    sample = MetricSample(timestamp, value, device_id, metric_type)
                    self.metric_histories[device_id][metric_type].add_sample(sample)
                    
                    # Check for alerts
                    if self.enable_alerting:
                        self._check_alert(sample)
        
        except Exception as e:
            self.logger.warning(f"Failed to collect metrics for device {device_id}: {e}")

    def _check_alert(self, sample: MetricSample):
        """Check if metric sample triggers an alert"""
        metric_thresholds = self.alert_thresholds.get(sample.metric_type, {})
        
        for level in [AlertLevel.CRITICAL, AlertLevel.ERROR, AlertLevel.WARNING]:
            threshold = metric_thresholds.get(level)
            if threshold is not None and sample.value >= threshold:
                alert = Alert(
                    level=level,
                    message=f"Device {sample.device_id} {sample.metric_type.value} at {sample.value:.1f} (threshold: {threshold})",
                    timestamp=sample.timestamp,
                    device_id=sample.device_id,
                    metric_type=sample.metric_type,
                    value=sample.value,
                    threshold=threshold
                )
                
                self._trigger_alert(alert)
                break  # Only trigger highest severity alert

    def _trigger_alert(self, alert: Alert):
        """Trigger an alert"""
        self.alert_history.append(alert)
        self.logger.log(
            logging.CRITICAL if alert.level == AlertLevel.CRITICAL else
            logging.ERROR if alert.level == AlertLevel.ERROR else
            logging.WARNING,
            alert.message
        )
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")

    def get_gpu_stats(self, device_id: int = 0) -> Dict[str, Any]:
        """Get comprehensive GPU statistics"""
        if device_id >= self.device_count:
            raise ValueError(f"Invalid device_id {device_id}, only {self.device_count} devices available")
        
        base_stats = {
            'device_id': device_id,
            'timestamp': time.time(),
            'gpu_utilization': None,
            'memory_utilization': None,
            'temperature': None,
            'power_watts': None,
            'memory_used_gb': 0,
            'memory_total_gb': 0,
            'memory_free_gb': 0,
            'compute_capability': None,
            'driver_version': None,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
        
        # Get PyTorch memory info
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(device_id)
                total = torch.cuda.get_device_properties(device_id).total_memory
                props = torch.cuda.get_device_properties(device_id)
                
                base_stats.update({
                    'memory_used_gb': allocated / (1024**3),
                    'memory_total_gb': total / (1024**3),
                    'memory_free_gb': (total - allocated) / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'device_name': props.name,
                    'multiprocessor_count': props.multi_processor_count
                })
            except Exception as e:
                self.logger.warning(f"Failed to get PyTorch memory info: {e}")
        
        # Get NVML stats if available
        if self.nvml_available:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                
                # Utilization
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    base_stats['gpu_utilization'] = utilization.gpu
                    base_stats['memory_utilization'] = utilization.memory
                except pynvml.NVMLError:
                    pass
                
                # Temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    base_stats['temperature'] = temperature
                except pynvml.NVMLError:
                    pass
                
                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    base_stats['power_watts'] = power
                except pynvml.NVMLError:
                    pass
                
                # Memory info from NVML (more accurate)
                try:
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    base_stats.update({
                        'memory_used_gb': memory.used / (1024**3),
                        'memory_total_gb': memory.total / (1024**3),
                        'memory_free_gb': memory.free / (1024**3)
                    })
                except pynvml.NVMLError:
                    pass
                
                # Driver version
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    base_stats['driver_version'] = driver_version
                except pynvml.NVMLError:
                    pass
                    
            except Exception as e:
                self.logger.warning(f"NVML stats collection failed for device {device_id}: {e}")
        
        return base_stats

    @contextmanager
    def profile_operation(self, operation_name: str, device_id: int = 0):
        """Context manager for profiling GPU operations"""
        if not self.enable_profiling:
            yield
            return
        
        profile_id = f"{operation_name}_{device_id}_{time.time()}"
        
        # Pre-operation metrics
        torch.cuda.synchronize(device_id)
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated(device_id) if torch.cuda.is_available() else 0
        start_stats = self.get_gpu_stats(device_id)
        
        self.active_profiles[profile_id] = {
            'operation_name': operation_name,
            'device_id': device_id,
            'start_time': start_time,
            'start_memory': start_memory,
            'start_stats': start_stats
        }
        
        try:
            yield profile_id
        finally:
            # Post-operation metrics
            torch.cuda.synchronize(device_id)
            end_time = time.perf_counter()
            end_memory = torch.cuda.memory_allocated(device_id) if torch.cuda.is_available() else 0
            end_stats = self.get_gpu_stats(device_id)
            
            if profile_id in self.active_profiles:
                profile_data = self.active_profiles.pop(profile_id)
                
                # Calculate metrics
                duration_ms = (end_time - start_time) * 1000
                memory_used_mb = (end_memory - start_memory) / (1024**2)
                peak_memory_mb = (torch.cuda.max_memory_allocated(device_id) - start_memory) / (1024**2) if torch.cuda.is_available() else 0
                
                # Analyze performance
                bottlenecks, recommendations = self._analyze_performance(
                    duration_ms, memory_used_mb, start_stats, end_stats
                )
                
                # Calculate efficiency score
                efficiency_score = self._calculate_efficiency_score(
                    duration_ms, memory_used_mb, start_stats, end_stats
                )
                
                # Create profile result
                profile = PerformanceProfile(
                    operation_name=operation_name,
                    device_id=device_id,
                    duration_ms=duration_ms,
                    memory_used_mb=memory_used_mb,
                    memory_peak_mb=peak_memory_mb,
                    gpu_utilization=end_stats.get('gpu_utilization'),
                    efficiency_score=efficiency_score,
                    bottlenecks=bottlenecks,
                    recommendations=recommendations
                )
                
                self.profile_history.append(profile)
                self.logger.debug(f"Profiled {operation_name}: {duration_ms:.2f}ms, {memory_used_mb:.1f}MB")

    def _analyze_performance(self, duration_ms: float, memory_used_mb: float, 
                           start_stats: Dict, end_stats: Dict) -> tuple[List[str], List[str]]:
        """Analyze performance and identify bottlenecks"""
        bottlenecks = []
        recommendations = []
        
        # Memory bottlenecks
        memory_usage_pct = end_stats.get('memory_used_gb', 0) / max(end_stats.get('memory_total_gb', 1), 1) * 100
        if memory_usage_pct > 90:
            bottlenecks.append("High memory usage")
            recommendations.append("Consider reducing batch size or using gradient checkpointing")
        
        # GPU utilization
        gpu_util = end_stats.get('gpu_utilization', 0)
        if gpu_util and gpu_util < 50:
            bottlenecks.append("Low GPU utilization")
            recommendations.append("Consider increasing batch size or optimizing data loading")
        
        # Temperature
        temperature = end_stats.get('temperature')
        if temperature and temperature > 80:
            bottlenecks.append("High temperature")
            recommendations.append("Check cooling and consider reducing workload")
        
        # Long duration heuristics
        if duration_ms > 1000:  # > 1 second
            recommendations.append("Consider operation optimization or parallelization")
        
        return bottlenecks, recommendations

    def _calculate_efficiency_score(self, duration_ms: float, memory_used_mb: float,
                                  start_stats: Dict, end_stats: Dict) -> float:
        """Calculate efficiency score (0-100)"""
        score = 100.0
        
        # GPU utilization component
        gpu_util = end_stats.get('gpu_utilization', 50)
        if gpu_util:
            util_score = min(100, gpu_util)
            score = score * 0.4 + util_score * 0.4
        
        # Memory efficiency component
        memory_total_gb = end_stats.get('memory_total_gb', 1)
        memory_efficiency = min(100, (memory_used_mb / 1024) / memory_total_gb * 100)
        score = score * 0.7 + memory_efficiency * 0.3
        
        # Temperature penalty
        temperature = end_stats.get('temperature')
        if temperature:
            if temperature > 85:
                score *= 0.8
            elif temperature > 75:
                score *= 0.9
        
        return max(0.0, min(100.0, score))

    def get_metric_statistics(self, device_id: int, metric_type: MetricType, 
                            seconds: Optional[float] = None) -> Dict[str, float]:
        """Get statistics for a specific metric"""
        history = self.metric_histories[device_id][metric_type]
        return history.get_statistics(seconds)

    def get_recent_alerts(self, device_id: Optional[int] = None, 
                         level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get recent alerts with optional filtering"""
        alerts = list(self.alert_history)
        
        if device_id is not None:
            alerts = [a for a in alerts if a.device_id == device_id]
        
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_performance_summary(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        devices = [device_id] if device_id is not None else list(range(self.device_count))
        
        summary = {
            'timestamp': time.time(),
            'monitoring_active': self.monitoring_active,
            'total_devices': self.device_count,
            'devices': {}
        }
        
        for dev_id in devices:
            device_summary = {
                'current_stats': self.get_gpu_stats(dev_id),
                'recent_alerts': len(self.get_recent_alerts(dev_id)),
                'metrics': {}
            }
            
            # Add metric statistics
            for metric_type in MetricType:
                if metric_type in self.metric_histories[dev_id]:
                    device_summary['metrics'][metric_type.value] = self.get_metric_statistics(
                        dev_id, metric_type, 300  # Last 5 minutes
                    )
            
            summary['devices'][dev_id] = device_summary
        
        return summary

    def benchmark_operation(self, operation_name: str, operation_func: Callable, 
                          num_runs: int = 10, warmup_runs: int = 3,
                          device_id: int = 0) -> Dict[str, Any]:
        """Benchmark an operation with multiple runs"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for benchmarking")
        
        # Warmup runs
        for _ in range(warmup_runs):
            with self.profile_operation(f"{operation_name}_warmup", device_id):
                operation_func()
        
        # Benchmark runs
        durations = []
        memory_usage = []
        
        for run in range(num_runs):
            with self.profile_operation(f"{operation_name}_run_{run}", device_id) as profile_id:
                start_memory = torch.cuda.memory_allocated(device_id)
                torch.cuda.synchronize(device_id)
                start_time = time.perf_counter()
                
                operation_func()
                
                torch.cuda.synchronize(device_id)
                end_time = time.perf_counter()
                end_memory = torch.cuda.memory_allocated(device_id)
                
                duration_ms = (end_time - start_time) * 1000
                memory_mb = (end_memory - start_memory) / (1024**2)
                
                durations.append(duration_ms)
                memory_usage.append(memory_mb)
        
        # Calculate statistics
        benchmark_results = {
            'operation_name': operation_name,
            'device_id': device_id,
            'num_runs': num_runs,
            'duration_ms': {
                'mean': statistics.mean(durations),
                'median': statistics.median(durations),
                'min': min(durations),
                'max': max(durations),
                'std': statistics.stdev(durations) if len(durations) > 1 else 0.0
            },
            'memory_mb': {
                'mean': statistics.mean(memory_usage),
                'median': statistics.median(memory_usage),
                'min': min(memory_usage),
                'max': max(memory_usage),
                'std': statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0.0
            },
            'throughput_ops_per_sec': 1000.0 / statistics.mean(durations) if durations else 0
        }
        
        # Store results
        self.benchmark_results[operation_name] = durations
        
        return benchmark_results

    def export_metrics(self, filepath: str, device_id: Optional[int] = None, 
                      format: str = 'json'):
        """Export metrics to file"""
        data = {
            'timestamp': time.time(),
            'export_format': format,
            'performance_summary': self.get_performance_summary(device_id),
            'alerts': [{
                'level': alert.level.value,
                'message': alert.message,
                'timestamp': alert.timestamp,
                'device_id': alert.device_id,
                'metric_type': alert.metric_type.value,
                'value': alert.value,
                'threshold': alert.threshold
            } for alert in self.alert_history],
            'profiles': [{
                'operation_name': profile.operation_name,
                'device_id': profile.device_id,
                'duration_ms': profile.duration_ms,
                'memory_used_mb': profile.memory_used_mb,
                'efficiency_score': profile.efficiency_score,
                'bottlenecks': profile.bottlenecks,
                'recommendations': profile.recommendations
            } for profile in self.profile_history]
        }
        
        with open(filepath, 'w') as f:
            if format.lower() == 'json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Metrics exported to {filepath}")

    def shutdown(self):
        """Shutdown performance monitor"""
        self.stop_monitoring()
        
        if self.nvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                self.logger.error(f"NVML shutdown error: {e}")
        
        self.logger.info("Performance monitor shutdown complete")

    def __del__(self):
        """Destructor cleanup"""
        try:
            self.shutdown()
        except Exception:
            pass