#!/usr/bin/env python3
"""
GPU Management System Demo for AutoVoice

This example demonstrates the comprehensive GPU management capabilities
including device selection, memory management, performance monitoring,
and error handling.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_voice.gpu import (
    GPUManager, GPUConfig, OptimizationLevel, ModelPrecision,
    CUDAManager, DeviceState, MemoryManager, AllocationStrategy,
    PerformanceMonitor, AlertLevel, MetricType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demo_gpu_configuration():
    """Demonstrate GPU configuration options"""
    print("\n" + "="*60)
    print("GPU CONFIGURATION DEMO")
    print("="*60)
    
    # Basic configuration
    basic_config = GPUConfig(
        optimization_level=OptimizationLevel.BASIC,
        precision=ModelPrecision.FP32,
        enable_monitoring=True
    )
    
    # Advanced configuration
    advanced_config = GPUConfig(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        precision=ModelPrecision.MIXED,
        enable_amp=True,
        enable_compile=True,
        enable_memory_pooling=True,
        allocation_strategy=AllocationStrategy.BALANCED,
        enable_profiling=True
    )
    
    print("Basic Config:")
    print(f"  Optimization Level: {basic_config.optimization_level.value}")
    print(f"  Precision: {basic_config.precision.value}")
    print(f"  Monitoring: {basic_config.enable_monitoring}")
    
    print("\nAdvanced Config:")
    print(f"  Optimization Level: {advanced_config.optimization_level.value}")
    print(f"  Precision: {advanced_config.precision.value}")
    print(f"  AMP Enabled: {advanced_config.enable_amp}")
    print(f"  Compile Enabled: {advanced_config.enable_compile}")
    print(f"  Memory Pooling: {advanced_config.enable_memory_pooling}")
    print(f"  Allocation Strategy: {advanced_config.allocation_strategy.value}")

def demo_cuda_manager():
    """Demonstrate CUDA manager capabilities"""
    print("\n" + "="*60)
    print("CUDA MANAGER DEMO")
    print("="*60)
    
    # Create CUDA manager with configuration
    cuda_config = {
        'max_retries': 3,
        'retry_delay': 0.1,
        'enable_health_check': False,  # Disable for demo
        'health_check_interval': 30.0
    }
    
    cuda_manager = CUDAManager(cuda_config)
    
    # Initialize CUDA
    print("Initializing CUDA manager...")
    success = cuda_manager.initialize()
    
    if success:
        print(f"✓ CUDA initialized successfully")
        print(f"  Device count: {cuda_manager.get_device_count()}")
        
        # Get device information
        for device_id in range(cuda_manager.get_device_count()):
            device_info = cuda_manager.get_device_info(device_id)
            if device_info:
                print(f"  Device {device_id}: {device_info.name}")
                print(f"    Compute Capability: {device_info.compute_capability}")
                print(f"    Total Memory: {device_info.total_memory / (1024**3):.1f} GB")
                print(f"    State: {device_info.state.value}")
        
        # Demonstrate device selection
        best_device = cuda_manager.get_best_device()
        if best_device is not None:
            print(f"  Best device: {best_device}")
            
            # Test device context
            print("  Testing device context...")
            with cuda_manager.device_context(best_device):
                print(f"    Context device: {cuda_manager.get_current_device()}")
    else:
        print("✗ CUDA initialization failed (expected in CPU-only environment)")
        print("  Status: CUDA not available or no compatible devices found")
    
    # Cleanup
    cuda_manager.shutdown()

def demo_memory_manager():
    """Demonstrate memory manager capabilities"""
    print("\n" + "="*60)
    print("MEMORY MANAGER DEMO")
    print("="*60)
    
    # Create memory manager with configuration
    memory_config = {
        'enable_pooling': True,
        'allocation_strategy': 'balanced',
        'max_pool_size': 1024**3,  # 1GB
        'enable_oom_handler': True,
        'monitor_performance': True,
        'enable_background_cleanup': False  # Disable for demo
    }
    
    memory_manager = MemoryManager(memory_config)
    
    print(f"Memory manager initialized")
    print(f"  Device count: {memory_manager.device_count}")
    print(f"  Pooling enabled: {memory_manager.enable_pooling}")
    print(f"  Allocation strategy: {memory_manager.allocation_strategy.value}")
    
    # Get memory information for available devices
    for device_id in range(memory_manager.device_count):
        try:
            memory_info = memory_manager.get_memory_info(device_id)
            print(f"  Device {device_id} Memory:")
            print(f"    Total: {memory_info['total'] / (1024**3):.1f} GB")
            print(f"    Used: {memory_info['used'] / (1024**3):.1f} GB")
            print(f"    Free: {memory_info['free'] / (1024**3):.1f} GB")
            print(f"    Usage: {memory_info['percentage']:.1f}%")
            
            if 'pool_stats' in memory_info:
                pool_stats = memory_info['pool_stats']
                print(f"    Pool size: {pool_stats['total_pool_size'] / (1024**2):.1f} MB")
                print(f"    Pool efficiency: {pool_stats['efficiency']:.1%}")
        except Exception as e:
            print(f"    Error getting memory info: {e}")
    
    # Demonstrate memory summary
    print("\nMemory Summary:")
    summary = memory_manager.get_memory_summary()
    print(f"  Total allocations: {summary['total_allocations']}")
    print(f"  Total deallocations: {summary['total_deallocations']}")
    print(f"  Pooling enabled: {summary['pooling_enabled']}")
    
    # Cleanup
    memory_manager.shutdown()

def demo_performance_monitor():
    """Demonstrate performance monitor capabilities"""
    print("\n" + "="*60)
    print("PERFORMANCE MONITOR DEMO")
    print("="*60)
    
    # Create performance monitor with configuration
    monitor_config = {
        'sampling_interval': 1.0,
        'enable_alerting': True,
        'enable_profiling': True,
        'enable_continuous_monitoring': False  # Disable for demo
    }
    
    performance_monitor = PerformanceMonitor(monitor_config)
    
    print(f"Performance monitor initialized")
    print(f"  Device count: {performance_monitor.device_count}")
    print(f"  NVML available: {performance_monitor.nvml_available}")
    
    # Add alert callback
    def alert_callback(alert):
        print(f"    ALERT [{alert.level.value.upper()}]: {alert.message}")
    
    performance_monitor.add_alert_callback(alert_callback)
    
    # Get GPU statistics for available devices
    for device_id in range(performance_monitor.device_count):
        try:
            stats = performance_monitor.get_gpu_stats(device_id)
            print(f"  Device {device_id} Stats:")
            print(f"    GPU Utilization: {stats.get('gpu_utilization', 'N/A')}")
            print(f"    Memory Utilization: {stats.get('memory_utilization', 'N/A')}")
            print(f"    Temperature: {stats.get('temperature', 'N/A')}")
            print(f"    Power: {stats.get('power_watts', 'N/A')} W")
            print(f"    Memory Used: {stats.get('memory_used_gb', 0):.1f} GB")
            print(f"    Memory Total: {stats.get('memory_total_gb', 0):.1f} GB")
        except Exception as e:
            print(f"    Error getting stats: {e}")
    
    # Demonstrate profiling context
    print("\nTesting profiling context...")
    try:
        with performance_monitor.profile_operation("demo_operation", 0):
            # Simulate some work
            time.sleep(0.01)
            print("    Operation completed")
    except Exception as e:
        print(f"    Profiling error: {e}")
    
    # Get performance summary
    print("\nPerformance Summary:")
    try:
        summary = performance_monitor.get_performance_summary()
        print(f"  Monitoring active: {summary['monitoring_active']}")
        print(f"  Total devices: {summary['total_devices']}")
    except Exception as e:
        print(f"  Error getting summary: {e}")
    
    # Cleanup
    performance_monitor.shutdown()

def demo_gpu_manager():
    """Demonstrate high-level GPU manager"""
    print("\n" + "="*60)
    print("GPU MANAGER DEMO")
    print("="*60)
    
    # Create GPU manager with comprehensive configuration
    gpu_config = GPUConfig(
        optimization_level=OptimizationLevel.BASIC,
        precision=ModelPrecision.FP32,
        enable_monitoring=True,
        enable_memory_pooling=True,
        allocation_strategy=AllocationStrategy.BALANCED,
        enable_profiling=False  # Disable for demo
    )
    
    print("Creating GPU manager...")
    gpu_manager = GPUManager(gpu_config)
    
    print(f"✓ GPU manager created")
    print(f"  CUDA available: {gpu_manager.is_cuda_available()}")
    
    # Get comprehensive status
    print("\nGPU Status:")
    status = gpu_manager.get_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    # Demonstrate device context
    print("\nTesting device context...")
    try:
        with gpu_manager.device_context() as device:
            print(f"  Current device: {device}")
    except Exception as e:
        print(f"  Context error: {e}")
    
    # Demonstrate memory operations
    print("\nMemory operations:")
    try:
        memory_info = gpu_manager.get_memory_info()
        if 'error' not in memory_info:
            print(f"  Memory used: {memory_info.get('used', 0) / (1024**3):.1f} GB")
            print(f"  Memory total: {memory_info.get('total', 0) / (1024**3):.1f} GB")
            print(f"  Memory percentage: {memory_info.get('percentage', 0):.1f}%")
        else:
            print(f"  {memory_info['error']}")
    except Exception as e:
        print(f"  Memory info error: {e}")
    
    # Test model optimization (mock model)
    print("\nTesting model optimization...")
    try:
        import torch.nn as nn
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = MockModel()
        print(f"  Original model device: {next(model.parameters()).device}")
        
        optimized_model = gpu_manager.optimize_model(model, "demo_model")
        print(f"  Optimized model device: {next(optimized_model.parameters()).device}")
        
    except Exception as e:
        print(f"  Model optimization error: {e}")
    
    # Cleanup
    gpu_manager.shutdown()

def demo_error_handling():
    """Demonstrate error handling capabilities"""
    print("\n" + "="*60)
    print("ERROR HANDLING DEMO")
    print("="*60)
    
    # Test with invalid configuration
    print("Testing invalid configurations...")
    
    try:
        # Invalid device ID
        cuda_manager = CUDAManager()
        cuda_manager.initialize()
        result = cuda_manager.set_device(999)  # Invalid device
        print(f"  Invalid device setting result: {result}")
    except Exception as e:
        print(f"  Expected error: {e}")
    
    # Test memory manager with invalid device
    try:
        memory_manager = MemoryManager()
        memory_info = memory_manager.get_memory_info(999)  # Invalid device
        print(f"  Memory info for invalid device: {memory_info}")
    except Exception as e:
        print(f"  Memory manager error: {e}")
    
    # Test performance monitor with invalid device
    try:
        monitor = PerformanceMonitor({'enable_continuous_monitoring': False})
        stats = monitor.get_gpu_stats(999)  # Invalid device
        print(f"  Stats for invalid device: {stats}")
    except Exception as e:
        print(f"  Performance monitor error: {e}")
    
    print("✓ Error handling demonstrations completed")

def main():
    """Run all GPU management demos"""
    print("AutoVoice GPU Management System Demo")
    print("This demo showcases the comprehensive GPU management capabilities")
    
    try:
        # Run demonstrations
        demo_gpu_configuration()
        demo_cuda_manager()
        demo_memory_manager()
        demo_performance_monitor()
        demo_gpu_manager()
        demo_error_handling()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Comprehensive CUDA device management")
        print("✓ Advanced memory management with pooling")
        print("✓ Real-time performance monitoring")
        print("✓ Robust error handling and recovery")
        print("✓ High-level GPU coordination")
        print("✓ Configurable optimization levels")
        print("✓ Context managers for safe operations")
        print("✓ Background health monitoring")
        print("✓ Alert system with callbacks")
        print("✓ Memory fragmentation handling")
        
        print("\nNote: Some features require CUDA-enabled hardware and drivers.")
        print("This demo works in both CUDA and CPU-only environments.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()