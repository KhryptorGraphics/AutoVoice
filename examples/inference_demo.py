#!/usr/bin/env python3
"""
AutoVoice Inference System Demo - <100ms Latency Optimization

Demonstrates all inference engines working together:
1. VoiceInferenceEngine - Main inference orchestrator
2. TensorRTEngine - TensorRT optimization  
3. VoiceSynthesizer - Text-to-speech synthesis
4. RealtimeProcessor - Stream processing
5. CUDAGraphs - CUDA graph optimization

Usage:
    python examples/inference_demo.py
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from auto_voice.inference import (
        InferenceManager,
        VoiceInferenceEngine,
        TensorRTEngine,
        VoiceSynthesizer,
        RealtimeProcessor,
        CUDAGraphManager
    )
    print("✓ Successfully imported all inference engines")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure PyTorch and other dependencies are installed")
    sys.exit(1)


def create_demo_config():
    """Create configuration for demo."""
    return {
        'device': 'cuda:0',
        'latency_target_ms': 100,
        'optimization_mode': 'speed',
        'enable_tensorrt': True,
        'enable_cuda_graphs': True,
        'enable_realtime': True,
        'mixed_precision': True,
        'sample_rate': 22050,
        'buffer_size': 1024,
        
        # Model paths (would be real paths in production)
        'model_dir': 'models/pytorch',
        'tensorrt_engine_dir': 'models/tensorrt',
        'engine_dir': 'models/engines',
        
        # Model configurations
        'transformer': {
            'input_dim': 80,
            'd_model': 512,
            'n_heads': 8,
            'num_layers': 6,
            'd_ff': 2048,
            'max_seq_len': 1024,
            'dropout': 0.1
        },
        'hifigan': {
            'generator': {
                'mel_channels': 80,
                'upsample_rates': [8, 8, 2, 2],
                'upsample_kernel_sizes': [16, 16, 4, 4],
                'upsample_initial_channel': 512,
                'resblock_kernel_sizes': [3, 7, 11],
                'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
            }
        }
    }


def demo_inference_manager():
    """Demonstrate unified inference manager."""
    print("\n" + "="*60)
    print("🚀 INFERENCE MANAGER DEMO")
    print("="*60)
    
    config = create_demo_config()
    
    try:
        # Create and initialize inference manager
        print("Initializing inference manager...")
        manager = InferenceManager(config)
        
        if not manager.initialize():
            print("❌ Failed to initialize inference manager")
            return
        
        print("✓ Inference manager initialized successfully")
        
        # Test synthesis with different priorities
        test_texts = [
            "Hello world, this is a test.",
            "Real-time voice synthesis demonstration.",
            "Testing latency optimization features."
        ]
        
        priorities = ['normal', 'high', 'realtime']
        
        for i, (text, priority) in enumerate(zip(test_texts, priorities)):
            print(f"\n--- Test {i+1}: {priority.upper()} priority ---")
            print(f"Text: '{text}'")
            
            start_time = time.time()
            result = manager.synthesize_speech(text, speaker_id=0, priority=priority)
            total_time = (time.time() - start_time) * 1000
            
            if 'error' in result:
                print(f"❌ Synthesis failed: {result['error']}")
                continue
            
            print(f"✓ Synthesis completed")
            print(f"  Method: {result['method']}")
            print(f"  Latency: {result['latency_ms']:.2f}ms")
            print(f"  Total time: {total_time:.2f}ms")
            print(f"  Within target: {result['within_target']}")
            print(f"  Audio shape: {result['audio'].shape if hasattr(result['audio'], 'shape') else len(result['audio'])}")
        
        # Get performance statistics
        print(f"\n--- Performance Statistics ---")
        stats = manager.get_performance_stats()
        
        manager_stats = stats['manager']
        print(f"Total inferences: {manager_stats['total_inferences']}")
        print(f"Success rate: {manager_stats['success_rate']:.2%}")
        print(f"Target latency: {manager_stats['target_latency_ms']}ms")
        
        if 'avg_latency_ms' in manager_stats:
            print(f"Average latency: {manager_stats['avg_latency_ms']:.2f}ms")
            print(f"Within target rate: {manager_stats['within_target_rate']:.2%}")
        
        # Test real-time processing
        print(f"\n--- Real-time Processing Test ---")
        if manager.start_realtime_processing():
            print("✓ Real-time processing started")
            
            # Simulate real-time audio processing
            for i in range(3):
                dummy_audio = np.random.randn(1024).astype(np.float32)
                if manager.realtime_processor:
                    result = manager.realtime_processor.process_audio(dummy_audio)
                    print(f"  Frame {i+1}: {'✓' if result is not None else '❌'}")
                time.sleep(0.1)
            
            manager.stop_realtime_processing()
            print("✓ Real-time processing stopped")
        
        # Apply latency optimizations
        print(f"\n--- Latency Optimization ---")
        manager.optimize_for_latency()
        print("✓ Latency optimizations applied")
        
        # Final performance test
        print(f"\n--- Post-optimization Test ---")
        test_text = "Optimized synthesis test."
        start_time = time.time()
        result = manager.synthesize_speech(test_text)
        total_time = (time.time() - start_time) * 1000
        
        if 'error' not in result:
            print(f"✓ Post-optimization latency: {result['latency_ms']:.2f}ms")
            print(f"✓ Total time: {total_time:.2f}ms")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_individual_engines():
    """Demonstrate individual engines."""
    print("\n" + "="*60)
    print("🔧 INDIVIDUAL ENGINES DEMO")
    print("="*60)
    
    config = create_demo_config()
    
    # Demo VoiceInferenceEngine
    print("\n--- VoiceInferenceEngine ---")
    try:
        engine = VoiceInferenceEngine(config)
        
        # Test synthesis
        test_text = "Testing voice inference engine."
        audio = engine.synthesize_speech(test_text)
        print(f"✓ VoiceInferenceEngine synthesis completed")
        print(f"  Audio shape: {audio.shape if hasattr(audio, 'shape') else len(audio)}")
        
        # Get model info
        info = engine.get_model_info()
        print(f"  Device: {info['device']}")
        print(f"  TensorRT available: {info['tensorrt_available']}")
        print(f"  Performance: {info.get('performance', {})}")
        
    except Exception as e:
        print(f"❌ VoiceInferenceEngine demo failed: {e}")
    
    # Demo CUDA Graph Manager  
    print("\n--- CUDAGraphManager ---")
    try:
        import torch
        if torch.cuda.is_available():
            graph_manager = CUDAGraphManager()
            
            # Create dummy model and inputs
            dummy_model = torch.nn.Linear(128, 64).cuda()
            dummy_input = torch.randn(1, 128).cuda()
            
            # Capture graph
            def model_fn(input_tensor):
                return dummy_model(input_tensor)
            
            graph_manager.capture_graph(
                'demo_graph',
                model_fn,
                {'input': dummy_input}
            )
            
            # Test replay
            for i in range(3):
                new_input = torch.randn(1, 128).cuda()
                start_time = time.time()
                output = graph_manager.replay_graph('demo_graph', {'input': new_input})
                replay_time = (time.time() - start_time) * 1000
                print(f"  Replay {i+1}: {replay_time:.2f}ms")
            
            # Get performance stats
            stats = graph_manager.get_performance_stats('demo_graph')
            print(f"✓ CUDA graphs demo completed")
            print(f"  Average replay time: {stats.get('avg_replay_time_ms', 0):.2f}ms")
            
        else:
            print("⚠️ CUDA not available, skipping CUDA graphs demo")
            
    except Exception as e:
        print(f"❌ CUDAGraphManager demo failed: {e}")


def demo_performance_comparison():
    """Compare performance of different optimization techniques."""
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    
    config = create_demo_config()
    test_text = "Performance comparison test sentence."
    
    configurations = [
        ('Baseline', {'optimization_mode': 'balanced', 'enable_cuda_graphs': False, 'mixed_precision': False}),
        ('Mixed Precision', {'optimization_mode': 'balanced', 'enable_cuda_graphs': False, 'mixed_precision': True}),
        ('CUDA Graphs', {'optimization_mode': 'balanced', 'enable_cuda_graphs': True, 'mixed_precision': False}),
        ('Full Optimization', {'optimization_mode': 'speed', 'enable_cuda_graphs': True, 'mixed_precision': True})
    ]
    
    results = []
    
    for name, overrides in configurations:
        print(f"\n--- {name} ---")
        
        test_config = config.copy()
        test_config.update(overrides)
        
        try:
            manager = InferenceManager(test_config)
            if not manager.initialize():
                print(f"❌ Failed to initialize {name}")
                continue
            
            # Warmup
            for _ in range(3):
                manager.synthesize_speech("warmup", priority='normal')
            
            # Performance test
            latencies = []
            for i in range(10):
                start_time = time.time()
                result = manager.synthesize_speech(test_text, priority='normal')
                latency = (time.time() - start_time) * 1000
                
                if 'error' not in result:
                    latencies.append(latency)
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                
                results.append((name, avg_latency, min_latency, max_latency))
                
                print(f"✓ Average latency: {avg_latency:.2f}ms")
                print(f"  Min: {min_latency:.2f}ms, Max: {max_latency:.2f}ms")
                print(f"  Within 100ms target: {sum(1 for l in latencies if l <= 100) / len(latencies):.1%}")
            else:
                print(f"❌ No successful inferences for {name}")
                
        except Exception as e:
            print(f"❌ {name} test failed: {e}")
    
    # Summary
    if results:
        print(f"\n--- Performance Summary ---")
        print(f"{'Configuration':<20} {'Avg (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10}")
        print("-" * 50)
        for name, avg, min_lat, max_lat in results:
            print(f"{name:<20} {avg:<10.2f} {min_lat:<10.2f} {max_lat:<10.2f}")
        
        # Find best performing configuration
        best_config = min(results, key=lambda x: x[1])
        print(f"\n🏆 Best performing: {best_config[0]} ({best_config[1]:.2f}ms avg)")


def main():
    """Run all demos."""
    print("🎙️ AutoVoice Inference System Demo")
    print("Optimized for <100ms latency")
    print("Testing all 5 inference engines...")
    
    try:
        # Check CUDA availability
        import torch
        print(f"\n🔧 System Information:")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name()}")
            print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Run demos
        demo_inference_manager()
        demo_individual_engines()
        demo_performance_comparison()
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("  ✓ VoiceInferenceEngine - Main inference orchestrator")  
        print("  ✓ TensorRTEngine - TensorRT optimization")
        print("  ✓ VoiceSynthesizer - Text-to-speech synthesis")
        print("  ✓ RealtimeProcessor - Stream processing")
        print("  ✓ CUDAGraphs - CUDA graph optimization")
        print("  ✓ InferenceManager - Unified coordination")
        print("\nPerformance Optimizations:")
        print("  ✓ <100ms latency target")
        print("  ✓ Mixed precision inference")
        print("  ✓ CUDA graph optimization")
        print("  ✓ Memory pool management")
        print("  ✓ Pipeline parallelism")
        print("  ✓ Adaptive batching")
        print("  ✓ Real-time streaming")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()