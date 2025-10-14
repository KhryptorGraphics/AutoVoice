"""
Real-time audio processing pipeline
"""
import torch
import numpy as np
from typing import Optional, Callable, Any
import threading
import queue
import time
import logging

class RealtimeProcessor:
    """Real-time audio processing pipeline"""

    def __init__(self, model: torch.nn.Module, device: str = 'cuda',
                 buffer_size: int = 1024, sample_rate: int = 22050):
        self.model = model
        self.device = device
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate

        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.running = False

        self.logger = logging.getLogger(__name__)

        # Warm up model
        self._warmup_model()

    def _warmup_model(self):
        """Warm up model with dummy data"""
        try:
            dummy_input = torch.randn(1, self.buffer_size, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            self.logger.info("Model warmed up successfully")
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")

    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get input data
                input_data = self.input_queue.get(timeout=0.1)

                # Process with model
                start_time = time.time()
                with torch.no_grad():
                    output = self.model(input_data)
                processing_time = time.time() - start_time

                # Put result in output queue
                self.output_queue.put(output, timeout=0.1)

                # Log performance metrics
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Processing time: {processing_time:.4f}s")

            except queue.Empty:
                continue
            except queue.Full:
                self.logger.warning("Output queue full, dropping frame")
                continue
            except Exception as e:
                self.logger.error(f"Processing error: {e}")

    def start(self):
        """Start real-time processing"""
        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info("Real-time processor started")

    def stop(self):
        """Stop real-time processing"""
        if not self.running:
            return

        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)

        # Clear queues
        while not self.input_queue.empty():
            self.input_queue.get_nowait()
        while not self.output_queue.empty():
            self.output_queue.get_nowait()

        self.logger.info("Real-time processor stopped")

    def process_audio(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process audio chunk and return result"""
        try:
            # Convert to tensor
            tensor_input = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.device)

            # Submit for processing
            self.input_queue.put(tensor_input, timeout=0.01)

            # Get result
            result = self.output_queue.get(timeout=0.05)
            return result.cpu().numpy().squeeze()

        except queue.Full:
            self.logger.warning("Input queue full, dropping frame")
            return None
        except queue.Empty:
            self.logger.warning("No output available")
            return None
        except Exception as e:
            self.logger.error(f"Process audio error: {e}")
            return None


class AsyncRealtimeProcessor:
    """Async version of real-time processor for integration with async frameworks."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda', **kwargs):
        self.processor = RealtimeProcessor(model, device, **kwargs)
        self.loop = None
        
    async def start(self):
        """Start async processing."""
        self.loop = asyncio.get_event_loop()
        await self.loop.run_in_executor(None, self.processor.start)
    
    async def stop(self):
        """Stop async processing."""
        if self.loop:
            await self.loop.run_in_executor(None, self.processor.stop)
    
    async def process_audio(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process audio chunk asynchronously."""
        if self.loop:
            return await self.loop.run_in_executor(
                None, self.processor.process_audio, audio_chunk
            )
        return None
    
    async def get_performance_stats(self) -> dict:
        """Get performance stats asynchronously."""
        if self.loop:
            return await self.loop.run_in_executor(
                None, self.processor.get_performance_stats
            )
        return {}