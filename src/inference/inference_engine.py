"""Inference engine for voice models."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union, List
import logging
import os

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Engine for model inference."""

    def __init__(self, model: nn.Module, device: torch.device,
                checkpoint_path: Optional[str] = None):
        """Initialize inference engine.

        Args:
            model: Model for inference
            device: Device for inference
            checkpoint_path: Path to model checkpoint
        """
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    @torch.no_grad()
    def infer(self, inputs: Union[torch.Tensor, Dict]) -> Union[torch.Tensor, Dict]:
        """Run inference on inputs.

        Args:
            inputs: Input tensor or dictionary

        Returns:
            Model outputs
        """
        self.model.eval()

        # Move inputs to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        elif isinstance(inputs, dict):
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(inputs) if isinstance(inputs, torch.Tensor) else self.model(**inputs)

        return outputs

    @torch.no_grad()
    def batch_infer(self, inputs: List, batch_size: int = 32) -> List:
        """Run inference on batch of inputs.

        Args:
            inputs: List of inputs
            batch_size: Batch size for inference

        Returns:
            List of outputs
        """
        self.model.eval()
        outputs = []

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]

            # Stack batch
            if isinstance(batch[0], torch.Tensor):
                batch = torch.stack(batch)
            elif isinstance(batch[0], dict):
                batch = {k: torch.stack([b[k] for b in batch])
                        for k in batch[0].keys()}

            # Infer
            batch_outputs = self.infer(batch)

            # Split outputs
            if isinstance(batch_outputs, torch.Tensor):
                outputs.extend(list(batch_outputs))
            elif isinstance(batch_outputs, dict):
                for j in range(len(batch)):
                    outputs.append({k: v[j] for k, v in batch_outputs.items()})

        return outputs

    def optimize_model(self, example_input: torch.Tensor):
        """Optimize model for inference.

        Args:
            example_input: Example input for tracing
        """
        self.model.eval()
        example_input = example_input.to(self.device)

        # Try to compile with torch.compile (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model)
            logger.info("Model compiled with torch.compile")
        except:
            logger.info("torch.compile not available")

        # Try to trace model
        try:
            traced_model = torch.jit.trace(self.model, example_input)
            self.model = traced_model
            logger.info("Model traced with TorchScript")
        except Exception as e:
            logger.warning(f"Could not trace model: {e}")

    def benchmark(self, input_shape: tuple, num_iterations: int = 100) -> Dict:
        """Benchmark inference performance.

        Args:
            input_shape: Shape of input tensor
            num_iterations: Number of iterations

        Returns:
            Benchmark results
        """
        import time

        # Create random input
        dummy_input = torch.randn(*input_shape, device=self.device)

        # Warmup
        for _ in range(10):
            _ = self.infer(dummy_input)

        # Benchmark
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()

        for _ in range(num_iterations):
            _ = self.infer(dummy_input)

        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = 1.0 / avg_time

        return {
            'total_time': total_time,
            'average_time': avg_time,
            'throughput': throughput,
            'num_iterations': num_iterations
        }


class StreamingInferenceEngine(InferenceEngine):
    """Streaming inference for real-time applications."""

    def __init__(self, *args, chunk_size: int = 1024,
                overlap: int = 256, **kwargs):
        """Initialize streaming inference.

        Args:
            chunk_size: Size of processing chunks
            overlap: Overlap between chunks
        """
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.buffer = None
        self.state = None

    @torch.no_grad()
    def stream_infer(self, chunk: torch.Tensor) -> torch.Tensor:
        """Process a chunk for streaming inference.

        Args:
            chunk: Input chunk

        Returns:
            Output chunk
        """
        self.model.eval()

        # Initialize buffer if needed
        if self.buffer is None:
            self.buffer = torch.zeros(
                (1, self.overlap), device=self.device
            )

        # Concatenate with buffer
        chunk = chunk.to(self.device)
        input_chunk = torch.cat([self.buffer, chunk], dim=-1)

        # Process chunk
        if hasattr(self.model, 'stream_forward'):
            output, self.state = self.model.stream_forward(input_chunk, self.state)
        else:
            output = self.model(input_chunk.unsqueeze(0)).squeeze(0)

        # Update buffer
        self.buffer = chunk[..., -self.overlap:]

        # Return output without overlap
        return output[..., self.overlap:]

    def reset_stream(self):
        """Reset streaming state."""
        self.buffer = None
        self.state = None