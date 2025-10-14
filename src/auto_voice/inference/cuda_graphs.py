"""CUDA graphs for optimized inference with <100ms latency targets."""

import logging
import torch
import time
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class CUDAGraphManager:
    """Manages CUDA graph capture and replay for optimized inference."""

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize CUDA graph manager."""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graphs = {}
        self.graph_inputs = {}
        self.graph_outputs = {}

    def capture_graph(self, name: str, model_fn: Callable,
                     static_inputs: Dict[str, torch.Tensor],
                     warmup_steps: int = 3) -> None:
        """
        Capture a CUDA graph for a model function.

        Args:
            name: Unique name for the graph
            model_fn: Function to capture (should take tensors as input)
            static_inputs: Static input tensors (will be used as placeholders)
            warmup_steps: Number of warmup steps before capture
        """
        logger.info(f"Capturing CUDA graph '{name}'")

        # Move inputs to device
        device_inputs = {}
        for key, tensor in static_inputs.items():
            device_inputs[key] = tensor.to(self.device)

        # Warmup runs
        logger.info(f"Performing {warmup_steps} warmup steps")
        with torch.cuda.device(self.device):
            for _ in range(warmup_steps):
                with torch.no_grad():
                    _ = model_fn(**device_inputs)
                torch.cuda.synchronize()

        # Capture graph
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph):
            with torch.no_grad():
                outputs = model_fn(**device_inputs)

        # Store graph and I/O tensors
        self.graphs[name] = graph
        self.graph_inputs[name] = device_inputs

        if isinstance(outputs, torch.Tensor):
            self.graph_outputs[name] = {'output': outputs}
        elif isinstance(outputs, (list, tuple)):
            self.graph_outputs[name] = {f'output_{i}': out for i, out in enumerate(outputs)}
        elif isinstance(outputs, dict):
            self.graph_outputs[name] = outputs
        else:
            self.graph_outputs[name] = {'output': outputs}

        logger.info(f"CUDA graph '{name}' captured successfully")

    def replay_graph(self, name: str, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Replay a captured CUDA graph with new inputs.

        Args:
            name: Name of the captured graph
            inputs: New input tensors (must have same shape as captured)

        Returns:
            Dictionary of output tensors
        """
        if name not in self.graphs:
            raise ValueError(f"Graph '{name}' not found. Available graphs: {list(self.graphs.keys())}")

        # Copy new inputs to graph input tensors
        for key, new_tensor in inputs.items():
            if key not in self.graph_inputs[name]:
                raise ValueError(f"Input '{key}' not found in graph '{name}'")

            graph_input = self.graph_inputs[name][key]
            if new_tensor.shape != graph_input.shape:
                raise ValueError(f"Input shape mismatch for '{key}': "
                               f"expected {graph_input.shape}, got {new_tensor.shape}")

            graph_input.copy_(new_tensor)

        # Replay graph
        self.graphs[name].replay()

        # Return copies of outputs
        outputs = {}
        for key, tensor in self.graph_outputs[name].items():
            outputs[key] = tensor.clone()

        return outputs

    def is_captured(self, name: str) -> bool:
        """Check if a graph is captured."""
        return name in self.graphs

    def list_graphs(self) -> List[str]:
        """List all captured graphs."""
        return list(self.graphs.keys())

    def clear_graph(self, name: str) -> None:
        """Clear a specific graph."""
        if name in self.graphs:
            del self.graphs[name]
            del self.graph_inputs[name]
            del self.graph_outputs[name]
            logger.info(f"Graph '{name}' cleared")

    def clear_all(self) -> None:
        """Clear all captured graphs."""
        self.graphs.clear()
        self.graph_inputs.clear()
        self.graph_outputs.clear()
        logger.info("All CUDA graphs cleared")


class GraphOptimizedModel:
    """Wrapper for models with CUDA graph optimization."""

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        """Initialize graph-optimized model wrapper."""
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_manager = CUDAGraphManager(self.device)
        self.model.to(self.device)
        self.model.eval()

    def capture_forward(self, sample_inputs: Dict[str, torch.Tensor],
                       graph_name: str = 'forward') -> None:
        """Capture forward pass as CUDA graph."""
        def forward_fn(**inputs):
            return self.model(**inputs)

        self.graph_manager.capture_graph(graph_name, forward_fn, sample_inputs)

    def forward(self, inputs: Dict[str, torch.Tensor],
                use_graph: bool = True, graph_name: str = 'forward') -> Dict[str, torch.Tensor]:
        """Forward pass with optional CUDA graph acceleration."""
        if use_graph and self.graph_manager.is_captured(graph_name):
            return self.graph_manager.replay_graph(graph_name, inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
                if isinstance(outputs, torch.Tensor):
                    return {'output': outputs}
                elif isinstance(outputs, (list, tuple)):
                    return {f'output_{i}': out for i, out in enumerate(outputs)}
                elif isinstance(outputs, dict):
                    return outputs
                else:
                    return {'output': outputs}