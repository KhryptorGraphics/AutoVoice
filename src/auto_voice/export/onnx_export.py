"""ONNX export functions for AutoVoice inference models.

Exports ContentEncoder projection, SoVitsSvc inference path, and
BigVGANGenerator with dynamic batch and sequence axes.
"""

import copy
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

OPSET_VERSION = 17


class _SoVitsSvcInference(nn.Module):
    """Wrapper exposing only the SoVitsSvc inference path for ONNX tracing.

    The full SoVitsSvc.forward() uses a `reverse` bool in FlowDecoder and
    optional spec input. ONNX tracing requires a fixed execution path, so
    this wrapper hard-codes the inference branch (spec=None, reverse=True).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.content_proj = model.content_proj
        self.pitch_proj = model.pitch_proj
        self.speaker_proj = model.speaker_proj
        self.flow = model.flow
        self.mel_decoder = model.mel_decoder

    def forward(self, content: torch.Tensor, pitch: torch.Tensor,
                speaker: torch.Tensor) -> torch.Tensor:
        """Inference-only forward: content+pitch+speaker -> mel.

        Args:
            content: [B, T, content_dim]
            pitch: [B, T, pitch_dim]
            speaker: [B, speaker_dim]

        Returns:
            mel_pred: [B, n_mels, T]
        """
        c = self.content_proj(content).transpose(1, 2)  # [B, hidden, T]
        p = self.pitch_proj(pitch).transpose(1, 2)      # [B, hidden, T]
        s = self.speaker_proj(speaker).unsqueeze(-1)     # [B, hidden, 1]

        h = c + p + s

        # Trace only the reverse flow path
        z = self._flow_reverse(h)
        mel_pred = self.mel_decoder(z)
        return mel_pred

    def _flow_reverse(self, x: torch.Tensor) -> torch.Tensor:
        """Run flow in reverse (inference direction)."""
        for flow in reversed(list(self.flow.flows)):
            x = flow(x, reverse=True)
        return x


def _set_eval_mode(model: nn.Module) -> nn.Module:
    """Set model to inference mode."""
    model.train(False)
    return model


def export_content_encoder(model: nn.Module, path: str,
                           seq_len: int = 100) -> str:
    """Export ContentEncoder projection to ONNX.

    Exports the projection sub-module (Linear or Conformer) that maps
    backend features [B, T, 256] to content space [B, T, output_size].
    The HuBERT/ContentVec feature extractors are NOT included (they
    should be exported separately or run as pre-processing).

    Args:
        model: ContentEncoder instance.
        path: Output ONNX file path.
        seq_len: Example sequence length for tracing.

    Returns:
        The output path string.
    """
    path = str(path)
    model = _set_eval_mode(copy.deepcopy(model).cpu())
    projection = model.projection

    dummy_input = torch.randn(1, seq_len, 256)

    torch.onnx.export(
        projection,
        dummy_input,
        path,
        opset_version=OPSET_VERSION,
        dynamo=False,
        input_names=['features'],
        output_names=['content'],
        dynamic_axes={
            'features': {0: 'batch', 1: 'sequence'},
            'content': {0: 'batch', 1: 'sequence'},
        },
    )
    logger.info(f"Exported ContentEncoder projection to {path}")
    return path


def export_sovits(model: nn.Module, path: str,
                  seq_len: int = 100) -> str:
    """Export SoVitsSvc inference path to ONNX.

    Wraps the model to trace only the inference branch (no posterior
    encoder, no training-time reparameterization). Flow runs in reverse.

    Args:
        model: SoVitsSvc instance.
        path: Output ONNX file path.
        seq_len: Example sequence length for tracing.

    Returns:
        The output path string.
    """
    path = str(path)
    model_copy = _set_eval_mode(copy.deepcopy(model).cpu())
    wrapper = _SoVitsSvcInference(model_copy)
    _set_eval_mode(wrapper)

    content_dim = model_copy.content_dim
    pitch_dim = model_copy.pitch_dim
    speaker_dim = model_copy.speaker_dim

    dummy_content = torch.randn(1, seq_len, content_dim)
    dummy_pitch = torch.randn(1, seq_len, pitch_dim)
    dummy_speaker = torch.randn(1, speaker_dim)

    torch.onnx.export(
        wrapper,
        (dummy_content, dummy_pitch, dummy_speaker),
        path,
        opset_version=OPSET_VERSION,
        dynamo=False,
        input_names=['content', 'pitch', 'speaker'],
        output_names=['mel_pred'],
        dynamic_axes={
            'content': {0: 'batch', 1: 'sequence'},
            'pitch': {0: 'batch', 1: 'sequence'},
            'speaker': {0: 'batch'},
            'mel_pred': {0: 'batch', 2: 'time'},
        },
    )
    logger.info(f"Exported SoVitsSvc inference to {path}")
    return path


def export_bigvgan(model: nn.Module, path: str,
                   mel_len: int = 100) -> str:
    """Export BigVGANGenerator to ONNX.

    Removes weight normalization parametrizations before export, since
    they introduce extra graph nodes that complicate ONNX optimization.

    Args:
        model: BigVGANGenerator instance.
        path: Output ONNX file path.
        mel_len: Example mel-spectrogram time dimension for tracing.

    Returns:
        The output path string.
    """
    path = str(path)
    model_copy = _set_eval_mode(copy.deepcopy(model).cpu())

    # Get num_mels before any weight_norm removal (ParametrizedConv1d
    # may lose .weight attribute after remove_parametrizations in PyTorch 2.11+)
    num_mels = model_copy.conv_pre.weight.shape[1]

    # Try to remove weight normalization for a cleaner ONNX graph.
    # If it fails (PyTorch parametrizations API issue), export with
    # weight_norm intact — functionally identical, slightly larger graph.
    try:
        model_copy.remove_weight_norm()
        # Verify the model still works after removal
        test_input = torch.randn(1, num_mels, 10)
        with torch.no_grad():
            model_copy(test_input)
    except (AttributeError, RuntimeError):
        logger.warning("weight_norm removal failed, exporting with parametrizations intact")
        model_copy = _set_eval_mode(copy.deepcopy(model).cpu())

    dummy_mel = torch.randn(1, num_mels, mel_len)

    torch.onnx.export(
        model_copy,
        dummy_mel,
        path,
        opset_version=OPSET_VERSION,
        dynamo=False,
        input_names=['mel'],
        output_names=['audio'],
        dynamic_axes={
            'mel': {0: 'batch', 2: 'time'},
            'audio': {0: 'batch', 2: 'time'},
        },
    )
    logger.info(f"Exported BigVGANGenerator to {path}")
    return path
