"""Model loading utilities for pre-trained weights.

Provides functions to load pre-trained models from various checkpoint formats:
- So-VITS-SVC (.pth files)
- HiFi-GAN (.ckpt files)
- HuBERT-Soft (.pt files)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


def load_sovits_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda',
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load So-VITS-SVC checkpoint from .pth file.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load onto
        config: Optional config override
        
    Returns:
        Dictionary with 'model_state_dict' and 'config'
    """
    try:
        logger.info(f"Loading So-VITS-SVC checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract state dict and config
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            ckpt_config = checkpoint.get('config', {})
        else:
            # Direct state dict
            state_dict = checkpoint
            ckpt_config = {}
        
        # Merge with provided config
        final_config = {**ckpt_config, **(config or {})}
        
        logger.info(f"Loaded checkpoint with {len(state_dict)} parameters")
        
        return {
            'model_state_dict': state_dict,
            'config': final_config
        }
        
    except Exception as e:
        logger.error(f"Failed to load So-VITS checkpoint: {e}")
        raise


def load_hifigan_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """Load HiFi-GAN checkpoint from .ckpt file.
    
    Args:
        checkpoint_path: Path to .ckpt checkpoint file
        device: Device to load onto
        
    Returns:
        State dictionary
    """
    try:
        logger.info(f"Loading HiFi-GAN checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract generator state dict
        if isinstance(checkpoint, dict):
            if 'generator' in checkpoint:
                state_dict = checkpoint['generator']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        logger.info(f"Loaded HiFi-GAN checkpoint with {len(state_dict)} parameters")
        
        return state_dict
        
    except Exception as e:
        logger.error(f"Failed to load HiFi-GAN checkpoint: {e}")
        raise


def load_hubert_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """Load HuBERT-Soft checkpoint from .pt file.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load onto
        
    Returns:
        State dictionary
    """
    try:
        logger.info(f"Loading HuBERT-Soft checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # HuBERT checkpoints can have various formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        logger.info(f"Loaded HuBERT checkpoint with {len(state_dict)} parameters")
        
        return state_dict
        
    except Exception as e:
        logger.error(f"Failed to load HuBERT checkpoint: {e}")
        raise


def load_pretrained_config(
    config_path: str = 'config/pretrained_models.yaml'
) -> Dict[str, Any]:
    """Load pre-trained model configuration from YAML.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dictionary with expanded paths
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Expand ${model_dir} variables
        model_dir = config.get('model_dir', 'models/pretrained')
        
        def expand_paths(obj):
            if isinstance(obj, str):
                return obj.replace('${model_dir}', model_dir)
            elif isinstance(obj, dict):
                return {k: expand_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [expand_paths(item) for item in obj]
            return obj
        
        expanded_config = expand_paths(config)
        
        logger.info(f"Loaded pre-trained model config from {config_path}")
        
        return expanded_config
        
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


def get_default_sovits_config() -> Dict[str, Any]:
    """Get default So-VITS-SVC configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'latent_dim': 192,
        'mel_channels': 80,
        'content_dim': 256,
        'pitch_dim': 192,
        'speaker_dim': 256,
        'hidden_channels': 192,
        'num_flows': 4,
        'use_only_mean': False,
        'sample_rate': 22050,
        'hop_length': 512,
        'n_fft': 2048,
        'win_length': 2048
    }
