#!/usr/bin/env python3
"""
Script to download all required models for Auto Voice Cloning.

Usage:
    python scripts/download_models.py [--model MODEL_NAME] [--force]

Options:
    --model MODEL_NAME  Download specific model only
    --force             Force re-download even if model exists
    --list              List available models
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from auto_voice.models import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_models(registry: ModelRegistry):
    """List all available models."""
    print("\nAvailable Models:")
    print("=" * 80)

    for model_name in registry.list_models():
        config = registry.get_config(model_name)
        downloaded = "✓" if registry.is_model_downloaded(model_name) else "✗"

        print(f"\n[{downloaded}] {model_name} (v{config.version})")
        print(f"    Type: {config.model_type.value}")
        print(f"    Description: {config.metadata.get('description', 'N/A')}")
        print(f"    GPU Required: {'Yes' if config.requires_gpu else 'No'}")
        print(f"    Min Memory: {config.min_memory_gb} GB")

        if config.metadata.get('model_size_mb'):
            print(f"    Size: ~{config.metadata['model_size_mb']} MB")

    print("\n" + "=" * 80)


def download_model(registry: ModelRegistry, model_name: str, force: bool = False):
    """Download a specific model."""
    config = registry.get_config(model_name)

    if registry.is_model_downloaded(model_name) and not force:
        logger.info(f"Model {model_name} already downloaded")
        return

    logger.info(f"Downloading model: {model_name}")
    logger.info(f"Type: {config.model_type.value}")
    logger.info(f"Version: {config.version}")

    try:
        # Load the model (this triggers download)
        if config.model_type.value == 'hubert':
            registry.load_hubert(model_name)
        elif config.model_type.value == 'hifigan':
            registry.load_hifigan(model_name)
        elif config.model_type.value == 'speaker_encoder':
            registry.load_speaker_encoder(model_name)

        logger.info(f"Successfully downloaded {model_name}")

    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        sys.exit(1)


def download_all_models(registry: ModelRegistry, force: bool = False):
    """Download all configured models."""
    models = registry.list_models()

    logger.info(f"Downloading {len(models)} models...")

    for i, model_name in enumerate(models, 1):
        logger.info(f"\n[{i}/{len(models)}] Processing {model_name}")
        download_model(registry, model_name, force)

    logger.info("\nAll models downloaded successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Download models for Auto Voice Cloning'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Download specific model only'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if model exists'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/',
        help='Directory to store models (default: models/)'
    )

    args = parser.parse_args()

    # Initialize registry
    registry = ModelRegistry(model_dir=args.model_dir, use_mock=False)

    # List models
    if args.list:
        list_models(registry)
        return

    # Download specific model
    if args.model:
        if args.model not in registry.list_models():
            logger.error(f"Unknown model: {args.model}")
            logger.info("Available models:")
            for name in registry.list_models():
                logger.info(f"  - {name}")
            sys.exit(1)

        download_model(registry, args.model, args.force)
    else:
        # Download all models
        download_all_models(registry, args.force)


if __name__ == '__main__':
    main()
