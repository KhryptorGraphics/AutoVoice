"""
Model downloading and loading utilities.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Any
from urllib.request import urlretrieve
from tqdm import tqdm

from .model_registry import ModelConfig

logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    """Progress bar for model downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class ModelDownloader:
    """
    Handles downloading models from remote URLs.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize downloader.

        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        url: str,
        destination: Path,
        sha256: Optional[str] = None,
        force: bool = False
    ) -> Path:
        """
        Download a file from URL to destination.

        Args:
            url: URL to download from
            destination: Local path to save file
            sha256: Optional SHA256 checksum to verify
            force: Force re-download even if file exists

        Returns:
            Path to downloaded file
        """
        destination = Path(destination)

        # Check if already downloaded
        if destination.exists() and not force:
            if sha256 and not self._verify_checksum(destination, sha256):
                logger.warning(f"Checksum mismatch for {destination}, re-downloading")
            else:
                logger.info(f"Model already downloaded: {destination}")
                return destination

        # Create parent directory
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress bar
        logger.info(f"Downloading from {url}")
        try:
            with DownloadProgressBar(
                unit='B',
                unit_scale=True,
                miniters=1,
                desc=destination.name
            ) as t:
                urlretrieve(url, destination, reporthook=t.update_to)

            logger.info(f"Downloaded to {destination}")

            # Verify checksum if provided
            if sha256 and not self._verify_checksum(destination, sha256):
                destination.unlink()
                raise ValueError(f"Checksum verification failed for {destination}")

            return destination

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if destination.exists():
                destination.unlink()
            raise

    def _verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        actual = sha256_hash.hexdigest()
        matches = actual.lower() == expected_sha256.lower()

        if not matches:
            logger.warning(f"Checksum mismatch: expected {expected_sha256}, got {actual}")

        return matches


class ModelLoader:
    """
    Loads neural models with automatic downloading.
    """

    def __init__(self, model_dir: Path):
        """
        Initialize model loader.

        Args:
            model_dir: Directory to store models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = ModelDownloader(model_dir)

    def load_hubert(self, config: ModelConfig) -> Any:
        """
        Load HuBERT model.

        Args:
            config: Model configuration

        Returns:
            Loaded HuBERT model
        """
        from .hubert_model import HuBERTModel

        # Download if needed
        if config.url and not config.local_path:
            model_path = self.model_dir / f"{config.name}_v{config.version}.pt"
            if not model_path.exists():
                self.downloader.download(
                    config.url,
                    model_path,
                    config.sha256
                )
            config.local_path = str(model_path)

        # Download config if needed
        config_path = None
        if config.config_url:
            config_path = self.model_dir / f"{config.name}_v{config.version}_config.json"
            if not config_path.exists():
                self.downloader.download(
                    config.config_url,
                    config_path
                )

        # Load model
        model = HuBERTModel(
            model_path=config.local_path,
            config_path=str(config_path) if config_path else None,
            use_mock=False
        )

        return model

    def load_hifigan(self, config: ModelConfig) -> Any:
        """
        Load HiFi-GAN model.

        Args:
            config: Model configuration

        Returns:
            Loaded HiFi-GAN model
        """
        from .hifigan_model import HiFiGANModel

        # Download if needed
        if config.url and not config.local_path:
            model_path = self.model_dir / f"{config.name}_v{config.version}.pt"
            if not model_path.exists():
                self.downloader.download(
                    config.url,
                    model_path,
                    config.sha256
                )
            config.local_path = str(model_path)

        # Download config if needed
        config_path = None
        if config.config_url:
            config_path = self.model_dir / f"{config.name}_v{config.version}_config.json"
            if not config_path.exists():
                self.downloader.download(
                    config.config_url,
                    config_path
                )

        # Load model
        model = HiFiGANModel(
            model_path=config.local_path,
            config_path=str(config_path) if config_path else None,
            use_mock=False
        )

        return model

    def load_speaker_encoder(self, config: ModelConfig) -> Any:
        """
        Load speaker encoder model.

        Args:
            config: Model configuration

        Returns:
            Loaded speaker encoder model
        """
        from .speaker_encoder import SpeakerEncoderModel

        # Download if needed
        if config.url and not config.local_path:
            model_path = self.model_dir / f"{config.name}_v{config.version}.ckpt"
            if not model_path.exists():
                self.downloader.download(
                    config.url,
                    model_path,
                    config.sha256
                )
            config.local_path = str(model_path)

        # Load model
        model = SpeakerEncoderModel(
            model_path=config.local_path,
            use_mock=False
        )

        return model
