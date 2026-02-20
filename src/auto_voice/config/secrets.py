"""Secrets management for API keys and sensitive configuration.

Task 3.5: Add secrets management (API keys, etc)

Secrets are loaded from:
1. Environment variables (AUTOVOICE_SECRET_* prefix)
2. Separate secrets file (JSON or YAML)
3. Docker secrets (/run/secrets/ directory)

Secrets are never logged in plain text.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Secret prefixes
ENV_SECRET_PREFIX = "AUTOVOICE_SECRET_"
DOCKER_SECRETS_DIR = "/run/secrets"


class SecretError(Exception):
    """Error accessing or loading secrets."""
    pass


class SecretsManager:
    """Manages sensitive configuration like API keys.

    Provides secure access to secrets with:
    - Automatic masking in logs
    - Multiple source support (env, file, Docker secrets)
    - Required secret validation
    """

    def __init__(
        self,
        secrets_file: Optional[str] = None,
        env_prefix: str = ENV_SECRET_PREFIX
    ):
        self.env_prefix = env_prefix
        self._secrets: Dict[str, str] = {}
        self._loaded_sources: list = []

        # Load from all sources
        self._load_from_env()

        if secrets_file:
            self._load_from_file(secrets_file)

        self._load_from_docker_secrets()

    def _load_from_env(self) -> None:
        """Load secrets from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                secret_name = key[len(self.env_prefix):].lower()
                self._secrets[secret_name] = value
                self._loaded_sources.append(f"env:{key}")
                logger.debug(f"Loaded secret '{secret_name}' from environment")

    def _load_from_file(self, path: str) -> None:
        """Load secrets from JSON or YAML file.

        Args:
            path: Path to secrets file
        """
        try:
            with open(path, 'r') as f:
                if path.endswith('.json'):
                    secrets = json.load(f)
                elif path.endswith(('.yaml', '.yml')):
                    secrets = yaml.safe_load(f) or {}
                else:
                    raise SecretError(f"Unsupported secrets file format: {path}")

            for key, value in secrets.items():
                if key not in self._secrets:  # Env vars take precedence
                    self._secrets[key.lower()] = str(value)

            self._loaded_sources.append(f"file:{path}")
            logger.info(f"Loaded secrets from {path}")

        except FileNotFoundError:
            logger.warning(f"Secrets file not found: {path}")
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise SecretError(f"Error parsing secrets file {path}: {e}") from e

    def _load_from_docker_secrets(self) -> None:
        """Load secrets from Docker secrets directory."""
        secrets_dir = Path(DOCKER_SECRETS_DIR)

        if not secrets_dir.exists():
            return

        for secret_file in secrets_dir.iterdir():
            if secret_file.is_file():
                secret_name = secret_file.name.lower()

                if secret_name not in self._secrets:  # Don't override existing
                    try:
                        value = secret_file.read_text().strip()
                        self._secrets[secret_name] = value
                        self._loaded_sources.append(f"docker:{secret_name}")
                        logger.debug(f"Loaded secret '{secret_name}' from Docker secrets")
                    except Exception as e:
                        logger.warning(f"Error reading Docker secret {secret_name}: {e}")

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value.

        Args:
            name: Secret name (case-insensitive)
            default: Default value if secret not found

        Returns:
            Secret value or default
        """
        return self._secrets.get(name.lower(), default)

    def get_required(self, name: str) -> str:
        """Get a required secret value.

        Args:
            name: Secret name (case-insensitive)

        Returns:
            Secret value

        Raises:
            SecretError: If secret is not found
        """
        value = self.get(name)
        if value is None:
            raise SecretError(f"Required secret '{name}' not found. "
                            f"Set via {self.env_prefix}{name.upper()} environment variable "
                            f"or add to secrets file.")
        return value

    def has(self, name: str) -> bool:
        """Check if a secret exists.

        Args:
            name: Secret name (case-insensitive)

        Returns:
            True if secret exists
        """
        return name.lower() in self._secrets

    def mask_string(self, value: str, visible_chars: int = 4) -> str:
        """Mask a string for safe logging.

        Args:
            value: String to mask
            visible_chars: Number of characters to show at end

        Returns:
            Masked string (e.g., '********abcd')
        """
        if len(value) <= visible_chars:
            return "*" * len(value)
        return "*" * (len(value) - visible_chars) + value[-visible_chars:]

    def __str__(self) -> str:
        """String representation with masked secrets."""
        masked = {k: self.mask_string(v) for k, v in self._secrets.items()}
        return f"SecretsManager(sources={self._loaded_sources}, secrets={masked})"

    def __repr__(self) -> str:
        """Representation with masked secrets."""
        return self.__str__()


def get_secret_manager() -> SecretsManager:
    """Get default secrets manager.

    Returns:
        Configured SecretsManager instance
    """
    secrets_file = os.environ.get("AUTOVOICE_SECRETS_FILE")
    return SecretsManager(secrets_file=secrets_file)
