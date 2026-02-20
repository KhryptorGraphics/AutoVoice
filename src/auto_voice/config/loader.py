"""Configuration loader supporting YAML files and environment variables.

Task 3.4: Support config from YAML or env vars
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .validator import ConfigValidator

logger = logging.getLogger(__name__)

# Environment variable prefix
ENV_PREFIX = "AUTOVOICE_"


class ConfigLoader:
    """Loads configuration from multiple sources with priority:

    1. Default values (lowest priority)
    2. YAML configuration file
    3. Environment variables (highest priority)

    Environment variable format:
    - AUTOVOICE_SERVER_PORT=8080 -> config["server"]["port"]
    - AUTOVOICE_DEBUG=true -> config["debug"]
    - Nested keys use double underscore: AUTOVOICE_GPU__DEVICE=cuda:1
    """

    def __init__(self, env_prefix: str = ENV_PREFIX):
        self.env_prefix = env_prefix
        self.validator = ConfigValidator()

    def load(self, defaults: Optional[Dict] = None) -> Dict[str, Any]:
        """Load configuration from all sources.

        Args:
            defaults: Default configuration values

        Returns:
            Merged configuration dictionary
        """
        config = defaults or {}

        # Load from environment variables (highest priority for overrides)
        env_config = self._load_from_env()
        config = self._merge_configs(config, env_config)

        return config

    def load_from_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise ConfigLoadError(f"Invalid YAML in {path}: {e}") from e

    def load_with_merge(
        self,
        defaults: Dict,
        yaml_config: Optional[Dict] = None,
        env_override: bool = True
    ) -> Dict[str, Any]:
        """Load configuration merging multiple sources.

        Args:
            defaults: Default configuration values
            yaml_config: Configuration loaded from YAML file
            env_override: Whether to apply environment variable overrides

        Returns:
            Merged configuration with priority: defaults < yaml < env
        """
        config = defaults.copy()

        # Merge YAML config (overrides defaults)
        if yaml_config:
            config = self._deep_merge(config, yaml_config)

        # Merge environment variables (highest priority)
        if env_override:
            env_config = self._load_from_env()
            config = self._deep_merge(config, env_config)

        # Apply defaults for any missing fields
        config = self.validator.apply_defaults(config)

        return config

    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary parsed from env vars
        """
        config = {}

        for key, value in os.environ.items():
            if not key.startswith(self.env_prefix):
                continue

            # Remove prefix and convert to lowercase
            config_key = key[len(self.env_prefix):].lower()

            # Handle nested keys (double underscore separator)
            if "__" in config_key:
                parts = config_key.split("__")
                self._set_nested_value(config, parts, self._convert_type(value))
            else:
                config[config_key] = self._convert_type(value)

        if config:
            logger.debug(f"Loaded {len(config)} values from environment variables")

        return config

    def _set_nested_value(self, config: Dict, keys: list, value: Any) -> None:
        """Set a value in nested dictionary structure.

        Args:
            config: Configuration dictionary to modify
            keys: List of nested keys
            value: Value to set
        """
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def _convert_type(self, value: str) -> Any:
        """Convert string value to appropriate type.

        Args:
            value: String value from environment variable

        Returns:
            Converted value (bool, int, float, or string)
        """
        # Boolean conversion
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False

        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _merge_configs(self, base: Dict, override: Dict) -> Dict[str, Any]:
        """Merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Configuration to merge (takes precedence)

        Returns:
            Merged configuration
        """
        return self._deep_merge(base.copy(), override)

    def _deep_merge(self, base: Dict, override: Dict) -> Dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary (modified in place)
            override: Dictionary to merge into base

        Returns:
            Merged dictionary
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base


class ConfigLoadError(Exception):
    """Error loading configuration."""
    pass
