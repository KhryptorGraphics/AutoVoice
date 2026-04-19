"""Tests for configuration validation on startup.

Task 3.3: Add config validation on startup
Task 3.4: Support config from YAML or env vars
Task 3.5: Add secrets management (API keys, etc)
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

import yaml


class TestConfigValidation:
    """Test configuration validation on startup."""

    def test_valid_config_loads(self):
        """Verify valid config loads without errors."""
        from auto_voice.config.validator import ConfigValidator

        valid_config = {
            "server": {
                "host": "0.0.0.0",
                "port": 5000,
                "workers": 1,
                "debug": False
            },
            "gpu": {
                "device": "cuda:0",
                "compute_capability": "11.0",
                "arch": "sm_110",
                "max_memory_fraction": 0.9,
                "enable_cuda_kernels": True
            }
        }

        validator = ConfigValidator()
        result = validator.validate(valid_config)

        assert result.is_valid, f"Config should be valid: {result.errors}"
        assert len(result.errors) == 0

    def test_invalid_port_fails_validation(self):
        """Verify invalid port number is rejected."""
        from auto_voice.config.validator import ConfigValidator

        invalid_config = {
            "server": {"port": 99999}  # Invalid port
        }

        validator = ConfigValidator()
        result = validator.validate(invalid_config)

        assert not result.is_valid
        assert any("port" in str(e).lower() for e in result.errors)

    def test_missing_optional_field_is_allowed(self):
        """Validator should allow omitted fields that have defaults."""
        from auto_voice.config.validator import ConfigValidator

        incomplete_config = {
            "server": {"host": "0.0.0.0"}
            # Missing port; validator applies defaults later during merge.
        }

        validator = ConfigValidator()
        result = validator.validate(incomplete_config)

        assert result.is_valid, result.errors

    def test_gpu_memory_fraction_range(self):
        """Verify GPU memory fraction must be 0-1."""
        from auto_voice.config.validator import ConfigValidator

        invalid_config = {
            "gpu": {"max_memory_fraction": 1.5}  # > 1 is invalid
        }

        validator = ConfigValidator()
        result = validator.validate(invalid_config)

        assert not result.is_valid


class TestEnvVarConfig:
    """Test configuration from environment variables."""

    def test_env_var_overrides_yaml(self):
        """Verify environment variables override YAML config."""
        from auto_voice.config.loader import ConfigLoader

        # Set env var
        with patch.dict(os.environ, {"AUTOVOICE_SERVER_PORT": "8080"}):
            loader = ConfigLoader()
            config = loader.load({"server": {"port": 5000}})

            assert config["server"]["port"] == 8080

    def test_env_var_prefix_filtering(self):
        """Verify only AUTOVOICE_ prefixed vars are loaded."""
        from auto_voice.config.loader import ConfigLoader

        with patch.dict(os.environ, {
            "AUTOVOICE_DEBUG": "true",
            "OTHER_VAR": "ignored"
        }):
            loader = ConfigLoader()
            env_config = loader._load_from_env()

            assert "debug" in env_config
            assert "other_var" not in env_config

    def test_nested_env_var_parsing(self):
        """Verify double underscore creates nested keys."""
        from auto_voice.config.loader import ConfigLoader

        with patch.dict(os.environ, {"AUTOVOICE_GPU__DEVICE": "cuda:1"}):
            loader = ConfigLoader()
            env_config = loader._load_from_env()

            assert env_config["gpu"]["device"] == "cuda:1"

    def test_type_conversion_from_env(self):
        """Verify types are converted from env strings."""
        from auto_voice.config.loader import ConfigLoader

        with patch.dict(os.environ, {
            "AUTOVOICE_SERVER_PORT": "8080",
            "AUTOVOICE_DEBUG": "true"
        }):
            loader = ConfigLoader()
            config = loader.load({})

            assert isinstance(config["server"]["port"], int)
            assert isinstance(config["debug"], bool)


class TestSecretsManagement:
    """Test secrets management for API keys."""

    def test_secret_loaded_from_file(self):
        """Verify secrets can be loaded from separate file."""
        from auto_voice.config.secrets import SecretsManager

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump({"api_key": "secret123"}, f)
            secret_file = f.name

        try:
            manager = SecretsManager(secrets_file=secret_file)
            assert manager.get("api_key") == "secret123"
        finally:
            os.unlink(secret_file)

    def test_secret_loaded_from_env(self):
        """Verify secrets can be loaded from environment."""
        from auto_voice.config.secrets import SecretsManager

        with patch.dict(os.environ, {"AUTOVOICE_SECRET_API_KEY": "env_secret"}):
            manager = SecretsManager()
            assert manager.get("api_key") == "env_secret"

    def test_secret_not_logged(self):
        """Verify secrets are masked in logs."""
        from auto_voice.config.secrets import SecretsManager

        manager = SecretsManager()
        manager._secrets = {"api_key": "super_secret"}

        str_repr = str(manager)
        assert "super_secret" not in str_repr
        assert "***" in str_repr or "redacted" in str_repr.lower()

    def test_required_secret_raises_if_missing(self):
        """Verify required secrets raise error if not found."""
        from auto_voice.config.secrets import SecretsManager, SecretError

        manager = SecretsManager()

        with pytest.raises(SecretError):
            manager.get_required("nonexistent_secret")


class TestConfigLoaderIntegration:
    """Test full config loading with all sources."""

    def test_load_from_yaml_file(self, tmp_path):
        """Verify config loads from YAML file."""
        from auto_voice.config.loader import ConfigLoader

        config_file = tmp_path / "config.yaml"
        config_data = {"server": {"port": 6000}}
        config_file.write_text(yaml.safe_dump(config_data))

        loader = ConfigLoader()
        config = loader.load_from_file(config_file)

        assert config["server"]["port"] == 6000

    def test_config_merge_priority(self):
        """Verify config merge priority: defaults < YAML < env."""
        from auto_voice.config.loader import ConfigLoader

        defaults = {"server": {"port": 5000, "host": "127.0.0.1"}}
        yaml_config = {"server": {"port": 6000}}

        with patch.dict(os.environ, {"AUTOVOICE_SERVER_HOST": "0.0.0.0"}):
            loader = ConfigLoader()
            config = loader.load_with_merge(defaults, yaml_config)

            # YAML overrides defaults
            assert config["server"]["port"] == 6000
            # Env overrides YAML
            assert config["server"]["host"] == "0.0.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
