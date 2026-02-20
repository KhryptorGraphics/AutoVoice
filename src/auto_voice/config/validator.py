"""Configuration validation using Pydantic-style validation.

Task 3.3: Add config validation on startup
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConfigValidator:
    """Validates AutoVoice configuration on startup."""

    # Schema definition for validation
    SCHEMA = {
        "server": {
            "required": False,
            "fields": {
                "host": {"type": str, "default": "0.0.0.0"},
                "port": {"type": int, "min": 1, "max": 65535, "default": 5000},
                "workers": {"type": int, "min": 1, "max": 32, "default": 1},
                "debug": {"type": bool, "default": False},
            }
        },
        "gpu": {
            "required": False,
            "fields": {
                "device": {"type": str, "default": "cuda:0"},
                "compute_capability": {"type": str, "default": "11.0"},
                "arch": {"type": str, "default": "sm_110"},
                "max_memory_fraction": {"type": float, "min": 0.0, "max": 1.0, "default": 0.9},
                "enable_cuda_kernels": {"type": bool, "default": True},
            }
        },
        "audio": {
            "required": False,
            "fields": {
                "sample_rate": {"type": int, "min": 8000, "max": 96000, "default": 22050},
                "hop_length": {"type": int, "min": 64, "max": 2048, "default": 512},
                "n_fft": {"type": int, "min": 256, "max": 8192, "default": 2048},
                "win_length": {"type": int, "min": 256, "max": 8192, "default": 2048},
                "n_mels": {"type": int, "min": 20, "max": 256, "default": 80},
            }
        },
        "models": {
            "required": False,
            "fields": {
                "pretrained_dir": {"type": str, "default": "models/pretrained"},
                "sovits_checkpoint": {"type": str, "default": "sovits5.0_main_1500.pth"},
                "hifigan_checkpoint": {"type": str, "default": "hifigan_ljspeech.ckpt"},
                "hubert_checkpoint": {"type": str, "default": "hubert-soft-0d54a1f4.pt"},
            }
        },
        "job_manager": {
            "required": False,
            "fields": {
                "enabled": {"type": bool, "default": True},
                "max_workers": {"type": int, "min": 1, "max": 16, "default": 4},
                "ttl_seconds": {"type": int, "min": 60, "max": 86400, "default": 3600},
                "in_progress_ttl_seconds": {"type": int, "min": 60, "max": 172800, "default": 7200},
            }
        },
        "voice_conversion": {
            "required": False,
            "fields": {
                "singing_conversion_enabled": {"type": bool, "default": True},
                "voice_cloning_enabled": {"type": bool, "default": True},
            }
        },
    }

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration against schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            ValidationResult with is_valid flag and any errors/warnings
        """
        errors = []
        warnings = []

        # Validate each section
        for section_name, section_schema in self.SCHEMA.items():
            if section_name not in config:
                if section_schema.get("required", False):
                    errors.append(f"Required section '{section_name}' is missing")
                continue

            section_data = config[section_name]
            if not isinstance(section_data, dict):
                errors.append(f"Section '{section_name}' must be a dictionary")
                continue

            # Validate fields within section
            for field_name, field_spec in section_schema.get("fields", {}).items():
                if field_name not in section_data:
                    if field_spec.get("required", False):
                        errors.append(f"Required field '{section_name}.{field_name}' is missing")
                    continue

                value = section_data[field_name]
                field_errors = self._validate_field(
                    f"{section_name}.{field_name}",
                    value,
                    field_spec
                )
                errors.extend(field_errors)

        # Check for unknown sections
        for section_name in config:
            if section_name not in self.SCHEMA:
                warnings.append(f"Unknown configuration section: '{section_name}'")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("Configuration validation passed")
        else:
            logger.error(f"Configuration validation failed with {len(errors)} errors")
            for error in errors:
                logger.error(f"  - {error}")

        for warning in warnings:
            logger.warning(f"  - {warning}")

        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def _validate_field(self, field_path: str, value: Any, spec: Dict) -> List[str]:
        """Validate a single field against its specification.

        Args:
            field_path: Dot-notation path to field (e.g., "server.port")
            value: Value to validate
            spec: Field specification dict

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        expected_type = spec.get("type")

        # Type validation
        if expected_type and not isinstance(value, expected_type):
            # Allow int for float fields
            if expected_type == float and isinstance(value, int):
                pass  # Valid
            else:
                errors.append(
                    f"Field '{field_path}' must be of type {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
                return errors

        # Range validation for numbers
        if isinstance(value, (int, float)):
            min_val = spec.get("min")
            max_val = spec.get("max")

            if min_val is not None and value < min_val:
                errors.append(
                    f"Field '{field_path}' value {value} is below minimum {min_val}"
                )

            if max_val is not None and value > max_val:
                errors.append(
                    f"Field '{field_path}' value {value} exceeds maximum {max_val}"
                )

        return errors

    def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to missing configuration fields.

        Args:
            config: Configuration dictionary (may be incomplete)

        Returns:
            Configuration with defaults applied
        """
        result = {}

        for section_name, section_schema in self.SCHEMA.items():
            section_data = config.get(section_name, {})
            result_section = {}

            # Apply field defaults
            for field_name, field_spec in section_schema.get("fields", {}).items():
                if field_name in section_data:
                    result_section[field_name] = section_data[field_name]
                elif "default" in field_spec:
                    result_section[field_name] = field_spec["default"]

            result[section_name] = result_section

        # Preserve any unknown sections as-is
        for key in config:
            if key not in self.SCHEMA:
                result[key] = config[key]

        return result
