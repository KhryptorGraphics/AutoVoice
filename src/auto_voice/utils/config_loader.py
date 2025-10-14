"""Configuration loader module for AutoVoice."""

import os
import json
import copy
from typing import Optional, Dict, Any
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    'audio': {
        'sample_rate': 22050,
        'channels': 1,
        'format': 'float32',
        'chunk_size': 1024,
        'buffer_size': 4096,
        'n_fft': 2048,
        'hop_length': 512,
        'win_length': 2048,
        'n_mels': 128,
        'fmin': 0,
        'fmax': 8000
    },
    'model': {
        'name': 'default',
        'checkpoint_path': None,
        'device': 'cuda',
        'dtype': 'float32',
        'max_length': 1000,
        'architecture': 'transformer',
        'hidden_size': 512,
        'num_layers': 12,
        'num_heads': 8,
        'dropout': 0.1,
        'max_sequence_length': 2048
    },
    'gpu': {
        'enabled': True,
        'device': 'cuda',
        'device_id': 0,
        'memory_fraction': 0.9,
        'allow_growth': True,
        'mixed_precision': True
    },
    'web': {
        'host': '0.0.0.0',
        'port': 5000,
        'debug': False,
        'cors_origins': '*',
        'max_content_length': 16 * 1024 * 1024,  # 16MB
        'workers': 4
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': None
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 100,
        'validation_split': 0.2,
        'checkpoint_interval': 10,
        'num_epochs': 100,
        'warmup_steps': 1000,
        'gradient_clip': 1.0,
        'save_interval': 1000,
        'validation_interval': 500
    },
    'inference': {
        'batch_size': 1,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'max_length': 1000,
        'repetition_penalty': 1.2
    },
    'server': {
        'host': '0.0.0.0',
        'port': 5000,
        'debug': False,
        'workers': 4
    }
}


def load_config_with_defaults() -> Dict[str, Any]:
    """Load configuration with default values.

    Returns:
        Dict containing default configuration
    """
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config_from_file(path: str, strict: bool = False) -> Dict[str, Any]:
    """Load configuration from a file.

    Args:
        path: Path to configuration file (JSON or Python)
        strict: If True, raise error if file doesn't exist

    Returns:
        Dict containing loaded configuration

    Raises:
        FileNotFoundError: If strict=True and file doesn't exist
        ValueError: If file format is invalid
    """
    config_path = Path(path)

    if not config_path.exists():
        if strict:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        return {}

    # Handle JSON files
    if config_path.suffix.lower() == '.json':
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {path}: {e}")

    # Handle Python files
    elif config_path.suffix.lower() == '.py':
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec and spec.loader:
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)

            # Extract config dict from module
            if hasattr(config_module, 'CONFIG'):
                return config_module.CONFIG
            elif hasattr(config_module, 'config'):
                return config_module.config
            else:
                # Try to extract all uppercase attributes
                config = {}
                for key in dir(config_module):
                    if key.isupper() and not key.startswith('_'):
                        config[key.lower()] = getattr(config_module, key)
                return config

    # Handle YAML files if PyYAML is available
    elif config_path.suffix.lower() in ['.yml', '.yaml']:
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ValueError(f"YAML support not available. Install PyYAML to load {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {path}: {e}")

    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge two configuration dictionaries.

    Args:
        base: Base configuration dict
        override: Override configuration dict

    Returns:
        Merged configuration dict (modifies base in-place)
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_configs(base[key], value)
        else:
            base[key] = value
    return base


def load_config_from_env(config: dict, prefix: str = 'AUTOVOICE_') -> dict:
    """Load configuration overrides from environment variables.

    Environment variable format:
    - AUTOVOICE_SECTION__KEY for nested values (double underscore)
    - AUTOVOICE_SECTION_KEY for nested values (single underscore - legacy support)
    - JSON strings are parsed for complex types
    - Example: AUTOVOICE_AUDIO__SAMPLE_RATE=44100 or AUTOVOICE_WEB_PORT=8080

    Args:
        config: Configuration dict to update
        prefix: Environment variable prefix

    Returns:
        Updated configuration dict

    Raises:
        ValueError: If environment variable format is invalid
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")

    try:
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Remove prefix and convert to lowercase
            key_path = env_key[len(prefix):].lower()

            # Handle both single and double underscore formats
            if '__' in key_path:
                # Double underscore format (preferred)
                keys = key_path.split('__')
            elif '_' in key_path:
                # Single underscore format (legacy support)
                keys = key_path.split('_', 1)  # Split into at most 2 parts
            else:
                # No underscore - top-level config
                keys = [key_path]

            # Navigate to the target location in config
            current = config
            for key in keys[:-1]:
                if not key:  # Skip empty keys
                    continue
                if key not in current:
                    current[key] = {}
                elif not isinstance(current[key], dict):
                    # Can't navigate further into non-dict
                    break
                current = current[key]

            # Set the value with type inference
            final_key = keys[-1]
            if not final_key:  # Skip empty final key
                continue

            # Try to parse as JSON first (handles arrays, objects, booleans, numbers)
            try:
                parsed_value = json.loads(env_value)
                current[final_key] = parsed_value
            except (json.JSONDecodeError, ValueError):
                # Check for special string values
                if env_value.lower() == 'true':
                    current[final_key] = True
                elif env_value.lower() == 'false':
                    current[final_key] = False
                elif env_value.lower() == 'none' or env_value.lower() == 'null':
                    current[final_key] = None
                else:
                    # Try to parse as number
                    try:
                        if '.' in env_value:
                            current[final_key] = float(env_value)
                        else:
                            current[final_key] = int(env_value)
                    except ValueError:
                        # Keep as string
                        current[final_key] = env_value

    except Exception as e:
        raise ValueError(f"Error processing environment variables: {e}")

    return config


def validate_config(config: dict) -> None:
    """Validate configuration structure and values.

    Args:
        config: Configuration dict to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")

    if not config:
        raise ValueError("Configuration cannot be empty")

    # Check required sections exist
    required_sections = ['audio', 'model', 'gpu', 'web', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
        if not isinstance(config[section], dict):
            raise ValueError(f"Config section '{section}' must be a dictionary")

    # Validate audio settings
    if 'audio' in config:
        audio = config['audio']
        if 'sample_rate' in audio:
            sr = audio['sample_rate']
            if not isinstance(sr, int) or sr <= 0:
                raise ValueError(f"Invalid sample_rate: {sr}")
        if 'channels' in audio:
            ch = audio['channels']
            if not isinstance(ch, int) or ch not in [1, 2]:
                raise ValueError(f"Invalid channels: {ch}. Must be 1 or 2")

    # Validate model settings
    if 'model' in config:
        model = config['model']
        if 'device' in model:
            device = model['device']
            if device not in ['cpu', 'cuda', 'mps', 'auto']:
                raise ValueError(f"Invalid device: {device}")

    # Validate web settings
    if 'web' in config:
        web = config['web']
        if 'port' in web:
            port = web['port']
            if not isinstance(port, int) or port < 0 or port > 65535:
                raise ValueError(f"Invalid port: {port}")
        if 'host' in web:
            host = web['host']
            if not isinstance(host, str):
                raise ValueError(f"Invalid host: {host}")

    # Validate GPU settings
    if 'gpu' in config:
        gpu = config['gpu']
        if 'device_id' in gpu:
            device_id = gpu['device_id']
            if not isinstance(device_id, int) or device_id < 0:
                raise ValueError(f"Invalid GPU device_id: {device_id}")
        if 'memory_fraction' in gpu:
            frac = gpu['memory_fraction']
            if not isinstance(frac, (int, float)) or frac <= 0 or frac > 1:
                raise ValueError(f"Invalid GPU memory_fraction: {frac}")

    # Validate logging settings
    if 'logging' in config:
        logging = config['logging']
        if 'level' in logging:
            level = logging['level']
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if level.upper() not in valid_levels:
                raise ValueError(f"Invalid logging level: {level}")


def load_config(config_path: Optional[str] = None, use_defaults: bool = True) -> dict:
    """Load configuration from file, environment, and defaults.

    Loading order:
    1. Start with default configuration (if use_defaults=True)
    2. Merge configuration from file (if config_path provided)
    3. Apply environment variable overrides
    4. Validate final configuration

    Args:
        config_path: Optional path to configuration file
        use_defaults: Whether to use default configuration as base

    Returns:
        Final merged and validated configuration dict

    Raises:
        ValueError: If configuration is invalid
    """
    # Start with defaults if requested
    if use_defaults:
        config = load_config_with_defaults()
    else:
        config = {}

    # Load and merge config file if provided
    if config_path and os.path.exists(config_path):
        file_config = load_config_from_file(config_path, strict=False)
        config = merge_configs(config, file_config)

    # Apply environment variable overrides
    config = load_config_from_env(config)

    # Validate final configuration
    validate_config(config)

    return config


# Keep backward compatibility
def override_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """Override configuration with environment variables (backward compatibility)"""
    return load_config_from_env(config)


# Export all functions
__all__ = [
    'load_config',
    'load_config_with_defaults',
    'load_config_from_file',
    'merge_configs',
    'load_config_from_env',
    'validate_config',
    'override_with_env',
    'DEFAULT_CONFIG'
]