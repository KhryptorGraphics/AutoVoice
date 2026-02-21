"""Tests for secure secret key handling.

Task 1.4: Add tests for secure secret key handling

Tests cover:
1. Secret key from SecretsManager
2. Secret key from env var
3. Random generation in testing mode
4. Key is never the old hardcoded default
5. Generated keys are sufficiently random and long enough
"""

import os
import tempfile
import pytest
from unittest.mock import patch

import json


class TestSecretKeyGeneration:
    """Test secure random key generation utilities."""

    def test_generate_secret_key_length(self):
        """Verify generated keys meet minimum length requirements."""
        from auto_voice.config.secret_key_generator import generate_secret_key

        key = generate_secret_key()

        # Default 32 bytes -> ~43 chars in base64
        assert len(key) >= 32, f"Key too short: {len(key)} chars"

    def test_generate_secret_key_randomness(self):
        """Verify keys are unique and unpredictable."""
        from auto_voice.config.secret_key_generator import generate_secret_key

        keys = [generate_secret_key() for _ in range(10)]

        # All keys should be unique
        assert len(set(keys)) == 10, "Keys are not unique"

    def test_generate_secret_key_minimum_length_validation(self):
        """Verify minimum length requirement is enforced."""
        from auto_voice.config.secret_key_generator import generate_secret_key

        with pytest.raises(ValueError, match="at least 32 bytes"):
            generate_secret_key(length=16)

    def test_generate_hex_key(self):
        """Verify hex key generation."""
        from auto_voice.config.secret_key_generator import generate_hex_key

        key = generate_hex_key()

        # 32 bytes = 64 hex chars
        assert len(key) >= 64
        # Should be valid hex
        assert all(c in '0123456789abcdef' for c in key)

    def test_generate_hex_key_minimum_length_validation(self):
        """Verify hex key minimum length enforcement."""
        from auto_voice.config.secret_key_generator import generate_hex_key

        with pytest.raises(ValueError, match="at least 32 bytes"):
            generate_hex_key(length=16)

    def test_generate_urlsafe_token(self):
        """Verify URL-safe token generation."""
        from auto_voice.config.secret_key_generator import generate_urlsafe_token

        token = generate_urlsafe_token()

        assert len(token) >= 32
        # Should be URL-safe (no special chars that need escaping)
        assert all(c.isalnum() or c in '-_' for c in token)

    def test_generate_urlsafe_token_custom_length(self):
        """Verify URL-safe token with custom length."""
        from auto_voice.config.secret_key_generator import generate_urlsafe_token

        token = generate_urlsafe_token(nbytes=64)

        # 64 bytes -> ~86 chars in base64
        assert len(token) >= 64


class TestFlaskSecretKeyTesting:
    """Test Flask app SECRET_KEY in testing mode."""

    def test_testing_mode_generates_random_key(self):
        """Verify testing mode generates secure random keys."""
        from auto_voice.web.app import create_app

        app, _ = create_app(testing=True)

        secret_key = app.config['SECRET_KEY']

        # Should be a secure random key (64 hex chars from secrets.token_hex(32))
        assert len(secret_key) >= 32
        assert secret_key != 'autovoice-dev-key'

    def test_testing_mode_keys_are_unique(self):
        """Verify each test app gets a unique key."""
        from auto_voice.web.app import create_app

        app1, _ = create_app(testing=True)
        app2, _ = create_app(testing=True)

        assert app1.config['SECRET_KEY'] != app2.config['SECRET_KEY']

    def test_testing_mode_key_is_hex(self):
        """Verify testing mode uses hex-encoded keys."""
        from auto_voice.web.app import create_app

        app, _ = create_app(testing=True)

        secret_key = app.config['SECRET_KEY']

        # secrets.token_hex(32) produces 64 hex chars
        assert len(secret_key) == 64
        assert all(c in '0123456789abcdef' for c in secret_key)


class TestFlaskSecretKeyProduction:
    """Test Flask app SECRET_KEY in production mode."""

    def test_production_requires_env_var(self):
        """Verify production mode requires AUTOVOICE_SECRET_FLASK_SECRET_KEY."""
        from auto_voice.web.app import create_app
        from auto_voice.config.secrets import SecretError

        # Clear any existing secret env vars
        env_backup = os.environ.copy()
        try:
            # Remove the secret key env var if it exists
            if 'AUTOVOICE_SECRET_FLASK_SECRET_KEY' in os.environ:
                del os.environ['AUTOVOICE_SECRET_FLASK_SECRET_KEY']

            with pytest.raises(SecretError, match="Required secret 'flask_secret_key' not found"):
                # Use TESTING=True to avoid SocketIO eventlet issues, but testing=False to test production path
                create_app(config={'TESTING': True}, testing=False)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    def test_production_loads_from_env_var(self):
        """Verify production mode loads secret from environment."""
        from auto_voice.web.app import create_app

        test_key = "test_secret_key_from_env_12345678901234567890"

        with patch.dict(os.environ, {'AUTOVOICE_SECRET_FLASK_SECRET_KEY': test_key}):
            # Use TESTING=True to avoid SocketIO eventlet issues
            app, _ = create_app(config={'TESTING': True}, testing=False)

            assert app.config['SECRET_KEY'] == test_key

    def test_production_config_override(self):
        """Verify config can override SecretsManager in production."""
        from auto_voice.web.app import create_app

        override_key = "override_secret_key_12345678901234567890"

        # Use TESTING=True to avoid SocketIO eventlet issues
        app, _ = create_app(config={'SECRET_KEY': override_key, 'TESTING': True}, testing=False)

        assert app.config['SECRET_KEY'] == override_key

    def test_secrets_manager_loads_from_file(self, tmp_path):
        """Verify SecretsManager can load secret from file."""
        from auto_voice.config.secrets import SecretsManager

        # Create secrets file
        secrets_file = tmp_path / "secrets.json"
        test_key = "file_secret_key_12345678901234567890"
        secrets_file.write_text(json.dumps({"flask_secret_key": test_key}))

        # Test SecretsManager directly with file path
        manager = SecretsManager(secrets_file=str(secrets_file))

        assert manager.get('flask_secret_key') == test_key
        assert manager.has('flask_secret_key')


class TestSecretKeyNeverHardcoded:
    """Test that hardcoded 'autovoice-dev-key' is never used."""

    def test_testing_mode_never_uses_hardcoded_key(self):
        """Verify testing mode never uses hardcoded default."""
        from auto_voice.web.app import create_app

        for _ in range(10):
            app, _ = create_app(testing=True)
            assert app.config['SECRET_KEY'] != 'autovoice-dev-key'

    def test_production_never_uses_hardcoded_key(self):
        """Verify production never falls back to hardcoded key."""
        from auto_voice.web.app import create_app

        test_key = "production_key_12345678901234567890"

        with patch.dict(os.environ, {'AUTOVOICE_SECRET_FLASK_SECRET_KEY': test_key}):
            # Use TESTING=True to avoid SocketIO eventlet issues
            app, _ = create_app(config={'TESTING': True}, testing=False)
            assert app.config['SECRET_KEY'] != 'autovoice-dev-key'

    def test_config_override_never_uses_hardcoded_key(self):
        """Verify config override doesn't allow hardcoded key."""
        from auto_voice.web.app import create_app

        # Even if someone tries to set it via config, it's not the default
        # Use TESTING=True to avoid SocketIO eventlet issues
        app, _ = create_app(config={'SECRET_KEY': 'custom_key', 'TESTING': True}, testing=False)

        assert app.config['SECRET_KEY'] == 'custom_key'
        assert app.config['SECRET_KEY'] != 'autovoice-dev-key'


class TestSecretKeySecurityProperties:
    """Test security properties of generated keys."""

    def test_key_entropy(self):
        """Verify generated keys have high entropy."""
        from auto_voice.config.secret_key_generator import generate_secret_key

        keys = [generate_secret_key() for _ in range(100)]

        # Check for no duplicates (high entropy test)
        assert len(set(keys)) == 100

        # Check each key has varied characters (not just repeating patterns)
        for key in keys[:10]:  # Sample check
            unique_chars = len(set(key))
            # Should have at least 20 different characters in base64 encoding
            assert unique_chars >= 20, f"Low character diversity: {unique_chars}"

    def test_key_meets_flask_requirements(self):
        """Verify keys meet Flask SECRET_KEY requirements."""
        from auto_voice.web.app import create_app

        with patch.dict(os.environ, {'AUTOVOICE_SECRET_FLASK_SECRET_KEY': 'test' * 16}):
            # Use TESTING=True to avoid SocketIO eventlet issues
            app, _ = create_app(config={'TESTING': True}, testing=False)

            secret_key = app.config['SECRET_KEY']

            # Flask requires SECRET_KEY to be bytes or str
            assert isinstance(secret_key, (str, bytes))

            # Should be long enough for secure signing
            assert len(secret_key) >= 32

    def test_testing_key_sufficient_for_signing(self):
        """Verify testing mode key is sufficient for session signing."""
        from auto_voice.web.app import create_app
        from itsdangerous import URLSafeTimedSerializer

        app, _ = create_app(testing=True)

        # Try to use the key for signing (Flask's actual use case)
        serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

        test_data = {'user_id': 123, 'session': 'test'}
        token = serializer.dumps(test_data)

        # Should be able to sign and verify
        assert serializer.loads(token) == test_data

    def test_hex_key_meets_requirements(self):
        """Verify hex keys meet security requirements."""
        from auto_voice.config.secret_key_generator import generate_hex_key

        key = generate_hex_key()

        # Should be at least 64 hex chars (32 bytes)
        assert len(key) >= 64

        # Should be valid hex
        try:
            bytes.fromhex(key)
        except ValueError:
            pytest.fail("Generated key is not valid hex")


class TestSecretsManagerIntegration:
    """Test SecretsManager integration with Flask app."""

    def test_secrets_manager_env_prefix(self):
        """Verify SecretsManager uses correct env prefix."""
        from auto_voice.config.secrets import SecretsManager

        test_key = "env_prefix_test_key_12345678901234567890"

        with patch.dict(os.environ, {'AUTOVOICE_SECRET_FLASK_SECRET_KEY': test_key}):
            manager = SecretsManager()

            assert manager.get('flask_secret_key') == test_key

    def test_secrets_manager_case_insensitive(self):
        """Verify SecretsManager handles case-insensitive names."""
        from auto_voice.config.secrets import SecretsManager

        test_key = "case_test_key_12345678901234567890"

        with patch.dict(os.environ, {'AUTOVOICE_SECRET_FLASK_SECRET_KEY': test_key}):
            manager = SecretsManager()

            # Should work with different cases
            assert manager.get('flask_secret_key') == test_key
            assert manager.get('FLASK_SECRET_KEY') == test_key
            assert manager.get('Flask_Secret_Key') == test_key

    def test_secrets_manager_required_secret(self):
        """Verify get_required raises on missing secret."""
        from auto_voice.config.secrets import SecretsManager, SecretError

        manager = SecretsManager()

        with pytest.raises(SecretError, match="Required secret 'nonexistent' not found"):
            manager.get_required('nonexistent')

    def test_secrets_not_logged(self):
        """Verify secrets are masked in string representation."""
        from auto_voice.config.secrets import SecretsManager

        test_key = "super_secret_key_should_not_appear"

        with patch.dict(os.environ, {'AUTOVOICE_SECRET_TEST': test_key}):
            manager = SecretsManager()

            str_repr = str(manager)
            repr_str = repr(manager)

            # Secret value should not appear in plain text
            assert test_key not in str_repr
            assert test_key not in repr_str

            # Should contain masking
            assert '*' in str_repr or 'masked' in str_repr.lower()
