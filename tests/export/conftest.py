"""Test configuration for export tests.

Pre-mocks TensorRT to avoid GPU dependencies while allowing code coverage.
"""
import sys
from unittest.mock import MagicMock

# NOTE: This conftest intentionally does NOT mock tensorrt at module load time
# because that causes issues with pytest-cov. Instead, we use @patch decorators
# in individual tests to mock TensorRT after imports complete.
