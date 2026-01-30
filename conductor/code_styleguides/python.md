# Python Style Guide

## Formatting

- **Formatter**: black (line length 88)
- **Import sorting**: isort (black-compatible profile)
- **Type checking**: mypy (strict mode)

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Modules | snake_case | `singing_conversion_pipeline.py` |
| Classes | PascalCase | `TRTEngineBuilder` |
| Functions | snake_case | `build_engine()` |
| Constants | UPPER_SNAKE | `OPSET_VERSION` |
| Private | `_prefix` | `_workspace_bytes` |
| Type vars | PascalCase | `TensorType` |

## Type Annotations

- All public functions must have full type annotations
- Use `from __future__ import annotations` for forward references
- Prefer `dict[str, Any]` over `Dict[str, Any]` (Python 3.12+)
- Use `Optional[X]` only when None is a valid value, not for unset params

## Error Handling

- **No fallback behavior**: Always `raise RuntimeError(...)`, never `pass` or return defaults
- Catch specific exceptions, never bare `except:`
- Error messages must include context (file path, tensor shape, etc.)
- Use `logger.error()` before raising for debugging

## Docstrings

- Google style docstrings for all public classes and functions
- Include Args, Returns, Raises sections
- Keep first line under 80 chars

```python
def build_engine(self, onnx_path: str, fp16: bool = True) -> trt.ICudaEngine:
    """Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to the ONNX model file.
        fp16: Enable FP16 precision.

    Returns:
        Compiled TensorRT engine.

    Raises:
        RuntimeError: If ONNX parsing or engine build fails.
    """
```

## Testing

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<behavior_description>`
- Use pytest fixtures, prefer `scope="module"` for expensive setup
- Assertions must check real values (shapes, dtypes, non-NaN)

## Imports

Order:
1. Standard library
2. Third-party packages
3. Local imports

```python
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import tensorrt as trt

from auto_voice.export.onnx_export import export_content_encoder
```

## ML/Audio Specific

- Tensor shapes: always document as comments (e.g., `# (batch, channels, time)`)
- Frame alignment: use `F.interpolate(x.transpose(1,2), size=target)` pattern
- Speaker embeddings: 256-dim (mean+std of 128 mels), L2-normalized
- Always set `model.train(False)` before inference/export
- Use `torch.no_grad()` context for inference paths
