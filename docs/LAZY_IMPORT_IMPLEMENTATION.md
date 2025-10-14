# Lazy Import Implementation

## Summary

Implemented true lazy imports across the AutoVoice package to prevent eager loading of heavy dependencies (torch, TensorRT) during module import.

## Changes Made

### 1. Removed Eager Initialization

**Files Modified:**
- `/src/auto_voice/models/__init__.py`
- `/src/auto_voice/audio/__init__.py`
- `/src/auto_voice/inference/__init__.py`
- `/src/auto_voice/__init__.py`

**What Was Removed:**
- `_initialize_names()` function that eagerly called `__getattr__()` for all names in `__all__`
- The invocation of `_initialize_names()` at module bottom

**Why This Fixes The Problem:**
The `_initialize_names()` function was looping through all exported names and calling `__getattr__()` at import time, which defeated the purpose of lazy loading by immediately importing all heavy modules.

### 2. Added TYPE_CHECKING Imports

**What Was Added:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .module import ClassName
    # ... other imports
```

**Purpose:**
- Provides type hints for static type checkers (mypy, pyright, etc.)
- These imports ONLY run during type checking, NOT at runtime
- Maintains IDE autocomplete and type safety

### 3. Retained Lazy Loading Mechanism

**What Was Kept:**
- `__all__` - Public API declaration
- `_module_cache` - Caching dictionary for loaded modules
- `__getattr__()` - PEP 562 lazy import mechanism that:
  - Checks cache first
  - Imports on-demand when attribute is accessed
  - Handles `ImportError` gracefully (returns `None` with warning)
  - Caches results for subsequent access

## How It Works

### Import Behavior

1. **Package Import (No Heavy Loading):**
   ```python
   import auto_voice  # Does NOT import torch
   import auto_voice.models  # Does NOT import torch
   ```

2. **Attribute Access (Lazy Loading):**
   ```python
   from auto_voice.models import VoiceTransformer  # NOW imports torch
   ```

3. **Caching:**
   - First access: imports and caches the module/class
   - Subsequent access: returns cached object (no re-import)

4. **Error Handling:**
   - If dependencies are missing, returns `None` with a warning
   - Allows package to work in environments without optional dependencies

### Benefits

1. **Faster Import Time:**
   - Package import doesn't trigger heavy dependency loading
   - Only loads what's actually used

2. **Graceful Degradation:**
   - Package works even if torch/TensorRT unavailable
   - Components that need them return `None` with warnings

3. **Type Safety:**
   - TYPE_CHECKING imports provide IDE support
   - Static type checkers work correctly
   - No runtime penalty

4. **PEP 562 Compliance:**
   - Uses official Python mechanism for lazy module attributes
   - `from package import name` syntax works correctly
   - No pre-binding of names needed

## Verification

The implementation was tested to verify:

✅ Package import does NOT load torch
✅ Submodule imports do NOT load torch
✅ Attribute access triggers lazy loading
✅ Caching works correctly
✅ Invalid attributes raise AttributeError
✅ Public APIs preserved in `__all__`
✅ TYPE_CHECKING imports work for type checkers

## Files Modified

1. `/src/auto_voice/models/__init__.py` - Removed eager init, added TYPE_CHECKING
2. `/src/auto_voice/audio/__init__.py` - Removed eager init, added TYPE_CHECKING
3. `/src/auto_voice/inference/__init__.py` - Removed eager init, added TYPE_CHECKING
4. `/src/auto_voice/__init__.py` - Complete refactor to lazy loading pattern

## Technical Details

### PEP 562 - Module __getattr__

The implementation uses [PEP 562](https://www.python.org/dev/peps/pep-0562/) which allows defining `__getattr__()` and `__dir__()` on modules:

- `__getattr__(name)` - Called when attribute not found in module dict
- Enables true lazy loading without pre-binding names
- `from module import name` triggers `__getattr__('name')`

### TYPE_CHECKING Pattern

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imports here ONLY run during static type checking
    # NOT executed at runtime
```

This is the standard pattern for avoiding circular imports and expensive type-only imports while maintaining type safety.

### Caching Strategy

- Simple dictionary cache: `_module_cache = {}`
- Cache key: attribute name (string)
- Cache value: imported object or `None`
- Thread-safe for most use cases (GIL protection)
- No cache invalidation needed (immutable after first load)

## Future Considerations

1. **Additional Modules:**
   - `/src/auto_voice/training/__init__.py` could benefit from same pattern
   - `/src/auto_voice/gpu/__init__.py` could benefit from same pattern
   - `/src/auto_voice/web/__init__.py` could benefit from same pattern

2. **Performance Monitoring:**
   - Track import time improvements
   - Monitor cache hit rates
   - Profile lazy loading overhead

3. **Documentation:**
   - Update user docs to explain lazy loading behavior
   - Document TYPE_CHECKING usage for contributors
   - Add examples of optional dependency handling
