# GLIBCXX_3.4.30 Fix - Applied Solution

## Problem
```
ImportError: /home/kp/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

This error blocked 10+ test modules from loading:
- scipy (scipy.stats, scipy.optimize)
- librosa
- sklearn
- torch modules
- torchaudio
- Other scientific computing libraries

## Root Cause
Anaconda's bundled `libstdc++.so.6` (version 11.2.0) only included GLIBCXX versions up to 3.4.29. Several packages (scipy, librosa, sklearn) were compiled against newer GLIBCXX versions (3.4.30+) and required a newer libstdc++.

## Solution Applied

### Method: Symlink to System libstdc++

**Step 1: Backup old library**
```bash
mv $CONDA_PREFIX/lib/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6.old
```

**Step 2: Create symlink to system library**
```bash
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6
```

**Step 3: Verify GLIBCXX versions**
```bash
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX
```

System libstdc++ provides GLIBCXX versions up to 3.4.33, which covers all required versions.

## Verification Results

All critical imports now work successfully:

```python
import scipy              # ✓ OK (v1.13.1)
import scipy.stats        # ✓ OK
import librosa            # ✓ OK (v0.10.2.post1)
import sklearn            # ✓ OK (v1.6.1)
import torch              # ✓ OK (v2.9.0+cu128)
import torchaudio         # ✓ OK
```

## Files Modified

- **Symlink created**: `/home/kp/anaconda3/lib/libstdc++.so.6` → `/usr/lib/x86_64-linux-gnu/libstdc++.so.6`
- **Backup created**: `/home/kp/anaconda3/lib/libstdc++.so.6.old`
- **Activation scripts**:
  - `/home/kp/anaconda3/etc/conda/activate.d/fix_libstdcxx.sh` (created but not needed with symlink)
  - `/home/kp/anaconda3/etc/conda/deactivate.d/fix_libstdcxx.sh` (created but not needed with symlink)

## Alternative Solutions (Not Used)

### 1. conda install (Attempted but slow)
```bash
conda install -c conda-forge libstdcxx-ng
```
This was attempted but environment solving took too long.

### 2. LD_LIBRARY_PATH (Created scripts)
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```
Activation scripts were created but the symlink method is cleaner and permanent.

## Impact

- **Before**: 10+ test modules failed to import
- **After**: All modules import successfully
- **Test suite**: Now ready to run complete validation

## Next Steps

1. Run the complete test suite to verify all tests pass
2. Monitor for any compatibility issues with the system libstdc++
3. Document this fix in the main validation guide

## Date Applied
2025-11-09

## System Information
- OS: Linux 6.6.87.2-microsoft-standard-WSL2
- Python: 3.13.5
- Conda: Anaconda base environment
- System libstdc++: 6.0.33 (GLIBCXX up to 3.4.33)
- Previous conda libstdc++: 11.2.0 (GLIBCXX up to 3.4.29)
