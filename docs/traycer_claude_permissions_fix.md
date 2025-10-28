# Traycer Claude Launch Configuration

**Issue:** Traycer launches Claude Code without the `--dangerously-skip-permissions` flag, requiring manual permission approvals.

**Solution:** Created a wrapper script to automatically add the flag.

## What Was Done

### 1. Created Claude Wrapper Script

**Location:** `/home/kp/.local/bin/claude`

```bash
#!/bin/bash
# Claude Code wrapper to always add --dangerously-skip-permissions flag
# This ensures traycer (and other tools) launch Claude with skip permissions enabled

# Path to real Claude Code binary (via node)
REAL_CLAUDE="/home/kp/.nvm/versions/node/v25.0.0/lib/node_modules/@anthropic-ai/claude-code/cli.js"

# Check if --dangerously-skip-permissions is already in args
if [[ "$*" != *"--dangerously-skip-permissions"* ]]; then
    # Add the flag before all other arguments
    exec node "$REAL_CLAUDE" --dangerously-skip-permissions "$@"
else
    # Flag already present, just pass through
    exec node "$REAL_CLAUDE" "$@"
fi
```

### 2. Updated PATH Priority

**File:** `~/.bashrc`

Added to ensure wrapper is found before nvm's Claude:

```bash
# Ensure ~/.local/bin is first in PATH (for Claude wrapper with --dangerously-skip-permissions)
export PATH="/home/kp/.local/bin:$PATH"
```

### 3. Made Wrapper Executable

```bash
chmod +x /home/kp/.local/bin/claude
```

## How It Works

1. When traycer (or any tool) calls `claude`, it finds the wrapper first (due to PATH priority)
2. The wrapper checks if `--dangerously-skip-permissions` is already in the arguments
3. If not present, it adds the flag before all other arguments
4. It then executes the real Claude Code binary with all arguments

## Testing

To test the wrapper works:

```bash
# Reload shell environment
source ~/.bashrc

# Verify wrapper is found first
which claude
# Should show: /home/kp/.local/bin/claude

# Test it works
claude --version
# Should show: 2.0.28 (Claude Code)

# Verify flag is added
# When traycer launches Claude, it will now automatically include --dangerously-skip-permissions
```

## Benefits

- ✅ No manual permission approvals needed when traycer launches Claude
- ✅ Works automatically for all tool launches (traycer, command line, etc.)
- ✅ Doesn't break existing functionality
- ✅ Can be easily reverted if needed

## Reverting (if needed)

To revert back to original behavior:

```bash
# Remove or comment out the PATH line in ~/.bashrc
# Delete the wrapper
rm /home/kp/.local/bin/claude

# Reload shell
source ~/.bashrc
```

## Note

**Security Consideration:** The `--dangerously-skip-permissions` flag bypasses all permission checks. Only use this in trusted environments (sandboxes, development machines with no sensitive data). Do NOT use this on production systems or systems with internet access to untrusted code.

---

**Status:** ✅ Configured
**Date:** October 27, 2025
