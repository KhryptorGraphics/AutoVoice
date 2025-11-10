# Phase 1 Report Generation Testing Guide

## Quick Verification

Run the automated verification script to confirm all improvements are in place:

```bash
./tests/verify_phase1_report_structure.sh
```

Expected output:
```
✓ Script syntax is valid
✓ REPORT_OUTPUT variable defined
✓ --report-out flag parsing present
✓ generate_report function exists
✓ Report writes to $REPORT_OUTPUT
✓ No FILLED file generation
✓ All required variables collected
✓ Dynamic checkbox generation present
✓ All required sections present
✓ Summary references correct report file

All verification tests passed!
```

## Full Integration Testing

### Prerequisites

1. Conda environment `autovoice_py312` must be active
2. PyTorch with CUDA support installed
3. CUDA toolkit installed (or ready to install)

### Test 1: Default Behavior

```bash
# Activate environment
conda activate autovoice_py312

# Run Phase 1 execution
./scripts/phase1_execute.sh
```

**Expected Results:**

1. Script executes all Phase 1 steps
2. `PHASE1_COMPLETION_REPORT.md` is created/updated
3. File contains fully populated content:
   - Execution date and duration
   - All checkboxes marked according to results
   - Python, PyTorch, CUDA versions
   - Extension file path and size
   - GPU information
4. No `PHASE1_COMPLETION_REPORT_FILLED.md` file exists
5. `PHASE1_EXECUTION_SUMMARY.txt` references correct report

**Verification Commands:**

```bash
# Check report exists and is populated
ls -lh PHASE1_COMPLETION_REPORT.md
grep "Overall Status:" PHASE1_COMPLETION_REPORT.md
grep "PyTorch Version:" PHASE1_COMPLETION_REPORT.md
grep "Extension file:" PHASE1_COMPLETION_REPORT.md

# Verify no duplicate file
ls PHASE1_COMPLETION_REPORT_FILLED.md 2>&1 | grep "No such file"

# Check summary references correct file
grep "PHASE1_COMPLETION_REPORT.md" PHASE1_EXECUTION_SUMMARY.txt
```

### Test 2: Custom Output Path

```bash
# Run with custom output path
./scripts/phase1_execute.sh --report-out /tmp/my_phase1_report.md
```

**Expected Results:**

1. Report is written to `/tmp/my_phase1_report.md`
2. Default `PHASE1_COMPLETION_REPORT.md` is NOT modified
3. Custom report has same complete content

**Verification Commands:**

```bash
# Check custom report exists
ls -lh /tmp/my_phase1_report.md

# Verify it's fully populated
grep "Overall Status:" /tmp/my_phase1_report.md
grep "PyTorch Version:" /tmp/my_phase1_report.md
```

### Test 3: Report Content Validation

After running Phase 1, validate the report contains all required sections:

```bash
# Check all major sections exist
grep "## Executive Summary" PHASE1_COMPLETION_REPORT.md
grep "## Pre-Flight Check Results" PHASE1_COMPLETION_REPORT.md
grep "## CUDA Toolkit Installation" PHASE1_COMPLETION_REPORT.md
grep "## CUDA Extension Build" PHASE1_COMPLETION_REPORT.md
grep "## Bindings Verification" PHASE1_COMPLETION_REPORT.md
grep "## PyTorch CUDA Validation" PHASE1_COMPLETION_REPORT.md
grep "## Environment Snapshot" PHASE1_COMPLETION_REPORT.md
grep "## Verification Checklist" PHASE1_COMPLETION_REPORT.md
grep "## Next Steps" PHASE1_COMPLETION_REPORT.md
grep "## Conclusion" PHASE1_COMPLETION_REPORT.md
```

### Test 4: Dynamic Checkbox Validation

Verify checkboxes are dynamically marked:

```bash
# Count marked checkboxes (should be > 0)
grep -c "\[x\]" PHASE1_COMPLETION_REPORT.md

# Check specific items
grep "\[x\] System CUDA toolkit installed" PHASE1_COMPLETION_REPORT.md
grep "\[x\] CUDA extensions built successfully" PHASE1_COMPLETION_REPORT.md
grep "\[x\] \`from auto_voice import cuda_kernels\` works" PHASE1_COMPLETION_REPORT.md
```

### Test 5: Variable Population

Verify all dynamic variables are populated (not placeholders):

```bash
# Should NOT find any placeholder text
! grep "\[TO BE FILLED\]" PHASE1_COMPLETION_REPORT.md
! grep "\[VERSION\]" PHASE1_COMPLETION_REPORT.md
! grep "\[PATH\]" PHASE1_COMPLETION_REPORT.md
! grep "\[TIME\]" PHASE1_COMPLETION_REPORT.md

# Should find actual values
grep "Python Version: 3\." PHASE1_COMPLETION_REPORT.md
grep "PyTorch Version: 2\." PHASE1_COMPLETION_REPORT.md
grep "CUDA Version: 12\." PHASE1_COMPLETION_REPORT.md
```

## Troubleshooting

### Issue: Script fails with "Unknown option"

**Cause:** Invalid command-line argument

**Solution:** Use correct syntax:
```bash
./scripts/phase1_execute.sh                           # Default
./scripts/phase1_execute.sh --report-out <path>       # Custom path
```

### Issue: Report not fully populated

**Cause:** Script may have failed before report generation

**Solution:** Check error messages and logs:
```bash
# Check if script completed
echo $?  # Should be 0 for success

# Review build log
cat build.log

# Check summary
cat PHASE1_EXECUTION_SUMMARY.txt
```

### Issue: Old FILLED file still exists

**Cause:** Leftover from previous implementation

**Solution:** Remove manually:
```bash
rm -f PHASE1_COMPLETION_REPORT_FILLED.md
```

## Success Criteria

✅ All verification tests pass
✅ `PHASE1_COMPLETION_REPORT.md` is fully populated
✅ No `PHASE1_COMPLETION_REPORT_FILLED.md` file exists
✅ All checkboxes dynamically marked
✅ Extension path and size included
✅ All system information present
✅ Custom output path works
✅ Summary references correct file

## Additional Resources

- Implementation details: `docs/PHASE1_REPORT_IMPLEMENTATION_SUMMARY.md`
- Improvements overview: `docs/PHASE1_REPORT_IMPROVEMENTS.md`
- Verification script: `tests/verify_phase1_report_structure.sh`

