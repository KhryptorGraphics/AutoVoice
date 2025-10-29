# Validation Quick Reference Card

## Quick Start

```bash
# Full validation suite (recommended)
bash scripts/run_full_validation.sh

# Docker validation only
bash scripts/test_docker_deployment.sh
```

## Script Locations

```
scripts/
├── test_docker_deployment.sh   # Docker validation (Comment 5)
└── run_full_validation.sh      # Full orchestration (Comment 6)
```

## Results Locations

```
validation_results/
├── docker_validation.log                    # Docker test output
└── validation_summary_<timestamp>.log       # Full validation summary
```

## Exit Codes

- `0` = All validations passed ✅
- `1` = One or more validations failed ❌

## What Each Script Does

### test_docker_deployment.sh

1. Build `autovoice:validation` image
2. Run container with GPU (if available)
3. Test health endpoints: `/health`, `/health/live`, `/health/ready`
4. Test GPU endpoint: `/api/v1/gpu_status`
5. Test API: `/api/v1/voice/profiles`
6. Run nvidia-smi in container
7. Check logs for errors
8. Cleanup automatically

**Time**: ~2-3 minutes

### run_full_validation.sh

1. Check environment (Python, GPU)
2. Generate test data
3. Run pytest system tests
4. Validate code quality
5. Validate integration
6. Validate documentation
7. Run Docker validation
8. Evaluate voice quality (optional)
9. Generate final report

**Time**: ~4-7 minutes

## Common Commands

```bash
# Check if scripts are executable
ls -l scripts/test_docker_deployment.sh scripts/run_full_validation.sh

# Make scripts executable (if needed)
chmod +x scripts/test_docker_deployment.sh scripts/run_full_validation.sh

# View latest Docker validation log
cat validation_results/docker_validation.log

# View latest full validation summary
ls -t validation_results/validation_summary_*.log | head -1 | xargs cat

# Clean validation results
rm -rf validation_results/
```

## CI/CD Integration

```yaml
# GitHub Actions
- name: Run validation
  run: bash scripts/run_full_validation.sh

- name: Upload results
  uses: actions/upload-artifact@v2
  with:
    name: validation-results
    path: validation_results/
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Docker not found | Install Docker: `curl -fsSL https://get.docker.com \| sh` |
| Permission denied | Make executable: `chmod +x scripts/*.sh` |
| Container won't start | Check logs in validation output |
| GPU not detected | Verify nvidia-smi works, check nvidia-docker |
| Validation skipped | Check if validator script exists |

## Requirements

**Docker Validation**:
- Docker installed
- Port 5000 available
- (Optional) NVIDIA GPU + nvidia-docker

**Full Validation**:
- Python 3.11+
- All dependencies installed
- Validator scripts present

## GPU vs CPU Behavior

| Feature | GPU Host | CPU Host |
|---------|----------|----------|
| Docker validation | Required | Optional |
| GPU status test | Required | Skipped |
| nvidia-smi | Executed | Skipped |
| Quality eval | Full | Quick mode |

## Validation Statistics

Full orchestrator tracks:
- Total validations attempted
- Passed validations ✓
- Failed validations ✗
- Skipped validations ⊘
- Success rate %
- Time per validation

## File Structure

```
project_root/
├── scripts/
│   ├── test_docker_deployment.sh       # 275 lines
│   └── run_full_validation.sh          # 255 lines
├── docs/
│   ├── validation_scripts_guide.md     # Full documentation
│   ├── validation_implementation_summary.md
│   └── VALIDATION_QUICK_REFERENCE.md   # This file
├── validation_results/                 # Auto-created
│   ├── docker_validation.log
│   └── validation_summary_*.log
└── Dockerfile                          # Used by Docker validation
```

## Key Features

### Docker Script
✅ Robust error handling
✅ GPU detection and conditional testing
✅ Comprehensive endpoint testing
✅ Automatic cleanup on exit
✅ Detailed logging
✅ Non-zero exit on failure

### Orchestrator Script
✅ Multi-phase validation
✅ Statistics tracking
✅ Optional vs. required phases
✅ Environment-aware execution
✅ Timestamped logs
✅ Graceful degradation

## Documentation

- **Full Guide**: `docs/validation_scripts_guide.md`
- **Implementation**: `docs/validation_implementation_summary.md`
- **Quick Reference**: This file

## Support

For detailed information, see:
- [Validation Scripts Guide](validation_scripts_guide.md)
- [Implementation Summary](validation_implementation_summary.md)
- [Quality Evaluation Guide](quality_evaluation_guide.md)

---

**Last Updated**: 2025-10-28
**Version**: 1.0
**Status**: Production Ready ✅
