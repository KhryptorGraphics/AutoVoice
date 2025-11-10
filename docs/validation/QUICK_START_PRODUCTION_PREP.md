# QUICK START: PRODUCTION PREPARATION
**AutoVoice - Fast Track to Production**

**Current Status:** 72/100 - NOT PRODUCTION READY
**Target:** 90/100 - PRODUCTION READY
**Timeline:** 2-4 weeks

---

## TL;DR - WHAT YOU NEED TO DO

### CRITICAL BLOCKERS (Fix First!)

```bash
# 1. Install missing dependencies (30 minutes)
pip install demucs pystoi pesq nisqa

# 2. Run tests to see what works (5 minutes)
pytest tests/ -v

# 3. Run benchmarks to validate performance (1 hour)
python scripts/run_comprehensive_benchmarks.py --quick

# 4. Check coverage (5 minutes)
pytest tests/ --cov=src --cov-report=html
```

**Then:** Work on improving test coverage from 9.16% to 80% (2-3 weeks)

---

## PHASE 1: IMMEDIATE FIXES (1-2 Days)

### Step 1: Install Dependencies

```bash
# Create fresh environment
conda create -n autovoice-prod python=3.12 -y
conda activate autovoice-prod

# Install core
pip install -r requirements.txt

# Install CRITICAL missing dependencies
pip install demucs
# OR if demucs fails:
pip install spleeter>=2.4.0,<3.0.0

# Install quality metrics
pip install pystoi pesq nisqa

# Optional: TensorRT for optimization
# pip install tensorrt>=8.6.0
```

### Step 2: Validate Installation

```bash
python scripts/validate_installation.py
```

Expected output:
```
✅ PyTorch 2.9.0+cu128
✅ CUDA 12.8
✅ demucs installed
✅ pystoi installed
✅ pesq installed
✅ nisqa installed
```

### Step 3: Run Full Test Suite

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Check what failed
pytest tests/ -v | grep FAILED
```

**Expected Result:** Most tests should now PASS (previously blocked)

### Step 4: Execute Benchmarks

```bash
# Quick benchmark (10-15 minutes)
python scripts/run_comprehensive_benchmarks.py --quick --gpu-id 0

# Full benchmark suite (1-2 hours)
python scripts/run_comprehensive_benchmarks.py --gpu-id 0
```

**Validate Against Targets:**
- Voice Conversion RTF: < 1.5x ✓
- TTS Latency (1s): < 100ms ✓
- GPU Speedup: > 3x ✓
- Pitch Accuracy: < 12 Hz ✓
- Speaker Similarity: > 0.85 ✓

---

## PHASE 2: TEST COVERAGE (1-2 Weeks)

### Current Status: 9.16% → Target: 80%

**Priority Modules to Test:**

1. **Memory Manager (18.65% coverage) - P0**
   ```bash
   pytest tests/test_memory_manager.py -v --cov=src/auto_voice/gpu/memory_manager.py
   ```

2. **Checkpoint Manager (18.89% coverage) - P0**
   ```bash
   pytest tests/test_checkpoint_manager.py -v --cov=src/auto_voice/training/checkpoint_manager.py
   ```

3. **Quality Metrics (23.49% coverage) - P1**
   ```bash
   pytest tests/test_quality_metrics.py -v --cov=src/auto_voice/utils/quality_metrics.py
   ```

### Coverage Improvement Strategy

**Week 1: Critical Modules (40% → 60%)**
```bash
# Add unit tests for:
- Memory management error handling
- Checkpoint save/restore edge cases
- Metrics calculation boundary conditions
- Performance monitoring failure modes
```

**Week 2: Integration & E2E (60% → 80%)**
```bash
# Add integration tests for:
- Full pipeline voice conversion
- Multi-GPU operations
- API endpoint workflows
- Error recovery scenarios
```

### Monitoring Progress

```bash
# Daily coverage check
pytest tests/ --cov=src --cov-report=term | grep TOTAL

# Generate HTML report
pytest tests/ --cov=src --cov-report=html
firefox htmlcov/index.html  # View detailed report
```

---

## PHASE 3: PERFORMANCE VALIDATION (1 Week)

### Load Testing

```bash
# 1. Start API server
python -m auto_voice.api.server

# 2. Run load tests (separate terminal)
python tests/load_test.py --users 50 --duration 300

# 3. Monitor metrics
curl http://localhost:8000/metrics
```

**Validate SLOs:**
- Concurrent Users: 50+ ✓
- Requests/Second: 100+ ✓
- P95 Latency: < 100ms ✓
- Error Rate: < 0.1% ✓

### Memory Leak Testing

```bash
# Run 24-hour stress test
python tests/stress_test.py --duration 86400 --check-memory

# Monitor GPU memory
watch -n 1 nvidia-smi
```

### Performance Profiling

```bash
# Profile CUDA kernels
python scripts/profile_cuda_kernels.py

# Profile full pipeline
python scripts/profile_performance.py --iterations 100
```

---

## PHASE 4: SECURITY & VALIDATION (1 Week)

### Security Audit

```bash
# Dependency vulnerability scan
pip-audit

# Code security scan
bandit -r src/

# Container security
docker scan autovoice:latest
```

### API Security Testing

```bash
# Authentication testing
python tests/security/test_auth.py

# Input validation fuzzing
python tests/security/test_input_validation.py

# Rate limiting
python tests/security/test_rate_limits.py
```

### Final Validation

```bash
# Run complete validation suite
python scripts/run_validation_suite.py

# Generate final report
python scripts/generate_validation_report.py
```

---

## DEPLOYMENT CHECKLIST

### Pre-Deployment (Must Complete)

```
☐ Test coverage ≥ 80%
☐ All performance benchmarks PASS
☐ Load testing completed (50+ users)
☐ Security audit PASS
☐ Dependencies validated
☐ Monitoring configured
☐ Alerts set up
☐ Runbooks documented
☐ Rollback plan tested
☐ Staging deployment successful
```

### Deployment Steps

**1. Staging Deployment**
```bash
# Build Docker image
docker build -t autovoice:staging .

# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# Validate health
curl http://staging.autovoice.com/health
```

**2. Production Deployment**
```bash
# Tag production image
docker tag autovoice:staging autovoice:v1.0.0

# Deploy with zero downtime
kubectl apply -f k8s/production/

# Monitor rollout
kubectl rollout status deployment/autovoice
```

**3. Post-Deployment Validation**
```bash
# Check health
curl https://autovoice.com/health

# Monitor metrics
open https://grafana.autovoice.com/

# Check logs
kubectl logs -f deployment/autovoice
```

---

## MONITORING & ALERTING

### Set Up Alerts

**Grafana Alerts:**
```yaml
- name: high_latency
  condition: p95_latency > 100ms
  for: 5m
  action: page_oncall

- name: error_rate
  condition: error_rate > 0.1%
  for: 5m
  action: page_oncall

- name: gpu_memory
  condition: gpu_memory > 90%
  for: 10m
  action: notify_team
```

### Health Checks

```bash
# Liveness probe
curl http://localhost:8000/health/live

# Readiness probe
curl http://localhost:8000/health/ready

# Metrics endpoint
curl http://localhost:8000/metrics
```

---

## TROUBLESHOOTING

### Common Issues

**Test Coverage Low**
```bash
# Find untested files
pytest --cov=src --cov-report=term-missing | grep "0%"

# Add tests for critical paths
pytest tests/ -k "test_critical" -v
```

**Benchmarks Failing**
```bash
# Check dependencies
python scripts/validate_installation.py

# Verify GPU
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Memory Issues**
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## QUICK VALIDATION COMMANDS

```bash
# 1. Check overall status
python scripts/validate_installation.py

# 2. Run critical tests
pytest tests/test_pipeline.py tests/test_api.py -v

# 3. Check coverage
pytest tests/ --cov=src --cov-report=term | grep TOTAL

# 4. Run quick benchmark
python scripts/run_comprehensive_benchmarks.py --quick

# 5. Validate metrics
curl http://localhost:8000/metrics | grep autovoice

# 6. Check logs
tail -f logs/autovoice.log
```

---

## TIMELINE SUMMARY

### Fast Track (2 Weeks)

**Week 1:**
- Days 1-2: Install deps, fix failing tests
- Days 3-4: Add critical unit tests (→ 60% coverage)
- Day 5: Run benchmarks, validate performance

**Week 2:**
- Days 1-2: Continue testing (→ 80% coverage)
- Days 3-4: Load testing, optimization
- Day 5: Staging deployment, final validation

### Standard Track (4 Weeks)

**Week 1:** Dependencies + basic testing
**Week 2:** Coverage improvement to 80%
**Week 3:** Performance and load testing
**Week 4:** Security audit + production prep

### Conservative Track (6-8 Weeks)

**Weeks 1-2:** Full test coverage (90%+)
**Weeks 3-4:** Comprehensive performance testing
**Weeks 5-6:** Security hardening and audit
**Weeks 7-8:** Staging validation + production rollout

---

## SUCCESS CRITERIA

### Production Ready When:

```
✅ Test coverage ≥ 80%
✅ All benchmarks PASS
✅ Load testing: 50+ concurrent users
✅ Security audit: No critical issues
✅ Staging deployment: Successful
✅ Monitoring: Configured and tested
✅ Documentation: Complete and reviewed
✅ Team: Trained and ready
```

### Production Score Target:

```
Current:  72/100  ⚠️
Target:   90/100  ✅
Gap:      -18 points

Breakdown:
- Architecture:      95  →  95  (maintain)
- Testing:           15  →  85  (+70 points)
- Documentation:     98  →  98  (maintain)
- Performance:       85  →  90  (+5 points)
- Dependencies:      45  →  95  (+50 points)
```

---

## NEXT STEPS

**RIGHT NOW:**
1. Install missing dependencies
2. Run full test suite
3. Execute benchmarks
4. Review results

**THIS WEEK:**
1. Fix all failing tests
2. Start coverage improvement
3. Begin load testing prep
4. Set up monitoring

**THIS MONTH:**
1. Achieve 80% coverage
2. Complete performance validation
3. Security audit
4. Production deployment

---

**For Detailed Information:**
- Full Report: `docs/validation/FINAL_PRODUCTION_READINESS_REPORT.md`
- Visual Dashboard: `docs/validation/PRODUCTION_READINESS_DASHBOARD.md`
- Test Guide: `docs/testing_guide.md`
- Deployment Guide: `docs/deployment_quick_reference.md`

---

**Questions?** Review the comprehensive reports or contact the development team.

**Ready to start?** Run: `pip install demucs pystoi pesq nisqa && pytest tests/ -v`
