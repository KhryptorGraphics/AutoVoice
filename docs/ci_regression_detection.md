# CI Regression Detection Implementation

## Overview

This document describes the automated quality regression detection system implemented in the GitHub Actions CI/CD pipeline.

## Components

### 1. GitHub Actions Workflow Job

**File**: `.github/workflows/quality_checks.yml`

**Job**: `quality-regression`

The regression detection job runs after the main quality validation job and performs the following steps:

1. **Download Baseline Metrics**: Attempts to download the baseline metrics from workflow artifacts or uses the file checked into the repository.

2. **Create Baseline if Missing**: If no baseline exists, creates a default baseline with initial quality and performance targets.

3. **Generate Test Data**: Creates synthetic test audio files using the same approach as the quality validation job.

4. **Run Regression Tests**: Executes the `TestQualityRegressionDetection` test class which:
   - Loads baseline metrics
   - Measures current quality and performance metrics
   - Compares current metrics against baseline thresholds
   - Detects regressions and warnings

5. **Generate Comparison Report**: Creates a markdown report showing:
   - Metric-by-metric comparison
   - Change percentages
   - Status indicators (✅ OK, ⚠️ Warning, ❌ REGRESSION)
   - Summary of detected issues

6. **Upload Artifacts**: Saves all regression analysis results, reports, and baseline files as workflow artifacts.

7. **Update Baseline (Main Branch Only)**: On successful runs on the main branch, updates the baseline metrics to reflect the new quality/performance standards.

8. **Comment on PRs**: Posts the regression analysis report as a comment on pull requests.

### 2. Regression Test Class

**File**: `tests/test_performance.py`

**Class**: `TestQualityRegressionDetection`

The test class includes three test methods:

#### `test_load_baseline_metrics()`
- Loads baseline metrics from file or creates defaults
- Validates baseline structure
- Prints baseline metadata (timestamp, commit)

#### `test_measure_current_metrics()`
- Generates synthetic test audio
- Measures quality metrics using `VoiceConversionEvaluator`
- Measures performance metrics (RTF on CPU/GPU)
- Collects and validates current metrics

#### `test_compare_against_baseline()`
- Compares current metrics against baseline
- Detects regressions based on metric-specific thresholds
- Identifies warnings for metrics approaching thresholds
- Saves results to JSON file
- Fails test if regressions detected

### 3. Baseline Metrics File

**File**: `.github/quality_baseline.json`

Structure:
```json
{
  "version": "1.0",
  "timestamp": "ISO 8601 timestamp",
  "branch": "branch name",
  "commit": "commit SHA",
  "metrics": {
    "pitch_rmse_hz": 10.0,
    "pitch_correlation": 0.80,
    "speaker_similarity": 0.85,
    "overall_quality_score": 0.75,
    "processing_rtf_cpu": 20.0,
    "processing_rtf_gpu": 5.0
  },
  "thresholds": {
    "pitch_rmse_hz_max_increase": 2.0,
    "pitch_correlation_min_decrease": 0.05,
    "speaker_similarity_min_decrease": 0.05,
    "overall_quality_min": 0.70,
    "rtf_max_increase_percent": 20.0
  }
}
```

### 4. Pytest Configuration

**File**: `tests/conftest.py`

Added command-line options:
- `--baseline-file`: Path to baseline metrics file (default: `.github/quality_baseline.json`)
- `--output-file`: Path to save regression results JSON (default: None)

Added marker:
- `@pytest.mark.quality`: For quality evaluation tests

## Metrics Tracked

### Quality Metrics

1. **Pitch RMSE (Hz)**: Root mean squared error of pitch tracking
   - Lower is better
   - Threshold: Max increase of 2.0 Hz

2. **Pitch Correlation**: Correlation coefficient between source and converted pitch
   - Higher is better (0-1 range)
   - Threshold: Max decrease of 0.05

3. **Speaker Similarity**: Similarity score between target and converted voice
   - Higher is better (0-1 range)
   - Threshold: Max decrease of 0.05

4. **Overall Quality Score**: Composite quality metric
   - Higher is better (0-1 range)
   - Threshold: Minimum of 0.70

### Performance Metrics

1. **Processing RTF (CPU)**: Real-time factor for CPU processing
   - Lower is better (1.0 = real-time)
   - Threshold: Max increase of 20%

2. **Processing RTF (GPU)**: Real-time factor for GPU processing
   - Lower is better
   - Threshold: Max increase of 20%

## Regression Detection Logic

### Metric-Specific Rules

1. **RMSE Metrics** (lower is better):
   - ❌ REGRESSION: Current > Baseline + max_increase
   - ⚠️ WARNING: Current > Baseline + (max_increase / 2)
   - ✅ OK: Current <= Baseline + (max_increase / 2)

2. **Correlation/Similarity Metrics** (higher is better):
   - ❌ REGRESSION: Current < Baseline - min_decrease
   - ⚠️ WARNING: Current < Baseline - (min_decrease / 2)
   - ✅ OK: Current >= Baseline - (min_decrease / 2)

3. **Quality Scores** (higher is better):
   - ❌ REGRESSION: Current < overall_quality_min
   - ⚠️ WARNING: Current < Baseline - 0.05
   - ✅ OK: Current >= Baseline - 0.05

4. **RTF Metrics** (lower is better):
   - ❌ REGRESSION: Change% > rtf_max_increase_percent
   - ⚠️ WARNING: Change% > (rtf_max_increase_percent / 2)
   - ✅ OK: Change% <= (rtf_max_increase_percent / 2)

## Workflow Behavior

### On Pull Requests

1. Download baseline from main branch artifacts or repository file
2. Run regression tests against current code
3. Generate comparison report
4. Post report as PR comment
5. **Fail the build if regressions detected**

### On Main Branch Push

1. Run regression tests
2. If tests pass, update baseline with new metrics
3. Upload updated baseline as artifact
4. Baseline is available for subsequent PR comparisons

## Usage

### Manual Local Testing

```bash
# Run regression tests locally
pytest tests/test_performance.py::TestQualityRegressionDetection \
  --baseline-file=.github/quality_baseline.json \
  --output-file=regression_results.json \
  -v

# View results
cat regression_results.json
```

### CI/CD Execution

The regression detection runs automatically:
- On every push to main/develop branches
- On every pull request to main/develop branches

### Updating Thresholds

Edit `.github/quality_baseline.json` to adjust acceptable regression thresholds:

```json
{
  "thresholds": {
    "pitch_rmse_hz_max_increase": 2.0,      // Increase for looser pitch RMSE requirements
    "pitch_correlation_min_decrease": 0.05, // Decrease for stricter correlation requirements
    "speaker_similarity_min_decrease": 0.05,
    "overall_quality_min": 0.70,
    "rtf_max_increase_percent": 20.0        // Adjust performance tolerance
  }
}
```

## Benefits

1. **Automated Quality Assurance**: Catches quality degradations before they reach production
2. **Performance Monitoring**: Tracks processing speed to prevent performance regressions
3. **Historical Tracking**: Maintains baseline history through artifacts
4. **Transparent**: Clear reporting of what changed and by how much
5. **Configurable**: Adjustable thresholds for different quality/speed priorities
6. **PR Feedback**: Immediate feedback on pull requests about quality impact

## Maintenance

### Resetting Baseline

If legitimate changes require a new baseline:

1. Update `.github/quality_baseline.json` manually
2. Commit the changes
3. Push to main branch
4. Subsequent comparisons will use the new baseline

### Troubleshooting

1. **Tests always failing**: Check if thresholds are too strict for current implementation
2. **No baseline found**: Ensure `.github/quality_baseline.json` exists in repository
3. **Metrics not updating**: Verify the `update baseline on main` step is running
4. **Inconsistent results**: Ensure test data generation uses consistent seed

## Future Enhancements

1. **Trend Analysis**: Track metrics over time to identify gradual degradation
2. **Multi-Architecture Baselines**: Separate baselines for different hardware configurations
3. **Statistical Significance**: Use statistical tests for regression detection
4. **Visual Reports**: Generate charts showing metric trends
5. **Configurable Presets**: Define different threshold profiles (strict, balanced, permissive)
