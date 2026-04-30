# Quality and UX Post-Release Milestone

This milestone turns the local/private readiness baseline into a quality-led
operator experience. It does not declare AutoVoice public- or commercial-
production-ready. The current system has evidence that it can boot, train,
convert, validate release evidence, and run hosted smoke checks, but those claims
must be regenerated for the candidate HEAD before a release decision.

## Evidence Baseline

- GitNexus was refreshed on 2026-04-28 during the production-readiness workstream.
- Public production full smoke passed against `https://autovoice.giggahost.com`.
- The smoke run proved upload, workflow review resolution, minimal LoRA
  training, conversion, mix download, and profile cleanup.
- Current benchmark evidence passes release gates with 8 pipelines, 6
  comparisons, and no promotable experimental candidates.
- The completion matrix passes, but its skip audit still records
  environment-gated CUDA, diarization, training, karaoke, TensorRT, and
  benchmark-audio E2E paths.
- The YouTube/source-ingest flow now has deterministic real-audio fixture
  coverage through `scripts/run_completion_matrix.py --real-audio`; the
  live-network YouTube smoke remains an explicit operator-owned opt-in lane.

Read this as historical or candidate-specific evidence unless the underlying
artifact embeds the current candidate git SHA. The current release status ladder
is defined in [current-truth.md](./current-truth.md).

## GitNexus Context

GitNexus identified the owning quality and UX surfaces as:

- `src/auto_voice/evaluation/benchmark_reporting.py`
- `scripts/build_benchmark_dashboard.py`
- `scripts/run_production_smoke.py`
- `frontend/src/pages/SystemStatusPage.tsx`
- `frontend/src/components/QualityMetricsDashboard.tsx`
- `frontend/src/components/SpectrogramViewer.tsx`
- `frontend/src/pages/ConversionWorkflowPage.tsx`
- `frontend/src/pages/ConversionHistoryPage.tsx`
- `frontend/src/pages/VoiceProfilePage.tsx`

## Child Work

- `AV-japt.1`: add quality metrics and stem assertions to production full smoke.
- `AV-japt.2`: surface benchmark and release evidence in the operator UI.
- `AV-japt.3`: add artifact comparison UX for conversion outputs.
- `AV-japt.4`: expand real-audio quality fixture coverage beyond minimal smoke.
  Initial coverage now includes review-gated YouTube ingest using tracked
  real-audio fixtures; broader conversion/quality fixture expansion remains
  independent post-release work.
- `AV-japt.5`: add target-voice accumulation and full-model training readiness UX.
- `AV-japt.6`: close environment-gated E2E confidence gaps for quality workflows.

## Completion Rule

This milestone is complete when the child backlog exists and each item has clear
acceptance criteria rooted in production evidence. Implementation of the child
issues should happen as independent post-release work, with GitNexus context and
benchmark evidence checked before each change.
