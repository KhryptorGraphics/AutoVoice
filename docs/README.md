# AutoVoice Docs Index

This file is the canonical entrypoint for repository documentation.

## Canonical Docs

These documents describe the current MVP behavior and should be treated as the source of truth:

- [../README.md](../README.md): runtime entrypoints, repo map, and current MVP scope
- [current-truth.md](./current-truth.md): canonical product boundary, runtime, docs, and historical artifact notes
- [swarm-operator-contract.md](./swarm-operator-contract.md): canonical `autovoice swarm` operating model, run taxonomy, GitNexus inputs, MemKraft writes, and lane completion rules
- [user-guide-voice-profiles.md](./user-guide-voice-profiles.md): source-artist and target-user workflow, training lifecycle, conversion, and karaoke usage
- [frontend-persistence-boundaries.md](./frontend-persistence-boundaries.md): what lives in browser `localStorage` versus what is owned by backend storage
- [api/README.md](./api/README.md): REST and WebSocket API entrypoint
- [deployment.md](./deployment.md): current compose, hosted preflight, release-candidate, and completion-matrix deployment contract
- [quality-ux-post-release.md](./quality-ux-post-release.md): post-release quality and operator-UX milestone backed by production smoke and benchmark evidence
- [troubleshooting.md](./troubleshooting.md): operational debugging and recovery notes

## Reference and Design Docs

These are useful for implementation context, but they are not the authoritative current-product spec:

- [continuous-learning-architecture.md](./continuous-learning-architecture.md): earlier architecture design track
- [seed-vc-architecture.md](./seed-vc-architecture.md): pipeline-specific reference
- [hq-svc-patterns.md](./hq-svc-patterns.md): HQ-SVC implementation notes
- [sota-voice-training.md](./sota-voice-training.md): research synthesis
- [smoothsinger-concepts.md](./smoothsinger-concepts.md): conceptual notes
- [security-review.md](./security-review.md): security review context
- [deployment-guide.md](./deployment-guide.md): deployment-oriented instructions
- [deployment-verification.md](./deployment-verification.md): reference notes for deployment verification details

## Historical Reports

Date-stamped readiness, coverage, and failure-analysis reports under `docs/` are historical artifacts. They document past investigation snapshots, not the current MVP truth. Use them for archaeology only after reading the canonical docs above.

Historical examples include:

- [test_coverage_improvement_20260202.md](./test_coverage_improvement_20260202.md)
- [test_coverage_conversion_quality_analyzer_20260202.md](./test_coverage_conversion_quality_analyzer_20260202.md)
- [test_trt_pipelines_coverage_20260202.md](./test_trt_pipelines_coverage_20260202.md)
- [test_coverage_audio_separation_20260202.md](./test_coverage_audio_separation_20260202.md)
- [test_failure_analysis_20260202.md](./test_failure_analysis_20260202.md)
- [augmentation_test_summary_20260202.md](./augmentation_test_summary_20260202.md)
