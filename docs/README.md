# AutoVoice Docs Index

This file is the canonical entrypoint for repository documentation.

## Canonical Docs

These documents describe the current MVP behavior and should be treated as the source of truth:

- [../README.md](../README.md): runtime entrypoints, repo map, and current MVP scope
- [current-truth.md](./current-truth.md): canonical product boundary, runtime, docs, and historical artifact notes
- [user-guide-voice-profiles.md](./user-guide-voice-profiles.md): source-artist and target-user workflow, training lifecycle, conversion, and karaoke usage
- [frontend-persistence-boundaries.md](./frontend-persistence-boundaries.md): what lives in browser `localStorage` versus what is owned by backend storage
- [api/README.md](./api/README.md): REST and WebSocket API entrypoint
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

## Historical Reports

Date-stamped readiness, coverage, and failure-analysis reports under `docs/` are historical artifacts. They document past investigation snapshots, not the current MVP truth. Use them for archaeology only after reading the canonical docs above.
