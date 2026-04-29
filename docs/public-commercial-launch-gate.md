# Public/Commercial Launch Gate

AutoVoice is not cleared for public multi-user or commercial launch until this
gate is satisfied. This document is the working checklist for `AV-3rfd.18`.

## Current Decision

Status: blocked pending external approval and hosted evidence.

Machine-readable gate: `GET /api/v1/public-commercial/readiness`.

The codebase now has local/operator controls for API-token auth, CORS allowlist
enforcement, media-consent attestations, path sandboxing, audit events, profile
export, profile purge, redacted public-mode responses, and review-gated YouTube
auto-ingest. Those controls are necessary, but not sufficient, for a public or
commercial launch.

## Evidence Collected On 2026-04-29

- Mocked browser closeout suite passed, including YouTube auto-ingest review and
  confirmation: `cd frontend && npm run test:e2e -- production-closeout.smoke.spec.ts`
- Live local browser suite passed against the `autovoice-thor` backend harness:
  `cd frontend && npm run test:e2e:live -- full-ui-local.live.spec.ts`
- A consent-cleared/public-domain YouTube candidate was probed with `yt-dlp`
  metadata only: `https://www.youtube.com/watch?v=26ilf0jO_ZM`
  (`5 Things: Black Holes`, NASA/Public Domain via Wikimedia review).
- Full real YouTube auto-ingest with Demucs plus diarization was not executed in
  this pass because it requires a rights-cleared operator-selected clip, model
  runtime assets, and hardware/runtime time budget. Do not treat metadata probing
  as audio pipeline evidence.

## Launch Blockers

- Account-level authentication and authorization are not implemented. API-token
  auth is suitable for a trusted operator or private gateway, not public users.
- Per-user storage isolation is not implemented. Public launch requires every
  media asset, profile, model, job, audit event, and export/delete operation to
  be scoped to a user or tenant.
- Persistent quotas and abuse controls are not implemented. Public launch needs
  upload/download limits, job concurrency limits, storage limits, rate policies,
  and abuse-review queues.
- Public ingress threat model and penetration-test evidence are missing.
- Legal/product approval is missing for voice consent, likeness/publicity
  rights, biometric privacy, copyright/source-media policy, platform terms, and
  prohibited impersonation rules.
- External deletion/export guarantees are not policy-approved. Current export
  and purge endpoints are operator controls, not a public-user data-governance
  program.
- Hosted observability, rollback, support, and incident-response evidence must
  be regenerated against the candidate public deployment.

## Machine-Readable Gate Inputs

The launch gate endpoint reports `ready: false` until all of these inputs are
configured and their linked evidence exists:

- `AUTOVOICE_ACCOUNT_AUTH_PROVIDER`: account auth provider, not operator API-token auth.
- `AUTOVOICE_TENANT_ISOLATION_ENABLED=true`: set only after cross-user isolation tests pass.
- `AUTOVOICE_QUOTA_BACKEND`: persistent quota backend, not in-memory/local limits.
- `AUTOVOICE_ABUSE_REVIEW_ENABLED=true`: set only after abuse review workflow is active.
- `AUTOVOICE_HOSTED_PUBLIC_EVIDENCE_PATH`: current-head hosted evidence artifact.
- `AUTOVOICE_LEGAL_APPROVAL_PATH`: signed legal/product approval artifact.
- `AUTOVOICE_PUBLIC_INGRESS_REVIEW_PATH`: completed threat-model/security-review artifact.

## Required Public/Commercial Evidence

Before closing `AV-3rfd.18`, archive evidence for the candidate commit under
ignored immutable `reports/` paths and link it from the release decision:

- authenticated hosted smoke for every user-facing route
- account-auth and authorization negative tests
- per-user isolation tests across profiles, media, jobs, exports, purges, audit
  events, and WebSocket/session channels
- quota/rate-limit tests for uploads, YouTube ingest, conversions, training, and
  downloads
- public-ingress security review and abuse-case review
- current-head benchmark, completion matrix, and hardware/model-lane evidence
- signed legal/product approval for voice, media, retention, deletion/export,
  and prohibited-use policy

## Closure Rule

`AV-3rfd.18` must remain open or blocked until all launch blockers above have
evidence. Closing it without policy approval and hosted public-lane evidence
would create a false production-readiness claim.
