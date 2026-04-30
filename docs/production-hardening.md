# Production Hardening Status

The current implementation adds enforceable public-mode gates but does not make AutoVoice automatically public-production-ready.

## Implemented Gates

- API-token middleware for hosted/operator endpoints when `AUTOVOICE_REQUIRE_API_AUTH=true` or `AUTOVOICE_PUBLIC_DEPLOYMENT=true`.
- Explicit CORS allowlist enforcement for Socket.IO and HTTP app startup in public mode.
- In-memory request rate limiting for public/authenticated mode.
- Strict server-path sandboxing for `audio_path` imports when public or strict mode is enabled.
- YouTube URL canonicalization to YouTube hosts and video IDs.
- Public-mode media consent/source-rights attestation checks for risky import/download paths.
- Review-gated YouTube ingest flow that downloads audio, writes vocal/instrumental stems,
  diarizes vocals, suggests source-profile matches, and applies assignments/creation only
  after operator confirmation.
- Deterministic local real-audio YouTube ingest evidence via
  `scripts/run_completion_matrix.py --real-audio`, plus a separate opt-in live
  YouTube smoke lane gated by `--live-youtube` and `AUTOVOICE_LIVE_YOUTUBE_URL`.
- Current-head benchmark provenance validation and `/tmp` source-bundle rejection.

## Remaining Blockers

- Full hardware/Jetson evidence must be regenerated for the current commit when hardware lanes are in scope.
- The latest pushed local-ready no-Docker completion matrix is green at
  `reports/completion/local-final-20260430T032639Z-25d484880f80/` for source git
  SHA `25d484880f803a551d457e6f893daf48b34506eb`. Any later release commit must
  regenerate matching evidence before being called current.
- Current benchmark latest artifacts must validate against the candidate source
  SHA. `reports/completion/latest/` remains a historical pointer until
  deliberately republished.
- MeanVC remains an experimental, explicit opt-in performance lane. Default local-only pytest now skips it with owner/action metadata unless `AUTOVOICE_MEANVC_FULL=1` and the required runtime assets are present.
- TensorRT runtime availability and TensorRT engine availability are separate states. The local `autovoice-thor` environment has TensorRT installed and the local-ready baseline found a complete canonical engine suite, but new hosts still require built engine artifacts under the configured engine directory.
- Multi-user SaaS requires account-level auth, per-user storage isolation, quotas, audit review, deletion/export APIs, and legal/policy approval.
