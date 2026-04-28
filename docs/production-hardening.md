# Production Hardening Status

The current implementation adds enforceable public-mode gates but does not make AutoVoice automatically public-production-ready.

## Implemented Gates

- API-token middleware for hosted/operator endpoints when `AUTOVOICE_REQUIRE_API_AUTH=true` or `AUTOVOICE_PUBLIC_DEPLOYMENT=true`.
- Explicit CORS allowlist enforcement for Socket.IO and HTTP app startup in public mode.
- In-memory request rate limiting for public/authenticated mode.
- Strict server-path sandboxing for `audio_path` imports when public or strict mode is enabled.
- YouTube URL canonicalization to YouTube hosts and video IDs.
- Public-mode media consent/source-rights attestation checks for risky import/download paths.
- Current-head benchmark provenance validation and `/tmp` source-bundle rejection.

## Remaining Blockers

- Full hardware/Jetson evidence must be regenerated for the current commit.
- The latest committed completion and benchmark evidence may still be stale until the matrix is rerun.
- Multi-user SaaS requires account-level auth, per-user storage isolation, quotas, audit review, deletion/export APIs, and legal/policy approval.
