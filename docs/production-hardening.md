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

- Full hardware/Jetson evidence must be regenerated for the current commit when hardware lanes are in scope.
- The current local/no-Docker completion matrix is green at `reports/completion/20260429T081623Z-222c8882/` for source git SHA `222c8882804d726fea2339e7831b7511bfb5a005`; rerun it after selecting the final release commit.
- Current benchmark latest artifacts validate against that source SHA, but `reports/completion/latest/` and `reports/release-evidence/latest/` remain historical pointers until deliberately republished.
- The full local pytest suite is not clean yet under all lanes: Docker deployment tests are intentionally outside the current local-only scope, and MeanVC performance remains below its recorded runtime contract on this machine.
- TensorRT runtime availability and TensorRT engine availability are separate states. The local `autovoice-thor` environment has TensorRT installed, but GPU optimization still requires built engine artifacts under the configured engine directory.
- Multi-user SaaS requires account-level auth, per-user storage isolation, quotas, audit review, deletion/export APIs, and legal/policy approval.
