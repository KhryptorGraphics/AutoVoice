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
- The current local/no-Docker completion matrix is green at `reports/completion/phase3-completion-20260429T121702Z-0324dc67/` for source git SHA `0324dc67623dc04f9b54e1858ab60c8fdede5f90`.
- Current benchmark latest artifacts validate against that source SHA. `reports/completion/latest/` remains a historical pointer until deliberately republished.
- MeanVC remains an experimental, explicit opt-in performance lane. Default local-only pytest now skips it with owner/action metadata unless `AUTOVOICE_MEANVC_FULL=1` and the required runtime assets are present.
- TensorRT runtime availability and TensorRT engine availability are separate states. The local `autovoice-thor` environment has TensorRT installed, but GPU optimization still requires built engine artifacts under the configured engine directory.
- Multi-user SaaS requires account-level auth, per-user storage isolation, quotas, audit review, deletion/export APIs, and legal/policy approval.
