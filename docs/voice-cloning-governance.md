# Voice Cloning Governance

AutoVoice is a local/operator-controlled tool by default. Public or commercial deployment requires additional governance beyond technical hardening.

## Required Public-Mode Attestations

When `AUTOVOICE_PUBLIC_DEPLOYMENT=true` or `AUTOVOICE_REQUIRE_MEDIA_CONSENT=true`, risky ingestion routes must include:

- `consent_confirmed=true`
- `source_media_policy_confirmed=true`

These attestations are a minimum safeguard. They do not resolve copyright, publicity/personality-right, biometric privacy, or platform terms questions by themselves.

## Allowed Source Policy

Hosted use should be limited to media and voices the operator or user owns, created, licensed, or is explicitly authorized to use. Do not use the service for deceptive impersonation, unauthorized celebrity/public-figure cloning, minors, or third-party commercial tracks without rights.

## Retention Matrix

| Artifact | Examples | Default storage | Deletion note |
| --- | --- | --- | --- |
| Source media | uploads, YouTube audio | compose data volumes | Clearing history does not prove media deletion |
| Derived media | stems, filtered vocals, diarized clips | compose data volumes | Delete with profile/media purge workflow |
| Profile data | embeddings, samples, adapters | profile/model volumes | Export/delete promises must include derived assets |
| Models/checkpoints | trained models, checkpoints | trained/checkpoint volumes | May contain voice identity information |
| Logs/audit | request IDs, operator events | log/app-state volumes | Keep long enough for abuse investigation |

## Operator Controls

Public mode now exposes structured operator controls for governance workflows:

- `GET /api/v1/audit/events` returns durable audit events with request IDs and resource metadata.
- `GET /api/v1/voice/profiles/<profile_id>/export` returns profile metadata, samples, and opaque asset references for user export workflows.
- `DELETE /api/v1/voice/profiles/<profile_id>/purge` removes the profile and any registered owned assets tracked by the app-state registry.
- Public-mode responses redact known filesystem path fields and replace them with `*_asset_id` fields backed by the server-side asset registry.

Profile export includes recent profile-scoped audit events plus a retention block
that states purge semantics. Profile purge removes profile-linked app-state
records and registered owned asset references, but durable audit records are
retained for abuse investigation and operator accountability.

## Launch Decisions Still Required

- Whether hosted YouTube ingestion is allowed at all.
- Which content classes are prohibited or require manual review.
- Retention periods for media, embeddings, models, and audit records.
- Export and deletion guarantees for external users.
- Account, quota, and abuse-review model for public use.
