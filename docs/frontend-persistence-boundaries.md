# Frontend Persistence Boundaries

AutoVoice uses two persistence domains:

- Backend-owned product state under `DATA_DIR`
- Browser-owned local preferences in `localStorage`

This boundary is intentional. Product state must be recoverable, consistent across pages, and visible to backend workflows. Browser state is limited to convenience preferences that are safe to keep local to one machine and one browser profile.

## Backend-Owned Product State

These domains are authoritative on the server and must not be treated as browser state:

- voice profiles, including `source_artist` and `target_user` roles
- uploaded samples and diarized segments
- LoRA adapters, full-model artifacts, and active model selection
- training jobs, training status, checkpoints, and promotion eligibility
- conversion jobs, stem assets, reassembly outputs, and conversion history
- karaoke sessions and live conversion runtime state
- backend audio routing, separation, and pitch configuration

The frontend must fetch these domains through `/api/v1/*` or Socket.IO rather than caching them as local product truth.

## Local-Only Browser Preferences

These keys are intentionally stored in `localStorage`:

| Storage key | Owner | Purpose |
|---|---|---|
| `autovoice_ui_config` | browser | theme, compact mode, advanced-controls visibility, refresh interval |
| `autovoice_notifications` | browser | browser notification toggles, local webhook definitions, enabled event list |
| `autovoice_preferred_pipeline` | browser | default pipeline preference for conversion forms |
| `autovoice_conversion_settings` | browser | last-used conversion form defaults |
| `autovoice_training_settings` | browser | reserved training form defaults |
| `autovoice_audio_settings` | browser | reserved client-side playback preferences |
| `autovoice_recent_profiles` | browser | quick-access recent profile list |
| `autovoice_view_preferences` | browser | layout and browsing preferences |
| `autovoice_debug_settings` | browser | diagnostics presentation preferences |

## Component Boundaries

The current frontend components follow this split:

- `NotificationSettings.tsx`: local-only browser notification preferences and webhook list
- `PipelineSelector.tsx`: local-only preferred default pipeline
- `SystemConfigPanel.tsx`: local-only UI settings, but backend-owned separation, pitch, and audio-router config
- `usePersistedState.ts`: shared hook for local-only browser preferences

## Import and Export Semantics

The System Configuration export includes both backend configuration and local UI preferences. Importing that file:

- updates backend-owned config through the API
- restores browser-owned UI preferences into `localStorage`

This does not change profile data, training history, adapters, or conversion history.
