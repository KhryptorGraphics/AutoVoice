import { io, Socket } from 'socket.io-client'

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1'
export const API_TOKEN_STORAGE_KEY = 'autovoice_api_token'

export function getApiAuthToken(): string | null {
  const envToken = import.meta.env.VITE_AUTOVOICE_API_TOKEN || import.meta.env.VITE_API_TOKEN
  if (typeof envToken === 'string' && envToken.trim()) {
    return envToken.trim()
  }
  if (typeof window === 'undefined') {
    return null
  }
  return window.localStorage.getItem(API_TOKEN_STORAGE_KEY)?.trim() || null
}

export function apiAuthHeaders(headers?: HeadersInit): Headers {
  const nextHeaders = new Headers(headers)
  const token = getApiAuthToken()
  if (token && !nextHeaders.has('Authorization')) {
    nextHeaders.set('Authorization', `Bearer ${token}`)
  }
  return nextHeaders
}

async function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const headers = apiAuthHeaders(init.headers)
  if (!(init.body instanceof FormData) && init.body !== undefined && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  return fetch(input, { ...init, headers })
}

function appendMediaAttestation(formData: FormData): void {
  formData.append('consent_confirmed', 'true')
  formData.append('source_media_policy_confirmed', 'true')
}

function mediaAttestationPayload(): { consent_confirmed: true; source_media_policy_confirmed: true } {
  return {
    consent_confirmed: true,
    source_media_policy_confirmed: true,
  }
}

// WebSocket event types
export type WSEventType =
  | 'conversion_progress'
  | 'conversion_complete'
  | 'conversion_error'
  | 'training_progress'
  | 'training_complete'
  | 'training_error'
  | 'training_paused'
  | 'training_resumed'
  | 'training_cancelled'
  | 'gpu_metrics'
  | 'model_loaded'
  | 'model_unloaded'

export interface WSEvent<T = unknown> {
  type: WSEventType
  timestamp: string
  data: T
}

export interface ConversionProgressEvent {
  job_id: string
  progress: number
  stage: 'separating' | 'encoding' | 'converting' | 'vocoding' | 'mixing'
  message?: string
}

export interface TrainingProgressEvent {
  job_id: string
  profile_id?: string
  epoch: number
  total_epochs: number
  step: number
  total_steps: number
  loss: number
  learning_rate: number
  progress_percent?: number
  gpu_metrics?: {
    available?: boolean
    memory_used_gb?: number
    memory_reserved_gb?: number
    memory_total_gb?: number
    utilization_percent?: number
  }
  quality_metrics?: {
    mos_proxy?: number
    speaker_similarity_proxy?: number
  }
  checkpoint_path?: string | null
  is_paused?: boolean
}

export type WSEventHandler<T = unknown> = (event: WSEvent<T>) => void

export type PipelineType =
  | 'realtime'
  | 'quality'
  | 'quality_seedvc'
  | 'realtime_meanvc'
  | 'quality_shortcut'

export type OfflinePipelineType =
  | 'realtime'
  | 'quality'
  | 'quality_seedvc'
  | 'quality_shortcut'

export type LivePipelineType = 'realtime' | 'realtime_meanvc'

export interface ConversionRecord {
  id: string
  status: 'queued' | 'processing' | 'in_progress' | 'complete' | 'completed' | 'error' | 'failed' | 'cancelled'
  progress?: number
  created_at: string
  started_at?: string
  completed_at?: string
  input_file: string
  profile_id: string
  preset: string
  duration?: number
  error?: string
  // Pipeline and adapter info
  pipeline_type?: PipelineType
  requested_pipeline?: PipelineType
  resolved_pipeline?: PipelineType
  runtime_backend?: 'pytorch' | 'tensorrt' | 'pytorch_full_model' | string
  adapter_type?: 'hq' | 'nvfp4' | 'unified'
  active_model_type?: ActiveModelType
  // Quality metrics
  processing_time_seconds?: number
  rtf?: number  // Real-time factor (processing_time / audio_duration)
  audio_duration_seconds?: number
  // Output URLs
  output_url?: string
  download_url?: string
  stem_urls?: Partial<Record<'vocals' | 'instrumental', string>>
  reassemble_url?: string
  // Additional fields used by ConversionHistoryPage
  timestamp?: Date
  isFavorite?: boolean
  targetVoice?: string
  quality?: string
  originalFileName?: string
  notes?: string
  resultUrl?: string
}

export interface YouTubeHistoryItem {
  id: string
  url: string
  title: string
  mainArtist: string | null
  featuredArtists: string[]
  hasDiarization: boolean
  numSpeakers: number
  timestamp: string
  audioPath: string | null
  filteredPath: string | null
  audioAssetId?: string | null
  filteredAssetId?: string | null
  videoId?: string | null
}

export type YouTubeHistorySaveItem =
  Omit<Partial<YouTubeHistoryItem>, 'audioPath' | 'filteredPath'> & {
    audioAssetId?: string | null
    filteredAssetId?: string | null
  }

export type ProfileRole = 'source_artist' | 'target_user'
export type ActiveModelType = 'base' | 'adapter' | 'full_model'

export interface ReadinessState {
  ready: boolean
  reason: string
  label?: string
  blockers?: string[]
  warnings?: string[]
  sample_count?: number
  clean_vocal_minutes?: number
  clean_vocal_seconds?: number
  remaining_seconds?: number
  remaining_minutes?: number
}

export interface ProfileReadiness {
  training?: ReadinessState
  conversion?: ReadinessState
  live_conversion?: ReadinessState
  diarization?: ReadinessState
}

export interface VoiceProfile {
  profile_id: string
  user_id?: string
  name?: string
  created_at: string
  created_from?: string
  profile_role?: ProfileRole
  sample_count: number
  training_sample_count?: number
  clean_vocal_seconds?: number
  clean_vocal_minutes?: number
  full_model_unlock_seconds?: number
  full_model_remaining_seconds?: number
  full_model_remaining_minutes?: number
  full_model_eligible?: boolean
  model_version?: string
  last_trained?: string
  quality_score?: number
  training_status?: TrainingStatusType
  has_trained_model?: boolean
  has_full_model?: boolean
  has_adapter_model?: boolean
  active_model_type?: ActiveModelType
  selected_adapter?: 'hq' | 'nvfp4' | 'unified' | null
  readiness?: ProfileReadiness
}

// Training status for voice profiles
export type TrainingStatusType = 'pending' | 'training' | 'ready' | 'failed'

export interface TrainingStatus {
  profile_id: string
  has_trained_model: boolean
  training_status: TrainingStatusType
}

// Adapter types for LoRA model selection
export type AdapterType = 'hq' | 'nvfp4'

export interface Adapter {
  type: AdapterType
  path: string
  size_kb: number
  epochs: number
  loss: number | null
  precision: string
  config: Record<string, unknown>
}

export interface AdapterListResponse {
  profile_id: string
  adapters: Adapter[]
  selected: AdapterType | null
  count: number
}

export interface AdapterMetrics {
  epochs: number
  loss: number | null
  precision: string
  trained_on: string | null
  architecture: {
    input_dim: number
    hidden_dim: number
    output_dim: number
    num_layers: number
    lora_rank: number
    lora_alpha: number
  }
  file_size_kb: number
  parameter_count: number
  parameter_count_formatted: string
  file_path: string
  modified_time: string
  performance: {
    relative_quality: string
    relative_speed: string
    memory_estimate_mb: number
  }
}

export interface AdapterMetricsResponse {
  profile_id: string
  profile_name: string
  adapters: Record<AdapterType, AdapterMetrics>
  adapter_count: number
  recommended: AdapterType | null
}

export interface TrainingJob {
  job_id: string
  profile_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at: string
  started_at?: string
  completed_at?: string
  progress: number
  sample_ids: string[]
  rejected_sample_ids?: string[]
  training_quality_summary?: Record<string, unknown>
  error?: string
  is_paused?: boolean
  results?: {
    initial_loss?: number
    final_loss?: number
    loss_curve?: number[]
    artifact_type?: 'adapter' | 'full_model'
    job_type?: 'lora' | 'full_model'
    initialization_mode?: 'scratch' | 'continue'
    resume_source?: 'scratch' | 'artifact' | 'checkpoint'
    resume_checkpoint?: string | null
    artifact_reused?: string | null
    requested_epochs?: number
    resumed_from_epoch?: number | null
    current_loss?: number
    current_epoch?: number
    current_step?: number
    latest_checkpoint?: string
    quality_metrics?: {
      mos_proxy?: number
      speaker_similarity_proxy?: number
    }
    gpu_metrics?: {
      available?: boolean
      memory_used_gb?: number
      memory_reserved_gb?: number
      memory_total_gb?: number
      utilization_percent?: number
    }
  }
}

export interface AppSettings {
  preferred_pipeline: 'realtime' | 'quality'
  preferred_offline_pipeline: OfflinePipelineType
  preferred_live_pipeline: LivePipelineType
  last_updated?: string | null
}

export interface PipelineStatusInfo {
  loaded: boolean
  memory_gb?: number
  latency_target_ms?: number
  sample_rate?: number
  description?: string
}

export interface PipelineStatusResponse {
  status: string
  timestamp: string
  pipelines: Record<string, PipelineStatusInfo>
}

export interface BenchmarkMetricStatus {
  value: number
  target_status: 'pass' | 'fail' | 'n/a' | string
}

export interface BenchmarkPipelineEvidence {
  title: string
  sample_count: number
  fixture_tier?: string
  fixture_suite?: string
  summary: Record<string, BenchmarkMetricStatus>
  source_bundle?: string
}

export interface BenchmarkDashboard {
  generated_at: string
  source_path?: string
  git_sha?: string | null
  git_sha_short?: string | null
  current_git_sha?: string | null
  current_git_sha_short?: string | null
  is_stale?: boolean
  provenance?: {
    schema_version?: number
    generator?: string
    git_sha?: string | null
    source_bundles?: string[]
  }
  target_hardware?: string
  canonical_pipelines?: {
    offline?: string
    live?: string
  }
  pipelines?: Record<string, BenchmarkPipelineEvidence>
  comparisons?: Record<string, {
    canonical_pipeline: string
    meets_or_beats_canonical: boolean
    quality_checks_passed: boolean
    latency_guard_passed: boolean
  }>
  promotable_candidates?: string[]
}

export interface ReleaseEvidence {
  generated_at: string
  source_path?: string
  current_git_sha?: string | null
  current_git_sha_short?: string | null
  is_stale?: boolean
  quality_gate_passed?: boolean
  ready?: boolean
  ready_for_release?: boolean
  status?: string
  git_sha?: string | null
  git_sha_short?: string | null
  output_dir?: string
  target_hardware?: string
  canonical_pipelines?: BenchmarkDashboard['canonical_pipelines']
  pipeline_count?: number
  comparison_count?: number
  promotable_candidates?: string[]
  fixture_tiers?: string[]
  provenance?: BenchmarkDashboard['provenance']
  health_url?: string
  quality_failures?: Array<Record<string, unknown>>
  blockers?: Array<string | Record<string, unknown>>
  artifacts?: Record<string, string>
  executed_lanes?: string[]
  lane_results?: Array<{
    name?: string
    lane?: string
    status?: string
    ok?: boolean
    details?: Record<string, unknown>
    artifacts?: Record<string, string>
    command?: string[]
    stderr?: string
  }>
  preflight_checks?: Array<{
    name?: string
    ok?: boolean
    skipped?: boolean
    stderr?: string
    stdout?: string
  }>
}

export interface TrainingTelemetryResponse {
  job: TrainingJob
  runtime_metrics: {
    epoch?: number
    total_epochs?: number
    step?: number
    total_steps?: number
    loss?: number
    learning_rate?: number
    progress_percent?: number
    gpu_metrics?: TrainingProgressEvent['gpu_metrics']
    quality_metrics?: TrainingProgressEvent['quality_metrics']
    checkpoint_path?: string | null
  }
  preview_available: boolean
  preview_sample_id?: string | null
}

// Full training configuration with all LoRA/EWC parameters
export interface TrainingConfig {
  // Training mode: 'lora' by default, 'full' unlocks after 30 minutes of clean user vocals
  training_mode: 'lora' | 'full'
  // Initialization mode for the chosen training type
  initialization_mode: 'scratch' | 'continue'
  // LoRA configuration (only used when training_mode='lora')
  lora_rank: number
  lora_alpha: number
  lora_dropout: number
  lora_target_modules: string[]
  // Training parameters
  learning_rate: number
  batch_size: number
  epochs: number
  warmup_steps: number
  max_grad_norm: number
  // Operator/runtime controls
  preset_id: string
  device_id: string
  precision: 'fp32' | 'fp16' | 'bf16'
  optimizer: 'adamw' | 'adam'
  weight_decay: number
  adam_beta1: number
  adam_beta2: number
  scheduler: 'exponential' | 'none'
  scheduler_gamma: number
  checkpoint_every_steps: number
  validation_split: number
  early_stopping_patience: number
  early_stopping_min_delta: number
  // EWC configuration (prevent catastrophic forgetting)
  use_ewc: boolean
  ewc_lambda: number
  // Prior preservation
  use_prior_preservation: boolean
  prior_loss_weight: number
}

export interface TrainingPreset {
  id: string
  label: string
  description: string
  requires_full_training?: boolean
  requires_existing_full_model?: boolean
  config: Partial<TrainingConfig>
}

export interface TrainingConfigOptions {
  schema_version: number
  defaults: TrainingConfig
  limits: Record<string, { min: number; max: number }>
  enums: {
    training_mode: Array<'lora' | 'full'>
    initialization_mode: Array<'scratch' | 'continue'>
    precision: Array<'fp32' | 'fp16' | 'bf16'>
    optimizer: Array<'adamw' | 'adam'>
    scheduler: Array<'exponential' | 'none'>
    lora_target_modules: string[]
  }
  presets: TrainingPreset[]
  devices: Array<{ id: string; label: string; available: boolean; reason?: string; memory_total_gb?: number }>
  full_model_unlock_minutes: number
}

// Default training config matching backend defaults
export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  training_mode: 'lora',
  initialization_mode: 'scratch',
  lora_rank: 8,
  lora_alpha: 16,
  lora_dropout: 0.1,
  lora_target_modules: ['q_proj', 'v_proj', 'content_encoder'],
  learning_rate: 1e-4,
  batch_size: 4,
  epochs: 100,
  warmup_steps: 100,
  max_grad_norm: 1.0,
  preset_id: 'custom',
  device_id: 'auto',
  precision: 'fp32',
  optimizer: 'adamw',
  weight_decay: 0.01,
  adam_beta1: 0.8,
  adam_beta2: 0.99,
  scheduler: 'exponential',
  scheduler_gamma: 0.999,
  checkpoint_every_steps: 1000,
  validation_split: 0,
  early_stopping_patience: 0,
  early_stopping_min_delta: 0,
  use_ewc: true,
  ewc_lambda: 1000.0,
  use_prior_preservation: false,
  prior_loss_weight: 0.5,
}

export interface AudioDevice {
  device_id: string
  name: string
  type: 'input' | 'output'
  sample_rate: number
  channels: number
  is_default: boolean
}

export interface DeviceConfig {
  input_device_id?: string
  output_device_id?: string
  sample_rate: number
  buffer_size?: number
}

export interface GPUDevice {
  index: number
  name: string
  memory_used: number
  memory_total: number
  memory_free?: number
  memory_used_gb?: number
  memory_total_gb?: number
  memory_free_gb?: number
  utilization?: number
  utilization_percent?: number | null
  temperature?: number
  temperature_c?: number | null
}

// Full GPU metrics response from /api/v1/gpu/metrics
export interface GPUMetrics {
  available: boolean
  device_count: number
  devices: GPUDevice[]
  note?: string
}

// Conversion configuration for song conversion
export type QualityPreset = 'draft' | 'fast' | 'balanced' | 'high' | 'studio'
export type EncoderBackend = 'hubert' | 'contentvec'
export type VocoderType = 'hifigan' | 'bigvgan'

export interface ConversionConfig {
  pipeline_type: OfflinePipelineType
  vocal_volume: number
  instrumental_volume: number
  pitch_shift: number
  preset: QualityPreset
  return_stems: boolean
  preserve_techniques: boolean
  encoder_backend: EncoderBackend
  vocoder_type: VocoderType
}

export const DEFAULT_CONVERSION_CONFIG: ConversionConfig = {
  pipeline_type: 'quality_seedvc',
  vocal_volume: 1.0,
  instrumental_volume: 0.9,
  pitch_shift: 0.0,
  preset: 'balanced',
  return_stems: false,
  preserve_techniques: true,
  encoder_backend: 'hubert',
  vocoder_type: 'hifigan',
}

// Quality preset details matching backend PRESETS
export const QUALITY_PRESETS: Record<QualityPreset, { n_steps: number; denoise: number; label: string }> = {
  draft: { n_steps: 10, denoise: 0.3, label: 'Draft (Fast)' },
  fast: { n_steps: 20, denoise: 0.5, label: 'Fast' },
  balanced: { n_steps: 50, denoise: 0.7, label: 'Balanced' },
  high: { n_steps: 100, denoise: 0.8, label: 'High Quality' },
  studio: { n_steps: 200, denoise: 0.9, label: 'Studio' },
}

// Training sample for voice profile
export interface TrainingSample {
  id: string
  profile_id: string
  profile_name?: string
  profile_role?: ProfileRole
  audio_path: string
  duration_seconds: number
  sample_rate: number
  created: string
  metadata?: Record<string, unknown>
  quality_metadata?: Record<string, unknown>
  extra_metadata?: Record<string, unknown>
  source?: string
  consent_status?: string
  rms_loudness?: number | null
  silence_ratio?: number | null
  clipping_ratio?: number | null
  trainable?: boolean
  quality_status?: 'pass' | 'warn' | 'fail' | 'unknown' | string
  issues?: string[]
  recommendations?: string[]
}

export interface SampleReviewResponse {
  samples: TrainingSample[]
  count: number
  summary: {
    trainable: number
    blocked: number
    quality_status_counts: Record<string, number>
  }
}

export interface DuplicateProfileCandidate {
  profile_id: string
  name: string
  profile_role?: ProfileRole
  similarity: number
}

export interface DuplicateProfileCheck {
  profile_id: string
  threshold: number
  candidates: DuplicateProfileCandidate[]
  duplicate_warning: boolean
}

export interface LocalProductionReadiness {
  ready: boolean
  git_sha?: string | null
  release_evidence_available: boolean
  checks: Array<{ id: string; label: string; ok: boolean }>
  commands: Record<string, string>
  paths: Record<string, string>
}

export interface BackupManifest {
  version: number
  git_sha?: string | null
  created_at: string
  included_paths: string[]
  files: Array<{ path: string; size_bytes: number; sha256: string }>
  checksums: Record<string, string>
  restore_warnings: string[]
}

export interface BackupExportResponse {
  status: string
  backup_path: string
  download_url?: string
  manifest: BackupManifest
}

export interface BackupImportResponse {
  status: 'dry_run' | 'applied' | string
  dry_run: boolean
  manifest: BackupManifest
  restore_warnings: string[]
  restored_paths: string[]
  restored_count: number
}

// Quality metrics for completed conversions
export interface QualityMetrics {
  pitch_accuracy: {
    rmse_hz: number
    correlation: number
    mean_error_cents: number
  }
  speaker_similarity: {
    cosine_similarity: number
    embedding_distance: number
  }
  naturalness: {
    spectral_distortion: number
    mos_estimate: number
  }
  intelligibility?: {
    stoi: number
    pesq: number
  }
}

// Audio router configuration for karaoke dual-channel output
export interface AudioRouterConfig {
  speaker_gain: number
  headphone_gain: number
  voice_gain: number
  instrumental_gain: number
  speaker_enabled: boolean
  headphone_enabled: boolean
  speaker_device: number | null
  headphone_device: number | null
  sample_rate: number
}

export const DEFAULT_AUDIO_ROUTER_CONFIG: AudioRouterConfig = {
  speaker_gain: 1.0,
  headphone_gain: 1.0,
  voice_gain: 1.0,
  instrumental_gain: 0.8,
  speaker_enabled: true,
  headphone_enabled: true,
  speaker_device: null,
  headphone_device: null,
  sample_rate: 24000,
}

// Loaded model info
export interface LoadedModel {
  type: string
  model_type: string
  name: string
  path?: string
  loaded?: boolean
  runtime_backend?: string
  device?: string
  status?: string
  memory_usage?: number
  memory_mb?: number
  loaded_at?: string
  source?: string
  artifact_path?: string
}

// TensorRT engine status
export interface TensorRTStatus {
  available: boolean
  runtime_available?: boolean
  runtime_version?: string
  runtime_error?: string | null
  engines_available?: boolean
  cuda_available?: boolean
  version?: string
  engines: {
    name: string
    path?: string
    directory?: string
    model?: string
    precision: 'fp32' | 'fp16' | 'int8'
    built_at?: string
    input_shape?: number[]
    optimized?: boolean
    suite_complete?: boolean
  }[]
  build_in_progress?: boolean
}

// User preset for saving conversion configurations
export interface UserPreset {
  id: string
  name: string
  config: Partial<ConversionConfig>
  created_at: string
  updated_at?: string
}

export interface KaraokePreflightResponse {
  ok: boolean
  issues: string[]
  warnings: string[]
  checks: {
    song_ready: boolean
    assets_ready: boolean
    pipeline_valid: boolean
    profile_ready: boolean
    voice_model_ready: boolean
    routing_ready: boolean
  }
  requested_pipeline: LivePipelineType
  active_model_type?: ActiveModelType | string | null
  audio_router_targets: {
    speaker_device: number | null
    headphone_device: number | null
  }
}

// Model checkpoint for version control
export interface Checkpoint {
  id: string
  profile_id: string
  version: string
  created_at: string
  epochs_trained: number
  final_loss: number
  is_active: boolean
  file_size_mb: number
  training_samples: number
  notes?: string
}

// Separation configuration for Demucs vocal isolation
export interface SeparationConfig {
  model: 'htdemucs' | 'htdemucs_ft' | 'mdx_extra'
  stems: ('vocals' | 'drums' | 'bass' | 'other')[]
  shifts: number // Number of random shifts for prediction (higher = better quality, slower)
  overlap: number // Overlap between chunks (0.0-0.99)
  segment_length: number | null // Segment length in seconds (null = full track)
  device: 'cuda' | 'cpu'
}

export const DEFAULT_SEPARATION_CONFIG: SeparationConfig = {
  model: 'htdemucs',
  stems: ['vocals'],
  shifts: 1,
  overlap: 0.25,
  segment_length: null,
  device: 'cuda',
}

// Pitch extraction configuration for CREPE/RMVPE
export interface PitchConfig {
  method: 'crepe' | 'rmvpe' | 'harvest' | 'dio'
  hop_length: number
  f0_min: number
  f0_max: number
  threshold: number // Confidence threshold for pitch detection
  use_gpu: boolean
}

export const DEFAULT_PITCH_CONFIG: PitchConfig = {
  method: 'rmvpe',
  hop_length: 160,
  f0_min: 50,
  f0_max: 1100,
  threshold: 0.3,
  use_gpu: true,
}

// Conversion job response (from POST /convert/song)
export interface ConversionJobResponse {
  status: 'queued' | 'processing' | 'success' | 'completed'
  job_id: string
  websocket_room?: string
  message?: string
  active_model_type?: ActiveModelType
  adapter_type?: 'hq' | 'nvfp4' | 'unified'
  stem_urls?: Partial<Record<'vocals' | 'instrumental', string>>
  reassemble_url?: string
  // When sync processing is used, result may be inline
  output_url?: string
  download_url?: string
  audio?: string
  format?: string
  sample_rate?: number
  duration?: number
  quality_metrics?: QualityMetrics
  requested_pipeline?: PipelineType
  resolved_pipeline?: PipelineType
  runtime_backend?: string
}

// Extended conversion status (from GET /convert/status/{job_id})
export interface ConversionStatusExtended extends ConversionRecord {
  // Timing info
  started_at?: string
  processing_time_seconds?: number
  // Quality metrics
  rtf?: number
  audio_duration_seconds?: number
  // Pipeline info used
  pipeline_type?: 'realtime' | 'quality' | 'quality_seedvc' | 'realtime_meanvc' | 'quality_shortcut'
  adapter_type?: 'hq' | 'nvfp4' | 'unified'
  active_model_type?: ActiveModelType
}

export type ConversionWorkflowStatus =
  | 'queued'
  | 'processing'
  | 'awaiting_review'
  | 'ready_for_training'
  | 'training_in_progress'
  | 'ready_for_conversion'
  | 'error'

export interface ConversionWorkflowResolvedProfile {
  profile_id: string
  name: string
  profile_role?: ProfileRole
  has_trained_model?: boolean
  active_model_type?: ActiveModelType
  sample_count?: number
  clean_vocal_minutes?: number
}

export interface ConversionWorkflowDiarizationState {
  status?: string
  diarization_id?: string | null
  num_speakers?: number
  dominant_speaker_id?: string | null
  speaker_assignments?: Array<{
    speaker_id: string
    duration_seconds?: number
    segment_count?: number
    resolved_profile_id?: string
    resolution?: string
    sample_paths?: string[]
  }>
}

export interface ConversionWorkflowCandidatePayload {
  role: ProfileRole
  speaker_id?: string
  duration_seconds?: number
  name?: string
  sample_paths?: string[]
  source_files?: string[]
}

export interface ConversionWorkflowReviewItem {
  review_id: string
  role: ProfileRole
  reason: string
  suggested_match?: {
    profile_id: string
    name: string
    profile_role?: ProfileRole
    similarity: number
  } | null
  candidate: ConversionWorkflowCandidatePayload
}

export interface ConversionWorkflowResolvedSourceProfile {
  profile_id: string
  name: string
  profile_role?: ProfileRole
  speaker_id?: string
  duration_seconds?: number
  status?: 'matched' | 'created' | 'review_required'
  suggested_match?: {
    profile_id: string
    name: string
    similarity: number
  } | null
}

export interface ConversionWorkflow {
  workflow_id: string
  status: ConversionWorkflowStatus
  stage: string
  progress: number
  artist_song: {
    filename: string
    path?: string
  }
  user_vocals: Array<{
    filename: string
    path?: string
  }>
  artist_vocals_path?: string | null
  instrumental_path?: string | null
  diarization_id?: string | null
  resolved_source_profiles: ConversionWorkflowResolvedSourceProfile[]
  resolved_target_profile_id?: string | null
  resolved_target_profile?: ConversionWorkflowResolvedProfile | null
  review_items: ConversionWorkflowReviewItem[]
  training_readiness: {
    ready: boolean
    reason: string
    sample_count?: number
    clean_vocal_minutes?: number
    blockers?: string[]
    warnings?: string[]
  }
  conversion_readiness: {
    ready: boolean
    reason: string
    blockers?: string[]
    warnings?: string[]
  }
  user_analysis?: Record<string, unknown>
  artist_analysis?: ConversionWorkflowDiarizationState | Record<string, unknown>
  readiness?: {
    training?: ReadinessState
    conversion?: ReadinessState
    live_conversion?: ReadinessState
  }
  current_training_job_id?: string | null
  created_at: string
  updated_at: string
  error?: string | null
}

// CUDA kernel metrics
export interface KernelMetric {
  name: string
  calls: number
  total_time_ms: number
  avg_time_ms: number
  min_time_ms?: number
  max_time_ms?: number
}

export interface HealthStatus {
  status: string
  components: Record<string, { status: string; details?: string }>
  uptime: number
  version: string
}

export interface SystemInfo {
  device: string
  cuda_available: boolean
  gpu_count: number
  python_version: string
  torch_version: string
  // Additional fields used by GPUMonitor
  gpu_available?: boolean
  gpu_name?: string
  gpu_utilization?: number
  gpu_memory_used?: number
  gpu_memory_total?: number
  gpu_temperature?: number
  model_loaded?: boolean
  models?: string[]
  status?: string
}

// Custom error classes for better error handling
export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public code?: string,
    public details?: Record<string, unknown>
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export class ConversionError extends ApiError {
  constructor(
    message: string,
    statusCode: number,
    public errorType: 'missing_adapter' | 'invalid_profile' | 'pipeline_error' | 'audio_error' | 'unknown',
    details?: Record<string, unknown>
  ) {
    super(message, statusCode, errorType, details)
    this.name = 'ConversionError'
  }

  static fromResponse(error: { error: string; code?: string; details?: Record<string, unknown> }, status: number): ConversionError {
    const message = error.error || 'Conversion failed'
    let errorType: ConversionError['errorType'] = 'unknown'

    // Detect error type from message or code
    if (message.includes('adapter') || message.includes('model not found')) {
      errorType = 'missing_adapter'
    } else if (message.includes('profile') || message.includes('Profile not found')) {
      errorType = 'invalid_profile'
    } else if (message.includes('pipeline') || message.includes('Pipeline')) {
      errorType = 'pipeline_error'
    } else if (message.includes('audio') || message.includes('Audio')) {
      errorType = 'audio_error'
    }

    return new ConversionError(message, status, errorType, error.details)
  }
}

class ApiService {
  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await apiFetch(`${API_BASE}${path}`, {
      ...options,
    })
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: response.statusText }))
      throw new ApiError(
        error.error || `HTTP ${response.status}`,
        response.status,
        error.code,
        error.details
      )
    }

    if (response.status === 204) {
      return undefined as T
    }

    const contentType = response.headers.get('content-type') ?? ''
    if (!contentType.includes('application/json')) {
      return undefined as T
    }

    const text = await response.text()
    return (text ? JSON.parse(text) : undefined) as T
  }

  async getHealth(): Promise<HealthStatus> {
    return this.request('/health')
  }

  async getSystemInfo(): Promise<SystemInfo> {
    return this.request('/system/info')
  }

  // Aliases for SystemStatusPage compatibility
  // Transforms nested backend response to flat structure expected by GPUMonitor
  async getSystemStatus(): Promise<SystemInfo> {
    const raw = await this.request<any>('/system/info')
    return {
      // Core fields
      device: raw.system?.platform ?? 'Unknown',
      cuda_available: raw.torch?.cuda_available ?? false,
      gpu_count: raw.torch?.device_count ?? 0,
      python_version: raw.system?.python_version ?? 'Unknown',
      torch_version: raw.torch?.version ?? 'Unknown',
      // GPUMonitor fields - transform nested torch data to flat structure
      gpu_available: raw.torch?.cuda_available ?? false,
      gpu_name: raw.torch?.device_name,
      status: raw.torch?.cuda_available ? 'ready' : 'no_gpu',
    }
  }

  async getModelsInfo(): Promise<{ models: LoadedModel[] }> {
    const models = await this.getLoadedModels()
    return { models }
  }

  async healthCheck(): Promise<{ status: string; gpu_available: boolean; models_loaded: boolean; uptime: number }> {
    const [health, models] = await Promise.all([this.getHealth(), this.getLoadedModels()])
    // Check if torch component details indicate CUDA is available
    const torchDetails = health.components?.torch?.details ?? ''
    const gpuAvailable = torchDetails.toLowerCase().includes('cuda') || health.components?.torch?.status === 'up'
    return {
      status: health.status,
      gpu_available: gpuAvailable,
      models_loaded: models.some((model) => model.loaded),
      uptime: health.uptime ?? 0,
    }
  }

  async getGPUMetrics(): Promise<GPUMetrics> {
    return this.request('/gpu/metrics')
  }

  async getPipelineStatus(): Promise<PipelineStatusResponse> {
    return this.request('/pipelines/status')
  }

  async getLatestBenchmarkDashboard(): Promise<BenchmarkDashboard> {
    return this.request('/reports/benchmarks/latest')
  }

  async getLatestReleaseEvidence(): Promise<ReleaseEvidence> {
    return this.request('/reports/release-evidence/latest')
  }

  async getLocalProductionReadiness(): Promise<LocalProductionReadiness> {
    return this.request('/readiness/local-production')
  }

  async getAppSettings(): Promise<AppSettings> {
    return this.request('/settings/app')
  }

  async updateAppSettings(settings: Partial<AppSettings>): Promise<AppSettings> {
    return this.request('/settings/app', {
      method: 'PATCH',
      body: JSON.stringify(settings),
    })
  }

  async karaokePreflight(payload: {
    song_id?: string
    profile_id?: string | null
    voice_model_id?: string | null
    pipeline_type: LivePipelineType
  }): Promise<KaraokePreflightResponse> {
    return this.request('/karaoke/preflight', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  }

  async getKernelMetrics(): Promise<KernelMetric[]> {
    const response = await this.request<{ kernels?: KernelMetric[]; note?: string }>('/kernels/metrics')
    return response.kernels ?? []
  }

  async getConversionMetrics(jobId: string): Promise<QualityMetrics> {
    return this.request(`/convert/metrics/${jobId}`)
  }

  async createVoiceProfile(audioFile: File, name?: string, userId?: string): Promise<VoiceProfile> {
    const formData = new FormData()
    formData.append('reference_audio', audioFile)
    appendMediaAttestation(formData)
    if (name) formData.append('name', name)
    if (userId) formData.append('user_id', userId)

    const response = await apiFetch(`${API_BASE}/voice/clone`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) throw new Error(`Upload failed: ${response.status}`)
    return response.json()
  }

  async listProfiles(userId?: string): Promise<VoiceProfile[]> {
    const params = userId ? `?user_id=${userId}` : ''
    return this.request(`/voice/profiles${params}`)
  }

  async checkDuplicateProfiles(profileId: string, threshold = 0.82): Promise<DuplicateProfileCheck> {
    return this.request(`/voice/profiles/duplicate-check?profile_id=${encodeURIComponent(profileId)}&threshold=${threshold}`)
  }

  async deleteProfile(profileId: string): Promise<void> {
    await this.request(`/voice/profiles/${profileId}`, { method: 'DELETE' })
  }

  async getProfileDetails(profileId: string): Promise<VoiceProfile & { training_history: TrainingJob[] }> {
    return this.request(`/voice/profiles/${profileId}`)
  }

  async renameProfile(profileId: string, name: string): Promise<VoiceProfile> {
    return this.request(`/voice/profiles/${profileId}`, {
      method: 'PATCH',
      body: JSON.stringify({ name }),
    })
  }

  async getTrainingStatus(profileId: string): Promise<TrainingStatus> {
    return this.request(`/voice/profiles/${profileId}/training-status`)
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Adapter Methods (LoRA model selection)
  // ─────────────────────────────────────────────────────────────────────────

  async getProfileAdapters(profileId: string): Promise<AdapterListResponse> {
    return this.request(`/voice/profiles/${profileId}/adapters`)
  }

  async selectAdapter(profileId: string, adapterType: AdapterType): Promise<{ status: string; selected: AdapterType }> {
    return this.request(`/voice/profiles/${profileId}/adapter/select`, {
      method: 'POST',
      body: JSON.stringify({ adapter_type: adapterType }),
    })
  }

  async getAdapterMetrics(profileId: string): Promise<AdapterMetricsResponse> {
    return this.request(`/voice/profiles/${profileId}/adapter/metrics`)
  }

  // Training Jobs
  async listTrainingJobs(profileId?: string): Promise<TrainingJob[]> {
    const params = profileId ? `?profile_id=${profileId}` : ''
    return this.request(`/training/jobs${params}`)
  }

  async getTrainingJob(jobId: string): Promise<TrainingJob> {
    return this.request(`/training/jobs/${jobId}`)
  }

  async getTrainingConfigOptions(): Promise<TrainingConfigOptions> {
    return this.request('/training/config-options')
  }

  async createTrainingJob(profileId: string, sampleIds: string[], config?: Partial<TrainingConfig>): Promise<TrainingJob> {
    return this.request('/training/jobs', {
      method: 'POST',
      body: JSON.stringify({ profile_id: profileId, sample_ids: sampleIds, config }),
    })
  }

  async cancelTrainingJob(jobId: string): Promise<void> {
    await this.request(`/training/jobs/${jobId}/cancel`, { method: 'POST' })
  }

  async pauseTrainingJob(jobId: string): Promise<TrainingJob> {
    return this.request(`/training/jobs/${jobId}/pause`, { method: 'POST' })
  }

  async resumeTrainingJob(jobId: string): Promise<TrainingJob> {
    return this.request(`/training/jobs/${jobId}/resume`, { method: 'POST' })
  }

  async getTrainingTelemetry(jobId: string): Promise<TrainingTelemetryResponse> {
    return this.request(`/training/jobs/${jobId}/telemetry`)
  }

  async getTrainingPreview(
    jobId: string,
    payload?: { profile_id?: string; sample_id?: string; duration_seconds?: number; offset_seconds?: number }
  ): Promise<Blob> {
    const response = await apiFetch(`${API_BASE}/training/preview/${jobId}`, {
      method: 'POST',
      body: JSON.stringify(payload ?? {}),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: response.statusText }))
      throw new ApiError(
        error.error || `HTTP ${response.status}`,
        response.status,
        error.code,
        error.details
      )
    }

    return response.blob()
  }

  // Audio Devices
  async listAudioDevices(): Promise<AudioDevice[]> {
    return this.request('/devices/list')
  }

  async getDeviceConfig(): Promise<DeviceConfig> {
    return this.request('/devices/config')
  }

  async setDeviceConfig(config: Partial<DeviceConfig>): Promise<DeviceConfig> {
    return this.request('/devices/config', {
      method: 'POST',
      body: JSON.stringify(config),
    })
  }

  async convertSong(
    audioFile: File,
    profileId: string,
    settings?: {
      preset?: string
      vocal_volume?: number
      instrumental_volume?: number
      pitch_shift?: number
      pipeline_type?: OfflinePipelineType
      adapter_type?: 'hq' | 'nvfp4' | 'unified'
      return_stems?: boolean
    }
  ): Promise<ConversionJobResponse> {
    const formData = new FormData()
    formData.append('audio', audioFile)
    formData.append('profile_id', profileId)
    appendMediaAttestation(formData)
    if (settings?.preset) formData.append('preset', settings.preset)
    if (settings?.vocal_volume != null) formData.append('vocal_volume', String(settings.vocal_volume))
    if (settings?.instrumental_volume != null) formData.append('instrumental_volume', String(settings.instrumental_volume))
    if (settings?.pitch_shift != null) formData.append('pitch_shift', String(settings.pitch_shift))
    if (settings?.pipeline_type) formData.append('pipeline_type', settings.pipeline_type)
    if (settings?.adapter_type) formData.append('adapter_type', settings.adapter_type)
    if (settings?.return_stems != null) formData.append('return_stems', String(settings.return_stems))

    const response = await apiFetch(`${API_BASE}/convert/song`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: `Conversion failed: ${response.status}` }))
      throw ConversionError.fromResponse(errorData, response.status)
    }

    return response.json()
  }

  async listConversionWorkflows(): Promise<ConversionWorkflow[]> {
    return this.request('/convert/workflows')
  }

  async getConversionWorkflow(workflowId: string): Promise<ConversionWorkflow> {
    return this.request(`/convert/workflows/${workflowId}`)
  }

  async createConversionWorkflow(
    artistSong: File,
    userVocalFiles: File[],
    options?: {
      target_profile_id?: string | null
      dominant_source_profile_id?: string | null
    }
  ): Promise<ConversionWorkflow> {
    const formData = new FormData()
    formData.append('artist_song', artistSong)
    appendMediaAttestation(formData)
    userVocalFiles.forEach((file) => formData.append('user_vocals', file))
    if (options?.target_profile_id) {
      formData.append('target_profile_id', options.target_profile_id)
    }
    if (options?.dominant_source_profile_id) {
      formData.append('dominant_source_profile_id', options.dominant_source_profile_id)
    }

    const response = await apiFetch(`${API_BASE}/convert/workflows`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: `Workflow creation failed: ${response.status}` }))
      throw new ApiError(
        error.error || `HTTP ${response.status}`,
        response.status,
        error.code,
        error.details
      )
    }
    return response.json()
  }

  async resolveConversionWorkflowMatch(
    workflowId: string,
    payload: {
      review_id: string
      resolution: 'use_suggested' | 'use_existing' | 'create_new'
      profile_id?: string
      name?: string
    }
  ): Promise<ConversionWorkflow> {
    return this.request(`/convert/workflows/${workflowId}/resolve-match`, {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  }

  async attachConversionWorkflowTrainingJob(workflowId: string, jobId: string): Promise<ConversionWorkflow> {
    return this.request(`/convert/workflows/${workflowId}/training-job`, {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId }),
    })
  }

  async convertWorkflow(
    workflowId: string,
    settings?: {
      preset?: string
      vocal_volume?: number
      instrumental_volume?: number
      pitch_shift?: number
      pipeline_type?: OfflinePipelineType
      adapter_type?: 'hq' | 'nvfp4' | 'unified'
      return_stems?: boolean
    }
  ): Promise<ConversionJobResponse> {
    return this.request(`/convert/workflows/${workflowId}/convert`, {
      method: 'POST',
      body: JSON.stringify(settings || {}),
    })
  }

  async getConversionStatus(jobId: string): Promise<ConversionRecord> {
    const status = await this.request<ConversionRecord>(`/convert/status/${jobId}`)
    return { ...status, id: (status as ConversionRecord).id ?? jobId }
  }

  async cancelConversion(jobId: string): Promise<void> {
    await this.request(`/convert/cancel/${jobId}`, { method: 'POST' })
  }

  async downloadResult(jobId: string): Promise<Blob> {
    const response = await apiFetch(`${API_BASE}/convert/download/${jobId}`)
    if (!response.ok) throw new Error(`Download failed: ${response.status}`)
    return response.blob()
  }

  async downloadConversionAsset(
    jobId: string,
    variant: 'mix' | 'vocals' | 'instrumental'
  ): Promise<Blob> {
    const response = await apiFetch(`${API_BASE}/convert/download/${jobId}?variant=${variant}`)
    if (!response.ok) throw new Error(`Download failed: ${response.status}`)
    return response.blob()
  }

  async reassembleConversion(jobId: string): Promise<Blob> {
    const response = await apiFetch(`${API_BASE}/convert/reassemble/${jobId}`)
    if (!response.ok) throw new Error(`Reassembly failed: ${response.status}`)
    return response.blob()
  }

  // Training Samples
  async uploadSample(profileId: string, audioFile: File, metadata?: Record<string, unknown>): Promise<TrainingSample> {
    const formData = new FormData()
    formData.append('audio', audioFile)
    appendMediaAttestation(formData)
    if (metadata) formData.append('metadata', JSON.stringify(metadata))

    const response = await apiFetch(`${API_BASE}/profiles/${profileId}/samples`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) throw new Error(`Upload failed: ${response.status}`)
    return response.json()
  }

  async listSamples(profileId: string): Promise<TrainingSample[]> {
    return this.request(`/profiles/${profileId}/samples`)
  }

  async listSampleReview(filters: { quality_status?: string; trainable?: boolean } = {}): Promise<SampleReviewResponse> {
    const params = new URLSearchParams()
    if (filters.quality_status) params.set('quality_status', filters.quality_status)
    if (typeof filters.trainable === 'boolean') params.set('trainable', String(filters.trainable))
    const query = params.toString()
    return this.request(`/samples/review${query ? `?${query}` : ''}`)
  }

  async getSample(profileId: string, sampleId: string): Promise<TrainingSample> {
    return this.request(`/profiles/${profileId}/samples/${sampleId}`)
  }

  async deleteSample(profileId: string, sampleId: string): Promise<void> {
    await this.request(`/profiles/${profileId}/samples/${sampleId}`, { method: 'DELETE' })
  }

  // Song Upload with Auto-Split (Demucs vocal separation)
  async uploadSongWithSplit(
    profileId: string,
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<{ job_id: string; song_id?: string }> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('profile_id', profileId)
    formData.append('auto_split', 'true')
    appendMediaAttestation(formData)

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = Math.round((event.loaded / event.total) * 100)
          onProgress(progress)
        }
      })

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText)
            resolve(response)
          } catch {
            reject(new Error('Invalid response format'))
          }
        } else {
          try {
            const error = JSON.parse(xhr.responseText)
            reject(new Error(error.error || `Upload failed: ${xhr.status}`))
          } catch {
            reject(new Error(`Upload failed: ${xhr.status}`))
          }
        }
      })

      xhr.addEventListener('error', () => {
        reject(new Error('Network error during upload'))
      })

      xhr.addEventListener('abort', () => {
        reject(new Error('Upload aborted'))
      })

      xhr.open('POST', `${API_BASE}/profiles/${profileId}/songs`)
      const token = getApiAuthToken()
      if (token) {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`)
      }
      xhr.send(formData)
    })
  }

  async getSeparationStatus(jobId: string): Promise<{
    status: 'pending' | 'processing' | 'complete' | 'error'
    progress: number
    message?: string
    error?: string
    vocals_path?: string
    instrumental_path?: string
  }> {
    return this.request(`/separation/status/${jobId}`)
  }

  // User Presets
  async listPresets(): Promise<UserPreset[]> {
    return this.request('/presets')
  }

  async getPreset(presetId: string): Promise<UserPreset> {
    return this.request(`/presets/${presetId}`)
  }

  async savePreset(name: string, config: Partial<ConversionConfig>): Promise<UserPreset> {
    return this.request('/presets', {
      method: 'POST',
      body: JSON.stringify({ name, config }),
    })
  }

  async updatePreset(presetId: string, updates: { name?: string; config?: Partial<ConversionConfig> }): Promise<UserPreset> {
    return this.request(`/presets/${presetId}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    })
  }

  async deletePreset(presetId: string): Promise<void> {
    await this.request(`/presets/${presetId}`, { method: 'DELETE' })
  }

  // Model Management
  async getLoadedModels(): Promise<LoadedModel[]> {
    const response = await this.request<{ models: LoadedModel[] }>('/models/loaded')
    return response.models.map((model) => ({
      ...model,
      type: model.type ?? model.model_type,
      memory_mb: model.memory_mb ?? ((model.memory_usage ?? 0) / (1024 * 1024)),
    }))
  }

  async loadModel(modelType: string, path?: string): Promise<LoadedModel> {
    return this.request('/models/load', {
      method: 'POST',
      body: JSON.stringify({ model_type: modelType, path }),
    })
  }

  async unloadModel(modelType: string): Promise<void> {
    await this.request('/models/unload', {
      method: 'POST',
      body: JSON.stringify({ model_type: modelType }),
    })
  }

  async rebuildTensorRT(precision: 'fp32' | 'fp16' | 'int8' = 'fp16'): Promise<{ status: string; duration_seconds: number }> {
    return this.request('/models/tensorrt/rebuild', {
      method: 'POST',
      body: JSON.stringify({ precision }),
    })
  }

  async getTensorRTStatus(): Promise<TensorRTStatus> {
    return this.request('/models/tensorrt/status')
  }

  async buildTensorRTEngines(
    models: string[],
    precision: 'fp32' | 'fp16' | 'int8'
  ): Promise<{ status: string; engines_built: string[] }> {
    return this.request('/models/tensorrt/build', {
      method: 'POST',
      body: JSON.stringify({ models, precision }),
    })
  }

  // Separation Configuration
  async getSeparationConfig(): Promise<SeparationConfig> {
    return this.request('/config/separation')
  }

  async updateSeparationConfig(config: Partial<SeparationConfig>): Promise<SeparationConfig> {
    return this.request('/config/separation', {
      method: 'PATCH',
      body: JSON.stringify(config),
    })
  }

  // Pitch Extraction Configuration
  async getPitchConfig(): Promise<PitchConfig> {
    return this.request('/config/pitch')
  }

  async updatePitchConfig(config: Partial<PitchConfig>): Promise<PitchConfig> {
    return this.request('/config/pitch', {
      method: 'PATCH',
      body: JSON.stringify(config),
    })
  }

  // Audio Router Configuration (for karaoke dual-channel)
  async getAudioRouterConfig(): Promise<AudioRouterConfig> {
    return this.request('/audio/router/config')
  }

  async updateAudioRouterConfig(config: Partial<AudioRouterConfig>): Promise<AudioRouterConfig> {
    return this.request('/audio/router/config', {
      method: 'PATCH',
      body: JSON.stringify(config),
    })
  }

  // Conversion History
  async getConversionHistory(profileId?: string): Promise<ConversionRecord[]> {
    const params = profileId ? `?profile_id=${profileId}` : ''
    return this.request(`/convert/history${params}`)
  }

  async deleteConversionRecord(id: string): Promise<void> {
    await this.request(`/convert/history/${id}`, { method: 'DELETE' })
  }

  async updateConversionRecord(id: string, updates: Partial<ConversionRecord>): Promise<ConversionRecord> {
    return this.request(`/convert/history/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    })
  }

  // Checkpoints
  async getCheckpoints(profileId: string): Promise<Checkpoint[]> {
    return this.request(`/profiles/${profileId}/checkpoints`)
  }

  async rollbackToCheckpoint(profileId: string, checkpointId: string): Promise<void> {
    await this.request(`/profiles/${profileId}/checkpoints/${checkpointId}/rollback`, {
      method: 'POST',
    })
  }

  async deleteCheckpoint(profileId: string, checkpointId: string): Promise<void> {
    await this.request(`/profiles/${profileId}/checkpoints/${checkpointId}`, {
      method: 'DELETE',
    })
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Diarization Methods
  // ─────────────────────────────────────────────────────────────────────────

  async diarizeAudio(file: File, numSpeakers?: number): Promise<DiarizationResult> {
    const formData = new FormData()
    formData.append('file', file)
    if (numSpeakers) {
      formData.append('num_speakers', numSpeakers.toString())
    }

    const response = await apiFetch(`${API_BASE}/audio/diarize`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.error || 'Diarization failed')
    }

    return response.json()
  }

  async diarizeAudioPath(audioPath: string, numSpeakers?: number): Promise<DiarizationResult> {
    return this.request<DiarizationResult>('/audio/diarize', {
      method: 'POST',
      body: JSON.stringify({ audio_path: audioPath, num_speakers: numSpeakers }),
    })
  }

  async assignDiarizationSegment(
    diarizationId: string,
    segmentIndex: number,
    profileId: string,
    extractAudio: boolean = true
  ): Promise<AssignSegmentResult> {
    return this.request<AssignSegmentResult>('/audio/diarize/assign', {
      method: 'POST',
      body: JSON.stringify({
        diarization_id: diarizationId,
        segment_index: segmentIndex,
        profile_id: profileId,
        extract_audio: extractAudio,
      }),
    })
  }

  async getProfileSegments(profileId: string): Promise<ProfileSegmentsResult> {
    return this.request<ProfileSegmentsResult>(`/profiles/${profileId}/segments`)
  }

  async autoCreateProfileFromDiarization(
    diarizationId: string,
    speakerId: string,
    name: string,
    userId?: string,
    extractSegments: boolean = true,
    options?: {
      profileRole?: ProfileRole
      metadata?: Record<string, unknown>
    }
  ): Promise<AutoCreateProfileResult> {
    return this.request<AutoCreateProfileResult>('/profiles/auto-create', {
      method: 'POST',
      body: JSON.stringify({
        diarization_id: diarizationId,
        speaker_id: speakerId,
        name,
        user_id: userId,
        extract_segments: extractSegments,
        profile_role: options?.profileRole,
        metadata: options?.metadata,
        ...mediaAttestationPayload(),
      }),
    })
  }

  async setSpeakerEmbedding(profileId: string, audioPath?: string, useSamples?: boolean): Promise<SpeakerEmbeddingResult> {
    return this.request<SpeakerEmbeddingResult>(`/profiles/${profileId}/speaker-embedding`, {
      method: 'POST',
      body: JSON.stringify({
        audio_path: audioPath,
        use_samples: useSamples,
      }),
    })
  }

  async getSpeakerEmbedding(profileId: string): Promise<SpeakerEmbeddingStatus> {
    return this.request<SpeakerEmbeddingStatus>(`/profiles/${profileId}/speaker-embedding`)
  }

  async filterSample(
    profileId: string,
    sampleId: string,
    similarityThreshold: number = 0.7
  ): Promise<FilterSampleResult> {
    return this.request<FilterSampleResult>(`/profiles/${profileId}/samples/${sampleId}/filter`, {
      method: 'POST',
      body: JSON.stringify({ similarity_threshold: similarityThreshold }),
    })
  }

  // ─────────────────────────────────────────────────────────────────────────
  // YouTube Methods
  // ─────────────────────────────────────────────────────────────────────────

  async getYouTubeVideoInfo(url: string): Promise<YouTubeVideoInfo> {
    return this.request<YouTubeVideoInfo>('/youtube/info', {
      method: 'POST',
      body: JSON.stringify({ url }),
    })
  }

  async downloadYouTubeAudio(
    url: string,
    options?: {
      format?: 'wav' | 'mp3' | 'flac'
      sample_rate?: number
      run_diarization?: boolean
      filter_to_main_artist?: boolean
    }
  ): Promise<YouTubeDownloadResult> {
    return this.request<YouTubeDownloadResult>('/youtube/download', {
      method: 'POST',
      body: JSON.stringify({
        url,
        format: options?.format ?? 'wav',
        sample_rate: options?.sample_rate ?? 44100,
        run_diarization: options?.run_diarization ?? false,
        filter_to_main_artist: options?.filter_to_main_artist ?? false,
        ...mediaAttestationPayload(),
      }),
    })
  }

  async startYouTubeIngest(
    url: string,
    options?: {
      format?: 'wav' | 'mp3' | 'flac'
      sample_rate?: number
    }
  ): Promise<YouTubeIngestJob> {
    return this.request<YouTubeIngestJob>('/youtube/ingest', {
      method: 'POST',
      body: JSON.stringify({
        url,
        format: options?.format ?? 'wav',
        sample_rate: options?.sample_rate ?? 44100,
        separate_vocals: true,
        run_diarization: true,
        match_existing_profiles: true,
        ...mediaAttestationPayload(),
      }),
    })
  }

  async getYouTubeIngest(jobId: string): Promise<YouTubeIngestJob> {
    return this.request<YouTubeIngestJob>(`/youtube/ingest/${jobId}`)
  }

  async confirmYouTubeIngest(
    jobId: string,
    decisions: YouTubeIngestDecision[]
  ): Promise<YouTubeIngestConfirmation> {
    return this.request<YouTubeIngestConfirmation>(`/youtube/ingest/${jobId}/confirm`, {
      method: 'POST',
      body: JSON.stringify({ decisions }),
    })
  }

  async exportLocalBackup(): Promise<BackupExportResponse> {
    return this.request('/backup/export', { method: 'POST' })
  }

  async importLocalBackupDryRun(backup: File): Promise<BackupImportResponse> {
    const formData = new FormData()
    formData.append('backup', backup)
    formData.append('apply', 'false')
    const response = await apiFetch(`${API_BASE}/backup/import`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) throw new Error(`Import dry-run failed: ${response.status}`)
    return response.json()
  }

  async importLocalBackupApply(backup: File): Promise<BackupImportResponse> {
    const formData = new FormData()
    formData.append('backup', backup)
    formData.append('apply', 'true')
    const response = await apiFetch(`${API_BASE}/backup/import`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) throw new Error(`Import apply failed: ${response.status}`)
    return response.json()
  }

  async addSampleFromPath(
    profileId: string,
    payload: {
      audio_path?: string | null
      audio_asset_id?: string | null
      metadata?: Record<string, unknown>
      skip_separation?: boolean
    }
  ): Promise<TrainingSample> {
    return this.request(`/profiles/${profileId}/samples/from-path`, {
      method: 'POST',
      body: JSON.stringify({
        ...mediaAttestationPayload(),
        ...payload,
      }),
    })
  }

  async getYouTubeHistory(limit?: number): Promise<YouTubeHistoryItem[]> {
    const params = limit ? `?limit=${limit}` : ''
    return this.request(`/youtube/history${params}`)
  }

  async saveYouTubeHistory(item: YouTubeHistorySaveItem): Promise<YouTubeHistoryItem> {
    return this.request('/youtube/history', {
      method: 'POST',
      body: JSON.stringify(item),
    })
  }

  async deleteYouTubeHistoryItem(id: string): Promise<void> {
    await this.request(`/youtube/history/${id}`, { method: 'DELETE' })
  }

  async clearYouTubeHistory(): Promise<void> {
    await this.request('/youtube/history', { method: 'DELETE' })
  }
}

// Diarization types
export interface DiarizationSegment {
  start: number
  end: number
  speaker_id: string
  confidence: number
  duration: number
}

export interface DiarizationResult {
  diarization_id: string
  audio_duration: number
  num_speakers: number
  segments: DiarizationSegment[]
  speaker_durations: Record<string, number>
}

export interface AssignSegmentResult {
  status: string
  profile_id: string
  segment_index: number
  extracted_path: string | null
  segment: DiarizationSegment
}

export interface ProfileSegmentsResult {
  profile_id: string
  total_segments: number
  total_duration: number
  training_samples: Array<{
    type: string
    sample_id: string
    vocals_path: string
    duration: number
    source_file: string
    created_at: string
  }>
  diarization_assignments: Array<{
    type: string
    segment_key: string
    audio_path: string
  }>
}

export interface AutoCreateProfileResult {
  profile_id: string
  name: string
  speaker_id: string
  profile_role: ProfileRole
  num_segments: number
  total_duration: number
  embedding_dim: number
  metadata?: Record<string, unknown>
  status: string
}

export interface SpeakerEmbeddingResult {
  profile_id: string
  embedding_dim: number
  source: string
  status: string
}

export interface SpeakerEmbeddingStatus {
  profile_id: string
  has_embedding: boolean
  embedding_dim: number | null
}

export interface FilterSampleResult {
  sample_id: string
  original_path: string
  filtered_path: string
  original_duration: number
  filtered_duration: number
  num_segments: number
  purity: number
  status: string
}

// ─────────────────────────────────────────────────────────────────────────
// YouTube Types
// ─────────────────────────────────────────────────────────────────────────

export interface YouTubeVideoInfo {
  success: boolean
  title: string
  duration: number
  main_artist: string | null
  featured_artists: string[]
  is_cover: boolean
  original_artist: string | null
  song_title: string | null
  thumbnail_url: string | null
  video_id: string | null
  error: string | null
}

export interface YouTubeDownloadResult {
  success: boolean
  audio_path: string | null
  audio_asset_id?: string | null
  audio_path_asset_id?: string | null
  title: string
  duration: number
  main_artist: string | null
  featured_artists: string[]
  is_cover: boolean
  original_artist: string | null
  song_title: string | null
  thumbnail_url: string | null
  video_id: string | null
  error: string | null
  diarization_id?: string
  speaker_durations?: Record<string, number>
  diarization_result?: {
    diarization_id?: string
    num_speakers: number
    speaker_durations?: Record<string, number>
    segments: Array<{
      speaker_id: string
      start: number
      end: number
      duration: number
    }>
  }
  diarization_error?: string
  filtered_audio_path?: string | null
  filtered_asset_id?: string | null
  filtered_audio_asset_id?: string | null
  main_speaker_id?: string | null
  filtered_duration?: number
}

export interface YouTubeProfileMatch {
  profile_id: string
  name: string
  profile_role?: ProfileRole
  similarity: number
  active_model_type?: string | null
  has_trained_model?: boolean
  sample_count?: number
}

export interface YouTubeSpeakerSuggestion {
  speaker_id: string
  suggested_name: string
  duration: number
  segment_count: number
  matches: YouTubeProfileMatch[]
  recommended_action: 'assign_existing' | 'create_new'
  recommended_profile_id?: string | null
  identity_confidence: 'voice_match' | 'metadata_unverified' | string
  duplicate_warning?: boolean
  duplicate_candidates?: YouTubeProfileMatch[]
  duplicate_threshold?: number
  match_error?: string | null
}

export interface YouTubeIngestJob {
  job_id: string
  job_type: 'youtube_ingest'
  status: 'queued' | 'running' | 'completed' | 'failed' | string
  progress: number
  stage?: string
  message?: string
  error?: string | null
  payload?: Record<string, unknown>
  result?: {
    job_id: string
    url: string
    metadata: {
      title: string
      duration: number
      main_artist: string | null
      featured_artists: string[]
      is_cover: boolean
      original_artist: string | null
      song_title: string | null
      thumbnail_url: string | null
      video_id: string | null
    }
    assets: {
      audio?: { path?: string | null; asset_id?: string | null }
      vocals?: { path?: string | null; asset_id?: string | null }
      instrumental?: { path?: string | null; asset_id?: string | null }
    }
    diarization_id: string
    diarization_result: NonNullable<YouTubeDownloadResult['diarization_result']> & {
      audio_duration?: number
    }
    suggestions: YouTubeSpeakerSuggestion[]
    review_required: boolean
  }
  confirmation?: YouTubeIngestConfirmation
}

export interface YouTubeIngestDecision {
  speaker_id: string
  action: 'assign_existing' | 'create_new' | 'skip'
  profile_id?: string
  name?: string
  metadata?: Record<string, unknown>
}

export interface YouTubeIngestConfirmation {
  job_id: string
  diarization_id: string
  applied: Array<{
    speaker_id: string
    action: 'assign_existing' | 'create_new'
    profile_id: string
    name?: string
    sample_id?: string
    duration?: number
  }>
  skipped: Array<{ speaker_id: string; action: 'skip' }>
  status: string
}

export const apiService = new ApiService()

// Alias for convenience
export const api = apiService

// WebSocket Manager for real-time updates
class WebSocketManager {
  private socket: Socket | null = null
  private handlers: Map<WSEventType, Set<WSEventHandler>> = new Map()

  connect(): void {
    if (this.socket?.connected) return

    if (this.socket) {
      this.socket.connect()
      return
    }

    this.socket = io(window.location.origin, {
      path: '/socket.io',
      transports: ['websocket'],
      auth: getApiAuthToken() ? { token: getApiAuthToken() } : undefined,
    })

    this.socket.on('connect', () => {
      console.log('WebSocket connected')
    })

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected')
    })

    ;([
      'conversion_progress',
      'conversion_complete',
      'conversion_error',
      'training_progress',
      'training_complete',
      'training_error',
      'training_paused',
      'training_resumed',
      'training_cancelled',
      'gpu_metrics',
      'model_loaded',
      'model_unloaded',
    ] as WSEventType[]).forEach((eventType) => {
      this.socket!.on(eventType, (data: unknown) => {
        this.dispatch({
          type: eventType,
          timestamp: new Date().toISOString(),
          data,
        })
      })
    })
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
  }

  subscribe<T = unknown>(eventType: WSEventType, handler: WSEventHandler<T>): () => void {
    this.connect()
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Set())
    }
    this.handlers.get(eventType)!.add(handler as WSEventHandler)

    // Return unsubscribe function
    return () => {
      this.handlers.get(eventType)?.delete(handler as WSEventHandler)
    }
  }

  private dispatch(event: WSEvent): void {
    const handlers = this.handlers.get(event.type)
    if (handlers) {
      handlers.forEach((handler) => handler(event))
    }
  }

  // Convenience methods for common subscriptions
  onConversionProgress(jobId: string, handler: (progress: ConversionProgressEvent) => void): () => void {
    return this.subscribe<ConversionProgressEvent>('conversion_progress', (event) => {
      if (event.data.job_id === jobId) {
        handler(event.data)
      }
    })
  }

  onTrainingProgress(jobId: string, handler: (progress: TrainingProgressEvent) => void): () => void {
    return this.subscribe<TrainingProgressEvent>('training_progress', (event) => {
      if (event.data.job_id === jobId) {
        handler(event.data)
      }
    })
  }

  onGPUMetrics(handler: (metrics: GPUMetrics) => void): () => void {
    return this.subscribe<GPUMetrics>('gpu_metrics', (event) => {
      handler(event.data as GPUMetrics)
    })
  }
}

export const wsManager = new WebSocketManager()
