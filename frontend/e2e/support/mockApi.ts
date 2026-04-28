import type { Page, Route } from '@playwright/test'

function jsonResponse(route: Route, payload: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(payload),
  })
}

export function createWavBuffer(sampleRate = 16000, durationSeconds = 1): Buffer {
  const samples = sampleRate * durationSeconds
  const data = Buffer.alloc(samples * 2)
  const header = Buffer.alloc(44)
  header.write('RIFF', 0)
  header.writeUInt32LE(36 + data.length, 4)
  header.write('WAVE', 8)
  header.write('fmt ', 12)
  header.writeUInt32LE(16, 16)
  header.writeUInt16LE(1, 20)
  header.writeUInt16LE(1, 22)
  header.writeUInt32LE(sampleRate, 24)
  header.writeUInt32LE(sampleRate * 2, 28)
  header.writeUInt16LE(2, 32)
  header.writeUInt16LE(16, 34)
  header.write('data', 36)
  header.writeUInt32LE(data.length, 40)
  return Buffer.concat([header, data])
}

type MockCommonApiOptions = {
  preferredPipeline?: 'realtime' | 'quality'
  karaokeUploadError?: string
  streamingStartError?: string
  voiceCloneError?: string
  apiToken?: string
}

export async function mockCommonApi(page: Page, options: MockCommonApiOptions = {}) {
  await page.addInitScript(({ streamingStartError, apiToken }) => {
    const globalAny = globalThis as typeof globalThis & {
      __AUTOVOICE_TEST_STREAMING__?: Record<string, unknown>
      AudioContext?: typeof AudioContext
    }

    if (apiToken) {
      window.localStorage.setItem('autovoice_api_token', apiToken)
    }

    globalAny.__AUTOVOICE_TEST_STREAMING__ = {
      connect: () => undefined,
      startSession: ({ pipelineType, options }: { pipelineType: string; options?: { profileId?: string; collectSamples?: boolean } }) => ({
        session_id: 'session-1',
        requested_pipeline: pipelineType,
        resolved_pipeline: pipelineType,
        runtime_backend: 'pytorch',
        target_profile_id: options?.profileId ?? null,
        source_voice_model_id: 'voice-model-1',
        active_model_type: options?.profileId ? 'adapter' : 'base',
        sample_collection_enabled: Boolean(options?.collectSamples),
        audio_router_targets: {
          speaker_device: 0,
          headphone_device: 1,
        },
      }),
      startStreaming: () => {
        if (streamingStartError) {
          throw new Error(streamingStartError)
        }
      },
      stopStreaming: () => undefined,
      endSession: () => undefined,
      disconnect: () => undefined,
    }

    class FakeMediaStreamTrack {
      stop() {}
    }

    class FakeMediaStream {
      getTracks() {
        return [new FakeMediaStreamTrack()]
      }
    }

    class FakeAudioNode {
      connect() {}
      disconnect() {}
    }

    class FakeScriptProcessorNode extends FakeAudioNode {
      onaudioprocess: ((event: { inputBuffer: { getChannelData: (_channel: number) => Float32Array } }) => void) | null = null
    }

    class FakeAudioContext {
      sampleRate: number

      constructor(config?: { sampleRate?: number }) {
        this.sampleRate = config?.sampleRate ?? 24000
      }

      createMediaStreamSource() {
        return new FakeAudioNode()
      }

      createScriptProcessor() {
        return new FakeScriptProcessorNode()
      }

      close() {
        return Promise.resolve()
      }
    }

    Object.defineProperty(globalAny, 'AudioContext', {
      configurable: true,
      writable: true,
      value: FakeAudioContext,
    })

    if (!navigator.mediaDevices) {
      Object.defineProperty(navigator, 'mediaDevices', {
        configurable: true,
        value: {},
      })
    }

    navigator.mediaDevices.getUserMedia = async () => new FakeMediaStream() as MediaStream
  }, { streamingStartError: options.streamingStartError ?? null, apiToken: options.apiToken ?? null })

  const profiles = [
    {
      profile_id: 'profile-1',
      name: 'Smoke Profile',
      created_at: '2026-04-18T00:00:00Z',
      sample_count: 3,
      training_sample_count: 3,
      clean_vocal_seconds: 2100,
      clean_vocal_minutes: 35,
      full_model_remaining_seconds: 0,
      full_model_remaining_minutes: 0,
      full_model_eligible: true,
      has_trained_model: true,
      has_full_model: false,
      has_adapter_model: true,
      active_model_type: 'adapter',
      profile_role: 'target_user',
      selected_adapter: 'hq',
      training_history: [],
    },
  ]

  const voiceModels = [
    {
      id: 'voice-model-1',
      name: 'Smoke Artist',
      type: 'pretrained',
    },
  ]
  const loadedModels = [
    {
      type: 'encoder',
      model_type: 'encoder',
      name: 'HuBERT Encoder',
      loaded: true,
      runtime_backend: 'pytorch',
      device: 'cuda:0',
      memory_usage: 805306368,
      loaded_at: '2026-04-18T00:00:00Z',
      source: 'registry',
    },
    {
      type: 'vocoder',
      model_type: 'vocoder',
      name: 'BigVGAN Vocoder',
      loaded: true,
      runtime_backend: 'tensorrt',
      device: 'cuda:0',
      memory_usage: 536870912,
      loaded_at: '2026-04-18T00:00:00Z',
      source: 'registry',
    },
  ]
  const builtEngines = [
    {
      name: 'encoder',
      precision: 'fp16',
      built_at: '2026-04-18T00:00:00Z',
    },
  ]

  let preferredPipeline: 'realtime' | 'quality' = options.preferredPipeline ?? 'quality'
  let preferredOfflinePipeline: 'realtime' | 'quality' | 'quality_seedvc' | 'quality_shortcut' = preferredPipeline
  let preferredLivePipeline: 'realtime' | 'realtime_meanvc' = 'realtime'
  let paused = false
  let separationPollCount = 0
  let speakerDevice = 0
  let headphoneDevice = 1
  let deviceConfig = {
    input_device_id: 'input-1',
    output_device_id: 'output-1',
    sample_rate: 48000,
  }
  let separationConfig = {
    model: 'htdemucs',
    stems: ['vocals'],
    shifts: 1,
    overlap: 0.25,
    segment_length: null,
    device: 'cuda',
  }
  let pitchConfig = {
    method: 'rmvpe',
    hop_length: 160,
    f0_min: 50,
    f0_max: 1100,
    threshold: 0.3,
    use_gpu: true,
  }
  let audioRouterConfig = {
    speaker_gain: 1.0,
    headphone_gain: 1.0,
    voice_gain: 1.0,
    instrumental_gain: 0.8,
    speaker_enabled: true,
    headphone_enabled: true,
    speaker_device: speakerDevice,
    headphone_device: headphoneDevice,
    sample_rate: 24000,
  }
  let youtubeInfoRequests = 0
  let youtubeDownloadRequests = 0
  let youtubeAddToProfileRequests = 0
  let conversionHistoryDeletes = 0
  let checkpointRollbacks = 0
  let checkpointDeletes = 0
  let lastTrainingPayload: Record<string, unknown> | null = null
  const authorizationHeaders: string[] = []
  const conversionRecords = [
    {
      id: 'history-1',
      status: 'complete',
      created_at: '2026-04-18T00:00:00Z',
      input_file: 'demo-vocal.wav',
      profile_id: 'profile-1',
      preset: 'Studio',
      duration: 132,
      pipeline_type: 'quality_seedvc',
      requested_pipeline: 'quality_seedvc',
      resolved_pipeline: 'quality_seedvc',
      runtime_backend: 'tensorrt',
      adapter_type: 'hq',
      active_model_type: 'adapter',
      processing_time_seconds: 61.2,
      rtf: 0.46,
      audio_duration_seconds: 132,
      resultUrl: '/outputs/history-1.wav',
      output_url: '/outputs/history-1.wav',
      download_url: '/outputs/history-1.wav',
      isFavorite: false,
      targetVoice: 'Smoke Profile',
      originalFileName: 'demo-vocal.wav',
    },
  ]
  const checkpoints = [
    {
      id: 'checkpoint-active',
      profile_id: 'profile-1',
      version: 'v2',
      created_at: '2026-04-18T01:00:00Z',
      epochs_trained: 12,
      final_loss: 0.21,
      is_active: true,
      file_size_mb: 96.2,
      training_samples: 3,
      notes: 'Current production adapter',
    },
    {
      id: 'checkpoint-rollback',
      profile_id: 'profile-1',
      version: 'v1',
      created_at: '2026-04-17T01:00:00Z',
      epochs_trained: 8,
      final_loss: 0.29,
      is_active: false,
      file_size_mb: 88.5,
      training_samples: 3,
      notes: 'Last known stable adapter',
    },
    {
      id: 'checkpoint-delete',
      profile_id: 'profile-1',
      version: 'v0',
      created_at: '2026-04-16T01:00:00Z',
      epochs_trained: 4,
      final_loss: 0.41,
      is_active: false,
      file_size_mb: 81.4,
      training_samples: 2,
      notes: 'Superseded experimental adapter',
    },
  ]

  await page.route('**/api/v1/health', async (route) => {
    authorizationHeaders.push(route.request().headers().authorization ?? '')
    return jsonResponse(route, {
      status: 'healthy',
      version: '1.0.0',
      timestamp: '2026-04-18T00:00:00Z',
      components: {
        torch: { status: 'up', details: 'cuda' },
        singing_pipeline: { status: 'up' },
      },
    })
  })

  await page.route('**/api/v1/system/info', async (route) => {
    authorizationHeaders.push(route.request().headers().authorization ?? '')
    return jsonResponse(route, {
      system: {
        platform: 'Jetson Thor',
        python_version: '3.11.9',
      },
      torch: {
        cuda_available: true,
        device_count: 1,
        device_name: 'NVIDIA Thor',
        version: '2.6.0',
      },
    })
  })

  await page.route('**/api/v1/reports/benchmarks/latest', async (route) => {
    authorizationHeaders.push(route.request().headers().authorization ?? '')
    return jsonResponse(route, {
      generated_at: '2026-04-18T00:00:00Z',
      target_hardware: 'NVIDIA Thor',
      provenance: {
        schema_version: 1,
        generator: 'mock-benchmark-dashboard',
        git_sha: 'abc123456789',
        source_bundles: ['reports/benchmarks/run-1/summary.json'],
      },
      canonical_pipelines: {
        offline: 'quality_seedvc',
        live: 'realtime',
      },
      pipelines: {
        realtime: {
          title: 'Realtime',
          sample_count: 3,
          fixture_tier: 'smoke',
          fixture_suite: 'operator-console',
          summary: {
            speaker_similarity_mean: { value: 0.91, target_status: 'pass' },
          },
          source_bundle: 'reports/benchmarks/run-1/summary.json',
        },
      },
      comparisons: {},
      promotable_candidates: [],
    })
  })

  await page.route('**/api/v1/reports/release-evidence/latest', async (route) => {
    authorizationHeaders.push(route.request().headers().authorization ?? '')
    return jsonResponse(route, {
      generated_at: '2026-04-18T00:00:00Z',
      status: 'blocked',
      ready_for_release: false,
      quality_gate_passed: true,
      target_hardware: 'NVIDIA Thor',
      git_sha_short: 'abc123456789',
      blockers: ['hardware validation lanes were not executed; rerun with --execute on Jetson'],
      artifacts: {
        completion_matrix: 'reports/completion/latest/completion_matrix.json',
        decision: 'reports/release-evidence/latest/release_decision.json',
      },
      lane_results: [
        {
          name: 'jetson-cuda-tensorrt',
          status: 'skipped',
          details: {
            reason: 'pass --hardware on Jetson/CUDA/TensorRT hosts',
            action: 'Run the hardware lane on Jetson.',
          },
        },
      ],
      preflight_checks: [
        {
          name: 'tegrastats',
          ok: false,
          stderr: 'tegrastats timed out',
        },
      ],
    })
  })

  await page.route('**/api/v1/settings/app', async (route) => {
    if (route.request().method() === 'PATCH') {
      const body = route.request().postDataJSON() as {
        preferred_pipeline?: 'realtime' | 'quality'
        preferred_offline_pipeline?: 'realtime' | 'quality' | 'quality_seedvc' | 'quality_shortcut'
        preferred_live_pipeline?: 'realtime' | 'realtime_meanvc'
      }
      if (body.preferred_pipeline) {
        preferredPipeline = body.preferred_pipeline
        preferredOfflinePipeline = body.preferred_pipeline
        if (body.preferred_pipeline === 'realtime') {
          preferredLivePipeline = 'realtime'
        }
      }
      if (body.preferred_offline_pipeline) {
        preferredOfflinePipeline = body.preferred_offline_pipeline
        preferredPipeline = preferredOfflinePipeline === 'realtime' ? 'realtime' : 'quality'
      }
      if (body.preferred_live_pipeline) {
        preferredLivePipeline = body.preferred_live_pipeline
      }
    }

    return jsonResponse(route, {
      preferred_pipeline: preferredPipeline,
      preferred_offline_pipeline: preferredOfflinePipeline,
      preferred_live_pipeline: preferredLivePipeline,
      last_updated: '2026-04-18T00:00:00Z',
    })
  })

  await page.route('**/api/v1/pipelines/status', async (route) => {
    return jsonResponse(route, {
      status: 'ok',
      timestamp: '2026-04-18T00:00:00Z',
      pipelines: {
        realtime: {
          loaded: true,
          memory_gb: 1.2,
          latency_target_ms: 100,
        },
        quality: {
          loaded: false,
          memory_gb: 2.8,
          latency_target_ms: 3000,
        },
        quality_seedvc: {
          loaded: true,
          memory_gb: 3.6,
          latency_target_ms: 1800,
        },
        realtime_meanvc: {
          loaded: false,
          memory_gb: 1.4,
          latency_target_ms: 80,
        },
      },
    })
  })

  await page.route('**/api/v1/devices/list', async (route) => {
    return jsonResponse(route, [
      {
        device_id: 'input-1',
        name: 'Studio Mic',
        type: 'input',
        sample_rate: 48000,
        channels: 1,
        is_default: true,
      },
      {
        device_id: 'output-1',
        name: 'Main Speakers',
        type: 'output',
        sample_rate: 48000,
        channels: 2,
        is_default: true,
      },
      {
        device_id: 'output-2',
        name: 'Control Headphones',
        type: 'output',
        sample_rate: 48000,
        channels: 2,
        is_default: false,
      },
    ])
  })

  await page.route('**/api/v1/devices/config', async (route) => {
    if (route.request().method() === 'POST') {
      deviceConfig = {
        ...deviceConfig,
        ...(route.request().postDataJSON() as Record<string, string | number | undefined>),
      }
    }
    return jsonResponse(route, deviceConfig)
  })

  await page.route('**/api/v1/models/loaded', async (route) => {
    return jsonResponse(route, { models: loadedModels })
  })

  await page.route('**/api/v1/models/load', async (route) => {
    const body = route.request().postDataJSON() as { model_type: string }
    const existing = loadedModels.find((model) => model.type === body.model_type)
    if (!existing) {
      loadedModels.push({
        type: body.model_type,
        model_type: body.model_type,
        name: `${body.model_type.toUpperCase()} Model`,
        loaded: true,
        runtime_backend: 'pytorch',
        device: 'cuda:0',
        memory_usage: 268435456,
        loaded_at: '2026-04-18T00:00:00Z',
        source: 'registry',
      })
    }
    return jsonResponse(route, loadedModels.find((model) => model.type === body.model_type) ?? loadedModels[0])
  })

  await page.route('**/api/v1/models/unload', async (route) => {
    const body = route.request().postDataJSON() as { model_type: string }
    const index = loadedModels.findIndex((model) => model.type === body.model_type)
    if (index >= 0) {
      loadedModels.splice(index, 1)
    }
    return route.fulfill({ status: 204 })
  })

  await page.route('**/api/v1/models/tensorrt/status', async (route) => {
    return jsonResponse(route, {
      available: true,
      version: '10.0',
      engines: builtEngines,
      build_in_progress: false,
    })
  })

  await page.route('**/api/v1/models/tensorrt/build', async (route) => {
    const body = route.request().postDataJSON() as { models: string[]; precision: 'fp32' | 'fp16' | 'int8' }
    for (const model of body.models) {
      const existing = builtEngines.find((engine) => engine.name === model)
      if (existing) {
        existing.precision = body.precision
      } else {
        builtEngines.push({
          name: model,
          precision: body.precision,
          built_at: '2026-04-18T00:00:00Z',
        })
      }
    }
    return jsonResponse(route, {
      status: 'ok',
      engines_built: body.models,
    })
  })

  await page.route('**/api/v1/models/tensorrt/rebuild', async (route) => {
    const body = route.request().postDataJSON() as { precision: 'fp32' | 'fp16' | 'int8' }
    builtEngines.splice(0, builtEngines.length, {
      name: 'encoder',
      precision: body.precision,
      built_at: '2026-04-18T00:00:00Z',
    })
    return jsonResponse(route, {
      status: 'ok',
      duration_seconds: 12.3,
    })
  })

  await page.route('**/api/v1/config/separation', async (route) => {
    if (route.request().method() === 'PATCH') {
      separationConfig = {
        ...separationConfig,
        ...(route.request().postDataJSON() as Record<string, unknown>),
      }
    }
    return jsonResponse(route, separationConfig)
  })

  await page.route('**/api/v1/config/pitch', async (route) => {
    if (route.request().method() === 'PATCH') {
      pitchConfig = {
        ...pitchConfig,
        ...(route.request().postDataJSON() as Record<string, unknown>),
      }
    }
    return jsonResponse(route, pitchConfig)
  })

  await page.route('**/api/v1/audio/router/config', async (route) => {
    if (route.request().method() === 'PATCH') {
      audioRouterConfig = {
        ...audioRouterConfig,
        ...(route.request().postDataJSON() as Record<string, unknown>),
      }
    }
    return jsonResponse(route, audioRouterConfig)
  })

  await page.route('**/api/v1/voice/profiles', async (route) => {
    return jsonResponse(route, profiles)
  })

  await page.route('**/api/v1/voice/profiles/profile-1', async (route) => {
    return jsonResponse(route, profiles[0])
  })

  await page.route('**/api/v1/voice/profiles/profile-1/adapters', async (route) => {
    return jsonResponse(route, {
      profile_id: 'profile-1',
      adapters: [
        {
          type: 'hq',
          path: '/tmp/profile-1_adapter.pt',
          size_kb: 128,
          epochs: 12,
          loss: 0.24,
          precision: 'fp16',
          config: {},
        },
      ],
      selected: 'hq',
      count: 1,
    })
  })

  await page.route('**/api/v1/profiles/profile-1/samples', async (route) => {
    return jsonResponse(route, [
      {
        id: 'sample-1',
        sample_id: 'sample-1',
        profile_id: 'profile-1',
        audio_path: '/tmp/sample-1.wav',
        file_path: '/tmp/sample-1.wav',
        duration_seconds: 12.5,
        sample_rate: 44100,
        created: '2026-04-18T00:00:00Z',
        metadata: { qa_status: 'pass' },
      },
      {
        id: 'sample-2',
        sample_id: 'sample-2',
        profile_id: 'profile-1',
        audio_path: '/tmp/sample-2.wav',
        file_path: '/tmp/sample-2.wav',
        duration_seconds: 18.25,
        sample_rate: 44100,
        created: '2026-04-18T00:05:00Z',
        metadata: { qa_status: 'pass' },
      },
      {
        id: 'sample-failed',
        sample_id: 'sample-failed',
        profile_id: 'profile-1',
        audio_path: '/tmp/sample-failed.wav',
        file_path: '/tmp/sample-failed.wav',
        duration_seconds: 9.5,
        sample_rate: 44100,
        created: '2026-04-18T00:10:00Z',
        metadata: { qa_status: 'fail' },
      },
    ])
  })

  await page.route('**/api/v1/profiles/profile-1/samples/from-path', async (route) => {
    youtubeAddToProfileRequests += 1
    return jsonResponse(route, {
      sample_id: 'sample-youtube-1',
      id: 'sample-youtube-1',
      profile_id: 'profile-1',
      audio_path: '/tmp/autovoice-youtube-smoke.wav',
      duration_seconds: 142,
      sample_rate: 44100,
      created: '2026-04-18T00:00:00Z',
      metadata: route.request().postDataJSON(),
    }, 201)
  })

  await page.route('**/api/v1/profiles/profile-1/checkpoints', async (route) => {
    return jsonResponse(route, checkpoints)
  })

  await page.route('**/api/v1/profiles/profile-1/checkpoints/checkpoint-rollback/rollback', async (route) => {
    checkpointRollbacks += 1
    checkpoints.forEach((checkpoint) => {
      checkpoint.is_active = checkpoint.id === 'checkpoint-rollback'
    })
    return route.fulfill({ status: 204 })
  })

  await page.route('**/api/v1/profiles/profile-1/checkpoints/checkpoint-delete', async (route) => {
    checkpointDeletes += 1
    const index = checkpoints.findIndex((checkpoint) => checkpoint.id === 'checkpoint-delete')
    if (index >= 0) {
      checkpoints.splice(index, 1)
    }
    return route.fulfill({ status: 204 })
  })

  await page.route(/\/api\/v1\/convert\/history(?:\?.*)?$/, async (route) => {
    return jsonResponse(route, conversionRecords)
  })

  await page.route('**/api/v1/convert/history/history-1', async (route) => {
    if (route.request().method() === 'DELETE') {
      conversionHistoryDeletes += 1
      const index = conversionRecords.findIndex((record) => record.id === 'history-1')
      if (index >= 0) {
        conversionRecords.splice(index, 1)
      }
      return route.fulfill({ status: 204 })
    }
    if (route.request().method() === 'PATCH') {
      const body = route.request().postDataJSON() as Record<string, unknown>
      Object.assign(conversionRecords[0], body)
      return jsonResponse(route, conversionRecords[0])
    }
    return jsonResponse(route, conversionRecords[0] ?? {})
  })

  await page.route('**/api/v1/youtube/info', async (route) => {
    youtubeInfoRequests += 1
    return jsonResponse(route, {
      success: true,
      title: 'Smoke Song',
      duration: 142,
      main_artist: 'Smoke Artist',
      featured_artists: [],
      is_cover: false,
      original_artist: null,
      song_title: 'Smoke Song',
      thumbnail_url: null,
      video_id: 'smoke-video',
      error: null,
    })
  })

  await page.route('**/api/v1/youtube/download', async (route) => {
    youtubeDownloadRequests += 1
    return jsonResponse(route, {
      success: true,
      audio_path: '/tmp/autovoice-youtube-smoke.wav',
      title: 'Smoke Song',
      duration: 142,
      main_artist: 'Smoke Artist',
      featured_artists: [],
      is_cover: false,
      original_artist: null,
      song_title: 'Smoke Song',
      thumbnail_url: null,
      video_id: 'smoke-video',
      error: null,
    })
  })

  await page.route(/\/api\/v1\/youtube\/history(?:\?.*)?$/, async (route) => {
    if (route.request().method() === 'DELETE') {
      return route.fulfill({ status: 204 })
    }
    return jsonResponse(route, [
      {
        id: 'youtube-history-1',
        url: 'https://youtu.be/smoke-video',
        title: 'Smoke Song',
        mainArtist: 'Smoke Artist',
        featuredArtists: [],
        hasDiarization: false,
        numSpeakers: 0,
        timestamp: '2026-04-18T00:00:00Z',
        audioPath: '/tmp/autovoice-youtube-smoke.wav',
        filteredPath: null,
        videoId: 'smoke-video',
      },
    ])
  })

  await page.route('**/api/v1/training/config-options', async (route) => {
    return jsonResponse(route, {
      schema_version: 1,
      defaults: {
        training_mode: 'lora',
        initialization_mode: 'scratch',
        lora_rank: 8,
        lora_alpha: 16,
        lora_dropout: 0.1,
        lora_target_modules: ['q_proj', 'v_proj', 'content_encoder'],
        learning_rate: 0.0001,
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
        ewc_lambda: 1000,
        use_prior_preservation: false,
        prior_loss_weight: 0.5,
      },
      limits: {},
      enums: {
        training_mode: ['lora', 'full'],
        initialization_mode: ['scratch', 'continue'],
        precision: ['fp32', 'fp16', 'bf16'],
        optimizer: ['adamw', 'adam'],
        scheduler: ['exponential', 'none'],
        lora_target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'content_encoder'],
      },
      presets: [
        {
          id: 'quality_lora',
          label: 'Quality LoRA',
          description: 'Higher-capacity LoRA for production offline conversion quality.',
          config: {
            training_mode: 'lora',
            initialization_mode: 'scratch',
            preset_id: 'quality_lora',
            lora_rank: 16,
            lora_alpha: 32,
            epochs: 120,
            learning_rate: 0.000075,
            batch_size: 2,
            precision: 'fp16',
            checkpoint_every_steps: 500,
          },
        },
      ],
      devices: [
        { id: 'auto', label: 'Auto CUDA device', available: true },
        { id: 'cuda:0', label: 'NVIDIA Thor', available: true, memory_total_gb: 64 },
      ],
      full_model_unlock_minutes: 30,
    })
  })

  await page.route(/\/api\/v1\/training\/jobs(?:\?.*)?$/, async (route) => {
    if (route.request().method() === 'POST') {
      lastTrainingPayload = route.request().postDataJSON() as Record<string, unknown>
      return jsonResponse(route, {
        job_id: 'job-created',
        profile_id: 'profile-1',
        status: 'running',
        created_at: '2026-04-18T00:00:00Z',
        started_at: '2026-04-18T00:01:00Z',
        progress: 1,
        sample_ids: (lastTrainingPayload.sample_ids as string[]) ?? [],
        is_paused: false,
        config: lastTrainingPayload.config,
      }, 201)
    }
    return jsonResponse(route, [
      {
        job_id: 'job-1',
        profile_id: 'profile-1',
        status: 'running',
        created_at: '2026-04-18T00:00:00Z',
        started_at: '2026-04-18T00:01:00Z',
        progress: 40,
        sample_ids: ['sample-1'],
        is_paused: paused,
        results: {
          current_loss: 0.24,
          current_epoch: 2,
          current_step: 1800,
          latest_checkpoint: '/tmp/checkpoint_step_1000.pth',
          quality_metrics: {
            mos_proxy: 4.1,
            speaker_similarity_proxy: 0.92,
          },
          gpu_metrics: {
            available: true,
            memory_used_gb: 3.2,
            utilization_percent: 71,
          },
        },
      },
    ])
  })

  await page.route('**/api/v1/training/jobs/job-1/telemetry', async (route) => {
    return jsonResponse(route, {
      job: {
        job_id: 'job-1',
        profile_id: 'profile-1',
        status: 'running',
        created_at: '2026-04-18T00:00:00Z',
        started_at: '2026-04-18T00:01:00Z',
        progress: 40,
        sample_ids: ['sample-1'],
        is_paused: paused,
      },
      runtime_metrics: {
        epoch: 2,
        total_epochs: 10,
        step: 18,
        total_steps: 40,
        loss: 0.24,
        learning_rate: 0.0001,
        gpu_metrics: {
          available: true,
          memory_used_gb: 3.2,
          utilization_percent: 71,
        },
        quality_metrics: {
          mos_proxy: 4.1,
          speaker_similarity_proxy: 0.92,
        },
        checkpoint_path: '/tmp/checkpoint_step_1000.pth',
      },
      preview_available: true,
      preview_sample_id: 'sample-1',
    })
  })

  await page.route('**/api/v1/training/jobs/job-1/pause', async (route) => {
    paused = true
    return jsonResponse(route, {
      job_id: 'job-1',
      profile_id: 'profile-1',
      status: 'running',
      created_at: '2026-04-18T00:00:00Z',
      started_at: '2026-04-18T00:01:00Z',
      progress: 40,
      sample_ids: ['sample-1'],
      is_paused: true,
    })
  })

  await page.route('**/api/v1/training/jobs/job-1/resume', async (route) => {
    paused = false
    return jsonResponse(route, {
      job_id: 'job-1',
      profile_id: 'profile-1',
      status: 'running',
      created_at: '2026-04-18T00:00:00Z',
      started_at: '2026-04-18T00:01:00Z',
      progress: 40,
      sample_ids: ['sample-1'],
      is_paused: false,
    })
  })

  await page.route('**/api/v1/training/jobs/job-1/cancel', async (route) => {
    return route.fulfill({ status: 204 })
  })

  await page.route('**/api/v1/training/preview/job-1', async (route) => {
    return route.fulfill({
      status: 200,
      contentType: 'audio/wav',
      body: createWavBuffer(),
    })
  })

  await page.route('**/api/v1/voice/clone', async (route) => {
    if (options.voiceCloneError) {
      return jsonResponse(route, { error: options.voiceCloneError }, 500)
    }

    const createdProfile = {
      profile_id: `profile-${profiles.length + 1}`,
      name: 'Created Smoke Profile',
      created_at: '2026-04-18T00:00:00Z',
      sample_count: 1,
      training_sample_count: 1,
      clean_vocal_seconds: 15,
      clean_vocal_minutes: 0.25,
      full_model_remaining_seconds: 1795,
      full_model_remaining_minutes: 29.9,
      full_model_eligible: false,
      has_trained_model: false,
      has_full_model: false,
      has_adapter_model: false,
      active_model_type: 'adapter',
      profile_role: 'target_user',
      selected_adapter: 'hq',
      training_history: [],
    }
    profiles.unshift(createdProfile)
    return jsonResponse(route, createdProfile, 201)
  })

  await page.route('**/api/v1/karaoke/upload', async (route) => {
    if (options.karaokeUploadError) {
      return jsonResponse(route, { error: options.karaokeUploadError }, 500)
    }

    return jsonResponse(route, {
      song_id: 'song-1',
      duration: 123,
      sample_rate: 44100,
      format: 'wav',
      status: 'uploaded',
    }, 201)
  })

  await page.route('**/api/v1/karaoke/separate', async (route) => {
    separationPollCount = 0
    return jsonResponse(route, {
      job_id: 'sep-1',
      song_id: 'song-1',
      status: 'processing',
      progress: 20,
      estimated_remaining: 4,
    }, 202)
  })

  await page.route('**/api/v1/karaoke/separate/sep-1', async (route) => {
    separationPollCount += 1
    if (separationPollCount < 2) {
      return jsonResponse(route, {
        job_id: 'sep-1',
        song_id: 'song-1',
        status: 'processing',
        progress: 55,
        estimated_remaining: 2,
      })
    }

    return jsonResponse(route, {
      job_id: 'sep-1',
      song_id: 'song-1',
      status: 'completed',
      progress: 100,
      vocals_ready: true,
      instrumental_ready: true,
      vocals_path: '/tmp/song-1_vocals.wav',
      instrumental_path: '/tmp/song-1_instrumental.wav',
    })
  })

  await page.route('**/api/v1/karaoke/devices', async (route) => {
    return jsonResponse(route, {
      devices: [
        {
          index: 0,
          name: 'Speakers',
          channels: 2,
          default_sample_rate: 48000,
          is_default: true,
        },
        {
          index: 1,
          name: 'Headphones',
          channels: 2,
          default_sample_rate: 48000,
          is_default: false,
        },
      ],
      count: 2,
    })
  })

  await page.route('**/api/v1/karaoke/devices/output', async (route) => {
    if (route.request().method() === 'POST') {
      const body = route.request().postDataJSON() as {
        speaker_device?: number
        headphone_device?: number
      }
      if (typeof body.speaker_device === 'number') {
        speakerDevice = body.speaker_device
      }
      if (typeof body.headphone_device === 'number') {
        headphoneDevice = body.headphone_device
      }
      audioRouterConfig = {
        ...audioRouterConfig,
        speaker_device: speakerDevice,
        headphone_device: headphoneDevice,
      }
    }

    return jsonResponse(route, {
      speaker_device: speakerDevice,
      headphone_device: headphoneDevice,
    })
  })

  await page.route('**/api/v1/karaoke/voice-models', async (route) => {
    return jsonResponse(route, {
      models: voiceModels,
      count: voiceModels.length,
    })
  })

  await page.route('**/api/v1/karaoke/voice-models/extract', async (route) => {
    const extracted = {
      id: `voice-model-${voiceModels.length + 1}`,
      name: 'Extracted Smoke Artist',
      type: 'extracted',
      source_song_id: 'song-1',
    }
    voiceModels.push(extracted)
    return jsonResponse(route, {
      model_id: extracted.id,
      name: extracted.name,
      type: extracted.type,
    }, 201)
  })

  await page.route('**/api/v1/karaoke/preflight', async (route) => {
    const body = route.request().postDataJSON() as { pipeline_type?: string }
    const pipelineType = body.pipeline_type ?? 'realtime'
    const ok = pipelineType === 'realtime' || pipelineType === 'realtime_meanvc'
    return jsonResponse(route, {
      ok,
      issues: ok ? [] : ['Live karaoke only supports the realtime or realtime_meanvc pipelines.'],
      warnings: [],
      checks: {
        song_ready: true,
        assets_ready: true,
        pipeline_valid: ok,
        profile_ready: true,
        voice_model_ready: true,
        routing_ready: true,
      },
      requested_pipeline: pipelineType,
      active_model_type: 'adapter',
      audio_router_targets: {
        speaker_device: speakerDevice,
        headphone_device: headphoneDevice,
      },
    })
  })

  await page.route('**/socket.io/**', async (route) => {
    await route.abort()
  })

  return {
    getPreferredPipeline: () => preferredOfflinePipeline,
    getPreferredLivePipeline: () => preferredLivePipeline,
    isPaused: () => paused,
    getSpeakerDevice: () => speakerDevice,
    getHeadphoneDevice: () => headphoneDevice,
    getProfileCount: () => profiles.length,
    getLoadedModelCount: () => loadedModels.length,
    getLoadedModelTypes: () => loadedModels.map((model) => model.type),
    getBuiltTensorRTEngines: () => builtEngines.map((engine) => `${engine.name}:${engine.precision}`),
    getYouTubeInfoRequests: () => youtubeInfoRequests,
    getYouTubeDownloadRequests: () => youtubeDownloadRequests,
    getYouTubeAddToProfileRequests: () => youtubeAddToProfileRequests,
    getConversionHistoryDeletes: () => conversionHistoryDeletes,
    getConversionRecordCount: () => conversionRecords.length,
    getCheckpointRollbacks: () => checkpointRollbacks,
    getCheckpointDeletes: () => checkpointDeletes,
    getCheckpointCount: () => checkpoints.length,
    getLastTrainingPayload: () => lastTrainingPayload,
    getAuthorizationHeaders: () => authorizationHeaders,
  }
}
