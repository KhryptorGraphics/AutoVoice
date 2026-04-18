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
}

export async function mockCommonApi(page: Page, options: MockCommonApiOptions = {}) {
  await page.addInitScript(({ streamingStartError }) => {
    const globalAny = globalThis as typeof globalThis & {
      __AUTOVOICE_TEST_STREAMING__?: Record<string, unknown>
      AudioContext?: typeof AudioContext
    }

    globalAny.__AUTOVOICE_TEST_STREAMING__ = {
      connect: () => undefined,
      startSession: () => ({ session_id: 'session-1' }),
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
  }, { streamingStartError: options.streamingStartError ?? null })

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

  let preferredPipeline: 'realtime' | 'quality' = options.preferredPipeline ?? 'quality'
  let paused = false
  let separationPollCount = 0
  let speakerDevice = 0
  let headphoneDevice = 1

  await page.route('**/api/v1/health', async (route) => {
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

  await page.route('**/api/v1/settings/app', async (route) => {
    if (route.request().method() === 'PATCH') {
      const body = route.request().postDataJSON() as { preferred_pipeline?: 'realtime' | 'quality' }
      if (body.preferred_pipeline) {
        preferredPipeline = body.preferred_pipeline
      }
    }

    return jsonResponse(route, {
      preferred_pipeline: preferredPipeline,
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
      },
    })
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
        metadata: {},
      },
    ])
  })

  await page.route(/\/api\/v1\/training\/jobs(?:\?.*)?$/, async (route) => {
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

  await page.route('**/socket.io/**', async (route) => {
    await route.abort()
  })

  return {
    getPreferredPipeline: () => preferredPipeline,
    isPaused: () => paused,
    getSpeakerDevice: () => speakerDevice,
    getHeadphoneDevice: () => headphoneDevice,
    getProfileCount: () => profiles.length,
  }
}
