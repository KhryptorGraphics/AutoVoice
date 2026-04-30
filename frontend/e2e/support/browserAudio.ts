import type { Page } from '@playwright/test'

export async function installBrowserAudioMocks(page: Page) {
  await page.addInitScript(() => {
    const globalAny = globalThis as typeof globalThis & {
      AudioContext?: typeof AudioContext
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

    class FakeAnalyserNode extends FakeAudioNode {
      fftSize = 1024

      getFloatTimeDomainData(data: Float32Array) {
        data.fill(0.1)
      }
    }

    class FakeBuffer {
      private readonly data = new Float32Array(2400)

      getChannelData() {
        return this.data
      }
    }

    class FakeBufferSource extends FakeAudioNode {
      buffer: FakeAudioBuffer | null = null
      start() {}
    }

    class FakeScriptProcessorNode extends FakeAudioNode {
      onaudioprocess: ((event: { inputBuffer: FakeBuffer }) => void) | null = null
    }

    class FakeAudioBuffer {
      private readonly data = new Float32Array(2400)

      getChannelData() {
        return this.data
      }
    }

    class FakeAudioContext {
      sampleRate: number
      destination: FakeAudioNode

      constructor(config?: { sampleRate?: number }) {
        this.sampleRate = config?.sampleRate ?? 24000
        this.destination = new FakeAudioNode()
      }

      createMediaStreamSource() {
        return new FakeAudioNode()
      }

      createAnalyser() {
        return new FakeAnalyserNode()
      }

      createScriptProcessor() {
        return new FakeScriptProcessorNode()
      }

      createBuffer() {
        return new FakeAudioBuffer()
      }

      createBufferSource() {
        return new FakeBufferSource()
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

    class FakeMediaRecorder {
      static isTypeSupported() {
        return true
      }

      state: 'inactive' | 'recording' = 'inactive'
      ondataavailable: ((event: BlobEvent) => void) | null = null
      onstop: (() => void) | null = null

      constructor(_stream: MediaStream, public readonly options?: MediaRecorderOptions) {}

      start() {
        this.state = 'recording'
      }

      stop() {
        if (this.state === 'inactive') return
        this.state = 'inactive'
        const data = new Blob([new Uint8Array([82, 73, 70, 70, 0, 0, 0, 0])], {
          type: this.options?.mimeType || 'audio/webm',
        })
        this.ondataavailable?.({ data } as BlobEvent)
        this.onstop?.()
      }
    }

    Object.defineProperty(globalThis, 'MediaRecorder', {
      configurable: true,
      writable: true,
      value: FakeMediaRecorder,
    })

    if (!navigator.mediaDevices) {
      Object.defineProperty(navigator, 'mediaDevices', {
        configurable: true,
        value: {},
      })
    }

    navigator.mediaDevices.enumerateDevices = async () => [
      {
        deviceId: 'browser-mic-1',
        kind: 'audioinput',
        label: 'Browser Headset Mic',
        groupId: 'browser-group-1',
        toJSON() { return this },
      },
      {
        deviceId: 'browser-output-1',
        kind: 'audiooutput',
        label: 'Browser Headphones',
        groupId: 'browser-group-1',
        toJSON() { return this },
      },
    ] as MediaDeviceInfo[]
    navigator.mediaDevices.getUserMedia = async () => new FakeMediaStream() as MediaStream

    Object.defineProperty(HTMLMediaElement.prototype, 'setSinkId', {
      configurable: true,
      writable: true,
      value: async function setSinkId(this: HTMLMediaElement, sinkId: string) {
        Object.defineProperty(this, 'sinkId', {
          configurable: true,
          value: sinkId,
        })
      },
    })
    HTMLMediaElement.prototype.play = async function play() {
      this.dispatchEvent(new Event('play'))
    }
  })
}
