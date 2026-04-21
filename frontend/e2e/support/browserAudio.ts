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

    if (!navigator.mediaDevices) {
      Object.defineProperty(navigator, 'mediaDevices', {
        configurable: true,
        value: {},
      })
    }

    navigator.mediaDevices.getUserMedia = async () => new FakeMediaStream() as MediaStream
  })
}
