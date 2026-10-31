export type BrowserAudioDevice = {
  deviceId: string
  label: string
  kind: MediaDeviceKind
}

export type BrowserAudioCapabilities = {
  hasMediaDevices: boolean
  hasGetUserMedia: boolean
  hasEnumerateDevices: boolean
  hasMediaRecorder: boolean
  hasOutputSelection: boolean
  isSecureContext: boolean
}

type AudioElementWithSink = HTMLAudioElement & {
  setSinkId?: (sinkId: string) => Promise<void>
}

export function getBrowserAudioCapabilities(audioElement?: HTMLAudioElement | null): BrowserAudioCapabilities {
  const mediaDevices = typeof navigator !== 'undefined' ? navigator.mediaDevices : undefined

  return {
    hasMediaDevices: Boolean(mediaDevices),
    hasGetUserMedia: Boolean(mediaDevices?.getUserMedia),
    hasEnumerateDevices: Boolean(mediaDevices?.enumerateDevices),
    hasMediaRecorder: typeof MediaRecorder !== 'undefined',
    hasOutputSelection: Boolean((audioElement as AudioElementWithSink | undefined)?.setSinkId),
    isSecureContext: typeof window === 'undefined' ? true : window.isSecureContext,
  }
}

export async function listBrowserAudioDevices(): Promise<BrowserAudioDevice[]> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return []
  }

  const devices = await navigator.mediaDevices.enumerateDevices()
  return devices
    .filter((device) => device.kind === 'audioinput' || device.kind === 'audiooutput')
    .map((device, index) => ({
      deviceId: device.deviceId,
      kind: device.kind,
      label: device.label || `${device.kind === 'audioinput' ? 'Microphone' : 'Output'} ${index + 1}`,
    }))
}

export async function requestBrowserMicrophone(deviceId?: string): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Browser microphone access is not available')
  }

  const audio: MediaTrackConstraints = {
    channelCount: 1,
    // Ask for full band. Without this the browser is free to take whatever the
    // device offers, and laptop digital-mic arrays routinely offer 16 kHz -
    // which reads as a brickwall at ~7.2 kHz in the take and cannot be undone
    // later. `ideal` so a device that only does 16 kHz still records.
    sampleRate: { ideal: 48000 },
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  }

  if (deviceId) {
    audio.deviceId = { exact: deviceId }
  }

  const stream = await navigator.mediaDevices.getUserMedia({ audio })
  // Requesting a constraint is not the same as getting it. Driver-level DSP
  // (Intel Smart Sound, Windows "audio enhancements", vendor apps) processes
  // the signal BEFORE the browser sees it, and the browser then reports
  // whatever it believes. A take measured 4.43 dB of 3-8 Hz level modulation
  // against ~1.7 for a natural voice - i.e. something was still compressing it
  // - and we had no way to tell whether the request had been honoured.
  // Read it back so the answer is in the console instead of inferred later.
  try {
    const track = stream.getAudioTracks()[0]
    const s = track?.getSettings?.() as MediaTrackSettings & {
      noiseSuppression?: boolean
      echoCancellation?: boolean
      autoGainControl?: boolean
    }
    if (s) {
      const on = (['noiseSuppression', 'echoCancellation', 'autoGainControl'] as const)
        .filter(k => s[k] === true)
      // eslint-disable-next-line no-console
      console.info('[mic] applied settings', {
        sampleRate: s.sampleRate, channelCount: s.channelCount, deviceId: s.deviceId,
        noiseSuppression: s.noiseSuppression, echoCancellation: s.echoCancellation,
        autoGainControl: s.autoGainControl,
      })
      if (on.length) {
        // eslint-disable-next-line no-console
        console.warn(
          `[mic] processing still ACTIVE despite being requested off: ${on.join(', ')}. ` +
          'This is applied below the browser - disable it in the OS sound settings ' +
          '(device Properties -> Advanced -> audio enhancements) or the vendor audio app.'
        )
      }
      // A 16 kHz stream is band-limited to 8 kHz no matter what the recording
      // context claims, and no later stage can restore it. Say so here, where
      // the device is still the explanation.
      if (typeof s.sampleRate === 'number' && s.sampleRate < 44100) {
        // eslint-disable-next-line no-console
        console.warn(
          `[mic] device is delivering ${s.sampleRate} Hz, so everything above ` +
          `${Math.round(s.sampleRate / 2000) / 1} kHz is already gone. Set the ` +
          'capture device to 48 kHz in the OS sound settings, or use a mic that offers it. ' +
          'Training data recorded this way teaches the model a 7 kHz voice.'
        )
      }
    }
  } catch {
    /* diagnostics only - never block capture */
  }
  return stream
}

export async function setAudioOutputDevice(
  audioElement: HTMLAudioElement,
  outputDeviceId: string
): Promise<'selected' | 'default' | 'unsupported'> {
  const withSink = audioElement as AudioElementWithSink
  if (!withSink.setSinkId) {
    return 'unsupported'
  }

  await withSink.setSinkId(outputDeviceId || '')
  return outputDeviceId ? 'selected' : 'default'
}

export function getSupportedRecordingMimeType(): string {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') {
    return ''
  }

  return [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
    'audio/mp4',
  ].find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) ?? ''
}

export function recordingExtensionForMimeType(mimeType: string): string {
  if (mimeType.includes('ogg')) return 'ogg'
  if (mimeType.includes('mp4')) return 'm4a'
  return 'webm'
}
