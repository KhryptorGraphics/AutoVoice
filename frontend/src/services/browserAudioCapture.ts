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
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  }

  if (deviceId) {
    audio.deviceId = { exact: deviceId }
  }

  return navigator.mediaDevices.getUserMedia({ audio })
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
