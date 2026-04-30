export type BrowserTakeQualityStatus = 'pass' | 'warn' | 'fail'

export interface BrowserTakeQualityReport {
  status: BrowserTakeQualityStatus
  issues: string[]
  recommendations: string[]
  durationSeconds: number
  blobSizeBytes: number
  decoded: boolean
  sampleRate?: number
  peakAmplitude?: number
  rmsAmplitude?: number
  activeRatio?: number
  clippingRatio?: number
}

interface BrowserTakeQualityOptions {
  minDurationSeconds?: number
  minPeakAmplitude?: number
  minRmsAmplitude?: number
  minActiveRatio?: number
}

type BrowserWindowWithWebkitAudio = Window &
  typeof globalThis & {
    webkitAudioContext?: typeof AudioContext
  }

const FAILURE_ISSUES = new Set([
  'take_too_short',
  'empty_recording',
  'decoded_audio_empty',
  'effectively_silent',
  'too_quiet',
  'silence_heavy',
])

async function decodeBlob(blob: Blob): Promise<AudioBuffer | null> {
  const AudioContextCtor =
    window.AudioContext ?? (window as BrowserWindowWithWebkitAudio).webkitAudioContext
  if (!AudioContextCtor) {
    return null
  }

  const context = new AudioContextCtor()
  try {
    if (typeof context.decodeAudioData !== 'function') {
      return null
    }
    return await context.decodeAudioData(await blob.arrayBuffer())
  } catch {
    return null
  } finally {
    await context.close().catch(() => undefined)
  }
}

function summarizeBuffer(audioBuffer: AudioBuffer) {
  const sampleCount = audioBuffer.length
  if (sampleCount <= 0 || audioBuffer.numberOfChannels <= 0) {
    return null
  }

  let peakAmplitude = 0
  let sumSquares = 0
  let activeSamples = 0
  let clippedSamples = 0

  for (let index = 0; index < sampleCount; index += 1) {
    let mixed = 0
    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
      mixed += audioBuffer.getChannelData(channel)[index] ?? 0
    }
    const sample = mixed / audioBuffer.numberOfChannels
    const absSample = Math.abs(sample)
    peakAmplitude = Math.max(peakAmplitude, absSample)
    sumSquares += sample * sample
    if (absSample >= 0.0001) {
      activeSamples += 1
    }
    if (absSample >= 0.98) {
      clippedSamples += 1
    }
  }

  return {
    sampleRate: audioBuffer.sampleRate,
    peakAmplitude,
    rmsAmplitude: Math.sqrt(sumSquares / sampleCount),
    activeRatio: activeSamples / sampleCount,
    clippingRatio: clippedSamples / sampleCount,
  }
}

export async function analyzeBrowserRecordingTake(
  blob: Blob,
  durationSeconds: number,
  options: BrowserTakeQualityOptions = {},
): Promise<BrowserTakeQualityReport> {
  const minDurationSeconds = options.minDurationSeconds ?? 1.0
  const minPeakAmplitude = options.minPeakAmplitude ?? 0.0001
  const minRmsAmplitude = options.minRmsAmplitude ?? 0.0005
  const minActiveRatio = options.minActiveRatio ?? 0.02
  const issues: string[] = []
  const recommendations: string[] = []

  if (durationSeconds < minDurationSeconds) {
    issues.push('take_too_short')
    recommendations.push(`Record at least ${minDurationSeconds.toFixed(0)} second before attaching the take.`)
  }
  if (blob.size <= 0) {
    issues.push('empty_recording')
    recommendations.push('Check the selected browser microphone and record again.')
  } else if (blob.size < 1024) {
    issues.push('recording_payload_small')
    recommendations.push('If this warning persists with real hardware, verify the browser can record audio.')
  }

  const decodedBuffer = blob.size > 0 ? await decodeBlob(blob) : null
  const decoded = Boolean(decodedBuffer)
  const summary = decodedBuffer ? summarizeBuffer(decodedBuffer) : null

  if (!decodedBuffer) {
    issues.push('decode_unavailable')
    recommendations.push('The server will decode this browser format before storing the training sample.')
  } else if (!summary) {
    issues.push('decoded_audio_empty')
    recommendations.push('Record again with an active microphone signal.')
  } else {
    if (summary.peakAmplitude < minPeakAmplitude) {
      issues.push('effectively_silent')
      recommendations.push('Record again with the microphone enabled and positioned closer to the singer.')
    }
    if (summary.rmsAmplitude < minRmsAmplitude) {
      issues.push('too_quiet')
      recommendations.push('Increase microphone gain or sing closer to the microphone.')
    }
    if (summary.activeRatio < minActiveRatio) {
      issues.push('silence_heavy')
      recommendations.push('Trim idle time and record a section with clear singing.')
    }
    if (summary.clippingRatio > 0.01) {
      issues.push('clipping_detected')
      recommendations.push('Lower microphone gain to avoid distorted training samples.')
    }
  }

  const status: BrowserTakeQualityStatus = issues.some((issue) => FAILURE_ISSUES.has(issue))
    ? 'fail'
    : issues.length > 0
      ? 'warn'
      : 'pass'

  return {
    status,
    issues,
    recommendations,
    durationSeconds,
    blobSizeBytes: blob.size,
    decoded,
    ...(summary ?? {}),
  }
}
