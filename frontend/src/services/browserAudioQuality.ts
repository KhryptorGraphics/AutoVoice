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
  /** Measured cliff in Hz; Nyquist when no band limit was found. */
  bandwidthHz?: number
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

/** In-place iterative radix-2 Cooley-Tukey FFT. `re`/`im` length must be a power of two. */
function fftInPlace(re: Float64Array, im: Float64Array): void {
  const n = re.length
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) {
      j ^= bit
    }
    j ^= bit
    if (i < j) {
      const tr = re[i]
      re[i] = re[j]
      re[j] = tr
      const ti = im[i]
      im[i] = im[j]
      im[j] = ti
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len
    const wr = Math.cos(angle)
    const wi = Math.sin(angle)
    const half = len >> 1
    for (let start = 0; start < n; start += len) {
      let cwr = 1
      let cwi = 0
      for (let k = 0; k < half; k += 1) {
        const a = start + k
        const b = a + half
        const vr = re[b] * cwr - im[b] * cwi
        const vi = re[b] * cwi + im[b] * cwr
        re[b] = re[a] - vr
        im[b] = im[a] - vi
        re[a] += vr
        im[a] += vi
        const nwr = cwr * wr - cwi * wi
        cwi = cwr * wi + cwi * wr
        cwr = nwr
      }
    }
  }
}

const CLIFF_FFT_SIZE = 8192
const CLIFF_PROBE_FRAMES = 8
const CLIFF_MIN_DROP_DB = 18.0

/**
 * Where the recording actually stops, found as the CLIFF rather than a level.
 *
 * Mirrors `effective_fmax` in the Python pipeline, for the same reason: an
 * absolute threshold ("last bin above -60 dB") is fooled the moment anything
 * lifts the noise floor, and reads a band-limited take as full band. A
 * brickwall has a signature no amplifier can fake - a sharp DROP - so look for
 * the steepest fall per 1.4x, above 3 kHz, over heavily time-medianed spectra.
 *
 * Returns Nyquist when no cliff is found, i.e. "not band-limited".
 */
function measureBandwidthHz(mono: Float32Array, sampleRate: number): number | undefined {
  const nyquist = sampleRate / 2
  if (mono.length < CLIFF_FFT_SIZE || nyquist <= 3000) {
    return undefined
  }

  // Frame parameters mirror `effective_fmax` in fork_voice_polish.py (n_fft
  // 8192, hop 2048, loud = within 25 dB of peak) so the client reaches the same
  // verdict as the server. The reported frequency can still differ by up to a
  // kilohertz, because the server picks its loud frames from librosa's RMS
  // framing rather than the analysis windows themselves; the band-limited /
  // full-band call does not move.
  const hop = CLIFF_FFT_SIZE >> 2
  const frameCount = Math.floor((mono.length - CLIFF_FFT_SIZE) / hop) + 1
  const frameRms = new Float64Array(frameCount)
  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0
    const offset = frame * hop
    for (let i = 0; i < CLIFF_FFT_SIZE; i += 1) {
      const v = mono[offset + i]
      sum += v * v
    }
    frameRms[frame] = Math.sqrt(sum / CLIFF_FFT_SIZE)
  }

  let peakRms = 0
  for (let frame = 0; frame < frameCount; frame += 1) {
    peakRms = Math.max(peakRms, frameRms[frame])
  }
  if (peakRms <= 0) {
    return undefined
  }
  const loudFloor = peakRms * 10 ** (-25 / 20)
  const loud: number[] = []
  for (let frame = 0; frame < frameCount; frame += 1) {
    if (frameRms[frame] > loudFloor) {
      loud.push(frame)
    }
  }
  if (loud.length === 0) {
    return undefined
  }
  const stride = Math.max(1, Math.floor(loud.length / CLIFF_PROBE_FRAMES))
  const picked: number[] = []
  for (let i = 0; i < loud.length && picked.length < CLIFF_PROBE_FRAMES; i += stride) {
    picked.push(loud[i])
  }

  const bins = (CLIFF_FFT_SIZE >> 1) + 1
  const spectra: Float64Array[] = []
  const window = new Float64Array(CLIFF_FFT_SIZE)
  for (let i = 0; i < CLIFF_FFT_SIZE; i += 1) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (CLIFF_FFT_SIZE - 1))
  }
  for (const frame of picked) {
    const re = new Float64Array(CLIFF_FFT_SIZE)
    const im = new Float64Array(CLIFF_FFT_SIZE)
    const offset = frame * hop
    for (let i = 0; i < CLIFF_FFT_SIZE; i += 1) {
      re[i] = mono[offset + i] * window[i]
    }
    fftInPlace(re, im)
    const mag = new Float64Array(bins)
    for (let bin = 0; bin < bins; bin += 1) {
      mag[bin] = Math.hypot(re[bin], im[bin])
    }
    spectra.push(mag)
  }
  if (spectra.length === 0) {
    return undefined
  }

  const median = new Float64Array(bins)
  const column = new Float64Array(spectra.length)
  for (let bin = 0; bin < bins; bin += 1) {
    for (let s = 0; s < spectra.length; s += 1) {
      column[s] = spectra[s][bin]
    }
    column.sort()
    median[bin] = column[column.length >> 1]
  }

  const level = new Float64Array(bins)
  let maxLevel = 0
  for (let bin = 0; bin < bins; bin += 1) {
    level[bin] = 20 * Math.log10(median[bin] + 1e-12)
    maxLevel = Math.max(maxLevel, level[bin])
  }
  const binHz = sampleRate / CLIFF_FFT_SIZE
  const band: number[] = []
  const bandMedianDb = (loHz: number, hiHz: number): number | null => {
    const lo = Math.max(0, Math.ceil(loHz / binHz))
    const hi = Math.min(bins - 1, Math.floor(hiHz / binHz))
    if (hi < lo) {
      return null
    }
    band.length = 0
    for (let bin = lo; bin <= hi; bin += 1) {
      band.push(level[bin])
    }
    band.sort((a, b) => a - b)
    return band[band.length >> 1] - maxLevel
  }

  let bestHz = nyquist
  let bestDrop = 0
  for (let lo = 3000; lo * 1.4 < nyquist; lo *= 1.15) {
    const below = bandMedianDb(lo * 0.85, lo * 1.05)
    const above = bandMedianDb(lo * 1.3, lo * 1.55)
    if (below === null || above === null) {
      continue
    }
    const drop = below - above
    if (drop > bestDrop) {
      bestDrop = drop
      bestHz = lo * 1.15
    }
  }
  return bestDrop >= CLIFF_MIN_DROP_DB ? bestHz : nyquist
}

const MIN_TRAINING_BANDWIDTH_HZ = 10000

function summarizeBuffer(audioBuffer: AudioBuffer) {
  const sampleCount = audioBuffer.length
  if (sampleCount <= 0 || audioBuffer.numberOfChannels <= 0) {
    return null
  }

  let peakAmplitude = 0
  let sumSquares = 0
  let activeSamples = 0
  let clippedSamples = 0
  const mono = new Float32Array(sampleCount)

  for (let index = 0; index < sampleCount; index += 1) {
    let mixed = 0
    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
      mixed += audioBuffer.getChannelData(channel)[index] ?? 0
    }
    const sample = mixed / audioBuffer.numberOfChannels
    mono[index] = sample
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
    bandwidthHz: measureBandwidthHz(mono, audioBuffer.sampleRate),
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
    // A take can pass every level check above and still be useless as training
    // data: a band-limited capture (Bluetooth HFP, a 16 kHz laptop mic array)
    // is loud, active and clip-free while carrying nothing above ~7 kHz. That
    // is how a 7.2 kHz take reported "pass", and the model then learned it.
    // Warn rather than fail - the take is still usable, and the render-side
    // `extend_bandwidth` stage reconstructs the missing octave.
    if (summary.bandwidthHz !== undefined && summary.bandwidthHz < MIN_TRAINING_BANDWIDTH_HZ) {
      issues.push('band_limited_capture')
      recommendations.push(
        `This take stops at ${(summary.bandwidthHz / 1000).toFixed(1)} kHz. ` +
        'Set the capture device to 48 kHz in the OS sound settings, or use a mic ' +
        'that offers it - training on this teaches the model a narrow voice.',
      )
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
