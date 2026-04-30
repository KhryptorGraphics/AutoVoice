import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CheckCircle, Headphones, Loader2, Mic, Play, RotateCcw, Save, Square, Trash2, Volume2 } from 'lucide-react'
import clsx from 'clsx'

import { apiService, type VoiceProfile } from '../services/api'
import { fetchSongAudio, type UploadedSong } from '../services/karaokeApi'
import {
  getBrowserAudioCapabilities,
  getSupportedRecordingMimeType,
  listBrowserAudioDevices,
  recordingExtensionForMimeType,
  requestBrowserMicrophone,
  setAudioOutputDevice,
  type BrowserAudioDevice,
} from '../services/browserAudioCapture'
import {
  analyzeBrowserRecordingTake,
  type BrowserTakeQualityReport,
} from '../services/browserAudioQuality'

const CALIBRATION_STORAGE_PREFIX = 'autovoice_singalong_calibration:'

interface BrowserSingAlongCaptureProps {
  song?: UploadedSong
  sourceAudioUrl?: string
  sourceId?: string
  sourceLabel?: string
  profiles: VoiceProfile[]
  disabled?: boolean
  onSampleAttached?: () => Promise<void> | void
}

function formatDuration(seconds: number) {
  const safeSeconds = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(safeSeconds / 60)
  const remainder = safeSeconds % 60
  return `${minutes}:${remainder.toString().padStart(2, '0')}`
}

function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop())
}

function sanitizeSourceId(sourceId: string) {
  const normalized = sourceId.trim().replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '')
  return normalized || 'browser-source'
}

function writeString(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index))
  }
}

function audioBufferToWav(buffer: AudioBuffer): Blob {
  const channelCount = buffer.numberOfChannels
  const sampleCount = buffer.length
  const bytesPerSample = 2
  const blockAlign = channelCount * bytesPerSample
  const wav = new ArrayBuffer(44 + sampleCount * blockAlign)
  const view = new DataView(wav)
  writeString(view, 0, 'RIFF')
  view.setUint32(4, 36 + sampleCount * blockAlign, true)
  writeString(view, 8, 'WAVE')
  writeString(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, channelCount, true)
  view.setUint32(24, buffer.sampleRate, true)
  view.setUint32(28, buffer.sampleRate * blockAlign, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, 16, true)
  writeString(view, 36, 'data')
  view.setUint32(40, sampleCount * blockAlign, true)
  let offset = 44
  for (let index = 0; index < sampleCount; index += 1) {
    for (let channel = 0; channel < channelCount; channel += 1) {
      const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[index] ?? 0))
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
      offset += bytesPerSample
    }
  }
  return new Blob([view], { type: 'audio/wav' })
}

async function decodeTakeBlob(blob: Blob): Promise<AudioBuffer | null> {
  const AudioContextCtor = window.AudioContext ?? (window as Window & typeof globalThis & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
  if (!AudioContextCtor) return null
  const context = new AudioContextCtor()
  try {
    return await context.decodeAudioData(await blob.arrayBuffer())
  } catch {
    return null
  } finally {
    await context.close().catch(() => undefined)
  }
}

export function BrowserSingAlongCapture({
  song,
  sourceAudioUrl,
  sourceId,
  sourceLabel,
  profiles,
  disabled = false,
  onSampleAttached,
}: BrowserSingAlongCaptureProps) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const levelFrameRef = useRef<number | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const recordingStartedAtRef = useRef<number | null>(null)
  const playbackStartedAtRef = useRef<number | null>(null)
  const captureOffsetMsRef = useRef<number | null>(null)
  const originalTrackPositionRef = useRef<number>(0)
  const takeUrlRef = useRef<string | null>(null)

  const [browserDevices, setBrowserDevices] = useState<BrowserAudioDevice[]>([])
  const [selectedInputId, setSelectedInputId] = useState('')
  const [selectedOutputId, setSelectedOutputId] = useState('')
  const [selectedProfileId, setSelectedProfileId] = useState('')
  const [songAudioUrl, setSongAudioUrl] = useState<string | null>(null)
  const [loadingSongAudio, setLoadingSongAudio] = useState(false)
  const [micReady, setMicReady] = useState(false)
  const [recording, setRecording] = useState(false)
  const [inputLevel, setInputLevel] = useState(0)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [takeBlob, setTakeBlob] = useState<Blob | null>(null)
  const [takeUrl, setTakeUrl] = useState<string | null>(null)
  const [takeDuration, setTakeDuration] = useState(0)
  const [takeQuality, setTakeQuality] = useState<BrowserTakeQualityReport | null>(null)
  const [qualityChecking, setQualityChecking] = useState(false)
  const [latencyCalibrationMs, setLatencyCalibrationMs] = useState(0)
  const [calibrationStatus, setCalibrationStatus] = useState<'missing' | 'pass' | 'warn'>('missing')
  const [alignedUploadReady, setAlignedUploadReady] = useState(false)
  const [alignmentWarning, setAlignmentWarning] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sinkStatus, setSinkStatus] = useState<'idle' | 'selected' | 'default' | 'unsupported' | 'error'>('idle')
  const [attaching, setAttaching] = useState(false)

  const targetProfiles = useMemo(
    () => profiles.filter((profile) => profile.profile_role !== 'source_artist'),
    [profiles],
  )
  const inputDevices = useMemo(
    () => browserDevices.filter((device) => device.kind === 'audioinput'),
    [browserDevices],
  )
  const outputDevices = useMemo(
    () => browserDevices.filter((device) => device.kind === 'audiooutput'),
    [browserDevices],
  )
  const capabilities = getBrowserAudioCapabilities(audioRef.current)
  const canRecord = capabilities.hasGetUserMedia && capabilities.hasMediaRecorder
  const audioSourceId = sanitizeSourceId(sourceId ?? song?.song_id ?? 'browser-source')
  const audioSourceLabel = sourceLabel ?? song?.song_id ?? 'Selected original audio'
  const calibrationStorageKey = `${CALIBRATION_STORAGE_PREFIX}${audioSourceId}`
  const inputDeviceLabel = inputDevices.find((device) => device.deviceId === selectedInputId)?.label ?? null
  const outputDeviceLabel = outputDevices.find((device) => device.deviceId === selectedOutputId)?.label ?? null
  const readinessReport = useMemo(() => {
    const checks = [
      { id: 'secure_context', label: 'HTTPS or localhost', passed: capabilities.isSecureContext },
      { id: 'mic_permission', label: 'Microphone permission', passed: micReady },
      { id: 'output_selection', label: 'Output device selectable or defaulted', passed: sinkStatus !== 'error' && sinkStatus !== 'idle' },
      { id: 'latency_calibration', label: 'Latency calibration', passed: calibrationStatus === 'pass' },
      { id: 'target_profile', label: 'Target profile selected', passed: Boolean(selectedProfileId) },
    ]
    const failed = checks.filter((check) => !check.passed)
    return {
      status: failed.length === 0 ? 'pass' : failed.length <= 2 ? 'warn' : 'fail',
      checks,
      failed_check_ids: failed.map((check) => check.id),
      latency_calibration_ms: latencyCalibrationMs,
      input_device_label: inputDeviceLabel,
      output_device_label: outputDeviceLabel,
    }
  }, [calibrationStatus, capabilities.isSecureContext, inputDeviceLabel, latencyCalibrationMs, micReady, outputDeviceLabel, selectedProfileId, sinkStatus])

  const refreshBrowserDevices = useCallback(async () => {
    const devices = await listBrowserAudioDevices()
    setBrowserDevices(devices)

    if (!selectedInputId) {
      setSelectedInputId(devices.find((device) => device.kind === 'audioinput')?.deviceId ?? '')
    }
    if (!selectedOutputId) {
      setSelectedOutputId(devices.find((device) => device.kind === 'audiooutput')?.deviceId ?? '')
    }
  }, [selectedInputId, selectedOutputId])

  const stopLevelMeter = useCallback(() => {
    if (levelFrameRef.current !== null) {
      window.cancelAnimationFrame(levelFrameRef.current)
      levelFrameRef.current = null
    }
    audioContextRef.current?.close().catch(() => undefined)
    audioContextRef.current = null
    setInputLevel(0)
  }, [])

  const startLevelMeter = useCallback((stream: MediaStream) => {
    stopLevelMeter()
    if (typeof AudioContext === 'undefined') {
      return
    }

    const context = new AudioContext()
    const analyser = context.createAnalyser()
    analyser.fftSize = 1024
    const source = context.createMediaStreamSource(stream)
    source.connect(analyser)
    audioContextRef.current = context

    const data = new Float32Array(analyser.fftSize)
    const tick = () => {
      analyser.getFloatTimeDomainData(data)
      let sum = 0
      for (const sample of data) {
        sum += sample * sample
      }
      setInputLevel(Math.min(1, Math.sqrt(sum / data.length) * 6))
      levelFrameRef.current = window.requestAnimationFrame(tick)
    }
    tick()
  }, [stopLevelMeter])

  const releaseTake = useCallback(() => {
    if (takeUrlRef.current) {
      URL.revokeObjectURL(takeUrlRef.current)
      takeUrlRef.current = null
    }
    setTakeUrl(null)
    setTakeBlob(null)
    setTakeDuration(0)
    setTakeQuality(null)
    setQualityChecking(false)
    setAlignedUploadReady(false)
    setAlignmentWarning(null)
    setStatus(null)
  }, [])

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop()
    }
    audioRef.current?.pause()
  }, [])

  useEffect(() => {
    if (targetProfiles.length > 0 && !targetProfiles.some((profile) => profile.profile_id === selectedProfileId)) {
      setSelectedProfileId(targetProfiles[0].profile_id)
    }
  }, [selectedProfileId, targetProfiles])

  useEffect(() => {
    void refreshBrowserDevices().catch(() => undefined)
  }, [refreshBrowserDevices])

  useEffect(() => {
    const raw = window.localStorage.getItem(calibrationStorageKey)
    if (!raw) {
      setLatencyCalibrationMs(0)
      setCalibrationStatus('missing')
      return
    }
    try {
      const parsed = JSON.parse(raw) as { latency_ms?: number }
      const latency = Number(parsed.latency_ms ?? 0)
      if (Number.isFinite(latency) && latency >= 0) {
        setLatencyCalibrationMs(latency)
        setCalibrationStatus(latency <= 180 ? 'pass' : 'warn')
      }
    } catch {
      setLatencyCalibrationMs(0)
      setCalibrationStatus('missing')
    }
  }, [calibrationStorageKey])

  useEffect(() => {
    let cancelled = false
    let objectUrl: string | null = null

    const loadSongAudio = async () => {
      releaseTake()
      setError(null)

      if (sourceAudioUrl) {
        setSongAudioUrl(sourceAudioUrl)
        setLoadingSongAudio(false)
        return
      }

      if (!song?.song_id) {
        setSongAudioUrl(null)
        setLoadingSongAudio(false)
        setError('Select an original audio file before recording.')
        return
      }

      setLoadingSongAudio(true)
      try {
        const blob = await fetchSongAudio(song.song_id)
        if (cancelled) return
        objectUrl = URL.createObjectURL(blob)
        setSongAudioUrl(objectUrl)
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load song audio')
          setSongAudioUrl(null)
        }
      } finally {
        if (!cancelled) {
          setLoadingSongAudio(false)
        }
      }
    }

    void loadSongAudio()

    return () => {
      cancelled = true
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl)
      }
    }
  }, [releaseTake, song?.song_id, sourceAudioUrl])

  useEffect(() => {
    return () => {
      stopRecording()
      stopStream(streamRef.current)
      stopLevelMeter()
      releaseTake()
    }
  }, [releaseTake, stopLevelMeter, stopRecording])

  useEffect(() => {
    if (!recording) {
      return undefined
    }

    const interval = window.setInterval(() => {
      if (recordingStartedAtRef.current) {
        setElapsedSeconds((Date.now() - recordingStartedAtRef.current) / 1000)
      }
    }, 250)

    return () => window.clearInterval(interval)
  }, [recording])

  const enableMicrophone = async () => {
    setError(null)
    setStatus(null)
    try {
      stopStream(streamRef.current)
      const stream = await requestBrowserMicrophone(selectedInputId || undefined)
      streamRef.current = stream
      setMicReady(true)
      startLevelMeter(stream)
      await refreshBrowserDevices()
      setStatus('Browser microphone is ready.')
    } catch (err) {
      setMicReady(false)
      setError(err instanceof Error ? err.message : 'Failed to access browser microphone')
    }
  }

  const applyOutputDevice = async (deviceId: string) => {
    setSelectedOutputId(deviceId)
    setError(null)

    if (!audioRef.current) {
      return
    }

    try {
      const result = await setAudioOutputDevice(audioRef.current, deviceId)
      setSinkStatus(result)
      if (result === 'unsupported') {
        setStatus('This browser does not support explicit output-device selection. Playback will use the system default.')
      } else {
        setStatus(result === 'selected' ? 'Browser output device selected.' : 'Browser output reset to system default.')
      }
    } catch (err) {
      setSinkStatus('error')
      setError(err instanceof Error ? err.message : 'Failed to select browser output device')
    }
  }

  const playTestTone = async () => {
    setError(null)
    try {
      const context = new AudioContext()
      const oscillator = context.createOscillator()
      const gain = context.createGain()
      oscillator.frequency.value = 880
      gain.gain.value = 0.08
      oscillator.connect(gain)
      gain.connect(context.destination)
      oscillator.start()
      oscillator.stop(context.currentTime + 0.18)
      window.setTimeout(() => void context.close().catch(() => undefined), 280)
      setStatus('Output test tone played.')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to play output test tone')
    }
  }

  const runLatencyCalibration = async () => {
    setError(null)
    const started = performance.now()
    await playTestTone()
    const estimatedLatency = Math.round(Math.max(0, performance.now() - started + 60))
    setLatencyCalibrationMs(estimatedLatency)
    setCalibrationStatus(estimatedLatency <= 180 ? 'pass' : 'warn')
    window.localStorage.setItem(calibrationStorageKey, JSON.stringify({
      latency_ms: estimatedLatency,
      input_device_label: inputDeviceLabel,
      output_device_label: outputDeviceLabel,
      calibrated_at: new Date().toISOString(),
    }))
    setStatus(`Latency calibration saved at ${estimatedLatency} ms.`)
  }

  const startRecording = async () => {
    if (!canRecord) {
      setError('This browser does not support microphone recording.')
      return
    }
    if (!songAudioUrl || !audioRef.current) {
      setError('Song audio is not ready yet.')
      return
    }
    if (!selectedProfileId) {
      setError('Select a target profile before recording.')
      return
    }

    setError(null)
    setStatus(null)
    releaseTake()

    try {
      if (!streamRef.current) {
        const stream = await requestBrowserMicrophone(selectedInputId || undefined)
        streamRef.current = stream
        setMicReady(true)
        startLevelMeter(stream)
        await refreshBrowserDevices()
      }

      await applyOutputDevice(selectedOutputId)

      const activeStream = streamRef.current
      if (!activeStream) {
        throw new Error('Browser microphone is not available')
      }

      const mimeType = getSupportedRecordingMimeType()
      chunksRef.current = []
      const recorder = new MediaRecorder(activeStream, mimeType ? { mimeType } : undefined)
      recorderRef.current = recorder

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }
      recorder.onstop = () => {
        const recordedType = mimeType || chunksRef.current[0]?.type || 'audio/webm'
        const blob = new Blob(chunksRef.current, { type: recordedType })
        const url = URL.createObjectURL(blob)
        takeUrlRef.current = url
        const duration = recordingStartedAtRef.current
          ? (Date.now() - recordingStartedAtRef.current) / 1000
          : elapsedSeconds
        setTakeBlob(blob)
        setTakeUrl(url)
        setTakeDuration(duration)
        setRecording(false)
        setElapsedSeconds(0)
        recordingStartedAtRef.current = null
        audioRef.current?.pause()
        if (audioRef.current) {
          audioRef.current.currentTime = 0
        }
        setTakeQuality(null)
        setQualityChecking(true)
        setStatus('Take recorded. Checking local quality before attach.')
        void decodeTakeBlob(blob).then((decodedBuffer) => {
          if (takeUrlRef.current !== url) return
          if (decodedBuffer) {
            setAlignedUploadReady(true)
            setAlignmentWarning(null)
          } else {
            setAlignedUploadReady(false)
            setAlignmentWarning('Browser could not decode this recording for aligned WAV export. The raw take will upload with alignment metadata.')
          }
        })
        void analyzeBrowserRecordingTake(blob, duration)
          .then((report) => {
            if (takeUrlRef.current !== url) return
            setTakeQuality(report)
            if (report.status === 'fail') {
              setError('Recorded take failed local quality checks. Discard it and record a longer, audible take.')
              setStatus(null)
            } else {
              setStatus(
                report.status === 'pass'
                  ? 'Take passed local quality checks. Preview it before attaching.'
                  : 'Take recorded with local quality warnings. Preview it before attaching.',
              )
            }
          })
          .catch(() => {
            if (takeUrlRef.current !== url) return
            setTakeQuality({
              status: 'warn',
              issues: ['quality_check_failed'],
              recommendations: ['Preview the take and retry if it sounds incorrect.'],
              durationSeconds: duration,
              blobSizeBytes: blob.size,
              decoded: false,
            })
            setStatus('Take recorded, but the browser quality check could not complete. Preview it before attaching.')
          })
          .finally(() => {
            if (takeUrlRef.current === url) {
              setQualityChecking(false)
            }
          })
      }

      const audio = audioRef.current
      audio.currentTime = 0
      originalTrackPositionRef.current = audio.currentTime
      recorder.start(1000)
      const captureStartedAt = performance.now()
      recordingStartedAtRef.current = Date.now()
      setRecording(true)
      await audio.play()
      playbackStartedAtRef.current = performance.now()
      captureOffsetMsRef.current = captureStartedAt - playbackStartedAtRef.current
    } catch (err) {
      if (recorderRef.current?.state === 'recording') {
        recorderRef.current.stop()
      }
      setRecording(false)
      setError(err instanceof Error ? err.message : 'Failed to start browser recording')
    }
  }

  const attachTake = async () => {
    if (!takeBlob || !selectedProfileId) {
      return
    }
    if (!takeQuality || takeQuality.status === 'fail') {
      setError('Record a take that passes local quality checks before attaching it.')
      return
    }

    setAttaching(true)
    setError(null)
    setStatus(null)
    try {
      const decodedBuffer = await decodeTakeBlob(takeBlob)
      const uploadBlob = decodedBuffer ? audioBufferToWav(decodedBuffer) : takeBlob
      const extension = decodedBuffer ? 'wav' : recordingExtensionForMimeType(takeBlob.type)
      const alignedUpload = Boolean(decodedBuffer)
      const alignmentOffsetMs = Math.round(
        (captureOffsetMsRef.current ?? 0) - latencyCalibrationMs,
      )
      const file = new File([uploadBlob], `browser-singalong-${audioSourceId}.${extension}`, {
        type: uploadBlob.type || takeBlob.type || 'audio/webm',
      })
      await apiService.uploadSample(selectedProfileId, file, {
        source: 'browser_singalong_capture',
        provenance: 'browser-client sing-along recording',
        source_file: audioSourceLabel,
        source_song_id: song?.song_id ?? null,
        source_audio_id: audioSourceId,
        duration_seconds: takeDuration,
        readiness_report: readinessReport,
        latency_calibration_ms: latencyCalibrationMs,
        alignment_offset_ms: alignmentOffsetMs,
        original_track_position_seconds: originalTrackPositionRef.current,
        input_device_label: inputDeviceLabel,
        output_device_label: outputDeviceLabel,
        aligned_upload: alignedUpload,
        alignment_warning: alignedUpload ? null : alignmentWarning,
        quality_metadata: {
          browser_capture: takeQuality,
        },
      })
      releaseTake()
      setStatus('Recorded take attached to the selected profile.')
      await onSampleAttached?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to attach recorded take')
    } finally {
      setAttaching(false)
    }
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6 lg:col-span-2" data-testid="browser-singalong-capture">
      <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Mic size={20} />
            Record Training Take From This Browser
          </h3>
          <p className="text-sm text-gray-400 mt-1">
            Play the full artist song through this browser&apos;s headphones while recording this browser&apos;s microphone.
          </p>
          <p className="mt-2 text-xs text-gray-500" data-testid="browser-capture-source-label">
            Original playback source: {audioSourceLabel}
          </p>
        </div>
        <div className="rounded-lg border border-blue-700 bg-blue-950/40 px-3 py-2 text-xs text-blue-100">
          Trusted LAN HTTPS required for remote browser mic access.
        </div>
      </div>

      {!capabilities.isSecureContext && (
        <div className="mt-4 rounded-lg border border-yellow-700 bg-yellow-950/40 p-3 text-sm text-yellow-100">
          Browser media devices require HTTPS on LAN. Use localhost or serve AutoVoice through HTTPS before recording.
        </div>
      )}

      <div className="mt-5 rounded-lg border border-gray-700 bg-gray-900/50 p-4" data-testid="singalong-readiness-flow">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div>
            <h4 className="text-sm font-semibold text-white">Device readiness</h4>
            <p className="mt-1 text-xs text-gray-400">
              Calibrate this browser before recording so the uploaded take carries playback and mic alignment metadata.
            </p>
          </div>
          <span className={clsx(
            'rounded-full px-3 py-1 text-xs font-medium',
            readinessReport.status === 'pass'
              ? 'bg-emerald-500/15 text-emerald-200'
              : readinessReport.status === 'warn'
                ? 'bg-amber-500/15 text-amber-200'
                : 'bg-red-500/15 text-red-200',
          )}>
            {readinessReport.status === 'pass' ? 'Ready' : readinessReport.status === 'warn' ? 'Needs review' : 'Blocked'}
          </span>
        </div>
        <div className="mt-4 grid grid-cols-1 gap-2 md:grid-cols-5">
          {readinessReport.checks.map((check) => (
            <div key={check.id} className="rounded border border-gray-800 bg-gray-950/70 p-3 text-xs">
              <div className={check.passed ? 'text-emerald-300' : 'text-gray-500'}>
                {check.passed ? 'Pass' : 'Pending'}
              </div>
              <div className="mt-1 text-gray-300">{check.label}</div>
            </div>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={playTestTone}
            className="rounded-lg bg-gray-700 px-3 py-2 text-sm hover:bg-gray-600"
          >
            Play Test Tone
          </button>
          <button
            type="button"
            onClick={runLatencyCalibration}
            className="rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium hover:bg-blue-500"
          >
            Calibrate Latency
          </button>
          <span className="self-center text-xs text-gray-400">
            Saved latency: {latencyCalibrationMs} ms · aligned WAV: {alignedUploadReady ? 'available' : 'pending'}
          </span>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div>
          <label className="mb-2 block text-sm text-gray-400">Target profile</label>
          <select
            value={selectedProfileId}
            onChange={(event) => setSelectedProfileId(event.target.value)}
            disabled={disabled || recording || targetProfiles.length === 0}
            data-testid="browser-capture-profile-select"
            className="w-full rounded bg-gray-700 p-3"
          >
            {targetProfiles.length === 0 && <option value="">Create a target profile first</option>}
            {targetProfiles.map((profile) => (
              <option key={profile.profile_id} value={profile.profile_id}>
                {profile.name || profile.profile_id} ({profile.sample_count} samples)
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-2 flex items-center gap-2 text-sm text-gray-400">
            <Mic size={14} />
            Browser microphone
          </label>
          <select
            value={selectedInputId}
            onChange={(event) => setSelectedInputId(event.target.value)}
            disabled={disabled || recording || !capabilities.hasEnumerateDevices}
            data-testid="browser-capture-input-select"
            className="w-full rounded bg-gray-700 p-3"
          >
            <option value="">System default microphone</option>
            {inputDevices.map((device) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-2 flex items-center gap-2 text-sm text-gray-400">
            <Headphones size={14} />
            Browser song output
          </label>
          <select
            value={selectedOutputId}
            onChange={(event) => applyOutputDevice(event.target.value)}
            disabled={disabled || recording || !capabilities.hasEnumerateDevices}
            data-testid="browser-capture-output-select"
            className="w-full rounded bg-gray-700 p-3"
          >
            <option value="">System default output</option>
            {outputDevices.map((device) => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-[1fr_220px]">
        <div className="rounded-lg border border-gray-700 bg-gray-900/50 p-4">
          <div className="mb-3 flex items-center justify-between text-sm text-gray-400">
            <span className="flex items-center gap-2">
              <Volume2 size={14} />
              Full artist song playback
            </span>
            {loadingSongAudio && (
              <span className="flex items-center gap-1">
                <Loader2 size={14} className="animate-spin" />
                Loading
              </span>
            )}
          </div>
          <audio
            ref={audioRef}
            src={songAudioUrl ?? undefined}
            controls
            onEnded={() => {
              if (recording) {
                stopRecording()
              }
            }}
            data-testid="browser-capture-song-audio"
            className="w-full"
          />
          <p className="mt-2 text-xs text-gray-500">
            Output selector status: {sinkStatus === 'idle' ? 'not selected yet' : sinkStatus}.
          </p>
        </div>

        <div className="rounded-lg border border-gray-700 bg-gray-900/50 p-4">
          <div className="mb-2 flex items-center justify-between text-sm text-gray-400">
            <span>Mic level</span>
            <span>{Math.round(inputLevel * 100)}%</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-gray-700">
            <div
              className={clsx('h-full transition-all', recording ? 'bg-red-500' : 'bg-green-500')}
              style={{ width: `${inputLevel * 100}%` }}
            />
          </div>
          <button
            onClick={enableMicrophone}
            disabled={disabled || recording || !capabilities.hasGetUserMedia}
            data-testid="browser-capture-enable-mic"
            className="mt-4 w-full rounded-lg bg-gray-700 px-4 py-2 text-sm hover:bg-gray-600 disabled:cursor-not-allowed disabled:text-gray-500"
          >
            {micReady ? 'Refresh Mic Permission' : 'Enable Browser Mic'}
          </button>
        </div>
      </div>

      <div className="mt-5 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="text-sm text-gray-400">
          {recording ? (
            <span className="text-red-300" data-testid="browser-capture-recording-status">
              Recording {formatDuration(elapsedSeconds)}
            </span>
          ) : takeBlob ? (
            <span>
              Recorded take: {formatDuration(takeDuration)} · {(takeBlob.size / 1024).toFixed(1)} KB
              {qualityChecking ? ' · checking quality' : ''}
            </span>
          ) : (
            <span>Enable the mic, select devices, then record a take.</span>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          {!recording ? (
            <button
              onClick={startRecording}
              disabled={disabled || !canRecord || !songAudioUrl || !selectedProfileId}
              data-testid="browser-capture-start"
              className="flex items-center gap-2 rounded-lg bg-red-600 px-4 py-2 font-medium hover:bg-red-700 disabled:bg-gray-700 disabled:text-gray-500"
            >
              <Play size={16} />
              Record Take
            </button>
          ) : (
            <button
              onClick={stopRecording}
              data-testid="browser-capture-stop"
              className="flex items-center gap-2 rounded-lg bg-gray-200 px-4 py-2 font-medium text-gray-900 hover:bg-white"
            >
              <Square size={16} />
              Stop Recording
            </button>
          )}

          {takeBlob && takeUrl && (
            <>
              <audio src={takeUrl} controls data-testid="browser-capture-take-preview" className="max-w-xs" />
              <button
                onClick={attachTake}
                disabled={attaching || qualityChecking || !takeQuality || takeQuality.status === 'fail'}
                data-testid="browser-capture-attach"
                className="flex items-center gap-2 rounded-lg bg-green-600 px-4 py-2 font-medium hover:bg-green-700 disabled:bg-gray-700"
              >
                {attaching ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
                Attach to Profile
              </button>
              <button
                onClick={releaseTake}
                disabled={attaching}
                data-testid="browser-capture-discard"
                className="flex items-center gap-2 rounded-lg bg-gray-700 px-4 py-2 hover:bg-gray-600 disabled:text-gray-500"
              >
                <Trash2 size={16} />
                Discard
              </button>
            </>
          )}

          <button
            onClick={() => void refreshBrowserDevices()}
            disabled={disabled || recording}
            className="flex items-center gap-2 rounded-lg bg-gray-700 px-4 py-2 hover:bg-gray-600 disabled:text-gray-500"
          >
            <RotateCcw size={16} />
            Refresh Devices
          </button>
        </div>
      </div>

      {status && (
        <div className="mt-4 flex items-center gap-2 rounded-lg border border-green-700 bg-green-950/30 p-3 text-sm text-green-100">
          <CheckCircle size={16} />
          {status}
        </div>
      )}
      {takeQuality && (
        <div
          className={clsx(
            'mt-4 rounded-lg border p-3 text-sm',
            takeQuality.status === 'fail'
              ? 'border-red-700 bg-red-950/40 text-red-100'
              : takeQuality.status === 'warn'
                ? 'border-yellow-700 bg-yellow-950/40 text-yellow-100'
                : 'border-green-700 bg-green-950/30 text-green-100',
          )}
          data-testid="browser-capture-quality-status"
        >
          <div className="font-medium">Local take check: {takeQuality.status}</div>
          {takeQuality.issues.length > 0 && (
            <div className="mt-1 text-xs">
              Issues: {takeQuality.issues.join(', ')}
            </div>
          )}
          {takeQuality.recommendations.length > 0 && (
            <div className="mt-1 text-xs">
              Next: {takeQuality.recommendations[0]}
            </div>
          )}
        </div>
      )}
      {alignmentWarning && (
        <div className="mt-4 rounded-lg border border-yellow-700 bg-yellow-950/40 p-3 text-sm text-yellow-100" data-testid="browser-capture-alignment-warning">
          {alignmentWarning}
        </div>
      )}
      {error && (
        <div className="mt-4 rounded-lg border border-red-700 bg-red-950/40 p-3 text-sm text-red-100" data-testid="browser-capture-error">
          {error}
        </div>
      )}
      {!canRecord && (
        <div className="mt-4 rounded-lg border border-yellow-700 bg-yellow-950/40 p-3 text-sm text-yellow-100">
          This browser cannot record microphone audio. Use a current Chromium, Edge, or Firefox build.
        </div>
      )}
    </div>
  )
}
