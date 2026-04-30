import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react'
import { Headphones, Loader2, Mic, Music, RefreshCw, UploadCloud } from 'lucide-react'

import { BrowserSingAlongCapture } from '../components/BrowserSingAlongCapture'
import { StatusBanner } from '../components/StatusBanner'
import { useToastContext } from '../contexts/ToastContext'
import { apiService, type SingAlongSource, type VoiceProfile } from '../services/api'

type BrowserAudioSource = {
  url: string
  id: string
  label: string
  detail: string
}

const ACCEPTED_AUDIO_TYPES = '.wav,.mp3,.m4a,.flac,.ogg,.webm,audio/*'

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 KB'
  }
  const units = ['B', 'KB', 'MB', 'GB']
  let size = bytes
  let unitIndex = 0
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }
  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`
}

function makeSourceId(file: File): string {
  return `${file.name}-${file.lastModified}`.replace(/[^a-z0-9._-]+/gi, '-')
}

export function SingAlongPage() {
  const [profiles, setProfiles] = useState<VoiceProfile[]>([])
  const [profilesLoading, setProfilesLoading] = useState(true)
  const [profilesError, setProfilesError] = useState<string | null>(null)
  const [source, setSource] = useState<BrowserAudioSource | null>(null)
  const [savedSources, setSavedSources] = useState<SingAlongSource[]>([])
  const [savedSourcesLoading, setSavedSourcesLoading] = useState(false)
  const [savedSourcesError, setSavedSourcesError] = useState<string | null>(null)
  const [selectedSavedSourceId, setSelectedSavedSourceId] = useState('')
  const [savedSourceApplying, setSavedSourceApplying] = useState(false)
  const { error: toastError } = useToastContext()

  const targetProfileCount = useMemo(
    () => profiles.filter((profile) => profile.profile_role !== 'source_artist').length,
    [profiles],
  )

  const loadProfiles = useCallback(async () => {
    setProfilesLoading(true)
    setProfilesError(null)
    try {
      setProfiles(await apiService.listProfiles())
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load voice profiles'
      setProfilesError(message)
      toastError(message)
    } finally {
      setProfilesLoading(false)
    }
  }, [toastError])

  useEffect(() => {
    void loadProfiles()
  }, [loadProfiles])

  const loadSavedSources = useCallback(async () => {
    setSavedSourcesLoading(true)
    setSavedSourcesError(null)
    try {
      const sources = await apiService.listSingAlongSources()
      setSavedSources(sources)
      setSelectedSavedSourceId((current) => current || sources[0]?.asset_id || '')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load saved originals'
      setSavedSourcesError(message)
    } finally {
      setSavedSourcesLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadSavedSources()
  }, [loadSavedSources])

  useEffect(() => {
    return () => {
      if (source?.url) {
        URL.revokeObjectURL(source.url)
      }
    }
  }, [source])

  const handleOriginalAudioChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }
    setSource({
      url: URL.createObjectURL(file),
      id: makeSourceId(file),
      label: file.name,
      detail: `${file.type || 'audio file'} · ${formatBytes(file.size)}`,
    })
  }

  const handleSavedSourceSelect = async () => {
    const savedSource = savedSources.find((candidate) => candidate.asset_id === selectedSavedSourceId)
    if (!savedSource) {
      return
    }
    setSavedSourceApplying(true)
    try {
      const blob = await apiService.fetchSingAlongSourceAudio(savedSource.asset_id)
      setSource({
        url: URL.createObjectURL(blob),
        id: savedSource.asset_id,
        label: savedSource.label || savedSource.filename,
        detail: `${savedSource.source || savedSource.kind} · ${savedSource.filename}`,
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load saved original'
      toastError(message)
    } finally {
      setSavedSourceApplying(false)
    }
  }

  return (
    <div className="space-y-6" data-testid="singalong-page">
      <div className="overflow-hidden rounded-2xl border border-sky-500/20 bg-gradient-to-br from-gray-800 via-slate-900 to-sky-950 p-6 shadow-xl">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-sky-400/30 bg-sky-400/10 px-3 py-1 text-sm text-sky-100">
              <Headphones className="h-4 w-4" />
              Browser audio I/O
            </div>
            <h1 className="text-3xl font-bold text-white">Sing-Along Recording Studio</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
              Play an original artist track through the headphones or speakers attached to this browser computer,
              record the singer through this browser computer&apos;s microphone, then attach the take to a voice profile.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="rounded-xl border border-white/10 bg-white/5 p-3">
              <div className="flex items-center gap-2 text-slate-300">
                <Music className="h-4 w-4 text-sky-300" />
                Original
              </div>
              <div className="mt-1 font-semibold text-white">Browser playback</div>
            </div>
            <div className="rounded-xl border border-white/10 bg-white/5 p-3">
              <div className="flex items-center gap-2 text-slate-300">
                <Mic className="h-4 w-4 text-emerald-300" />
                Singer
              </div>
              <div className="mt-1 font-semibold text-white">Browser mic</div>
            </div>
          </div>
        </div>
      </div>

      <StatusBanner
        tone="info"
        title="Device selection happens in the browser"
        message="The output and input lists come from the computer using this web page. On another LAN machine, open AutoVoice over HTTPS and this page will use that machine's headphones, speakers, and mics."
        testId="singalong-browser-routing-banner"
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <section className="rounded-xl border border-gray-700 bg-gray-800 p-5">
          <label className="mb-3 block text-sm font-semibold text-gray-200" htmlFor="singalong-original-file">
            Original artist audio
          </label>
          <div className="rounded-xl border border-dashed border-sky-500/40 bg-gray-900/60 p-5">
            <input
              id="singalong-original-file"
              type="file"
              accept={ACCEPTED_AUDIO_TYPES}
              onChange={handleOriginalAudioChange}
              data-testid="singalong-original-file-input"
              className="block w-full cursor-pointer rounded-lg border border-gray-700 bg-gray-900 text-sm text-gray-300 file:mr-4 file:border-0 file:bg-sky-600 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white hover:file:bg-sky-500"
            />
            <div className="mt-4 flex items-start gap-3 text-sm text-gray-300">
              <UploadCloud className="mt-0.5 h-5 w-5 text-sky-300" />
              <div>
                <div className="font-medium text-white">
                  {source ? source.label : 'Choose the full song the singer will hear.'}
                </div>
                <div className="mt-1 text-xs text-gray-400">
                  {source
                    ? source.detail
                    : 'Supported formats depend on the current browser, usually WAV, MP3, M4A, FLAC, OGG, and WebM.'}
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 rounded-xl border border-gray-700 bg-gray-900/70 p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <label className="text-sm font-semibold text-gray-200" htmlFor="singalong-saved-original">
                Saved original audio
              </label>
              <button
                type="button"
                onClick={loadSavedSources}
                disabled={savedSourcesLoading}
                className="inline-flex items-center gap-2 rounded-lg border border-gray-600 px-3 py-1.5 text-xs font-medium text-gray-200 hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-60"
                aria-label="Refresh saved original audio"
              >
                {savedSourcesLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                Refresh
              </button>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row">
              <select
                id="singalong-saved-original"
                value={selectedSavedSourceId}
                onChange={(event) => setSelectedSavedSourceId(event.target.value)}
                disabled={savedSourcesLoading || savedSources.length === 0}
                className="min-w-0 flex-1 rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 disabled:opacity-60"
                data-testid="singalong-saved-original-select"
              >
                {savedSources.length === 0 ? (
                  <option value="">No saved originals</option>
                ) : (
                  savedSources.map((savedSource) => (
                    <option key={savedSource.asset_id} value={savedSource.asset_id}>
                      {savedSource.label || savedSource.filename}
                    </option>
                  ))
                )}
              </select>
              <button
                type="button"
                onClick={handleSavedSourceSelect}
                disabled={!selectedSavedSourceId || savedSourceApplying}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-sky-600 px-4 py-2 text-sm font-semibold text-white hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-60"
                data-testid="singalong-use-saved-original"
              >
                {savedSourceApplying ? <Loader2 className="h-4 w-4 animate-spin" /> : <Music className="h-4 w-4" />}
                Use saved
              </button>
            </div>
            {savedSourcesError && <p className="mt-2 text-xs text-red-300">{savedSourcesError}</p>}
          </div>
        </section>

        <aside className="rounded-xl border border-gray-700 bg-gray-800 p-5">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-400">Profile readiness</h2>
          <div className="mt-3 text-3xl font-bold text-white">{profilesLoading ? '-' : targetProfileCount}</div>
          <p className="mt-1 text-sm text-gray-400">target profiles available for attaching the recorded take</p>
          {profilesError && (
            <div className="mt-4">
              <StatusBanner tone="danger" title="Profiles unavailable" message={profilesError} compact />
            </div>
          )}
          {!profilesLoading && !profilesError && targetProfileCount === 0 && (
            <div className="mt-4">
              <StatusBanner
                tone="warning"
                title="Create a target profile first"
                message="Recording works after at least one target-user voice profile exists."
                compact
              />
            </div>
          )}
        </aside>
      </div>

      {source ? (
        <BrowserSingAlongCapture
          sourceAudioUrl={source.url}
          sourceId={source.id}
          sourceLabel={source.label}
          profiles={profiles}
          disabled={profilesLoading}
          onSampleAttached={loadProfiles}
        />
      ) : (
        <div className="rounded-xl border border-gray-700 bg-gray-800 p-8 text-center" data-testid="singalong-empty-state">
          <Music className="mx-auto mb-3 h-10 w-10 text-gray-500" />
          <h2 className="text-lg font-semibold text-white">Select original audio to show recording controls</h2>
          <p className="mx-auto mt-2 max-w-2xl text-sm text-gray-400">
            After you choose a song, this page shows the original playback controls, output-device selector,
            microphone selector, record/stop controls, take preview, and attach-to-profile action.
          </p>
        </div>
      )}
    </div>
  )
}
