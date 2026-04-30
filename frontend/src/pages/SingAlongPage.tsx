import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react'
import { Headphones, Mic, Music, UploadCloud } from 'lucide-react'

import { BrowserSingAlongCapture } from '../components/BrowserSingAlongCapture'
import { StatusBanner } from '../components/StatusBanner'
import { useToastContext } from '../contexts/ToastContext'
import { apiService, type VoiceProfile } from '../services/api'

type BrowserAudioSource = {
  file: File
  url: string
  id: string
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
      file,
      url: URL.createObjectURL(file),
      id: makeSourceId(file),
    })
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
                  {source ? source.file.name : 'Choose the full song the singer will hear.'}
                </div>
                <div className="mt-1 text-xs text-gray-400">
                  {source
                    ? `${source.file.type || 'audio file'} · ${formatBytes(source.file.size)}`
                    : 'Supported formats depend on the current browser, usually WAV, MP3, M4A, FLAC, OGG, and WebM.'}
                </div>
              </div>
            </div>
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
          sourceLabel={source.file.name}
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
