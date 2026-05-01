import { useState, useEffect } from 'react'
import { Youtube, Search, Download, Music, Users, Loader2, AlertCircle, CheckCircle, User, Plus, UserPlus, History, Info } from 'lucide-react'
import { api, getApiAuthToken, YouTubeVideoInfo, YouTubeDownloadResult, VoiceProfile, type YouTubeHistoryItem, type YouTubeIngestDecision, type YouTubeIngestJob } from '../services/api'
import { useToastContext } from '../contexts/ToastContext'

type Stage = 'idle' | 'fetching' | 'info' | 'downloading' | 'ingesting' | 'diarizing' | 'complete' | 'error'
type DownloadStep = 'download' | 'diarize' | 'complete'

interface ArtistToCreate {
  name: string
  speakerId?: string
  selected: boolean
}

function buildDefaultIngestDecisions(job: YouTubeIngestJob): Record<string, YouTubeIngestDecision> {
  const next: Record<string, YouTubeIngestDecision> = {}
  for (const suggestion of job.result?.suggestions ?? []) {
    const shouldAssign = suggestion.recommended_action === 'assign_existing' && suggestion.recommended_profile_id
    next[suggestion.speaker_id] = {
      speaker_id: suggestion.speaker_id,
      action: shouldAssign ? 'assign_existing' : 'create_new',
      profile_id: shouldAssign ? suggestion.recommended_profile_id ?? undefined : undefined,
      name: shouldAssign ? undefined : suggestion.suggested_name,
      metadata: {
        source: 'youtube_ingest_review',
        suggested_name: suggestion.suggested_name,
        identity_confidence: suggestion.identity_confidence,
      },
    }
  }
  return next
}

export function YouTubeDownloadPage() {
  const toast = useToastContext()
  const [url, setUrl] = useState('')
  const [stage, setStage] = useState<Stage>('idle')
  const [error, setError] = useState<string | null>(null)
  const [videoInfo, setVideoInfo] = useState<YouTubeVideoInfo | null>(null)
  const [downloadResult, setDownloadResult] = useState<YouTubeDownloadResult | null>(null)
  const [profiles, setProfiles] = useState<VoiceProfile[]>([])
  const [selectedProfileId, setSelectedProfileId] = useState<string>('')
  const [runDiarization, setRunDiarization] = useState(false)
  const [audioFormat, setAudioFormat] = useState<'wav' | 'mp3' | 'flac'>('wav')
  const [artistsToCreate, setArtistsToCreate] = useState<ArtistToCreate[]>([])
  const [creatingProfiles, setCreatingProfiles] = useState(false)
  const [downloadStep, setDownloadStep] = useState<DownloadStep>('download')
  const [filterToMainArtist, setFilterToMainArtist] = useState(false)
  const [downloadHistory, setDownloadHistory] = useState<YouTubeHistoryItem[]>([])
  const [ingestJob, setIngestJob] = useState<YouTubeIngestJob | null>(null)
  const [ingestDecisions, setIngestDecisions] = useState<Record<string, YouTubeIngestDecision>>({})
  const [confirmingIngest, setConfirmingIngest] = useState(false)
  const [clearingHistory, setClearingHistory] = useState(false)

  // Load profiles on mount
  useEffect(() => {
    loadProfiles()
    void loadDownloadHistory()
  }, [])

  useEffect(() => {
    if (!ingestJob || !['queued', 'running'].includes(ingestJob.status)) return

    const timer = window.setInterval(() => {
      void api.getYouTubeIngest(ingestJob.job_id).then((job) => {
        setIngestJob(job)
        if (job.status === 'completed' && job.result) {
          setStage('info')
          setIngestDecisions(buildDefaultIngestDecisions(job))
          void loadDownloadHistory()
        }
        if (job.status === 'failed') {
          setStage('error')
          setError(job.error || 'YouTube auto-ingest failed')
        }
      }).catch((err) => {
        setStage('error')
        setError(err instanceof Error ? err.message : 'Failed to poll YouTube auto-ingest')
      })
    }, 2000)

    return () => window.clearInterval(timer)
  }, [ingestJob])

  const loadProfiles = async () => {
    try {
      const profileList = await api.listProfiles()
      setProfiles(profileList)
    } catch (err) {
      console.error('Failed to load profiles:', err)
    }
  }

  const loadDownloadHistory = async () => {
    try {
      const history = await api.getYouTubeHistory(20)
      setDownloadHistory(history)
    } catch (err) {
      console.error('Failed to load download history:', err)
    }
  }

  const handleClearHistory = async () => {
    setClearingHistory(true)
    try {
      await api.clearYouTubeHistory()
      setDownloadHistory([])
      await loadDownloadHistory()
      toast.success('YouTube history cleared')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to clear YouTube history'
      toast.error(message)
    } finally {
      setClearingHistory(false)
    }
  }

  const handleFetchInfo = async () => {
    if (!url.trim()) {
      setError('Please enter a YouTube URL')
      return
    }

    setStage('fetching')
    setError(null)
    setVideoInfo(null)
    setDownloadResult(null)
    setIngestJob(null)
    setIngestDecisions({})

    try {
      const info = await api.getYouTubeVideoInfo(url)
      if (!info.success) {
        throw new Error(info.error || 'Failed to fetch video info')
      }
      setVideoInfo(info)
      setStage('info')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch video info')
      setStage('error')
    }
  }

  const handleStartAutoIngestFromUrl = async () => {
    if (!url.trim()) {
      setError('Please enter a YouTube URL')
      return
    }

    setStage('ingesting')
    setError(null)
    setVideoInfo(null)
    setDownloadResult(null)
    setIngestJob(null)
    setIngestDecisions({})

    try {
      const job = await api.startYouTubeIngest(url, { format: audioFormat })
      setIngestJob(job)
      toast.success('YouTube auto-ingest started')
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to start YouTube auto-ingest'
      setError(errorMsg)
      toast.error(errorMsg)
      setStage('error')
    }
  }

  const handleDownload = async () => {
    if (!videoInfo) return

    setStage('downloading')
    setDownloadStep('download')
    setError(null)
    setArtistsToCreate([])

    try {
      // If diarization is enabled, show a brief step indicator
      if (runDiarization) {
        // Simulate download step timing for visual feedback
        setTimeout(() => setDownloadStep('diarize'), 5000)
      }

      const result = await api.downloadYouTubeAudio(url, {
        format: audioFormat,
        run_diarization: runDiarization,
        filter_to_main_artist: filterToMainArtist,
      })

      if (!result.success) {
        throw new Error(result.error || 'Download failed')
      }

      setDownloadResult(result)

      // Build list of artists to potentially create profiles for
      const artists: ArtistToCreate[] = []

      // Add main artist
      if (result.main_artist) {
        artists.push({
          name: result.main_artist,
          speakerId: result.diarization_result ? 'SPEAKER_00' : undefined,
          selected: true,
        })
      }

      // Add featured artists
      if (result.featured_artists) {
        result.featured_artists.forEach((artist, idx) => {
          artists.push({
            name: artist,
            speakerId: result.diarization_result ? `SPEAKER_0${idx + 1}` : undefined,
            selected: false,
          })
        })
      }

      // If diarization found more speakers than named artists, add unknown speakers
      if (result.diarization_result) {
        const numNamed = artists.length
        const numSpeakers = result.diarization_result.num_speakers
        for (let i = numNamed; i < numSpeakers; i++) {
          artists.push({
            name: `Unknown Speaker ${i + 1}`,
            speakerId: `SPEAKER_0${i}`,
            selected: false,
          })
        }
      }

      setArtistsToCreate(artists)

      await loadDownloadHistory()

      toast.success('Download completed successfully!')
      setStage('complete')
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Download failed'
      setError(errorMsg)
      toast.error(errorMsg)
      setStage('error')
    }
  }

  const handleStartIngest = async () => {
    if (!url.trim()) return

    setStage('ingesting')
    setError(null)
    setIngestJob(null)
    setIngestDecisions({})

    try {
      const job = await api.startYouTubeIngest(url, { format: audioFormat })
      setIngestJob(job)
      toast.success('YouTube auto-ingest started')
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to start YouTube auto-ingest'
      setError(errorMsg)
      toast.error(errorMsg)
      setStage('error')
    }
  }

  const updateIngestDecision = (speakerId: string, updates: Partial<YouTubeIngestDecision>) => {
    setIngestDecisions(prev => ({
      ...prev,
      [speakerId]: {
        ...prev[speakerId],
        speaker_id: speakerId,
        ...updates,
      },
    }))
  }

  const handleConfirmIngest = async () => {
    if (!ingestJob?.result) return
    const decisions = Object.values(ingestDecisions)
    if (decisions.length === 0) {
      setError('No speaker decisions to confirm')
      return
    }

    setConfirmingIngest(true)
    setError(null)

    try {
      const confirmation = await api.confirmYouTubeIngest(ingestJob.job_id, decisions)
      const refreshed = await api.getYouTubeIngest(ingestJob.job_id)
      setIngestJob({ ...refreshed, confirmation })
      await loadProfiles()
      toast.success(`Applied ${confirmation.applied.length} speaker decision(s)`)
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to confirm YouTube ingest'
      setError(errorMsg)
      toast.error(errorMsg)
    } finally {
      setConfirmingIngest(false)
    }
  }

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const formatTimestamp = (timestamp: string): string => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMins / 60)
    const diffDays = Math.floor(diffHours / 24)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  const handleAddToProfile = async () => {
    if (!downloadResult || !selectedProfileId) return
    const audioAssetId = downloadResult.audio_asset_id ?? downloadResult.audio_path_asset_id ?? null
    const useAssetId = Boolean(getApiAuthToken() && audioAssetId)
    if (!useAssetId && !downloadResult.audio_path) return

    try {
      const sample = await api.addSampleFromPath(selectedProfileId, {
        audio_asset_id: useAssetId ? audioAssetId : undefined,
        audio_path: useAssetId ? undefined : downloadResult.audio_path,
        metadata: {
          title: downloadResult.title,
          video_id: downloadResult.video_id,
          main_artist: downloadResult.main_artist,
          featured_artists: downloadResult.featured_artists,
        },
      })
      toast.success(`Added "${downloadResult.title}" as sample ${sample.id} to profile`)

      // Refresh profiles to update sample count
      await loadProfiles()
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to add to profile'
      setError(errorMsg)
      toast.error(errorMsg)
    }
  }

  const toggleArtistSelection = (index: number) => {
    setArtistsToCreate(prev => prev.map((artist, i) =>
      i === index ? { ...artist, selected: !artist.selected } : artist
    ))
  }

  const handleCreateSelectedProfiles = async () => {
    const selected = artistsToCreate.filter(a => a.selected)
    if (selected.length === 0) {
      setError('Please select at least one artist')
      return
    }

    const diarizationId =
      downloadResult?.diarization_id || downloadResult?.diarization_result?.diarization_id
    if (!diarizationId) {
      setError('Run diarization first so source artist profiles can be created from detected singers.')
      return
    }

    setCreatingProfiles(true)
    setError(null)

    try {
      const createdProfiles: string[] = []

      for (const artist of selected) {
        try {
          if (!artist.speakerId) continue
          await api.autoCreateProfileFromDiarization(
            diarizationId,
            artist.speakerId,
            artist.name,
            undefined,
            true,
            {
              profileRole: 'source_artist',
              metadata: {
                source: 'youtube_download',
                title: downloadResult?.title,
                video_id: downloadResult?.video_id,
                main_artist: downloadResult?.main_artist,
                featured_artists: downloadResult?.featured_artists,
                song_title: downloadResult?.song_title,
              },
            }
          )
          createdProfiles.push(artist.name)
        } catch (err) {
          console.error(`Failed to create profile for ${artist.name}:`, err)
        }
      }

      // Refresh profiles
      await loadProfiles()

      if (createdProfiles.length > 0) {
        toast.success(`Created ${createdProfiles.length} source profile(s): ${createdProfiles.join(', ')}`)
      }

      // Clear selection
      setArtistsToCreate(prev => prev.map(a => ({ ...a, selected: false })))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create profiles')
    } finally {
      setCreatingProfiles(false)
    }
  }

  const resetState = () => {
    setStage('idle')
    setUrl('')
    setVideoInfo(null)
    setDownloadResult(null)
    setIngestJob(null)
    setIngestDecisions({})
    setError(null)
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6 flex items-center gap-3">
        <Youtube className="text-red-500" />
        YouTube Download
      </h1>

      {/* URL Input */}
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <label htmlFor="youtube-url" className="block text-sm text-gray-400 mb-1 flex items-center gap-1">
          YouTube URL
          <span title="Supports YouTube videos and playlists. Works with youtube.com, youtu.be, and music.youtube.com URLs" className="cursor-help">
            <Info size={12} className="text-gray-500" />
          </span>
        </label>
        <p className="text-xs text-gray-500 mb-2">
          Paste a YouTube URL to start the full reviewed ingest pipeline. Download-only remains available after fetching info.
        </p>
        <div className="flex gap-3">
          <input
            id="youtube-url"
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            disabled={stage === 'fetching' || stage === 'downloading' || stage === 'ingesting'}
            onKeyDown={(e) => e.key === 'Enter' && handleStartAutoIngestFromUrl()}
            aria-label="YouTube URL input"
          />
          <button
            onClick={handleStartAutoIngestFromUrl}
            disabled={stage === 'fetching' || stage === 'downloading' || stage === 'ingesting'}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded-lg flex items-center gap-2"
          >
            {stage === 'ingesting' ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Users size={20} />
            )}
            Auto Ingest
          </button>
          <button
            onClick={handleFetchInfo}
            disabled={stage === 'fetching' || stage === 'downloading' || stage === 'ingesting'}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-600 rounded-lg flex items-center gap-2"
          >
            {stage === 'fetching' ? <Loader2 size={20} className="animate-spin" /> : <Search size={20} />}
            Manual
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 mb-6 flex items-start gap-3">
          <AlertCircle className="text-red-400 flex-shrink-0 mt-0.5" size={20} />
          <div>
            <p className="font-medium text-red-300">Error</p>
            <p className="text-red-200">{error}</p>
          </div>
        </div>
      )}

      {/* Video Info Card */}
      {videoInfo && (stage === 'info' || stage === 'downloading' || stage === 'ingesting' || stage === 'complete') && (
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <div className="flex gap-6">
            {/* Thumbnail */}
            {videoInfo.thumbnail_url && (
              <img
                src={videoInfo.thumbnail_url}
                alt={videoInfo.title}
                className="w-48 h-auto rounded-lg flex-shrink-0"
              />
            )}

            <div className="flex-1">
              <h2 className="text-xl font-semibold mb-2">{videoInfo.title}</h2>

              <div className="flex flex-wrap gap-4 text-sm text-gray-400 mb-4">
                <span className="flex items-center gap-1">
                  <Music size={16} />
                  {formatDuration(videoInfo.duration)}
                </span>
                {videoInfo.video_id && (
                  <span className="text-gray-500">ID: {videoInfo.video_id}</span>
                )}
              </div>

              {/* Artist Info */}
              <div className="space-y-2">
                {videoInfo.main_artist && (
                  <div className="flex items-center gap-2">
                    <User size={16} className="text-blue-400" />
                    <span className="text-gray-300">Main Artist:</span>
                    <span className="font-medium text-white">{videoInfo.main_artist}</span>
                  </div>
                )}

                {videoInfo.featured_artists.length > 0 && (
                  <div className="flex items-start gap-2">
                    <Users size={16} className="text-purple-400 mt-0.5" />
                    <span className="text-gray-300">Featured:</span>
                    <div className="flex flex-wrap gap-2">
                      {videoInfo.featured_artists.map((artist, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 bg-purple-900/50 text-purple-300 rounded-full text-sm"
                        >
                          {artist}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {videoInfo.is_cover && (
                  <div className="flex items-center gap-2 text-yellow-400">
                    <Music size={16} />
                    <span>Cover of: {videoInfo.original_artist || 'Unknown'}</span>
                  </div>
                )}

                {videoInfo.song_title && (
                  <div className="text-gray-400 text-sm">
                    Song: "{videoInfo.song_title}"
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Download Options */}
          {stage === 'info' && (
            <div className="mt-6 pt-6 border-t border-gray-700">
              <h3 className="font-medium mb-4">Advanced Manual Options</h3>

              <div className="space-y-4 mb-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="audio-format" className="block text-sm text-gray-400 mb-1 flex items-center gap-1">
                      Audio Format
                      <span title="WAV: Uncompressed, best quality for training. MP3: Compressed, smaller file size. FLAC: Lossless compression" className="cursor-help">
                        <Info size={12} className="text-gray-500" />
                      </span>
                    </label>
                    <select
                      id="audio-format"
                      value={audioFormat}
                      onChange={(e) => setAudioFormat(e.target.value as 'wav' | 'mp3' | 'flac')}
                      className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:border-blue-500"
                      aria-label="Select audio format"
                    >
                      <option value="wav">WAV (Best for training)</option>
                      <option value="mp3">MP3</option>
                      <option value="flac">FLAC</option>
                    </select>
                  </div>

                  <div className="flex items-end">
                    <label htmlFor="run-diarization" className="flex items-center gap-2 cursor-pointer">
                      <input
                        id="run-diarization"
                        type="checkbox"
                        checked={runDiarization}
                        onChange={(e) => {
                          setRunDiarization(e.target.checked)
                          if (!e.target.checked) {
                            setFilterToMainArtist(false)
                          }
                        }}
                        className="w-4 h-4 rounded bg-gray-700 border-gray-600"
                        aria-label="Enable speaker diarization to separate different voices"
                      />
                      <span className="text-sm flex items-center gap-1">
                        Run speaker diarization
                        <span title="Automatically detects and separates different speakers in the audio. Useful for songs with multiple artists or featured vocals" className="cursor-help">
                          <Info size={12} className="text-gray-500" />
                        </span>
                        {videoInfo.featured_artists.length > 0 && (
                          <span className="text-purple-400 ml-1">
                            (Recommended - {videoInfo.featured_artists.length + 1} artists detected)
                          </span>
                        )}
                      </span>
                    </label>
                  </div>
                </div>

                {/* Filter to main artist toggle - only show when diarization is enabled */}
                {runDiarization && videoInfo.main_artist && (
                  <div className="bg-gray-700/50 rounded-lg p-3">
                    <label htmlFor="filter-main-artist" className="flex items-center gap-2 cursor-pointer">
                      <input
                        id="filter-main-artist"
                        type="checkbox"
                        checked={filterToMainArtist}
                        onChange={(e) => setFilterToMainArtist(e.target.checked)}
                        className="w-4 h-4 rounded bg-gray-700 border-gray-600"
                        aria-label={`Filter audio to only ${videoInfo.main_artist}'s voice`}
                      />
                      <div>
                        <span className="text-sm font-medium flex items-center gap-1">
                          Filter to "{videoInfo.main_artist}" only
                          <span title="Creates a clean voice profile by isolating only the main artist's vocals. Removes featured artists, backing vocals, and other speakers" className="cursor-help">
                            <Info size={12} className="text-gray-500" />
                          </span>
                        </span>
                        <p className="text-xs text-gray-400 mt-0.5">
                          Only keep audio segments from the main artist, removing featured artists and other voices
                        </p>
                      </div>
                    </label>
                  </div>
                )}
              </div>

              <div className="grid gap-3 md:grid-cols-2">
                <button
                  onClick={handleDownload}
                  className="w-full px-4 py-3 bg-green-600 hover:bg-green-700 rounded-lg flex items-center justify-center gap-2 font-medium"
                >
                  <Download size={20} />
                  Download Audio
                </button>
                <button
                  onClick={handleStartIngest}
                  className="w-full px-4 py-3 bg-red-600 hover:bg-red-700 rounded-lg flex items-center justify-center gap-2 font-medium"
                >
                  <Users size={20} />
                  Auto Ingest + Match Profiles
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-3">
                Auto ingest downloads audio, splits vocals/instrumental, diarizes the vocals, and suggests profile matches. It will not create or modify profiles until you review and confirm.
              </p>
            </div>
          )}

          {/* Auto ingest progress */}
          {ingestJob && ['queued', 'running'].includes(ingestJob.status) && (
            <div className="mt-6 pt-6 border-t border-gray-700">
              <h3 className="font-medium mb-3 flex items-center gap-2">
                <Loader2 size={18} className="animate-spin text-red-400" />
                Auto Ingest Running
              </h3>
              <div className="flex items-center justify-between text-sm text-gray-400 mb-2">
                <span>{ingestJob.message || ingestJob.stage || 'Processing...'}</span>
                <span>{ingestJob.progress || 0}%</span>
              </div>
              <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-red-500 rounded-full transition-all duration-500"
                  style={{ width: `${Math.max(5, Math.min(100, ingestJob.progress || 5))}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Current stage: {ingestJob.stage || ingestJob.status}
              </p>
            </div>
          )}

          {/* Downloading Progress */}
          {stage === 'downloading' && (
            <div className="mt-6 pt-6 border-t border-gray-700">
              <div className="space-y-4">
                {/* Progress steps */}
                <div className="flex items-center gap-4 justify-center">
                  {/* Download step */}
                  <div className="flex items-center gap-2">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      downloadStep === 'download' ? 'bg-blue-600' : 'bg-green-600'
                    }`}>
                      {downloadStep === 'download' ? (
                        <Loader2 size={16} className="animate-spin text-white" />
                      ) : (
                        <CheckCircle size={16} className="text-white" />
                      )}
                    </div>
                    <span className={downloadStep === 'download' ? 'text-blue-400 font-medium' : 'text-green-400'}>
                      Download
                    </span>
                  </div>

                  <div className={`w-12 h-0.5 ${downloadStep !== 'download' ? 'bg-green-500' : 'bg-gray-600'}`} />

                  {/* Diarization step */}
                  <div className="flex items-center gap-2">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      downloadStep === 'diarize' ? 'bg-purple-600' :
                      downloadStep === 'complete' ? 'bg-green-600' :
                      runDiarization ? 'bg-gray-600' : 'bg-gray-700'
                    } ${!runDiarization ? 'opacity-50' : ''}`}>
                      {downloadStep === 'diarize' ? (
                        <Loader2 size={16} className="animate-spin text-white" />
                      ) : downloadStep === 'complete' && runDiarization ? (
                        <CheckCircle size={16} className="text-white" />
                      ) : (
                        <Users size={16} className={runDiarization ? 'text-gray-300' : 'text-gray-500'} />
                      )}
                    </div>
                    <span className={
                      downloadStep === 'diarize' ? 'text-purple-400 font-medium' :
                      downloadStep === 'complete' && runDiarization ? 'text-green-400' :
                      runDiarization ? 'text-gray-400' : 'text-gray-500 line-through'
                    }>
                      {runDiarization ? 'Diarization' : 'Skip'}
                    </span>
                  </div>

                  <div className={`w-12 h-0.5 ${downloadStep === 'complete' ? 'bg-green-500' : 'bg-gray-600'}`} />

                  {/* Complete step */}
                  <div className="flex items-center gap-2">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      downloadStep === 'complete' ? 'bg-green-600' : 'bg-gray-700'
                    }`}>
                      <CheckCircle size={16} className={downloadStep === 'complete' ? 'text-white' : 'text-gray-500'} />
                    </div>
                    <span className={downloadStep === 'complete' ? 'text-green-400' : 'text-gray-500'}>
                      Complete
                    </span>
                  </div>
                </div>

                {/* Status message */}
                <p className="text-center text-gray-400">
                  {downloadStep === 'download' && 'Downloading audio from YouTube...'}
                  {downloadStep === 'diarize' && 'Running speaker diarization...'}
                  {downloadStep === 'complete' && 'Processing complete!'}
                  {downloadStep === 'download' && runDiarization && (
                    <span className="block text-sm text-gray-500 mt-1">
                      Speaker diarization will run after download
                    </span>
                  )}
                  {downloadStep === 'diarize' && (
                    <span className="block text-sm text-gray-500 mt-1">
                      Detecting and separating speakers...
                    </span>
                  )}
                </p>

                {/* Animated progress bar */}
                <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      downloadStep === 'download' ? 'bg-blue-500 animate-pulse' :
                      downloadStep === 'diarize' ? 'bg-purple-500 animate-pulse' :
                      'bg-green-500'
                    }`}
                    style={{
                      width: downloadStep === 'download' ? '40%' :
                             downloadStep === 'diarize' ? '75%' : '100%'
                    }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Auto Ingest Review */}
      {ingestJob?.status === 'completed' && ingestJob.result && (
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <CheckCircle className="text-green-400" size={24} />
            <div>
              <h3 className="text-lg font-medium">Auto Ingest Ready for Review</h3>
              <p className="text-sm text-gray-400">
                Review speaker-to-profile decisions before anything is created or assigned.
              </p>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3 mb-4">
            <div className="bg-gray-700 rounded-lg p-3">
              <p className="text-xs text-gray-400">Vocals Stem</p>
              <p className="text-sm font-medium text-white">
                {ingestJob.result.assets.vocals?.asset_id ? 'Registered' : 'Saved'}
              </p>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <p className="text-xs text-gray-400">Instrumental Stem</p>
              <p className="text-sm font-medium text-white">
                {ingestJob.result.assets.instrumental?.asset_id ? 'Registered' : 'Saved'}
              </p>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <p className="text-xs text-gray-400">Speakers</p>
              <p className="text-sm font-medium text-white">
                {ingestJob.result.diarization_result.num_speakers}
              </p>
            </div>
          </div>

          <div className="space-y-4">
            {ingestJob.result.suggestions.map((suggestion) => {
              const decision = ingestDecisions[suggestion.speaker_id] || {
                speaker_id: suggestion.speaker_id,
                action: 'create_new' as const,
                name: suggestion.suggested_name,
              }
              const bestMatch = suggestion.matches[0]

              return (
                <div key={suggestion.speaker_id} className="bg-gray-700 rounded-lg p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                    <div>
                      <h4 className="font-medium flex items-center gap-2">
                        <Users size={16} className="text-purple-400" />
                        {suggestion.speaker_id}
                      </h4>
                      <p className="text-sm text-gray-400">
                        {formatDuration(suggestion.duration)} across {suggestion.segment_count} segment(s)
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-300">{suggestion.suggested_name}</p>
                      <p className="text-xs text-gray-500">
                        {suggestion.identity_confidence === 'voice_match'
                          ? 'Voice match suggested'
                          : 'Metadata label, unverified'}
                      </p>
                    </div>
                  </div>

                  {bestMatch && (
                    <div className="bg-gray-800/70 rounded p-3 mb-3 text-sm">
                      <span className="text-gray-400">Best existing match: </span>
                      <span className="font-medium">{bestMatch.name}</span>
                      <span className="text-gray-500"> ({Math.round(bestMatch.similarity * 100)}%)</span>
                    </div>
                  )}

                  {suggestion.duplicate_warning && (
                    <div className="bg-yellow-900/30 border border-yellow-700 rounded p-3 mb-3 text-sm text-yellow-100">
                      Likely duplicate source profile detected at or above 82% similarity. Review the selected existing profile before creating a new one.
                    </div>
                  )}

                  {suggestion.match_error && (
                    <p className="text-sm text-yellow-300 mb-3">
                      Voice matching warning: {suggestion.match_error}
                    </p>
                  )}

                  <div className="grid gap-3 md:grid-cols-3">
                    <div>
                      <label htmlFor={`decision-${suggestion.speaker_id}`} className="block text-xs text-gray-400 mb-1">
                        Action
                      </label>
                      <select
                        id={`decision-${suggestion.speaker_id}`}
                        value={decision.action}
                        onChange={(event) => {
                          const action = event.target.value as YouTubeIngestDecision['action']
                          updateIngestDecision(suggestion.speaker_id, {
                            action,
                            profile_id: action === 'assign_existing'
                              ? suggestion.recommended_profile_id || bestMatch?.profile_id || ''
                              : undefined,
                            name: action === 'create_new' ? suggestion.suggested_name : undefined,
                          })
                        }}
                        className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:border-blue-500"
                      >
                        <option value="assign_existing">Assign to existing profile</option>
                        <option value="create_new">Create new source profile</option>
                        <option value="skip">Skip this speaker</option>
                      </select>
                    </div>

                    {decision.action === 'assign_existing' && (
                      <div className="md:col-span-2">
                        <label htmlFor={`profile-${suggestion.speaker_id}`} className="block text-xs text-gray-400 mb-1">
                          Existing profile
                        </label>
                        <select
                          id={`profile-${suggestion.speaker_id}`}
                          value={decision.profile_id || ''}
                          onChange={(event) => updateIngestDecision(suggestion.speaker_id, { profile_id: event.target.value })}
                          className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:border-blue-500"
                        >
                          <option value="">Select a source profile...</option>
                          {profiles.map((profile) => (
                            <option key={profile.profile_id} value={profile.profile_id}>
                              {profile.name || profile.profile_id}
                            </option>
                          ))}
                        </select>
                      </div>
                    )}

                    {decision.action === 'create_new' && (
                      <div className="md:col-span-2">
                        <label htmlFor={`name-${suggestion.speaker_id}`} className="block text-xs text-gray-400 mb-1">
                          New source profile name
                        </label>
                        <input
                          id={`name-${suggestion.speaker_id}`}
                          value={decision.name || ''}
                          onChange={(event) => updateIngestDecision(suggestion.speaker_id, { name: event.target.value })}
                          className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:border-blue-500"
                          placeholder="Artist or speaker name"
                        />
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          {ingestJob.confirmation ? (
            <div className="mt-4 bg-green-900/30 border border-green-700 rounded-lg p-3">
              <p className="text-sm text-green-300">
                Applied {ingestJob.confirmation.applied.length} decision(s); skipped {ingestJob.confirmation.skipped.length}.
              </p>
            </div>
          ) : (
            <button
              onClick={handleConfirmIngest}
              disabled={confirmingIngest}
              className="mt-4 w-full px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg flex items-center justify-center gap-2 font-medium"
            >
              {confirmingIngest ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  Applying Decisions...
                </>
              ) : (
                <>
                  <UserPlus size={18} />
                  Confirm Reviewed Profile Actions
                </>
              )}
            </button>
          )}
        </div>
      )}

      {/* Download Complete */}
      {stage === 'complete' && downloadResult && (
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <CheckCircle className="text-green-400" size={24} />
            <h3 className="text-lg font-medium">Download Complete</h3>
          </div>

          <div className="bg-gray-700 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-400 mb-1">Audio saved to:</p>
            <p className="font-mono text-sm break-all">{downloadResult.audio_path}</p>
          </div>

          {/* Diarization Results */}
          {downloadResult.diarization_result && (
            <div className="bg-gray-700 rounded-lg p-4 mb-4">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <Users size={18} className="text-purple-400" />
                Speaker Diarization Results
              </h4>
              <p className="text-sm text-gray-400 mb-3">
                Detected {downloadResult.diarization_result.num_speakers} speaker(s)
              </p>
              <div className="space-y-2">
                {downloadResult.diarization_result.segments.slice(0, 5).map((seg, idx) => (
                  <div key={idx} className="flex items-center gap-3 text-sm">
                    <span className="px-2 py-0.5 bg-purple-900/50 text-purple-300 rounded">
                      {seg.speaker_id}
                    </span>
                    <span className="text-gray-400">
                      {formatDuration(seg.start)} - {formatDuration(seg.end)}
                    </span>
                    <span className="text-gray-500">
                      ({formatDuration(seg.duration)})
                    </span>
                  </div>
                ))}
                {downloadResult.diarization_result.segments.length > 5 && (
                  <p className="text-sm text-gray-500">
                    ...and {downloadResult.diarization_result.segments.length - 5} more segments
                  </p>
                )}
              </div>
            </div>
          )}

          {downloadResult.diarization_error && (
            <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-3 mb-4">
              <p className="text-sm text-yellow-300">
                Diarization failed: {downloadResult.diarization_error}
              </p>
            </div>
          )}

          {/* Auto-Create Profiles Section */}
          {artistsToCreate.length > 0 && (
            <div className="bg-gray-700 rounded-lg p-4 mb-4">
              <h4 className="font-medium mb-3 flex items-center gap-2">
                <UserPlus size={18} className="text-green-400" />
                Create Voice Profiles
              </h4>
              <p className="text-sm text-gray-400 mb-3">
                Select artists to create voice profiles from this audio:
              </p>
              <div className="space-y-2 mb-4">
                {artistsToCreate.map((artist, idx) => (
                  <label
                    key={idx}
                    htmlFor={`artist-${idx}`}
                    className="flex items-center gap-3 p-2 rounded hover:bg-gray-600/50 cursor-pointer"
                  >
                    <input
                      id={`artist-${idx}`}
                      type="checkbox"
                      checked={artist.selected}
                      onChange={() => toggleArtistSelection(idx)}
                      className="w-4 h-4 rounded bg-gray-600 border-gray-500"
                      aria-label={`Create voice profile for ${artist.name}`}
                    />
                    <span className={artist.selected ? 'text-white' : 'text-gray-300'}>
                      {artist.name}
                    </span>
                    {artist.speakerId && (
                      <span className="text-xs text-gray-500">
                        ({artist.speakerId})
                      </span>
                    )}
                    {idx === 0 && videoInfo?.main_artist && (
                      <span className="text-xs px-2 py-0.5 bg-blue-900/50 text-blue-300 rounded">
                        Main Artist
                      </span>
                    )}
                  </label>
                ))}
              </div>
              <button
                onClick={handleCreateSelectedProfiles}
                disabled={creatingProfiles || artistsToCreate.filter(a => a.selected).length === 0}
                className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg flex items-center justify-center gap-2"
              >
                {creatingProfiles ? (
                  <>
                    <Loader2 size={18} className="animate-spin" />
                    Creating Profiles...
                  </>
                ) : (
                  <>
                    <UserPlus size={18} />
                    Create Selected Profiles ({artistsToCreate.filter(a => a.selected).length})
                  </>
                )}
              </button>
            </div>
          )}

          {/* Actions */}
          <div className="space-y-4">
            <div>
              <label htmlFor="profile-select" className="block text-sm text-gray-400 mb-2 flex items-center gap-1">
                Add to Existing Profile
                <span title="Add this audio as a training sample to an existing voice profile" className="cursor-help">
                  <Info size={12} className="text-gray-500" />
                </span>
              </label>
              <div className="flex gap-4">
                <select
                  id="profile-select"
                  value={selectedProfileId}
                  onChange={(e) => setSelectedProfileId(e.target.value)}
                  className="flex-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:border-blue-500"
                  aria-label="Select voice profile to add audio to"
                >
                  <option value="">Select a profile to add to...</option>
                  {profiles.map((p) => (
                    <option key={p.profile_id} value={p.profile_id}>
                      {p.name || p.profile_id}
                    </option>
                  ))}
                </select>
                <button
                  onClick={handleAddToProfile}
                  disabled={!selectedProfileId}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg flex items-center gap-2"
                >
                  <Plus size={18} />
                  Add to Profile
                </button>
              </div>
            </div>

            <button
              onClick={resetState}
              className="w-full px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded-lg"
            >
              Download Another Video
            </button>
          </div>
        </div>
      )}

      {/* Download History */}
      {downloadHistory.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium flex items-center gap-2">
              <History size={20} className="text-gray-400" />
              Recent Downloads
            </h3>
            <button
              type="button"
              onClick={() => {
                void handleClearHistory()
              }}
              disabled={clearingHistory}
              className="text-sm text-gray-500 hover:text-gray-300 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {clearingHistory ? 'Clearing...' : 'Clear History'}
            </button>
          </div>

          <div className="space-y-3">
            {downloadHistory.slice(0, 5).map((item) => (
              <div
                key={item.id}
                className="bg-gray-700/50 rounded-lg p-3 flex items-start gap-3"
              >
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{item.title}</p>
                  <div className="flex flex-wrap gap-2 text-sm text-gray-400 mt-1">
                    {item.mainArtist && (
                      <span className="flex items-center gap-1">
                        <User size={12} />
                        {item.mainArtist}
                      </span>
                    )}
                    {item.featuredArtists.length > 0 && (
                      <span className="text-purple-400">
                        +{item.featuredArtists.length} featured
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0">
                  {/* Diarization status badge */}
                  {item.hasDiarization ? (
                    <span className="px-2 py-1 bg-purple-900/50 text-purple-300 rounded text-xs flex items-center gap-1">
                      <Users size={12} />
                      {item.numSpeakers} speakers
                    </span>
                  ) : (
                    <span className="px-2 py-1 bg-gray-600 text-gray-400 rounded text-xs">
                      No diarization
                    </span>
                  )}

                  {/* Filtered badge */}
                  {item.filteredPath && (
                    <span className="px-2 py-1 bg-green-900/50 text-green-300 rounded text-xs">
                      Filtered
                    </span>
                  )}
                </div>

                <span className="text-xs text-gray-500 flex-shrink-0">
                    {formatTimestamp(item.timestamp)}
                </span>
              </div>
            ))}

            {downloadHistory.length > 5 && (
              <p className="text-center text-sm text-gray-500">
                +{downloadHistory.length - 5} more downloads
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
