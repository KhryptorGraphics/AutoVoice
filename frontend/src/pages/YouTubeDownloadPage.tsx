import { useState, useEffect } from 'react'
import { Youtube, Search, Download, Music, Users, Loader2, AlertCircle, CheckCircle, User, Plus, UserPlus, History, Info } from 'lucide-react'
import { api, YouTubeVideoInfo, YouTubeDownloadResult, VoiceProfile, type YouTubeHistoryItem } from '../services/api'
import { useToastContext } from '../contexts/ToastContext'

type Stage = 'idle' | 'fetching' | 'info' | 'downloading' | 'diarizing' | 'complete' | 'error'
type DownloadStep = 'download' | 'diarize' | 'complete'

interface ArtistToCreate {
  name: string
  speakerId?: string
  selected: boolean
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

  // Load profiles on mount
  useEffect(() => {
    loadProfiles()
    void loadDownloadHistory()
  }, [])

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

  const handleFetchInfo = async () => {
    if (!url.trim()) {
      setError('Please enter a YouTube URL')
      return
    }

    setStage('fetching')
    setError(null)
    setVideoInfo(null)
    setDownloadResult(null)

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
    if (!downloadResult?.audio_path || !selectedProfileId) return

    try {
      const response = await fetch(`/api/v1/profiles/${selectedProfileId}/samples/from-path`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_path: downloadResult.audio_path,
          metadata: {
            title: downloadResult.title,
            video_id: downloadResult.video_id,
            main_artist: downloadResult.main_artist,
            featured_artists: downloadResult.featured_artists,
          },
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Failed to add to profile')
      }

      const sample = await response.json()
      alert(`Added "${downloadResult.title}" as sample ${sample.sample_id} to profile`)

      // Refresh profiles to update sample count
      await loadProfiles()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add to profile')
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

    setCreatingProfiles(true)
    setError(null)

    try {
      const createdProfiles: string[] = []

      for (const artist of selected) {
        try {
          // For now, create a basic profile with the name
          // In a full implementation, we would extract the speaker's segments
          // and use them as the reference audio
          const response = await fetch('/api/v1/voice/clone', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              name: artist.name,
              from_youtube: true,
              audio_path: downloadResult?.audio_path,
              speaker_id: artist.speakerId,
            }),
          })

          if (response.ok) {
            await response.json() // consume response
            createdProfiles.push(artist.name)
          }
        } catch (err) {
          console.error(`Failed to create profile for ${artist.name}:`, err)
        }
      }

      // Refresh profiles
      await loadProfiles()

      if (createdProfiles.length > 0) {
        alert(`Created ${createdProfiles.length} profile(s): ${createdProfiles.join(', ')}`)
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
          Paste any YouTube video URL to download audio for voice conversion training
        </p>
        <div className="flex gap-3">
          <input
            id="youtube-url"
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            disabled={stage === 'fetching' || stage === 'downloading'}
            onKeyDown={(e) => e.key === 'Enter' && handleFetchInfo()}
            aria-label="YouTube URL input"
          />
          <button
            onClick={handleFetchInfo}
            disabled={stage === 'fetching' || stage === 'downloading'}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg flex items-center gap-2"
          >
            {stage === 'fetching' ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Search size={20} />
            )}
            Fetch Info
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
      {videoInfo && (stage === 'info' || stage === 'downloading' || stage === 'complete') && (
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
              <h3 className="font-medium mb-4">Download Options</h3>

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

              <button
                onClick={handleDownload}
                className="w-full px-4 py-3 bg-green-600 hover:bg-green-700 rounded-lg flex items-center justify-center gap-2 font-medium"
              >
                <Download size={20} />
                Download Audio
              </button>
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
              onClick={() => {
                void api.clearYouTubeHistory().then(() => setDownloadHistory([]))
              }}
              className="text-sm text-gray-500 hover:text-gray-300"
            >
              Clear History
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
