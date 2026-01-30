import { useState, useEffect } from 'react'
import { Youtube, Search, Download, Music, Users, Loader2, AlertCircle, CheckCircle, User, Plus } from 'lucide-react'
import { api, YouTubeVideoInfo, YouTubeDownloadResult, VoiceProfile } from '../services/api'

type Stage = 'idle' | 'fetching' | 'info' | 'downloading' | 'diarizing' | 'complete' | 'error'

export function YouTubeDownloadPage() {
  const [url, setUrl] = useState('')
  const [stage, setStage] = useState<Stage>('idle')
  const [error, setError] = useState<string | null>(null)
  const [videoInfo, setVideoInfo] = useState<YouTubeVideoInfo | null>(null)
  const [downloadResult, setDownloadResult] = useState<YouTubeDownloadResult | null>(null)
  const [profiles, setProfiles] = useState<VoiceProfile[]>([])
  const [selectedProfileId, setSelectedProfileId] = useState<string>('')
  const [runDiarization, setRunDiarization] = useState(false)
  const [audioFormat, setAudioFormat] = useState<'wav' | 'mp3' | 'flac'>('wav')

  // Load profiles on mount
  useEffect(() => {
    loadProfiles()
  }, [])

  const loadProfiles = async () => {
    try {
      const profileList = await api.listProfiles()
      setProfiles(profileList)
    } catch (err) {
      console.error('Failed to load profiles:', err)
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
    setError(null)

    try {
      const result = await api.downloadYouTubeAudio(url, {
        format: audioFormat,
        run_diarization: runDiarization,
      })

      if (!result.success) {
        throw new Error(result.error || 'Download failed')
      }

      setDownloadResult(result)
      setStage('complete')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed')
      setStage('error')
    }
  }

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const handleAddToProfile = async () => {
    if (!downloadResult?.audio_path || !selectedProfileId) return

    try {
      // For now, we would need to create an endpoint to add an existing file as a sample
      // This is a placeholder - the actual implementation would use a new API endpoint
      alert(`TODO: Add ${downloadResult.audio_path} to profile ${selectedProfileId}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add to profile')
    }
  }

  const handleCreateProfile = async () => {
    if (!downloadResult || !videoInfo) return

    const artistName = videoInfo.main_artist || 'Unknown Artist'
    try {
      // Create a new profile from the diarization result
      if (downloadResult.diarization_result && downloadResult.diarization_result.num_speakers > 0) {
        // Use auto-create from diarization
        alert(`TODO: Create profile for "${artistName}" using diarization segments`)
      } else {
        alert(`TODO: Create profile for "${artistName}" from downloaded audio`)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create profile')
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
        <label className="block text-sm font-medium text-gray-300 mb-2">
          YouTube URL
        </label>
        <div className="flex gap-3">
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            disabled={stage === 'fetching' || stage === 'downloading'}
            onKeyDown={(e) => e.key === 'Enter' && handleFetchInfo()}
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

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Audio Format</label>
                  <select
                    value={audioFormat}
                    onChange={(e) => setAudioFormat(e.target.value as 'wav' | 'mp3' | 'flac')}
                    className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                    <option value="wav">WAV (Best for training)</option>
                    <option value="mp3">MP3</option>
                    <option value="flac">FLAC</option>
                  </select>
                </div>

                <div className="flex items-end">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={runDiarization}
                      onChange={(e) => setRunDiarization(e.target.checked)}
                      className="w-4 h-4 rounded bg-gray-700 border-gray-600"
                    />
                    <span className="text-sm">
                      Run speaker diarization
                      {videoInfo.featured_artists.length > 0 && (
                        <span className="text-purple-400 ml-1">
                          (Recommended - {videoInfo.featured_artists.length + 1} artists detected)
                        </span>
                      )}
                    </span>
                  </label>
                </div>
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
              <div className="flex items-center justify-center gap-3 text-blue-400">
                <Loader2 size={24} className="animate-spin" />
                <span>Downloading and processing audio...</span>
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

          {/* Actions */}
          <div className="space-y-4">
            <div className="flex gap-4">
              <select
                value={selectedProfileId}
                onChange={(e) => setSelectedProfileId(e.target.value)}
                className="flex-1 bg-gray-700 border border-gray-600 rounded px-3 py-2"
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

            {videoInfo?.main_artist && (
              <button
                onClick={handleCreateProfile}
                className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg flex items-center justify-center gap-2"
              >
                <User size={18} />
                Create Profile for "{videoInfo.main_artist}"
              </button>
            )}

            <button
              onClick={resetState}
              className="w-full px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded-lg"
            >
              Download Another Video
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
