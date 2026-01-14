import { useState, useRef, useEffect, useCallback } from 'react'
import { User, Plus, Trash2, Edit2, Music, Upload, X, Check, AlertCircle, Loader, Zap, Mic, Play, Square, Volume2, Brain, Trophy } from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService, VoiceProfile } from '../services/api'
import { VoiceProfileTester } from '../components/VoiceProfileTester'

interface VocalCoachingTip {
  type: 'critical' | 'warning' | 'info' | 'success'
  icon: string
  title: string
  message: string
  suggestion: string
}

// Note: CoachingSession, VocalProgress, AutoAccompaniment, ProfessionalEffectsChain
// interfaces reserved for future advanced features

function VocalCoachingAssistant({
  vocalQualityScore,
  noiseLevel,
  signalToNoise,
  pitchStability,
  clippingDetected,
  audioLevel,
  isRecording
}: {
  vocalQualityScore: number
  noiseLevel: number
  signalToNoise: number
  pitchStability: number
  clippingDetected: boolean
  audioLevel: number
  isRecording: boolean
}) {
  const [coachingTips, setCoachingTips] = useState<VocalCoachingTip[]>([])
  const [currentTipIndex, setCurrentTipIndex] = useState(0)
  const [showTipDetails, setShowTipDetails] = useState(false)

  // AI-powered coaching logic
  const analyzeAndCoach = useCallback(() => {
    const newTips: VocalCoachingTip[] = []

    // Critical issues
    if (clippingDetected) {
      newTips.push({
        type: 'critical',
        icon: '🔊',
        title: 'Audio Clipping Detected',
        message: 'Your recording is distorting due to excessive volume!',
        suggestion: 'Move back from the microphone or reduce your singing volume.'
      })
    }

    // Volume coaching
    if (audioLevel < 0.1 && isRecording) {
      newTips.push({
        type: 'warning',
        icon: '🔉',
        title: 'Low Volume Detected',
        message: 'Your voice is too quiet for optimal recording.',
        suggestion: 'Sing louder or move closer to the microphone.'
      })
    }

    // Noise coaching
    if (noiseLevel > 0.5) {
      newTips.push({
        type: 'warning',
        icon: '🗣️',
        title: 'High Background Noise',
        message: 'Too much background noise is affecting recording quality.',
        suggestion: 'Find a quieter location or use noise reduction.'
      })
    }

    // SNR coaching
    if (signalToNoise < 10) {
      newTips.push({
        type: 'warning',
        icon: '📻',
        title: 'Poor Signal Quality',
        message: 'Signal-to-noise ratio is too low.',
        suggestion: 'Reduce background noise or increase microphone gain.'
      })
    }

    // Pitch stability coaching
    if (pitchStability < 0.6) {
      newTips.push({
        type: 'info',
        icon: '🎵',
        title: 'Pitch Stability Practice',
        message: 'Your pitch varies significantly while singing.',
        suggestion: 'Practice singing long tones on a single note for consistency.'
      })
    }

    // Quality score coaching
    if (vocalQualityScore < 0.7 && isRecording) {
      newTips.push({
        type: 'info',
        icon: '⭐',
        title: 'Quality Score Improvement',
        message: 'Your recording quality could be improved.',
        suggestion: 'Focus on steady volume, pitch stability, and minimal background noise.'
      })
    }

    // Success coaching
    if (vocalQualityScore > 0.9 && pitchStability > 0.8 && audioLevel > 0.2) {
      newTips.push({
        type: 'success',
        icon: '🎉',
        title: 'Excellent Recording!',
        message: 'Your vocal quality is outstanding!',
        suggestion: 'Keep singing with this technique - you\'re recording professionally!'
      })
    }

    // Proximity coaching
    if (audioLevel > 0.8 && signalToNoise > 20) {
      newTips.push({
        type: 'info',
        icon: '📏',
        title: 'Microphone Distance Tip',
        message: 'You\'re very close to the microphone.',
        suggestion: 'Try moving back 6-8 inches for the sweet spot.'
      })
    }

    // Perfect recording message
    if (vocalQualityScore > 0.85 && pitchStability > 0.75 && signalToNoise > 15 && !clippingDetected) {
      newTips.push({
        type: 'success',
        icon: '🏆',
        title: 'Professional Quality!',
        message: 'This recording meets studio standards!',
        suggestion: 'You\'re ready for professional voice conversion!'
      })
    }

    setCoachingTips(newTips)

    // Show most important tip first
    if (newTips.length > 0) {
      const criticalTips = newTips.filter(t => t.type === 'critical')
      const warningTips = newTips.filter(t => t.type === 'warning')
      const successTips = newTips.filter(t => t.type === 'success')

      if (criticalTips.length > 0) {
        setCurrentTipIndex(0)
      } else if (warningTips.length > 0) {
        setCurrentTipIndex(newTips.indexOf(warningTips[0]))
      } else if (successTips.length > 0) {
        setCurrentTipIndex(newTips.indexOf(successTips[0]))
      }
    }
  }, [vocalQualityScore, noiseLevel, signalToNoise, pitchStability, clippingDetected, audioLevel, isRecording])

  // Coaching analysis runs every 2 seconds during recording
  useEffect(() => {
    if (isRecording) {
      analyzeAndCoach()
      const interval = setInterval(analyzeAndCoach, 2000)
      return () => clearInterval(interval)
    } else {
      setCoachingTips([]) // Clear tips when not recording
    }
  }, [isRecording, analyzeAndCoach])

  if (coachingTips.length === 0 && !isRecording) {
    return null
  }

  const currentTip = coachingTips[currentTipIndex]
  const tipColorClass = {
    critical: 'bg-red-50 border-red-200 text-red-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    info: 'bg-blue-50 border-blue-200 text-blue-800',
    success: 'bg-green-50 border-green-200 text-green-800'
  }[currentTip?.type || 'info']

  return (
    <div className={`border-2 rounded-lg p-4 mb-4 ${tipColorClass}`}>
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          {isRecording ? (
            <Brain className="w-8 h-8 animate-pulse" />
          ) : (
            <Brain className="w-8 h-8" />
          )}
        </div>

        <div className="flex-1">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold flex items-center space-x-2">
              <span>AI Vocal Coach</span>
              {currentTip && <span>{currentTip.icon}</span>}
            </h3>

            {coachingTips.length > 1 && (
              <div className="flex space-x-1">
                <button
                  onClick={() => setCurrentTipIndex(prev => Math.max(0, prev - 1))}
                  disabled={currentTipIndex === 0}
                  className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-50"
                >
                  ‹
                </button>
                <span className="text-xs text-gray-500 px-2">
                  {currentTipIndex + 1} of {coachingTips.length}
                </span>
                <button
                  onClick={() => setCurrentTipIndex(prev => Math.min(coachingTips.length - 1, prev + 1))}
                  disabled={currentTipIndex === coachingTips.length - 1}
                  className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-50"
                >
                  ›
                </button>
              </div>
            )}
          </div>

          {currentTip ? (
            <div>
              <h4 className="font-medium mb-1">{currentTip.title}</h4>
              <p className="text-sm mb-2">{currentTip.message}</p>
              <p className="text-sm font-medium mb-2">{currentTip.suggestion}</p>

              <button
                onClick={() => setShowTipDetails(!showTipDetails)}
                className="text-xs underline hover:no-underline"
              >
                {showTipDetails ? 'Hide' : 'Show'} detailed explanation
              </button>

              {showTipDetails && (
                <div className="mt-3 text-xs space-y-2 border-t pt-3">
                  <div>
                    <strong>Why this matters:</strong> Professional singers and audio engineers
                    pay close attention to these metrics to achieve broadcast-quality recordings.
                  </div>
                  <div>
                    <strong>Measuring technique:</strong>
                    <ul className="list-disc list-inside mt-1 space-y-1">
                      <li>Quality Score = weighted combination of volume, SNR, pitch stability, and clipping</li>
                      <li>SNR measures how much clear signal you have above background noise</li>
                      <li>Pitch stability indicates how steady your note control is</li>
                      <li>Volume should be consistent (not too quiet or loud)</li>
                    </ul>
                  </div>
                  <div>
                    <strong>Pro tip:</strong> Scrolling through tips during recording helps you
                    understand what each metric means and how to improve it live!
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div>
              <h4 className="font-medium mb-1">AI Coaching Ready</h4>
              <p className="text-sm">
                Start recording to receive real-time vocal coaching and performance feedback!
              </p>
            </div>
          )}

          {/* Recording Quality Summary */}
          {isRecording && (
            <div className="mt-4 pt-3 border-t">
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <div className="flex justify-between mb-1">
                    <span>Overall Quality:</span>
                    <span className="font-medium">{Math.round(vocalQualityScore * 100)}%</span>
                  </div>
                  <div className="w-full h-2 bg-gray-200 rounded-full">
                    <div
                      className={`h-full rounded-full ${
                        vocalQualityScore > 0.8 ? 'bg-green-500' :
                        vocalQualityScore > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${vocalQualityScore * 100}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span>Live Feedback:</span>
                    <span className="font-medium">
                      {clippingDetected ? '⚠️ Clipping' :
                       audioLevel < 0.1 ? '🔉 Too quiet' :
                       noiseLevel > 0.5 ? '📢 Noisy' :
                       '✅ Great!'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Achievement Badges */}
          {isRecording && vocalQualityScore > 0.85 && (
            <div className="mt-3 flex items-center space-x-2">
              <Trophy className="w-4 h-4 text-yellow-500 animate-bounce" />
              <span className="text-xs font-medium text-yellow-700">
                🎖️ Professional Grade Recording!
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export function VoiceProfilesPage() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showSingAlongModal, setShowSingAlongModal] = useState(false)
  const [showEditModal, setShowEditModal] = useState(false)
  const [showTester, setShowTester] = useState(false)
  const [selectedProfile, setSelectedProfile] = useState<VoiceProfile | null>(null)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [profileToDelete, setProfileToDelete] = useState<VoiceProfile | null>(null)

  const queryClient = useQueryClient()

  // Fetch voice profiles
  const { data: profiles = [], isLoading, error } = useQuery({
    queryKey: ['voiceProfiles'],
    queryFn: () => apiService.getVoiceProfiles(),
  })

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (profileId: string) => apiService.deleteVoiceProfile(profileId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['voiceProfiles'] })
      setShowDeleteConfirm(false)
      setProfileToDelete(null)
    },
  })

  const handleDelete = (profile: VoiceProfile) => {
    setProfileToDelete(profile)
    setShowDeleteConfirm(true)
  }

  const confirmDelete = () => {
    if (profileToDelete) {
      deleteMutation.mutate(profileToDelete.id)
    }
  }

  const handleEdit = (profile: VoiceProfile) => {
    setSelectedProfile(profile)
    setShowEditModal(true)
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center space-x-3">
            <User className="w-8 h-8 text-purple-600" />
            <span>Voice Profiles</span>
          </h1>
          <p className="text-gray-600 mt-2">
            Manage your voice profiles for singing voice conversion
          </p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={() => setShowCreateModal(true)}
            className="bg-purple-600 hover:bg-purple-700 text-white font-semibold px-6 py-3 rounded-lg transition-colors flex items-center space-x-2"
          >
            <Upload className="w-5 h-5" />
            <span>Upload Audio</span>
          </button>
          <button
            onClick={() => setShowSingAlongModal(true)}
            className="bg-green-600 hover:bg-green-700 text-white font-semibold px-6 py-3 rounded-lg transition-colors flex items-center space-x-2"
          >
            <Mic className="w-5 h-5" />
            <span>Sing Along</span>
          </button>
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="bg-white rounded-lg shadow-lg p-12 text-center">
          <Loader className="w-12 h-12 text-purple-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading voice profiles...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold text-red-900">Error Loading Profiles</h3>
            <p className="text-red-700 text-sm mt-1">{(error as Error).message}</p>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && !error && profiles.length === 0 && (
        <div className="bg-white rounded-lg shadow-lg p-12 text-center">
          <div className="max-w-md mx-auto">
            <div className="p-4 bg-purple-100 rounded-full inline-block mb-4">
              <Music className="w-12 h-12 text-purple-600" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              No Voice Profiles Yet
            </h2>
            <p className="text-gray-600 mb-6">
              Create your first voice profile by uploading a 30-60 second audio sample.
              This will be used to convert singing voices in songs.
            </p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="bg-purple-600 hover:bg-purple-700 text-white font-semibold px-6 py-3 rounded-lg transition-colors inline-flex items-center space-x-2"
            >
              <Plus className="w-5 h-5" />
              <span>Create Your First Profile</span>
            </button>
          </div>
        </div>
      )}

      {/* Profiles Grid */}
      {!isLoading && !error && profiles.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {profiles.map((profile: VoiceProfile) => (
            <ProfileCard
              key={profile.id}
              profile={profile}
              onEdit={handleEdit}
              onDelete={handleDelete}
              onTest={() => {
                setSelectedProfile(profile)
                setShowTester(true)
              }}
            />
          ))}
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <CreateProfileModal
          onClose={() => setShowCreateModal(false)}
          onSuccess={() => {
            setShowCreateModal(false)
            queryClient.invalidateQueries({ queryKey: ['voiceProfiles'] })
          }}
        />
      )}

      {/* Edit Modal */}
      {showEditModal && selectedProfile && (
        <EditProfileModal
          profile={selectedProfile}
          onClose={() => {
            setShowEditModal(false)
            setSelectedProfile(null)
          }}
          onSuccess={() => {
            setShowEditModal(false)
            setSelectedProfile(null)
            queryClient.invalidateQueries({ queryKey: ['voiceProfiles'] })
          }}
        />
      )}

      {/* Delete Confirmation */}
      {showDeleteConfirm && profileToDelete && (
        <DeleteConfirmModal
          profile={profileToDelete}
          onConfirm={confirmDelete}
          onCancel={() => {
            setShowDeleteConfirm(false)
            setProfileToDelete(null)
          }}
          isDeleting={deleteMutation.isPending}
        />
      )}

      {/* Sing-Along Profile Creation */}
      {showSingAlongModal && (
        <SingAlongProfileModal
          onClose={() => setShowSingAlongModal(false)}
          onSuccess={() => {
            setShowSingAlongModal(false)
            queryClient.invalidateQueries({ queryKey: ['voiceProfiles'] })
          }}
        />
      )}

      {/* Voice Profile Tester */}
      {showTester && selectedProfile && (
        <VoiceProfileTester
          profile={selectedProfile}
          onClose={() => {
            setShowTester(false)
            setSelectedProfile(null)
          }}
        />
      )}
    </div>
  )
}

// Sing-Along Profile Creation Modal
interface SingAlongProfileModalProps {
  onClose: () => void
  onSuccess: () => void
}

function SingAlongProfileModal({ onClose, onSuccess }: SingAlongProfileModalProps) {
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [userId, setUserId] = useState('')
  const [profileName, setProfileName] = useState('')
  const [description, setDescription] = useState('')

  // Recording state
  const [isPlaying, setIsPlaying] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [audioLevel, setAudioLevel] = useState(0)
  const [recordingTime, setRecordingTime] = useState(0)

  // Audio quality monitoring
  const [noiseLevel, setNoiseLevel] = useState(0)
  const [clippingDetected, setClippingDetected] = useState(false)
  const [vocalQualityScore, setVocalQualityScore] = useState(0)
  const [pitchStability, setPitchStability] = useState(0)
  const [signalToNoise, setSignalToNoise] = useState(0)

  // Recording features (prefixed with _ to suppress unused warnings - future use)
  const [_countdown, _setCountdown] = useState(0)
  const [_recordingChunks, _setRecordingChunks] = useState<Blob[]>([])
  const [_recordingStats, _setRecordingStats] = useState<any>(null)
  const [waveformData, setWaveformData] = useState<number[]>([])

  // Controls
  const [micGain, setMicGain] = useState(1.0)
  const [monitorInput, setMonitorInput] = useState(false)
  const [noiseReduction, setNoiseReduction] = useState(true)
  const [reverbEffect, setReverbEffect] = useState(false)

  // Recorder and playback refs
  const songAudioRef = useRef<HTMLAudioElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recordedChunksRef = useRef<Blob[]>([])
  const analyserRef = useRef<AnalyserNode | null>(null)
  const recordingStartTimeRef = useRef<number>(0)
  const animationFrameRef = useRef<number>()

  const [error, setError] = useState<string | null>(null)

  const createMutation = useMutation({
    mutationFn: async (recordedBlob: Blob) => {
      if (!profileName.trim()) {
        throw new Error('Profile name is required')
      }

      // Convert blob to file
      const recordedFile = new File([recordedBlob], 'singing_recording.wav', { type: 'audio/wav' })

      const formData = new FormData()
      formData.append('audio', recordedFile)
      formData.append('profile_name', profileName.trim())
      if (userId) formData.append('user_id', userId)
      if (description) formData.append('description', description)

      return apiService.createVoiceProfile(formData)
    },
    onSuccess: () => {
      onSuccess()
    },
    onError: (err: any) => {
      setError(err.response?.data?.message || err.message || 'Failed to create profile')
    },
  })

  // Enhanced audio quality analysis
  const analyzeAudioQuality = (stream: MediaStream) => {
    const audioContext = new AudioContext()
    const source = audioContext.createMediaStreamSource(stream)
    const analyser = audioContext.createAnalyser()
    analyser.fftSize = 1024
    analyser.smoothingTimeConstant = 0.8
    analyserRef.current = analyser
    source.connect(analyser)

    const dataArray = new Uint8Array(analyser.frequencyBinCount)
    const buffer = new Float32Array(dataArray.length)

    let history: number[] = []
    let clippingCount = 0
    let totalSamples = 0

    const analyze = () => {
      if (analyserRef.current && isRecording) {
        analyserRef.current.getFloatFrequencyData(buffer)

        // Calculate RMS level
        const rms = Math.sqrt(buffer.reduce((sum, val) => sum + val * val, 0) / buffer.length)
        const level = (rms + 100) / 100 // Normalize to 0-1 (accounting for dB)
        setAudioLevel(Math.max(0, Math.min(1, level)))

        // Calculate noise floor (quietest frequencies)
        const quietest = buffer.slice(0, Math.floor(buffer.length * 0.1))
        const noiseFloor = quietest.reduce((a, b) => a + b, 0) / quietest.length
        setNoiseLevel(Math.abs(noiseFloor))

        // Signal-to-noise ratio (simplified)
        const signalPower = Math.pow(10, rms / 10)
        const noisePower = Math.pow(10, noiseFloor / 10)
        const snr = signalPower > noisePower ? 10 * Math.log10(signalPower / noisePower) : 0
        setSignalToNoise(Math.max(0, Math.min(50, snr)))

        // Pitch stability (simplified using dominant frequency)
        const dominantFreq = buffer.indexOf(Math.max(...buffer)) * (audioContext.sampleRate / analyser.frequencyBinCount)
        history.push(dominantFreq)
        if (history.length > 10) history.shift()

        const variation = history.length > 1 ?
          history.reduce((acc, freq, i, arr) => i > 0 ? acc + Math.abs(freq - arr[i-1]) : 0, 0) / (history.length - 1) : 0
        setPitchStability(Math.max(0, Math.min(1, 100 / (variation + 1))))

        // Clipping detection (simplified - check for saturation)
        const peakLevel = Math.max(...buffer.map(Math.abs))
        const isClipping = peakLevel > 0.95 // Very close to max
        setClippingDetected(isClipping)

        if (isClipping) clippingCount++
        totalSamples++

        // Vocal quality score (weighted combination of metrics)
        const qualityScore = (
          (level * 0.3) +                      // Volume
          Math.min(1, snr / 15) * 0.3 +       // SNR (good >=15dB)
          (1 - (clippingCount / totalSamples)) * 0.2 + // No clipping
          pitchStability * 0.2               // Pitch consistency
        )
        setVocalQualityScore(Math.max(0, Math.min(1, qualityScore)))

        // Build waveform data
        const waveformSample = level
        setWaveformData(prev => {
          const newData = [...prev, waveformSample]
          return newData.slice(-200) // Keep last 200 samples
        })

        animationFrameRef.current = requestAnimationFrame(analyze)
      }
    }
    analyze()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith('audio/')) {
      setAudioFile(file)
      setError(null)

      // Load audio to get duration
      const audio = new Audio(URL.createObjectURL(file))
      audio.onloadedmetadata = () => setDuration(audio.duration)
      audio.onerror = () => setError('Invalid audio file')
    }
  }

  const startPlayback = async () => {
    if (!songAudioRef.current || !audioFile) return

    try {
      await songAudioRef.current.play()
      setIsPlaying(true)
    } catch (err) {
      setError('Failed to play audio file')
    }
  }

  const stopPlayback = () => {
    if (songAudioRef.current) {
      songAudioRef.current.pause()
      songAudioRef.current.currentTime = 0
    }
    setIsPlaying(false)
    setCurrentTime(0)
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 44100,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      })

      // Analyze audio for visual feedback
      analyzeAudioQuality(stream)

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })
      mediaRecorderRef.current = mediaRecorder
      recordedChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstart = () => {
        recordingStartTimeRef.current = Date.now()
        setRecordingTime(0)
      }

      mediaRecorder.onstop = () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current)
        }
        setAudioLevel(0)
        setRecordingTime(0)
      }

      mediaRecorder.start(100) // Collect data every 100ms
      setIsRecording(true)
      setError(null)

      // Start playback simultaneously
      setTimeout(() => startPlayback(), 500) // Small delay to sync

    } catch (err) {
      setError('Could not access microphone. Please check permissions.')
    }
  }

  const stopRecording = async () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop())

      setIsRecording(false)
      setIsPlaying(false)
      stopPlayback()

      // Create recorded blob and submit
      setTimeout(() => {
        const recordedBlob = new Blob(recordedChunksRef.current, { type: 'audio/webm' })
        createMutation.mutate(recordedBlob)
      }, 200) // Brief delay to ensure recording is complete
    }
  }

  // Update time displays
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | undefined
    if (isRecording && recordingStartTimeRef.current) {
      interval = setInterval(() => {
        setRecordingTime((Date.now() - recordingStartTimeRef.current) / 1000)
      }, 100)
    }
    return () => clearInterval(interval)
  }, [isRecording])

  useEffect(() => {
    if (songAudioRef.current) {
      const updateTime = () => setCurrentTime(songAudioRef.current!.currentTime)
      songAudioRef.current.addEventListener('timeupdate', updateTime)
      songAudioRef.current.addEventListener('ended', () => setIsPlaying(false))

      return () => {
        songAudioRef.current?.removeEventListener('timeupdate', updateTime)
        songAudioRef.current?.removeEventListener('ended', () => setIsPlaying(false))
      }
    }
  }, [audioFile])

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[95vh] overflow-y-auto">
        <div className="p-6 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-900 flex items-center space-x-3">
            <Mic className="w-6 h-6 text-purple-600" />
            <span>Sing Along & Create Profile</span>
          </h2>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
            disabled={createMutation.isPending}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Instructions */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2 flex items-center space-x-2">
              <Mic className="w-4 h-4" />
              <span>How It Works</span>
            </h3>
            <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
              <li>Upload a song you want to sing along with</li>
              <li>Listen to a preview and adjust volume</li>
              <li>Click "Start Singing" when you're ready</li>
              <li>Sing along with the song as it plays</li>
              <li>The system will analyze your voice and create a profile</li>
              <li>Your profile will be ready for voice conversion!</li>
            </ol>
          </div>

          {/* Step 1: Upload Song */}
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-900">Step 1: Choose a Song</h3>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-purple-400 transition-colors">
              <input
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                className="hidden"
                id="song-upload"
              />
              <label htmlFor="song-upload" className="cursor-pointer flex flex-col items-center">
                <Music className="w-12 h-12 text-gray-400 mb-3" />
                {audioFile ? (
                  <div>
                    <p className="text-sm font-medium text-gray-900">{audioFile.name}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      Duration: {duration.toFixed(1)}s • {(audioFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                ) : (
                  <div>
                    <p className="text-sm font-medium text-gray-900">Click to upload song</p>
                    <p className="text-xs text-gray-500 mt-1">WAV, MP3, FLAC up to 50MB</p>
                  </div>
                )}
              </label>
            </div>

            {/* Audio Preview */}
            {audioFile && (
              <audio
                ref={songAudioRef}
                src={URL.createObjectURL(audioFile)}
                preload="metadata"
                className="hidden"
              />
            )}

            {audioFile && (
              <div className="flex items-center justify-center space-x-4 bg-gray-50 rounded-lg p-4">
                <button
                  type="button"
                  onClick={isPlaying ? stopPlayback : startPlayback}
                  className="p-3 bg-purple-600 hover:bg-purple-700 text-white rounded-full transition-colors"
                  disabled={!audioFile}
                >
                  {isPlaying ? <Square className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                </button>
                <div className="flex-1">
                  <div className="w-full h-2 bg-gray-200 rounded-full">
                    <div
                      className="h-full bg-purple-600 rounded-full transition-all"
                      style={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-600 mt-1">
                    <span>{Math.floor(currentTime / 60)}:{(currentTime % 60).toString().padStart(2, '0')}</span>
                    <span>{Math.floor(duration / 60)}:{(duration % 60).toString().padStart(2, '0')}</span>
                  </div>
                </div>
                <Volume2 className="w-5 h-5 text-gray-400" />
              </div>
            )}
          </div>

          {/* Step 2: Profile Details */}
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-900">Step 2: Profile Information</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Profile Name *
                </label>
                <input
                  type="text"
                  value={profileName}
                  onChange={(e) => setProfileName(e.target.value)}
                  placeholder="e.g., My Singing Voice"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  disabled={createMutation.isPending}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  User ID (Optional)
                </label>
                <input
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder="user123"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  disabled={createMutation.isPending}
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description (Optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe your singing style or voice characteristics"
                rows={3}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
                disabled={createMutation.isPending}
              />
            </div>
          </div>

          {/* Step 3: Recording Interface */}
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-900">Step 3: Record Your Singing</h3>

            {!isRecording && !createMutation.isPending && (
              <div className="text-center py-8">
                <div className="p-4 bg-red-100 rounded-full inline-block mb-4">
                  <Mic className="w-16 h-16 text-red-600" />
                </div>
                <p className="text-gray-600 mb-4 text-lg">
                  When you're ready, click "Start Singing" and begin singing along with your chosen song
                </p>
                <p className="text-sm text-gray-500 mb-6">
                  Make sure your microphone is working and you're in a quiet environment
                </p>
                <button
                  onClick={startRecording}
                  className="px-8 py-4 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg transition-colors flex items-center space-x-2 mx-auto"
                  disabled={!audioFile || !profileName.trim()}
                >
                  <Mic className="w-5 h-5" />
                  <span>Start Singing & Recording</span>
                </button>
                {(!audioFile || !profileName.trim()) && (
                  <p className="text-sm text-gray-500 mt-2">
                    {!audioFile ? 'Please upload a song first' : 'Please enter a profile name'}
                  </p>
                )}
              </div>
            )}

                {/* AI Vocal Coaching Assistant */}
                <VocalCoachingAssistant
                  vocalQualityScore={vocalQualityScore}
                  noiseLevel={noiseLevel}
                  signalToNoise={signalToNoise}
                  pitchStability={pitchStability}
                  clippingDetected={clippingDetected}
                  audioLevel={audioLevel}
                  isRecording={isRecording}
                />

                {/* Recording Interface */}
                {(isRecording || recordingTime > 0) && (
                  <div className="space-y-6">
                    {/* Recording Status */}
                    <div className="bg-gradient-to-r from-red-50 to-purple-50 border-2 border-red-200 rounded-lg p-6">
                      <div className="text-center mb-6">
                        <div className={`inline-block p-4 rounded-full ${isRecording ? 'bg-red-100 animate-pulse' : 'bg-gray-100'}`}>
                          <Mic className="w-16 h-16 text-red-600" />
                        </div>
                        <h4 className="font-semibold text-lg mt-4">
                          {isRecording ? 'Recording your singing...' : 'Recording Complete'}
                        </h4>
                        <p className="text-gray-600">
                          Recording time: {recordingTime.toFixed(1)}s
                        </p>
                      </div>

                      {/* Live Audio Visualization */}
                      {isRecording && (
                        <div className="mb-6">
                          {/* Audio Level Bars */}
                          <div className="flex items-center justify-center space-x-1 mb-4">
                            {Array.from({ length: 15 }).map((_, i) => (
                              <div
                                key={i}
                                className="w-2 bg-gradient-to-t from-purple-400 to-pink-400 rounded-full transition-all duration-150"
                                style={{
                                  height: `${4 + (audioLevel * 100)}px`,
                                  opacity: audioLevel > (i / 15) ? 1 : 0.3,
                                  transform: `scaleY(${1 + audioLevel * 0.2})`
                                }}
                              />
                            ))}
                          </div>

                          {/* Waveform Visualization */}
                          {waveformData.length > 0 && (
                            <div className="bg-black rounded-lg p-4 mb-4">
                              <canvas
                                className="w-full h-20"
                                style={{
                                  background: 'linear-gradient(to right, #1a1a2e, #16213e, #0f3460)'
                                }}
                                ref={(canvas) => {
                                  if (canvas) {
                                    const ctx = canvas.getContext('2d')
                                    if (ctx) {
                                      ctx.clearRect(0, 0, canvas.width, canvas.height)
                                      ctx.strokeStyle = '#8b5cf6'
                                      ctx.lineWidth = 2
                                      ctx.beginPath()

                                      waveformData.forEach((amplitude, index) => {
                                        const x = (index / waveformData.length) * canvas.width
                                        const y = canvas.height / 2 + (amplitude - 0.5) * canvas.height * 0.8
                                        if (index === 0) ctx.moveTo(x, y)
                                        else ctx.lineTo(x, y)
                                      })

                                      ctx.stroke()
                                    }
                                  }
                                }}
                              />
                            </div>
                          )}
                        </div>
                      )}

                      {/* Audio Quality Indicators */}
                      {isRecording && (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                          {/* Vocal Quality Score */}
                          <div className="bg-white rounded-lg p-4 text-center border-2 border-purple-200">
                            <div className="text-2xl font-bold text-purple-600">
                              {Math.round(vocalQualityScore * 100)}%
                            </div>
                            <div className="text-sm text-gray-600">Quality Score</div>
                            <div className="w-full h-2 bg-gray-200 rounded-full mt-2">
                              <div
                                className={`h-full rounded-full ${
                                  vocalQualityScore > 0.8 ? 'bg-green-500' :
                                  vocalQualityScore > 0.6 ? 'bg-yellow-500' :
                                  'bg-red-500'
                                }`}
                                style={{ width: `${vocalQualityScore * 100}%` }}
                              />
                            </div>
                          </div>

                          {/* Noise Level */}
                          <div className="bg-white rounded-lg p-4 text-center border-2 border-blue-200">
                            <div className="text-2xl font-bold text-blue-600">
                              {noiseLevel.toFixed(2)}
                            </div>
                            <div className="text-sm text-gray-600">Noise Floor</div>
                            <div className="w-full h-2 bg-gray-200 rounded-full mt-2">
                              <div
                                className={`h-full rounded-full ${
                                  noiseLevel < 0.3 ? 'bg-green-500' :
                                  noiseLevel < 0.6 ? 'bg-yellow-500' :
                                  'bg-red-500'
                                }`}
                                style={{ width: `${noiseLevel * 100 * 3}%` }}
                              />
                            </div>
                          </div>

                          {/* Signal-to-Noise Ratio */}
                          <div className="bg-white rounded-lg p-4 text-center border-2 border-green-200">
                            <div className="text-2xl font-bold text-green-600">
                              {signalToNoise.toFixed(0)}dB
                            </div>
                            <div className="text-sm text-gray-600">SNR</div>
                            <div className="w-full h-2 bg-gray-200 rounded-full mt-2">
                              <div
                                className={`h-full rounded-full ${
                                  signalToNoise > 20 ? 'bg-green-500' :
                                  signalToNoise > 10 ? 'bg-yellow-500' :
                                  'bg-red-500'
                                }`}
                                style={{ width: `${Math.min(signalToNoise / 30 * 100, 100)}%` }}
                              />
                            </div>
                          </div>

                          {/* Pitch Stability */}
                          <div className="bg-white rounded-lg p-4 text-center border-2 border-orange-200">
                            <div className="text-2xl font-bold text-orange-600">
                              {Math.round(pitchStability * 100)}%
                            </div>
                            <div className="text-sm text-gray-600">Pitch Stability</div>
                            <div className="w-full h-2 bg-gray-200 rounded-full mt-2">
                              <div
                                className={`h-full rounded-full ${
                                  pitchStability > 0.7 ? 'bg-green-500' :
                                  pitchStability > 0.5 ? 'bg-yellow-500' :
                                  'bg-red-500'
                                }`}
                                style={{ width: `${pitchStability * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Recording Statistics */}
                      {recordingTime > 0 && (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                          <div className="text-center">
                            <div className="text-lg font-bold text-gray-900">{recordingTime.toFixed(1)}s</div>
                            <div className="text-sm text-gray-600">Duration</div>
                          </div>
                          <div className="text-center">
                            <div className="text-lg font-bold text-gray-900">{recordedChunksRef.current.length}</div>
                            <div className="text-sm text-gray-600">Chunks</div>
                          </div>
                          <div className="text-center">
                            <div className="text-lg font-bold text-gray-900">
                              {(recordedChunksRef.current.reduce((acc: any, chunk: any) => acc + chunk.size, 0) / 1024 / 1024).toFixed(1)}MB
                            </div>
                            <div className="text-sm text-gray-600">Data Size</div>
                          </div>
                          <div className="text-center">
                            <div className="text-lg font-bold text-gray-900">44.1kHz</div>
                            <div className="text-sm text-gray-600">Sample Rate</div>
                          </div>
                        </div>
                      )}

                      {/* Recording Controls */}
                      <div className="flex justify-center space-x-4">
                        <button
                          onClick={stopRecording}
                          className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors flex items-center space-x-2"
                          disabled={!isRecording}
                        >
                          <Square className="w-5 h-5" />
                          <span>Stop Recording</span>
                        </button>
                      </div>
                    </div>

                {/* Audio Controls Panel */}
                <div className="bg-gray-50 rounded-lg p-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Microphone Gain: {(micGain * 100).toFixed(0)}%
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="2.0"
                      step="0.1"
                      value={micGain}
                      onChange={(e) => setMicGain(parseFloat(e.target.value))}
                      className="w-full"
                      disabled={isRecording}
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-4">
                      Effects
                    </label>
                    <div className="space-y-2">
                      <label className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={noiseReduction}
                          onChange={(e) => setNoiseReduction(e.target.checked)}
                          disabled={isRecording}
                          className="rounded"
                        />
                        <span className="text-sm">Noise Reduction</span>
                      </label>
                      <label className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={reverbEffect}
                          onChange={(e) => setReverbEffect(e.target.checked)}
                          disabled={isRecording}
                          className="rounded"
                        />
                        <span className="text-sm">Reverb</span>
                      </label>
                      <label className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={monitorInput}
                          onChange={(e) => setMonitorInput(e.target.checked)}
                          disabled={isRecording || !isRecording}
                          className="rounded"
                        />
                        <span className="text-sm">Hear Yourself</span>
                      </label>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-4">
                      Recording Tips
                    </label>
                    <ul className="text-xs text-gray-600 space-y-1">
                      <li>• Stay 6-12 inches from mic</li>
                      <li>• Sing at consistent volume</li>
                      <li>• Avoid sudden movements</li>
                      <li>• Check for red clips warning</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {/* Processing State */}
            {createMutation.isPending && (
              <div className="bg-purple-50 border-2 border-purple-200 rounded-lg p-6 text-center">
                <Loader className="w-12 h-12 text-purple-600 animate-spin mx-auto mb-4" />
                <h4 className="font-semibold text-purple-900">Creating Your Voice Profile</h4>
                <p className="text-purple-700">
                  Analyzing your singing and generating profile...
                </p>
              </div>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-semibold text-red-900">Error</h3>
                <p className="text-red-700 text-sm mt-1">{error}</p>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end space-x-3 pt-6 border-t border-gray-200">
            <button
              type="button"
              onClick={onClose}
              className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              disabled={isRecording || createMutation.isPending}
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// Profile Card Component
interface ProfileCardProps {
  profile: VoiceProfile
  onEdit: (profile: VoiceProfile) => void
  onDelete: (profile: VoiceProfile) => void
  onTest: () => void
}

function ProfileCard({ profile, onEdit, onDelete, onTest }: ProfileCardProps) {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-purple-100 rounded-lg">
            <User className="w-6 h-6 text-purple-600" />
          </div>
          <div>
            <h3 className="font-bold text-gray-900">{profile.name}</h3>
            <p className="text-sm text-gray-500">
              {new Date(profile.created_at).toLocaleDateString()}
            </p>
          </div>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={onTest}
            className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
            title="Test profile"
          >
            <Zap className="w-4 h-4" />
          </button>
          <button
            onClick={() => onEdit(profile)}
            className="p-2 text-gray-600 hover:text-purple-600 hover:bg-purple-50 rounded-lg transition-colors"
            title="Edit profile"
          >
            <Edit2 className="w-4 h-4" />
          </button>
          <button
            onClick={() => onDelete(profile)}
            className="p-2 text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
            title="Delete profile"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {profile.description && (
        <p className="text-gray-600 text-sm mb-4">{profile.description}</p>
      )}

      <div className="space-y-2">
        {profile.sample_duration && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">Sample Duration:</span>
            <span className="font-medium text-gray-900">
              {profile.sample_duration.toFixed(1)}s
            </span>
          </div>
        )}

        {profile.vocal_range && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">Vocal Range:</span>
            <span className="font-medium text-gray-900">
              {profile.vocal_range.min_note} - {profile.vocal_range.max_note}
            </span>
          </div>
        )}

        {profile.characteristics && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">Characteristics:</span>
            <span className="font-medium text-gray-900">
              {profile.characteristics.gender}, {profile.characteristics.age_range}
            </span>
          </div>
        )}

        {profile.embedding_quality !== undefined && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">Quality:</span>
            <div className="flex items-center space-x-2">
              <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full ${
                    profile.embedding_quality > 0.8
                      ? 'bg-green-500'
                      : profile.embedding_quality > 0.6
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                  }`}
                  style={{ width: `${profile.embedding_quality * 100}%` }}
                />
              </div>
              <span className="font-medium text-gray-900">
                {(profile.embedding_quality * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Create Profile Modal
interface CreateProfileModalProps {
  onClose: () => void
  onSuccess: () => void
}

function CreateProfileModal({ onClose, onSuccess }: CreateProfileModalProps) {
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [userId, setUserId] = useState('')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const createMutation = useMutation({
    mutationFn: async () => {
      if (!audioFile) throw new Error('No audio file selected')

      const formData = new FormData()
      formData.append('audio', audioFile)
      if (userId) formData.append('user_id', userId)

      return apiService.createVoiceProfile(formData, setUploadProgress)
    },
    onSuccess: () => {
      onSuccess()
    },
    onError: (err: any) => {
      setError(err.response?.data?.message || err.message || 'Failed to create profile')
    },
  })

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setAudioFile(file)
      setError(null)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!audioFile) {
      setError('Please select an audio file')
      return
    }
    createMutation.mutate()
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-900">Create Voice Profile</h2>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
            disabled={createMutation.isPending}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Instructions */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">Recording Guidelines</h3>
            <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside">
              <li>Upload a 30-60 second audio sample</li>
              <li>Use clear, high-quality audio (WAV, MP3, FLAC)</li>
              <li>Include varied singing (different notes and dynamics)</li>
              <li>Avoid background noise and music</li>
              <li>Record in a quiet environment</li>
            </ul>
          </div>

          {/* File Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Audio Sample *
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-purple-400 transition-colors">
              <input
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                className="hidden"
                id="audio-upload"
                disabled={createMutation.isPending}
              />
              <label
                htmlFor="audio-upload"
                className="cursor-pointer flex flex-col items-center"
              >
                <Upload className="w-12 h-12 text-gray-400 mb-3" />
                {audioFile ? (
                  <div>
                    <p className="text-sm font-medium text-gray-900">{audioFile.name}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {(audioFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                ) : (
                  <div>
                    <p className="text-sm font-medium text-gray-900">
                      Click to upload audio file
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      WAV, MP3, FLAC up to 100MB
                    </p>
                  </div>
                )}
              </label>
            </div>
          </div>

          {/* User ID (Optional) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              User ID (Optional)
            </label>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="Enter user identifier"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              disabled={createMutation.isPending}
            />
            <p className="text-xs text-gray-500 mt-1">
              Optional identifier to organize profiles by user
            </p>
          </div>

          {/* Upload Progress */}
          {createMutation.isPending && uploadProgress > 0 && (
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-600">Uploading...</span>
                <span className="font-medium text-gray-900">{uploadProgress}%</span>
              </div>
              <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-600 transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-semibold text-red-900">Error</h3>
                <p className="text-red-700 text-sm mt-1">{error}</p>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
            <button
              type="button"
              onClick={onClose}
              className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              disabled={createMutation.isPending}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              disabled={!audioFile || createMutation.isPending}
            >
              {createMutation.isPending ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  <span>Creating...</span>
                </>
              ) : (
                <>
                  <Check className="w-4 h-4" />
                  <span>Create Profile</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

// Edit Profile Modal
interface EditProfileModalProps {
  profile: VoiceProfile
  onClose: () => void
  onSuccess: () => void
}

function EditProfileModal({ profile, onClose, onSuccess }: EditProfileModalProps) {
  const [name, setName] = useState(profile.name)
  const [description, setDescription] = useState(profile.description || '')
  const [error, setError] = useState<string | null>(null)

  const updateMutation = useMutation({
    mutationFn: async () => {
      return apiService.updateVoiceProfile(profile.id, {
        name,
        description,
      })
    },
    onSuccess: () => {
      onSuccess()
    },
    onError: (err: any) => {
      setError(err.response?.data?.message || err.message || 'Failed to update profile')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) {
      setError('Profile name is required')
      return
    }
    updateMutation.mutate()
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full">
        <div className="p-6 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-900">Edit Voice Profile</h2>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
            disabled={updateMutation.isPending}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Profile Name *
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter profile name"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              disabled={updateMutation.isPending}
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Description (Optional)
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Add a description for this profile"
              rows={3}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
              disabled={updateMutation.isPending}
            />
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-semibold text-red-900">Error</h3>
                <p className="text-red-700 text-sm mt-1">{error}</p>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
            <button
              type="button"
              onClick={onClose}
              className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              disabled={updateMutation.isPending}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              disabled={!name.trim() || updateMutation.isPending}
            >
              {updateMutation.isPending ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  <span>Saving...</span>
                </>
              ) : (
                <>
                  <Check className="w-4 h-4" />
                  <span>Save Changes</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

// Delete Confirmation Modal
interface DeleteConfirmModalProps {
  profile: VoiceProfile
  onConfirm: () => void
  onCancel: () => void
  isDeleting: boolean
}

function DeleteConfirmModal({ profile, onConfirm, onCancel, isDeleting }: DeleteConfirmModalProps) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full">
        <div className="p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-3 bg-red-100 rounded-full">
              <AlertCircle className="w-6 h-6 text-red-600" />
            </div>
            <h2 className="text-xl font-bold text-gray-900">Delete Voice Profile</h2>
          </div>

          <p className="text-gray-600 mb-6">
            Are you sure you want to delete the voice profile <strong>"{profile.name}"</strong>?
            This action cannot be undone.
          </p>

          <div className="flex justify-end space-x-3">
            <button
              onClick={onCancel}
              className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              disabled={isDeleting}
            >
              Cancel
            </button>
            <button
              onClick={onConfirm}
              className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              disabled={isDeleting}
            >
              {isDeleting ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  <span>Deleting...</span>
                </>
              ) : (
                <>
                  <Trash2 className="w-4 h-4" />
                  <span>Delete Profile</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
