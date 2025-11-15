import { useState } from 'react'
import { Play, Pause, Upload, Loader } from 'lucide-react'
import { apiService, VoiceProfile } from '../services/api'

interface VoiceProfileTesterProps {
  profile: VoiceProfile
  onClose: () => void
}

export function VoiceProfileTester({ profile, onClose }: VoiceProfileTesterProps) {
  const [testAudio, setTestAudio] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      if (file.size > 50 * 1024 * 1024) {
        setError('File size must be less than 50MB')
        return
      }
      setTestAudio(file)
      setError(null)
    }
  }

  const handleTest = async () => {
    if (!testAudio) {
      setError('Please select an audio file')
      return
    }

    setIsProcessing(true)
    setError(null)

    try {
      const response = await apiService.testVoiceProfile(profile.id, testAudio)
      setResult(response.output_url || response.result_url)
    } catch (err: any) {
      setError(err.message || 'Failed to test voice profile')
    } finally {
      setIsProcessing(false)
    }
  }

  const handlePlayResult = () => {
    if (result) {
      const audio = new Audio(result)
      if (isPlaying) {
        audio.pause()
        setIsPlaying(false)
      } else {
        audio.play()
        setIsPlaying(true)
        audio.onended = () => setIsPlaying(false)
      }
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Test Voice Profile</h2>
        <p className="text-gray-600 mb-6">Profile: {profile.name}</p>

        {/* File Upload */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Select Audio File</label>
          <label className="flex items-center justify-center w-full px-4 py-3 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-primary-500 transition-colors">
            <div className="flex items-center space-x-2">
              <Upload className="w-5 h-5 text-gray-400" />
              <span className="text-sm text-gray-600">
                {testAudio ? testAudio.name : 'Click to select audio file'}
              </span>
            </div>
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </label>
          <p className="text-xs text-gray-500 mt-2">Max 50MB • MP3, WAV, FLAC supported</p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-sm font-medium text-green-900 mb-3">Test Result</p>
            <button
              onClick={handlePlayResult}
              className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
            >
              {isPlaying ? (
                <>
                  <Pause className="w-5 h-5" />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  <span>Play Result</span>
                </>
              )}
            </button>
            <a
              href={result}
              download={`test_${profile.name}.wav`}
              className="mt-2 block text-center text-sm text-green-700 hover:text-green-800 font-medium"
            >
              Download Result
            </a>
          </div>
        )}

        {/* Buttons */}
        <div className="flex space-x-3">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors font-medium"
          >
            Close
          </button>
          <button
            onClick={handleTest}
            disabled={!testAudio || isProcessing}
            className="flex-1 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-300 text-white rounded-lg transition-colors font-medium flex items-center justify-center space-x-2"
          >
            {isProcessing ? (
              <>
                <Loader className="w-5 h-5 animate-spin" />
                <span>Testing...</span>
              </>
            ) : (
              <span>Test Profile</span>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

