import { useState, useEffect, useCallback } from 'react'
import { User, Plus, Trash2, RefreshCw, ChevronRight, XCircle, Loader2, Upload, Mic, Play, CheckCircle2, Clock, AlertCircle, Users } from 'lucide-react'
import { apiService, VoiceProfile, TrainingJob, TrainingConfig, DEFAULT_TRAINING_CONFIG, TrainingSample, TrainingStatusType } from '../services/api'
import { TrainingConfigPanel } from '../components/TrainingConfigPanel'
import { TrainingJobQueue } from '../components/TrainingJobQueue'
import { LossCurveChart } from '../components/LossCurveChart'
import { AddSongButton } from '../components/AddSongButton'
import { LiveTrainingMonitor } from '../components/LiveTrainingMonitor'
import { TrainingSampleUpload } from '../components/TrainingSampleUpload'
import clsx from 'clsx'

// Training status badge component
function TrainingStatusBadge({ status }: { status?: TrainingStatusType }) {
  if (!status) status = 'pending'

  const configs: Record<TrainingStatusType, { label: string; icon: React.ReactNode; className: string }> = {
    pending: {
      label: 'Not Trained',
      icon: <Clock size={12} />,
      className: 'bg-gray-600 text-gray-300',
    },
    training: {
      label: 'Training...',
      icon: <Loader2 size={12} className="animate-spin" />,
      className: 'bg-yellow-600 text-yellow-100',
    },
    ready: {
      label: 'Ready',
      icon: <CheckCircle2 size={12} />,
      className: 'bg-green-600 text-green-100',
    },
    failed: {
      label: 'Failed',
      icon: <AlertCircle size={12} />,
      className: 'bg-red-600 text-red-100',
    },
  }

  const config = configs[status]

  return (
    <span className={clsx('inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium', config.className)}>
      {config.icon}
      {config.label}
    </span>
  )
}

interface ProfileDetailProps {
  profile: VoiceProfile
  onBack: () => void
  onDelete: () => void
}

function ProfileDetail({ profile, onBack, onDelete }: ProfileDetailProps) {
  const [samples, setSamples] = useState<TrainingSample[]>([])
  const [loading, setLoading] = useState(true)
  const [isDeleting, setIsDeleting] = useState(false)
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG)
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null)
  const [startingTraining, setStartingTraining] = useState(false)
  const [activeTab, setActiveTab] = useState<'samples' | 'config' | 'jobs' | 'segments'>('samples')
  const [showAdvancedUpload, setShowAdvancedUpload] = useState(false)
  const [assignedSegments, setAssignedSegments] = useState<Array<{ type: string; segment_key: string; audio_path: string }>>([])
  const [loadingSegments, setLoadingSegments] = useState(false)

  useEffect(() => {
    const fetchSamples = async () => {
      try {
        const data = await apiService.listSamples(profile.profile_id)
        setSamples(data)
      } catch (error) {
        console.error('Failed to fetch samples:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchSamples()
  }, [profile.profile_id])

  // Fetch assigned diarization segments when segments tab is active
  useEffect(() => {
    if (activeTab !== 'segments') return

    const fetchSegments = async () => {
      setLoadingSegments(true)
      try {
        const result = await apiService.getProfileSegments(profile.profile_id)
        setAssignedSegments(result.diarization_assignments || [])
      } catch (error) {
        console.error('Failed to fetch segments:', error)
        setAssignedSegments([])
      } finally {
        setLoadingSegments(false)
      }
    }
    fetchSegments()
  }, [profile.profile_id, activeTab])

  const handleDelete = async () => {
    if (!confirm(`Delete profile "${profile.name || profile.profile_id}"? This cannot be undone.`)) return
    setIsDeleting(true)
    try {
      await apiService.deleteProfile(profile.profile_id)
      onDelete()
    } catch (error) {
      alert(`Failed to delete profile: ${error}`)
      setIsDeleting(false)
    }
  }

  const handleStartTraining = async () => {
    if (samples.length === 0) {
      alert('No samples available. Upload samples first.')
      return
    }
    setStartingTraining(true)
    try {
      const sampleIds = samples.map(s => s.id)
      await apiService.createTrainingJob(profile.profile_id, sampleIds, trainingConfig)
      setActiveTab('jobs')
    } catch (error) {
      alert(`Failed to start training: ${error}`)
    } finally {
      setStartingTraining(false)
    }
  }

  const handleUploadSample = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      const sample = await apiService.uploadSample(profile.profile_id, file)
      setSamples(prev => [...prev, sample])
    } catch (error) {
      alert(`Failed to upload sample: ${error}`)
    }
    e.target.value = ''
  }

  const handleDeleteSample = async (sampleId: string) => {
    if (!confirm('Delete this sample?')) return
    try {
      await apiService.deleteSample(profile.profile_id, sampleId)
      setSamples(prev => prev.filter(s => s.id !== sampleId))
    } catch (error) {
      alert(`Failed to delete sample: ${error}`)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <button
          onClick={onBack}
          className="text-gray-400 hover:text-white"
        >
          ← Back
        </button>
        <h2 className="text-2xl font-bold">{profile.name || profile.profile_id}</h2>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Samples</div>
          <div className="text-2xl font-bold">{samples.length}</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Training Status</div>
          <div className="mt-1">
            <TrainingStatusBadge status={profile.training_status} />
          </div>
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Model Version</div>
          <div className="text-2xl font-bold">{profile.model_version || 'Base'}</div>
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Quality Score</div>
          <div className="text-2xl font-bold">
            {profile.quality_score ? `${(profile.quality_score * 100).toFixed(0)}%` : 'N/A'}
          </div>
        </div>
      </div>

      {/* Tab navigation */}
      <div className="flex gap-1 p-1 bg-gray-800 rounded-lg">
        {(['samples', 'segments', 'config', 'jobs'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={clsx(
              'flex-1 px-4 py-2 text-sm font-medium rounded transition-colors capitalize',
              activeTab === tab
                ? 'bg-gray-700 text-white'
                : 'text-gray-400 hover:text-white'
            )}
          >
            {tab === 'samples' ? `Samples (${samples.length})` :
             tab === 'segments' ? 'Diarized Segments' :
             tab === 'jobs' ? 'Training Jobs' : 'Config'}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'samples' && (
        <div className="bg-gray-800 rounded-lg p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Training Samples</h3>
            <div className="flex items-center gap-2">
              <AddSongButton
                profileId={profile.profile_id}
                onSongAdded={() => {
                  // Refresh samples after song is added and split
                  apiService.listSamples(profile.profile_id).then(setSamples)
                }}
              />
              <button
                onClick={() => setShowAdvancedUpload(!showAdvancedUpload)}
                className={clsx(
                  'flex items-center gap-2 px-4 py-2 rounded',
                  showAdvancedUpload
                    ? 'bg-purple-600 hover:bg-purple-700'
                    : 'bg-gray-600 hover:bg-gray-500'
                )}
                title="Upload with speaker detection"
              >
                <Users size={16} />
                Smart Upload
              </button>
              <label className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded cursor-pointer">
                <Upload size={16} />
                Quick Upload
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleUploadSample}
                  className="hidden"
                />
              </label>
            </div>
          </div>

          {/* Advanced upload with diarization */}
          {showAdvancedUpload && (
            <TrainingSampleUpload
              profileId={profile.profile_id}
              profileName={profile.name}
              onSampleAdded={(sample) => {
                setSamples(prev => [...prev, sample])
                setShowAdvancedUpload(false)
              }}
            />
          )}
          {loading ? (
            <div className="flex items-center gap-2 text-gray-400">
              <Loader2 className="animate-spin" size={16} />
              Loading...
            </div>
          ) : samples.length === 0 ? (
            <p className="text-gray-500">No samples yet. Upload audio samples to train this voice profile.</p>
          ) : (
            <div className="space-y-2">
              {samples.map(sample => (
                <div key={sample.id} className="flex items-center justify-between p-3 bg-gray-750 rounded">
                  <div className="flex items-center gap-3">
                    <Mic size={16} className="text-gray-400" />
                    <div>
                      <div className="text-sm">{sample.audio_path.split('/').pop()}</div>
                      <div className="text-xs text-gray-500">
                        {sample.duration_seconds.toFixed(1)}s · {sample.sample_rate}Hz
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteSample(sample.id)}
                    className="p-1 text-gray-400 hover:text-red-400"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {activeTab === 'config' && (
        <div className="space-y-4">
          <TrainingConfigPanel
            config={trainingConfig}
            onChange={setTrainingConfig}
          />
          <button
            onClick={handleStartTraining}
            disabled={startingTraining || samples.length === 0}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed rounded"
          >
            {startingTraining ? (
              <Loader2 className="animate-spin" size={16} />
            ) : (
              <Play size={16} />
            )}
            Start Training ({samples.length} samples)
          </button>
        </div>
      )}

      {activeTab === 'jobs' && (
        <div className="space-y-4">
          {/* Live Training Monitor - shown when a job is running */}
          {selectedJob?.status === 'running' && (
            <LiveTrainingMonitor
              jobId={selectedJob.job_id}
              profileId={profile.profile_id}
              onComplete={() => {
                // Refresh selected job to update status
                apiService.getTrainingJob(selectedJob.job_id).then(setSelectedJob)
              }}
            />
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <TrainingJobQueue
              profileId={profile.profile_id}
              onJobSelect={setSelectedJob}
            />
            {selectedJob && selectedJob.status !== 'running' && (
              <LossCurveChart job={selectedJob} />
            )}
          </div>
        </div>
      )}

      {activeTab === 'segments' && (
        <div className="bg-gray-800 rounded-lg p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Diarized Audio Segments</h3>
            <p className="text-sm text-gray-400">
              Segments assigned to this profile from speaker diarization
            </p>
          </div>

          {loadingSegments ? (
            <div className="flex items-center gap-2 text-gray-400">
              <Loader2 className="animate-spin" size={16} />
              Loading segments...
            </div>
          ) : assignedSegments.length === 0 ? (
            <div className="text-center py-8">
              <Users className="mx-auto text-gray-500 mb-3" size={48} />
              <p className="text-gray-400 mb-2">No diarized segments assigned to this profile.</p>
              <p className="text-sm text-gray-500">
                Use the <a href="/diarization" className="text-blue-400 hover:underline">Diarization</a> page
                to analyze multi-speaker audio and assign segments to profiles.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {assignedSegments.map((segment, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 bg-gray-750 rounded">
                  <div className="flex items-center gap-3">
                    <Users size={16} className="text-purple-400" />
                    <div>
                      <div className="text-sm">{segment.audio_path?.split('/').pop() || segment.segment_key}</div>
                      <div className="text-xs text-gray-500">
                        Type: {segment.type} · Key: {segment.segment_key}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Delete button */}
      <div className="flex justify-end pt-4 border-t border-gray-700">
        <button
          onClick={handleDelete}
          disabled={isDeleting}
          className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:opacity-50 rounded"
        >
          {isDeleting ? <Loader2 className="animate-spin" size={16} /> : <Trash2 size={16} />}
          Delete Profile
        </button>
      </div>
    </div>
  )
}

function CreateProfileForm({ onCreated }: { onCreated: (profile: VoiceProfile) => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return

    setCreating(true)
    setError(null)

    try {
      // Pass name as second parameter - backend will store it directly
      const profile = await apiService.createVoiceProfile(file, name || undefined)
      onCreated(profile)
      setFile(null)
      setName('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create profile')
    } finally {
      setCreating(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-gray-800 rounded-lg p-6 space-y-4">
      <h3 className="text-lg font-semibold">Create New Profile</h3>

      <div>
        <label className="block text-sm text-gray-400 mb-1">Profile Name (optional)</label>
        <input
          type="text"
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="My Voice"
          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500"
        />
      </div>

      <div>
        <label className="block text-sm text-gray-400 mb-1">Voice Sample</label>
        <div className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center">
          {file ? (
            <div className="flex items-center justify-center gap-2">
              <Mic size={20} className="text-green-400" />
              <span>{file.name}</span>
              <button
                type="button"
                onClick={() => setFile(null)}
                className="text-red-400 hover:text-red-300"
              >
                <XCircle size={16} />
              </button>
            </div>
          ) : (
            <>
              <Upload className="mx-auto h-10 w-10 text-gray-500 mb-2" />
              <p className="text-gray-400 text-sm mb-2">Upload audio sample (10-30 seconds)</p>
              <input
                type="file"
                accept="audio/*"
                onChange={e => setFile(e.target.files?.[0] || null)}
                className="hidden"
                id="profile-audio"
              />
              <label
                htmlFor="profile-audio"
                className="inline-block px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded cursor-pointer"
              >
                Select File
              </label>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="text-red-400 text-sm">{error}</div>
      )}

      <button
        type="submit"
        disabled={!file || creating}
        className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded"
      >
        {creating ? <Loader2 className="animate-spin" size={16} /> : <Plus size={16} />}
        Create Profile
      </button>
    </form>
  )
}

export function VoiceProfilePage() {
  const [profiles, setProfiles] = useState<VoiceProfile[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedProfile, setSelectedProfile] = useState<VoiceProfile | null>(null)
  const [showCreateForm, setShowCreateForm] = useState(false)

  const fetchProfiles = useCallback(async () => {
    try {
      const data = await apiService.listProfiles()
      setProfiles(data)
    } catch (error) {
      console.error('Failed to fetch profiles:', error)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchProfiles()
  }, [fetchProfiles])

  const handleProfileCreated = (profile: VoiceProfile) => {
    setProfiles(prev => [profile, ...prev])
    setShowCreateForm(false)
  }

  const handleProfileDeleted = () => {
    setSelectedProfile(null)
    fetchProfiles()
  }

  if (selectedProfile) {
    return (
      <div className="max-w-4xl mx-auto">
        <ProfileDetail
          profile={selectedProfile}
          onBack={() => setSelectedProfile(null)}
          onDelete={handleProfileDeleted}
        />
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Voice Profiles</h1>
        <div className="flex gap-2">
          <button
            onClick={fetchProfiles}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="Refresh"
          >
            <RefreshCw size={20} />
          </button>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
          >
            <Plus size={16} />
            New Profile
          </button>
        </div>
      </div>

      {showCreateForm && (
        <div className="mb-6">
          <CreateProfileForm onCreated={handleProfileCreated} />
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="animate-spin text-gray-400" size={32} />
        </div>
      ) : profiles.length === 0 ? (
        <div className="bg-gray-800 rounded-lg p-8 text-center">
          <User className="mx-auto h-12 w-12 text-gray-500 mb-3" />
          <h3 className="text-lg font-medium mb-2">No Voice Profiles</h3>
          <p className="text-gray-400 mb-4">Create your first voice profile to get started.</p>
          <button
            onClick={() => setShowCreateForm(true)}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
          >
            <Plus size={16} />
            Create Profile
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          {profiles.map(profile => (
            <button
              key={profile.profile_id}
              onClick={() => setSelectedProfile(profile)}
              className="w-full flex items-center justify-between p-4 bg-gray-800 hover:bg-gray-750 rounded-lg text-left transition-colors"
            >
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center">
                  <User size={20} className="text-gray-400" />
                </div>
                <div>
                  <div className="font-medium">{profile.name || profile.profile_id}</div>
                  <div className="text-sm text-gray-400">
                    {profile.sample_count} samples
                    {profile.last_trained && ` · Trained ${new Date(profile.last_trained).toLocaleDateString()}`}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <TrainingStatusBadge status={profile.training_status} />
                {profile.quality_score && (
                  <div className="text-sm">
                    <span className="text-gray-400">Quality:</span>{' '}
                    <span className={clsx(
                      profile.quality_score > 0.8 ? 'text-green-400' :
                      profile.quality_score > 0.6 ? 'text-yellow-400' : 'text-red-400'
                    )}>
                      {(profile.quality_score * 100).toFixed(0)}%
                    </span>
                  </div>
                )}
                <ChevronRight size={20} className="text-gray-500" />
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
