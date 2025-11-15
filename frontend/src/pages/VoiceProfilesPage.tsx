import { useState } from 'react'
import { User, Plus, Trash2, Edit2, Music, Upload, X, Check, AlertCircle, Loader, Zap } from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService, VoiceProfile } from '../services/api'
import { VoiceProfileTester } from '../components/VoiceProfileTester'

export function VoiceProfilesPage() {
  const [showCreateModal, setShowCreateModal] = useState(false)
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
        <button
          onClick={() => setShowCreateModal(true)}
          className="bg-purple-600 hover:bg-purple-700 text-white font-semibold px-6 py-3 rounded-lg transition-colors flex items-center space-x-2"
        >
          <Plus className="w-5 h-5" />
          <span>Create Profile</span>
        </button>
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

