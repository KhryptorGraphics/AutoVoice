import { User, Check } from 'lucide-react'
import clsx from 'clsx'
import { VoiceProfile } from '../services/api'

interface VoiceProfileSelectorProps {
  profiles: VoiceProfile[]
  selectedProfile: VoiceProfile | null
  onSelect: (profile: VoiceProfile) => void
  isLoading?: boolean
}

export function VoiceProfileSelector({
  profiles,
  selectedProfile,
  onSelect,
  isLoading = false,
}: VoiceProfileSelectorProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (profiles.length === 0) {
    return (
      <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <User className="w-12 h-12 text-gray-400 mx-auto mb-3" />
        <p className="text-gray-600 font-medium">No voice profiles available</p>
        <p className="text-sm text-gray-500 mt-1">
          Create a voice profile first to use singing voice conversion
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {profiles.map((profile) => {
        const isSelected = selectedProfile?.id === profile.id

        return (
          <button
            key={profile.id}
            onClick={() => onSelect(profile)}
            className={clsx(
              'w-full p-4 rounded-lg border-2 transition-all text-left',
              'hover:border-primary-300 hover:bg-primary-50/50',
              isSelected
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-200 bg-white'
            )}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-3 flex-1">
                <div
                  className={clsx(
                    'p-2 rounded-lg',
                    isSelected ? 'bg-primary-100' : 'bg-gray-100'
                  )}
                >
                  <User
                    className={clsx(
                      'w-6 h-6',
                      isSelected ? 'text-primary-600' : 'text-gray-600'
                    )}
                  />
                </div>

                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-gray-900 truncate">
                    {profile.name}
                  </h3>
                  {profile.description && (
                    <p className="text-sm text-gray-600 mt-1 line-clamp-2">
                      {profile.description}
                    </p>
                  )}
                  <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                    {profile.sample_duration && (
                      <span>
                        {Math.round(profile.sample_duration)}s sample
                      </span>
                    )}
                    <span>
                      Created {new Date(profile.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>

              {isSelected && (
                <div className="ml-3 flex-shrink-0">
                  <div className="p-1 bg-primary-600 rounded-full">
                    <Check className="w-4 h-4 text-white" />
                  </div>
                </div>
              )}
            </div>
          </button>
        )
      })}
    </div>
  )
}

