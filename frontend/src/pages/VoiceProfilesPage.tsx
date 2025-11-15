import { User, Plus } from 'lucide-react'

export function VoiceProfilesPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center space-x-3">
          <User className="w-8 h-8 text-primary-600" />
          <span>Voice Profiles</span>
        </h1>
        <p className="text-gray-600 mt-2">
          Manage your voice profiles for singing voice conversion
        </p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-12 text-center">
        <div className="max-w-md mx-auto">
          <div className="p-4 bg-gray-100 rounded-full inline-block mb-4">
            <Plus className="w-12 h-12 text-gray-400" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            Voice Profile Management
          </h2>
          <p className="text-gray-600 mb-6">
            This page will allow you to create, edit, and delete voice profiles.
            Upload voice samples to create new profiles for singing voice conversion.
          </p>
          <button className="bg-primary-600 hover:bg-primary-700 text-white font-semibold px-6 py-3 rounded-lg transition-colors">
            Create Voice Profile
          </button>
        </div>
      </div>
    </div>
  )
}

