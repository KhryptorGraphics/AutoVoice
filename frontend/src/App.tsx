import { Routes, Route, NavLink } from 'react-router-dom'
import { Music, History, Activity, Mic, HelpCircle, User, Users, Youtube } from 'lucide-react'
import { ConversionHistoryPage } from './pages/ConversionHistoryPage'
import { SystemStatusPage } from './pages/SystemStatusPage'
import { KaraokePage } from './pages/KaraokePage'
import { VoiceProfilePage } from './pages/VoiceProfilePage'
import { DiarizationResultsPage } from './pages/DiarizationResultsPage'
import { YouTubeDownloadPage } from './pages/YouTubeDownloadPage'
import HelpPage from './pages/HelpPage'
import clsx from 'clsx'

function ConvertPage() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Voice Conversion</h1>
      <div className="bg-gray-800 rounded-lg p-6">
        <p className="text-gray-400">
          Upload a song and select a voice profile to convert.
        </p>
        <div className="mt-6 space-y-4">
          <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center">
            <Music className="mx-auto h-12 w-12 text-gray-500 mb-3" />
            <p className="text-gray-400">Drop audio file here or click to upload</p>
            <input
              type="file"
              accept="audio/*"
              className="hidden"
              id="audio-upload"
            />
            <label
              htmlFor="audio-upload"
              className="mt-3 inline-block px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded cursor-pointer"
            >
              Select File
            </label>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const navItems = [
    { to: '/', label: 'Convert', icon: Music },
    { to: '/karaoke', label: 'Karaoke', icon: Mic },
    { to: '/profiles', label: 'Profiles', icon: User },
    { to: '/youtube', label: 'YouTube', icon: Youtube },
    { to: '/diarization', label: 'Diarization', icon: Users },
    { to: '/history', label: 'History', icon: History },
    { to: '/system', label: 'System', icon: Activity },
    { to: '/help', label: 'Help', icon: HelpCircle },
  ]

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 flex items-center h-14">
          <span className="text-xl font-bold text-blue-400 mr-8">AutoVoice</span>
          <div className="flex gap-1">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-2 px-3 py-2 rounded text-sm',
                    isActive
                      ? 'bg-gray-700 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                  )
                }
              >
                <Icon size={16} />
                {label}
              </NavLink>
            ))}
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<ConvertPage />} />
          <Route path="/karaoke" element={<KaraokePage />} />
          <Route path="/profiles" element={<VoiceProfilePage />} />
          <Route path="/youtube" element={<YouTubeDownloadPage />} />
          <Route path="/diarization" element={<DiarizationResultsPage />} />
          <Route path="/history" element={<ConversionHistoryPage />} />
          <Route path="/system" element={<SystemStatusPage />} />
          <Route path="/help" element={<HelpPage />} />
        </Routes>
      </main>
    </div>
  )
}
