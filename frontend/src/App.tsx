import { Routes, Route, NavLink } from 'react-router-dom'
import { Activity, HelpCircle, History, Mic, Music, User, Users, Youtube } from 'lucide-react'
import clsx from 'clsx'

import { ToastProvider } from './contexts/ToastContext'
import { ConversionHistoryPage } from './pages/ConversionHistoryPage'
import { ConversionWorkflowPage } from './pages/ConversionWorkflowPage'
import HelpPage from './pages/HelpPage'
import { KaraokePage } from './pages/KaraokePage'
import { DiarizationResultsPage } from './pages/DiarizationResultsPage'
import { SystemStatusPage } from './pages/SystemStatusPage'
import { VoiceProfilePage } from './pages/VoiceProfilePage'
import { YouTubeDownloadPage } from './pages/YouTubeDownloadPage'

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
    <ToastProvider position="top-right">
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
            <Route path="/" element={<ConversionWorkflowPage />} />
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
    </ToastProvider>
  )
}
