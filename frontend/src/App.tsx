import { useState } from 'react'
import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import {
  Activity,
  Headphones,
  HelpCircle,
  History,
  ListChecks,
  Menu,
  Mic,
  Music,
  User,
  Users,
  X,
  Youtube,
} from 'lucide-react'
import clsx from 'clsx'

import { ToastProvider } from './contexts/ToastContext'
import { ErrorBoundary } from './components/ErrorBoundary'
import { ConversionHistoryPage } from './pages/ConversionHistoryPage'
import { ConversionWorkflowPage } from './pages/ConversionWorkflowPage'
import HelpPage from './pages/HelpPage'
import { KaraokePage } from './pages/KaraokePage'
import { DiarizationResultsPage } from './pages/DiarizationResultsPage'
import { SingAlongPage } from './pages/SingAlongPage'
import { SampleInboxPage } from './pages/SampleInboxPage'
import { SystemStatusPage } from './pages/SystemStatusPage'
import { VoiceProfilePage } from './pages/VoiceProfilePage'
import { YouTubeDownloadPage } from './pages/YouTubeDownloadPage'

const navGroups = [
  {
    heading: 'Create',
    items: [
      { to: '/', label: 'Convert', icon: Music },
      { to: '/karaoke', label: 'Karaoke', icon: Mic },
      { to: '/singalong', label: 'Sing Along', icon: Headphones },
    ],
  },
  {
    heading: 'Sources',
    items: [
      { to: '/youtube', label: 'YouTube', icon: Youtube },
      { to: '/samples', label: 'Samples', icon: ListChecks },
      { to: '/diarization', label: 'Diarization', icon: Users },
    ],
  },
  {
    heading: 'Library',
    items: [
      { to: '/profiles', label: 'Profiles', icon: User },
      { to: '/history', label: 'History', icon: History },
    ],
  },
  {
    heading: 'System',
    items: [
      { to: '/system', label: 'Status', icon: Activity },
      { to: '/help', label: 'Help', icon: HelpCircle },
    ],
  },
]

export default function App() {
  const location = useLocation()
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

  const navContent = (
    <nav className="flex h-full flex-col gap-6 overflow-y-auto px-3 py-4">
      <div className="flex items-center justify-between px-2">
        <span className="text-xl font-bold text-blue-400">AutoVoice</span>
        <button
          type="button"
          onClick={() => setMobileNavOpen(false)}
          className="rounded p-1 text-gray-400 hover:text-white md:hidden"
          aria-label="Close navigation"
        >
          <X size={20} />
        </button>
      </div>
      {navGroups.map((group) => (
        <div key={group.heading} className="flex flex-col gap-1">
          <div className="px-3 pb-1 text-xs font-semibold uppercase tracking-wider text-gray-500">
            {group.heading}
          </div>
          {group.items.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              onClick={() => setMobileNavOpen(false)}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 rounded px-3 py-2 text-sm transition-colors',
                  isActive
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-400 hover:bg-gray-700/50 hover:text-white'
                )
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </div>
      ))}
    </nav>
  )

  return (
    <ToastProvider position="top-right">
      <div className="flex min-h-screen bg-gray-900 text-white">
        {/* Desktop sidebar */}
        <aside className="sticky top-0 hidden h-screen w-56 shrink-0 flex-col border-r border-gray-700 bg-gray-800 md:flex">
          {navContent}
        </aside>

        {/* Mobile drawer */}
        {mobileNavOpen && (
          <div className="fixed inset-0 z-40 md:hidden">
            <div
              className="absolute inset-0 bg-black/50"
              onClick={() => setMobileNavOpen(false)}
              aria-hidden="true"
            />
            <aside className="absolute left-0 top-0 h-full w-64 border-r border-gray-700 bg-gray-800">
              {navContent}
            </aside>
          </div>
        )}

        <div className="flex min-w-0 flex-1 flex-col">
          {/* Mobile top bar */}
          <header className="flex h-14 items-center gap-3 border-b border-gray-700 bg-gray-800 px-4 md:hidden">
            <button
              type="button"
              onClick={() => setMobileNavOpen(true)}
              className="rounded p-1 text-gray-300 hover:text-white"
              aria-label="Open navigation"
            >
              <Menu size={22} />
            </button>
            <span className="text-lg font-bold text-blue-400">AutoVoice</span>
          </header>

          <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8">
            <ErrorBoundary resetKeys={[location.pathname]}>
              <Routes>
                <Route path="/" element={<ConversionWorkflowPage />} />
                <Route path="/karaoke" element={<KaraokePage />} />
                <Route path="/singalong" element={<SingAlongPage />} />
                <Route path="/profiles" element={<VoiceProfilePage />} />
                <Route path="/samples" element={<SampleInboxPage />} />
                <Route path="/youtube" element={<YouTubeDownloadPage />} />
                <Route path="/diarization" element={<DiarizationResultsPage />} />
                <Route path="/history" element={<ConversionHistoryPage />} />
                <Route path="/system" element={<SystemStatusPage />} />
                <Route path="/help" element={<HelpPage />} />
              </Routes>
            </ErrorBoundary>
          </main>
        </div>
      </div>
    </ToastProvider>
  )
}
