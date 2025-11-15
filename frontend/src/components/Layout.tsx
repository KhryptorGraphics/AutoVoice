import { Link, useLocation } from 'react-router-dom'
import { Music, User, Activity, Home } from 'lucide-react'
import clsx from 'clsx'

interface LayoutProps {
  children: React.ReactNode
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation()

  const navigation = [
    { name: 'Home', href: '/', icon: Home },
    { name: 'Singing Conversion', href: '/singing-conversion', icon: Music },
    { name: 'Voice Profiles', href: '/voice-profiles', icon: User },
    { name: 'System Status', href: '/system-status', icon: Activity },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-primary-600 rounded-lg">
                <Music className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">AutoVoice</h1>
                <p className="text-xs text-gray-500">Singing Voice Conversion</p>
              </div>
            </div>

            <nav className="flex space-x-1">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href
                const Icon = item.icon

                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={clsx(
                      'flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors',
                      isActive
                        ? 'bg-primary-100 text-primary-700'
                        : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{item.name}</span>
                  </Link>
                )
              })}
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="min-h-[calc(100vh-8rem)]">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <p>
              © 2025 AutoVoice. High-performance GPU-accelerated voice synthesis.
            </p>
            <div className="flex items-center space-x-4">
              <a
                href="https://github.com/KhryptorGraphics/AutoVoice"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-primary-600 transition-colors"
              >
                GitHub
              </a>
              <a
                href="/api/v1/health"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-primary-600 transition-colors"
              >
                API Status
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

