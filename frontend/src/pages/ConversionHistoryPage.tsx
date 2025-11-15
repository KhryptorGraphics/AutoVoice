import { useState, useEffect } from 'react'
import { Clock, Download, Play, Trash2, Music } from 'lucide-react'

interface ConversionRecord {
  id: string
  originalFileName: string
  targetVoice: string
  timestamp: Date
  duration: number
  quality: string
  resultUrl?: string
}

export function ConversionHistoryPage() {
  const [history, setHistory] = useState<ConversionRecord[]>([])
  const [filter, setFilter] = useState<'all' | 'today' | 'week' | 'month'>('all')

  useEffect(() => {
    // Load history from localStorage
    const savedHistory = localStorage.getItem('conversionHistory')
    if (savedHistory) {
      const parsed = JSON.parse(savedHistory)
      setHistory(parsed.map((item: any) => ({
        ...item,
        timestamp: new Date(item.timestamp),
      })))
    }
  }, [])

  const handleDelete = (id: string) => {
    const newHistory = history.filter((item) => item.id !== id)
    setHistory(newHistory)
    localStorage.setItem('conversionHistory', JSON.stringify(newHistory))
  }

  const handleClearAll = () => {
    if (confirm('Are you sure you want to clear all history?')) {
      setHistory([])
      localStorage.removeItem('conversionHistory')
    }
  }

  const filteredHistory = history.filter((item) => {
    const now = new Date()
    const diff = now.getTime() - item.timestamp.getTime()
    const dayInMs = 24 * 60 * 60 * 1000

    switch (filter) {
      case 'today':
        return diff < dayInMs
      case 'week':
        return diff < 7 * dayInMs
      case 'month':
        return diff < 30 * dayInMs
      default:
        return true
    }
  })

  const formatDate = (date: Date) => {
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(diff / 3600000)
    const days = Math.floor(diff / 86400000)

    if (minutes < 1) return 'Just now'
    if (minutes < 60) return `${minutes}m ago`
    if (hours < 24) return `${hours}h ago`
    if (days < 7) return `${days}d ago`
    return date.toLocaleDateString()
  }

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center space-x-3">
          <Clock className="w-8 h-8 text-primary-600" />
          <span>Conversion History</span>
        </h1>
        <p className="text-gray-600 mt-2">View and manage your past conversions</p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6">
        {/* Filters */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex space-x-2">
            {(['all', 'today', 'week', 'month'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  filter === f
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
          {history.length > 0 && (
            <button
              onClick={handleClearAll}
              className="text-red-600 hover:text-red-700 font-medium"
            >
              Clear All
            </button>
          )}
        </div>

        {/* History List */}
        {filteredHistory.length === 0 ? (
          <div className="text-center py-12">
            <Music className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-500 text-lg">No conversions yet</p>
            <p className="text-gray-400 text-sm mt-2">
              Your conversion history will appear here
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredHistory.map((item) => (
              <div
                key={item.id}
                className="border border-gray-200 rounded-lg p-4 hover:border-primary-300 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <Music className="w-5 h-5 text-primary-600" />
                      <div>
                        <h3 className="font-semibold text-gray-900">
                          {item.originalFileName}
                        </h3>
                        <p className="text-sm text-gray-600">
                          Voice: {item.targetVoice} • Duration: {formatDuration(item.duration)} •
                          Quality: {item.quality}
                        </p>
                        <p className="text-xs text-gray-400 mt-1">
                          {formatDate(item.timestamp)}
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    {item.resultUrl && (
                      <>
                        <button
                          onClick={() => {
                            const audio = new Audio(item.resultUrl)
                            audio.play()
                          }}
                          className="p-2 text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
                          title="Play"
                        >
                          <Play className="w-5 h-5" />
                        </button>
                        <a
                          href={item.resultUrl}
                          download
                          className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                          title="Download"
                        >
                          <Download className="w-5 h-5" />
                        </a>
                      </>
                    )}
                    <button
                      onClick={() => handleDelete(item.id)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

