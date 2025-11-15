import { useState, useEffect } from 'react'
import { Clock, Download, Play, Trash2, Music, Star, Search, Filter, Tag, FileText } from 'lucide-react'
import { ConversionRecord } from '../services/api'

interface FilterOptions {
  timeRange: 'all' | 'today' | 'week' | 'month'
  quality: 'all' | 'draft' | 'fast' | 'balanced' | 'high' | 'studio'
  favorites: boolean
  voice: string
}

export function ConversionHistoryPage() {
  const [history, setHistory] = useState<ConversionRecord[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const [filters, setFilters] = useState<FilterOptions>({
    timeRange: 'all',
    quality: 'all',
    favorites: false,
    voice: 'all',
  })
  const [editingNotes, setEditingNotes] = useState<string | null>(null)
  const [notesText, setNotesText] = useState('')

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

  const handleToggleFavorite = (id: string) => {
    const updated = history.map((item) =>
      item.id === id ? { ...item, isFavorite: !item.isFavorite } : item
    )
    setHistory(updated)
    localStorage.setItem('conversionHistory', JSON.stringify(updated))
  }

  const handleSaveNotes = (id: string) => {
    const updated = history.map((item) =>
      item.id === id ? { ...item, notes: notesText } : item
    )
    setHistory(updated)
    localStorage.setItem('conversionHistory', JSON.stringify(updated))
    setEditingNotes(null)
  }

  const handleClearAll = () => {
    if (confirm('Are you sure you want to clear all history?')) {
      setHistory([])
      localStorage.removeItem('conversionHistory')
    }
  }

  const getUniqueVoices = () => {
    const voices = new Set(history.map((item) => item.targetVoice))
    return Array.from(voices).sort()
  }

  const filteredHistory = history.filter((item) => {
    const now = new Date()
    const diff = now.getTime() - item.timestamp.getTime()
    const dayInMs = 24 * 60 * 60 * 1000

    // Time range filter
    let timeMatch = true
    switch (filters.timeRange) {
      case 'today':
        timeMatch = diff < dayInMs
        break
      case 'week':
        timeMatch = diff < 7 * dayInMs
        break
      case 'month':
        timeMatch = diff < 30 * dayInMs
        break
    }

    // Quality filter
    const qualityMatch = filters.quality === 'all' || item.quality === filters.quality

    // Favorites filter
    const favoritesMatch = !filters.favorites || item.isFavorite

    // Voice filter
    const voiceMatch = filters.voice === 'all' || item.targetVoice === filters.voice

    // Search query filter
    const searchMatch =
      searchQuery === '' ||
      item.originalFileName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.targetVoice.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.notes?.toLowerCase().includes(searchQuery.toLowerCase())

    return timeMatch && qualityMatch && favoritesMatch && voiceMatch && searchMatch
  }).sort((a, b) => {
    // Sort favorites first, then by date
    if (a.isFavorite && !b.isFavorite) return -1
    if (!a.isFavorite && b.isFavorite) return 1
    return b.timestamp.getTime() - a.timestamp.getTime()
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
        {/* Search Bar */}
        <div className="mb-6">
          <div className="relative">
            <Search className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search by filename, voice, or notes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
        </div>

        {/* Filters */}
        <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
          <div className="flex space-x-2 flex-wrap">
            {(['all', 'today', 'week', 'month'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilters({ ...filters, timeRange: f })}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  filters.timeRange === f
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              <Filter className="w-5 h-5" />
              <span>Advanced Filters</span>
            </button>

            <button
              onClick={() => setFilters({ ...filters, favorites: !filters.favorites })}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                filters.favorites
                  ? 'bg-yellow-100 text-yellow-700'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Star className="w-5 h-5" />
              <span>Favorites</span>
            </button>

            {history.length > 0 && (
              <button
                onClick={handleClearAll}
                className="text-red-600 hover:text-red-700 font-medium"
              >
                Clear All
              </button>
            )}
          </div>
        </div>

        {/* Advanced Filters */}
        {showFilters && (
          <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Quality</label>
                <select
                  value={filters.quality}
                  onChange={(e) => setFilters({ ...filters, quality: e.target.value as any })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="all">All Qualities</option>
                  <option value="draft">Draft</option>
                  <option value="fast">Fast</option>
                  <option value="balanced">Balanced</option>
                  <option value="high">High</option>
                  <option value="studio">Studio</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Voice</label>
                <select
                  value={filters.voice}
                  onChange={(e) => setFilters({ ...filters, voice: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="all">All Voices</option>
                  {getUniqueVoices().map((voice) => (
                    <option key={voice} value={voice}>
                      {voice}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}

        {/* History List */}
        {filteredHistory.length === 0 ? (
          <div className="text-center py-12">
            <Music className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-500 text-lg">
              {searchQuery || Object.values(filters).some((v) => v !== 'all' && v !== false)
                ? 'No conversions match your filters'
                : 'No conversions yet'}
            </p>
            <p className="text-gray-400 text-sm mt-2">
              {searchQuery || Object.values(filters).some((v) => v !== 'all' && v !== false)
                ? 'Try adjusting your search or filters'
                : 'Your conversion history will appear here'}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredHistory.map((item) => (
              <div
                key={item.id}
                className={`border rounded-lg p-4 transition-colors ${
                  item.isFavorite ? 'border-yellow-300 bg-yellow-50' : 'border-gray-200 hover:border-primary-300'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-3 flex-1">
                    <Music className="w-5 h-5 text-primary-600 flex-shrink-0" />
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <h3 className="font-semibold text-gray-900">{item.originalFileName}</h3>
                        {item.isFavorite && <Star className="w-4 h-4 text-yellow-500 fill-yellow-500" />}
                      </div>
                      <p className="text-sm text-gray-600">
                        Voice: {item.targetVoice} • Duration: {formatDuration(item.duration)} • Quality:{' '}
                        <span className="font-medium">{item.quality}</span>
                      </p>
                      <p className="text-xs text-gray-400 mt-1">{formatDate(item.timestamp)}</p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-1 flex-shrink-0">
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
                      onClick={() => handleToggleFavorite(item.id)}
                      className={`p-2 rounded-lg transition-colors ${
                        item.isFavorite
                          ? 'text-yellow-600 bg-yellow-50'
                          : 'text-gray-400 hover:text-yellow-600 hover:bg-yellow-50'
                      }`}
                      title={item.isFavorite ? 'Remove from favorites' : 'Add to favorites'}
                    >
                      <Star className={`w-5 h-5 ${item.isFavorite ? 'fill-current' : ''}`} />
                    </button>
                    <button
                      onClick={() => {
                        setEditingNotes(item.id)
                        setNotesText(item.notes || '')
                      }}
                      className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                      title="Add notes"
                    >
                      <FileText className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => handleDelete(item.id)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>

                {/* Notes Section */}
                {editingNotes === item.id ? (
                  <div className="mt-3 p-3 bg-white border border-blue-200 rounded-lg">
                    <textarea
                      value={notesText}
                      onChange={(e) => setNotesText(e.target.value)}
                      placeholder="Add notes about this conversion..."
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                      rows={2}
                    />
                    <div className="flex justify-end space-x-2 mt-2">
                      <button
                        onClick={() => setEditingNotes(null)}
                        className="px-3 py-1 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={() => handleSaveNotes(item.id)}
                        className="px-3 py-1 text-sm bg-blue-600 text-white hover:bg-blue-700 rounded-lg transition-colors"
                      >
                        Save
                      </button>
                    </div>
                  </div>
                ) : item.notes ? (
                  <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded text-sm text-gray-700">
                    <p className="font-medium text-blue-900 mb-1">Notes:</p>
                    <p>{item.notes}</p>
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

