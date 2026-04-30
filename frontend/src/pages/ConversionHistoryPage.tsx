import { useState, useEffect, useCallback } from 'react'
import { Clock, Download, Play, Trash2, Music, Star, Search, Filter, FileText, GitCompare } from 'lucide-react'
import { apiService, type ConversionRecord } from '../services/api'
import { PipelineBadge, type PipelineType } from '../components/PipelineSelector'
import { AdapterBadge } from '../components/AdapterSelector'
import { useToastContext } from '../contexts/ToastContext'
import { ConfirmActionButton } from '../components/ConfirmActionButton'
import { StatusBanner } from '../components/StatusBanner'

interface FilterOptions {
  timeRange: 'all' | 'today' | 'week' | 'month'
  quality: 'all' | 'draft' | 'fast' | 'balanced' | 'high' | 'studio'
  favorites: boolean
  voice: string
}

const UNKNOWN_DATE = new Date(0)

function textOrEmpty(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function firstText(...values: unknown[]): string {
  for (const value of values) {
    const text = textOrEmpty(value)
    if (text) return text
  }
  return ''
}

function coerceDate(...values: unknown[]): Date {
  for (const value of values) {
    if (value instanceof Date && Number.isFinite(value.getTime())) {
      return value
    }
    if (typeof value === 'string' || typeof value === 'number') {
      const date = new Date(value)
      if (Number.isFinite(date.getTime())) {
        return date
      }
    }
  }
  return UNKNOWN_DATE
}

function normalizeHistoryRecord(item: ConversionRecord): ConversionRecord {
  const originalFileName = firstText(item.originalFileName, item.input_file, item.id) || 'Untitled conversion'
  const targetVoice = firstText(item.targetVoice, item.profile_id) || 'Unknown voice'
  const quality = firstText(item.quality, item.preset) || 'N/A'
  const resultUrl = firstText(item.resultUrl, item.output_url, item.download_url) || undefined

  return {
    ...item,
    input_file: firstText(item.input_file, originalFileName) || originalFileName,
    originalFileName,
    targetVoice,
    quality,
    resultUrl,
    timestamp: coerceDate(item.timestamp, item.completed_at, item.created_at),
  }
}

function recordTitle(item: ConversionRecord): string {
  return firstText(item.originalFileName, item.input_file, item.id) || 'Untitled conversion'
}

function recordVoice(item: ConversionRecord): string {
  return firstText(item.targetVoice, item.profile_id) || 'Unknown voice'
}

function recordQuality(item: ConversionRecord): string {
  return firstText(item.quality, item.preset) || 'N/A'
}

export function ConversionHistoryPage() {
  const [history, setHistory] = useState<ConversionRecord[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [pageError, setPageError] = useState<string | null>(null)
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
  const { error: toastError, success: toastSuccess } = useToastContext()

  const loadHistory = useCallback(async () => {
    setIsLoading(true)
    setPageError(null)
    try {
      const records = await apiService.getConversionHistory()
      setHistory(records.map(normalizeHistoryRecord))
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load conversion history'
      setPageError(message)
      toastError(message)
    } finally {
      setIsLoading(false)
    }
  }, [toastError])

  useEffect(() => {
    void loadHistory()
  }, [loadHistory])

  const handleDelete = async (id: string) => {
    try {
      await apiService.deleteConversionRecord(id)
      setHistory((prev) => prev.filter((item) => item.id !== id))
      toastSuccess('Conversion record deleted')
    } catch (error) {
      toastError(error instanceof Error ? error.message : 'Failed to delete conversion record')
    }
  }

  const handleToggleFavorite = async (id: string) => {
    const item = history.find((entry) => entry.id === id)
    if (!item) return
    try {
      const updatedRecord = await apiService.updateConversionRecord(id, {
        isFavorite: !item.isFavorite,
      })
      setHistory((prev) => prev.map((entry) => (
        entry.id === id ? normalizeHistoryRecord({ ...entry, ...updatedRecord }) : entry
      )))
      toastSuccess(updatedRecord.isFavorite ? 'Marked as favorite' : 'Removed from favorites')
    } catch (error) {
      toastError(error instanceof Error ? error.message : 'Failed to update favorite')
    }
  }

  const handleSaveNotes = async (id: string) => {
    try {
      const updatedRecord = await apiService.updateConversionRecord(id, { notes: notesText })
      setHistory((prev) => prev.map((item) => (
        item.id === id ? normalizeHistoryRecord({ ...item, ...updatedRecord }) : item
      )))
      setEditingNotes(null)
      toastSuccess('Notes saved')
    } catch (error) {
      toastError(error instanceof Error ? error.message : 'Failed to save notes')
    }
  }

  const handleClearAll = async () => {
    try {
      await Promise.all(history.map((item) => apiService.deleteConversionRecord(item.id)))
      setHistory([])
      toastSuccess('Conversion history cleared')
    } catch (error) {
      toastError(error instanceof Error ? error.message : 'Failed to clear conversion history')
    }
  }

  const getUniqueVoices = () => {
    const voices = new Set(history.map(recordVoice).filter(Boolean))
    return Array.from(voices).sort()
  }

  const filteredHistory = history.filter((item) => {
    const now = new Date()
    const itemTime = coerceDate(item.timestamp, item.created_at)
    const diff = now.getTime() - itemTime.getTime()
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
    const qualityMatch = filters.quality === 'all' || recordQuality(item) === filters.quality

    // Favorites filter
    const favoritesMatch = !filters.favorites || item.isFavorite

    // Voice filter
    const voiceMatch = filters.voice === 'all' || recordVoice(item) === filters.voice

    // Search query filter
    const normalizedQuery = searchQuery.toLowerCase()
    const searchMatch =
      searchQuery === '' ||
      recordTitle(item).toLowerCase().includes(normalizedQuery) ||
      recordVoice(item).toLowerCase().includes(normalizedQuery) ||
      textOrEmpty(item.notes).toLowerCase().includes(normalizedQuery)

    return timeMatch && qualityMatch && favoritesMatch && voiceMatch && searchMatch
  }).sort((a, b) => {
    // Sort favorites first, then by date
    if (a.isFavorite && !b.isFavorite) return -1
    if (!a.isFavorite && b.isFavorite) return 1
    const timeA = coerceDate(a.timestamp, a.created_at).getTime()
    const timeB = coerceDate(b.timestamp, b.created_at).getTime()
    return timeB - timeA
  })

  const formatDate = (date: Date) => {
    if (!Number.isFinite(date.getTime()) || date.getTime() === 0) {
      return 'Unknown date'
    }
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
    <div className="max-w-6xl mx-auto px-4 py-8" data-testid="history-page">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white flex items-center space-x-3">
          <Clock className="w-8 h-8 text-primary-600" />
          <span>Conversion History</span>
        </h1>
        <p className="text-gray-300 mt-2">View and manage your past conversions</p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6">
        {pageError && (
          <div className="mb-6">
            <StatusBanner
              tone="danger"
              title="History unavailable"
              message={pageError}
            />
          </div>
        )}

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
              <ConfirmActionButton
                label="Clear all"
                confirmLabel="Clear history"
                confirmMessage="Delete every conversion record from the local history list?"
                onConfirm={handleClearAll}
                variant="danger"
                testId="history-clear-all"
              />
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
        {isLoading ? (
          <div className="py-12 text-center text-gray-500">Loading history...</div>
        ) : filteredHistory.length === 0 ? (
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
                        <h3 className="font-semibold text-gray-900">{recordTitle(item)}</h3>
                        {item.isFavorite && <Star className="w-4 h-4 text-yellow-500 fill-yellow-500" />}
                      </div>
                      <p className="text-sm text-gray-600">
                        Voice: {recordVoice(item)} • Duration: {formatDuration(item.duration ?? 0)} • Quality:{' '}
                        <span className="font-medium">{recordQuality(item)}</span>
                      </p>
                      {/* Pipeline and adapter badges */}
                      <div className="flex items-center gap-2 mt-1">
                        {item.pipeline_type && (
                          <PipelineBadge pipeline={item.pipeline_type as PipelineType} />
                        )}
                        {item.adapter_type && item.adapter_type !== 'unified' && (
                          <AdapterBadge adapterType={item.adapter_type as 'hq' | 'nvfp4'} />
                        )}
                        {item.rtf !== undefined && (
                          <span className="text-xs text-gray-500">
                            RTF: {item.rtf.toFixed(2)}x
                          </span>
                        )}
                        {item.processing_time_seconds !== undefined && (
                          <span className="text-xs text-gray-500">
                            Processed in {item.processing_time_seconds.toFixed(1)}s
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-gray-400 mt-1">{formatDate(coerceDate(item.timestamp, item.created_at))}</p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-1 flex-shrink-0">
                    {(item.resultUrl ?? item.output_url ?? item.download_url) && (
                      <>
                        <button
                          onClick={() => {
                            const audio = new Audio(item.resultUrl ?? item.output_url ?? item.download_url!)
                            audio.play()
                          }}
                          className="p-2 text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
                          title="Play"
                        >
                          <Play className="w-5 h-5" />
                        </button>
                        <a
                          href={item.resultUrl ?? item.output_url ?? item.download_url}
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

                {(item.stem_urls?.vocals || item.stem_urls?.instrumental || item.reassemble_url) && (
                  <ArtifactComparison record={item} />
                )}

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

function ArtifactComparison({ record }: { record: ConversionRecord }) {
  const mixUrl = firstText(record.resultUrl, record.output_url, record.download_url) || undefined
  const artifacts = [
    { key: 'source', label: 'Original source', url: firstText(record.input_file) || undefined, tone: 'bg-gray-100 text-gray-700' },
    { key: 'mix', label: 'Converted mix', url: mixUrl, tone: 'bg-green-50 text-green-700' },
    { key: 'vocals', label: 'Converted vocals', url: firstText(record.stem_urls?.vocals) || undefined, tone: 'bg-cyan-50 text-cyan-700' },
    { key: 'instrumental', label: 'Instrumental', url: firstText(record.stem_urls?.instrumental) || undefined, tone: 'bg-amber-50 text-amber-700' },
    { key: 'reassemble', label: 'Reassembled mix', url: firstText(record.reassemble_url) || undefined, tone: 'bg-blue-50 text-blue-700' },
  ]

  return (
    <div className="mt-3 rounded-lg border border-gray-200 bg-gray-50 p-3" data-testid={`artifact-comparison-${record.id}`}>
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
        <GitCompare className="h-4 w-4" />
        Artifact comparison
      </div>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2 lg:grid-cols-3">
        {artifacts.map((artifact) => (
          <div key={artifact.key} className="rounded-md border border-gray-200 bg-white p-2">
            <div className="text-xs font-medium text-gray-500">{artifact.label}</div>
            {artifact.url ? (
              <div className="mt-2 flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => {
                    const audio = new Audio(artifact.url)
                    audio.play()
                  }}
                  className="rounded bg-primary-50 p-1.5 text-primary-700 hover:bg-primary-100"
                  title={`Play ${artifact.label}`}
                >
                  <Play className="h-4 w-4" />
                </button>
                <a
                  href={artifact.url}
                  download
                  className={`rounded px-2 py-1 text-xs font-medium transition-colors ${artifact.tone}`}
                >
                  Download
                </a>
              </div>
            ) : (
              <div className="mt-2 rounded bg-gray-100 px-2 py-1 text-xs text-gray-500">
                Not available for this conversion
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
