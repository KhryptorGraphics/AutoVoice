import { useState, useMemo } from 'react'
import {
  Play, Pause, Download, Trash2, Star, StarOff,
  ChevronDown, ChevronUp, Search, Columns, AlertCircle,
  CheckCircle, Clock, Loader2, BarChart2, History
} from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService, ConversionRecord } from '../services/api'
import { PipelineBadge } from './PipelineSelector'
import { AdapterBadge } from './AdapterSelector'
import clsx from 'clsx'

interface ConversionHistoryTableProps {
  profileId?: string
  onSelect?: (record: ConversionRecord) => void
  onCompare?: (records: [ConversionRecord, ConversionRecord]) => void
}

type SortField = 'created_at' | 'duration' | 'status' | 'pipeline_type'
type SortDirection = 'asc' | 'desc'

const statusConfig: Record<ConversionRecord['status'], { icon: typeof CheckCircle; color: string; label: string }> = {
  queued: { icon: Clock, color: 'text-yellow-400', label: 'Queued' },
  processing: { icon: Loader2, color: 'text-blue-400', label: 'Processing' },
  complete: { icon: CheckCircle, color: 'text-green-400', label: 'Complete' },
  completed: { icon: CheckCircle, color: 'text-green-400', label: 'Complete' },
  error: { icon: AlertCircle, color: 'text-red-400', label: 'Failed' },
  cancelled: { icon: AlertCircle, color: 'text-gray-400', label: 'Cancelled' },
}

export function ConversionHistoryTable({ profileId, onSelect, onCompare }: ConversionHistoryTableProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<ConversionRecord['status'] | 'all'>('all')
  const [sortField, setSortField] = useState<SortField>('created_at')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [playingId, setPlayingId] = useState<string | null>(null)
  const [audioElements] = useState<Map<string, HTMLAudioElement>>(new Map())
  const queryClient = useQueryClient()

  const { data: records, isLoading } = useQuery({
    queryKey: ['conversionHistory', profileId],
    queryFn: () => apiService.getConversionHistory(profileId),
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiService.deleteConversionRecord(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['conversionHistory'] }),
  })

  const toggleFavoriteMutation = useMutation({
    mutationFn: ({ id, isFavorite }: { id: string; isFavorite: boolean }) =>
      apiService.updateConversionRecord(id, { isFavorite }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['conversionHistory'] }),
  })

  const filteredRecords = useMemo(() => {
    if (!records) return []

    return records
      .filter(r => {
        if (statusFilter !== 'all' && r.status !== statusFilter) return false
        if (searchQuery) {
          const query = searchQuery.toLowerCase()
          return (
            r.input_file.toLowerCase().includes(query) ||
            r.profile_id.toLowerCase().includes(query) ||
            r.targetVoice?.toLowerCase().includes(query)
          )
        }
        return true
      })
      .sort((a, b) => {
        let cmp = 0
        switch (sortField) {
          case 'created_at':
            cmp = new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
            break
          case 'duration':
            cmp = (a.duration || 0) - (b.duration || 0)
            break
          case 'status':
            cmp = a.status.localeCompare(b.status)
            break
          case 'pipeline_type':
            cmp = (a.pipeline_type || '').localeCompare(b.pipeline_type || '')
            break
        }
        return sortDirection === 'asc' ? cmp : -cmp
      })
  }, [records, statusFilter, searchQuery, sortField, sortDirection])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('desc')
    }
  }

  const toggleSelection = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const handlePlayToggle = (record: ConversionRecord) => {
    if (!record.resultUrl) return

    if (playingId === record.id) {
      const audio = audioElements.get(record.id)
      audio?.pause()
      setPlayingId(null)
    } else {
      // Pause any currently playing
      if (playingId) {
        audioElements.get(playingId)?.pause()
      }

      let audio = audioElements.get(record.id)
      if (!audio) {
        audio = new Audio(record.resultUrl)
        audio.onended = () => setPlayingId(null)
        audioElements.set(record.id, audio)
      }
      audio.play()
      setPlayingId(record.id)
    }
  }

  const handleCompare = () => {
    if (selectedIds.size !== 2 || !onCompare) return
    const selected = filteredRecords.filter(r => selectedIds.has(r.id))
    if (selected.length === 2) {
      onCompare([selected[0], selected[1]])
    }
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString()
  }

  const formatDuration = (seconds?: number) => {
    if (!seconds) return '-'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null
    return sortDirection === 'asc' ? <ChevronUp size={12} /> : <ChevronDown size={12} />
  }

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <History size={18} className="text-blue-400" />
            <h3 className="font-semibold">Conversion History</h3>
          </div>
          <span className="text-sm text-gray-500">
            {filteredRecords.length} records
          </span>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-3">
          <div className="flex-1 relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            <input
              type="text"
              placeholder="Search by filename or voice..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm"
            />
          </div>

          <select
            value={statusFilter}
            onChange={e => setStatusFilter(e.target.value as typeof statusFilter)}
            className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
          >
            <option value="all">All Status</option>
            <option value="complete">Complete</option>
            <option value="processing">Processing</option>
            <option value="queued">Queued</option>
            <option value="error">Failed</option>
          </select>

          {selectedIds.size === 2 && onCompare && (
            <button
              onClick={handleCompare}
              className="flex items-center gap-2 px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm"
            >
              <Columns size={14} />
              Compare Selected
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      {isLoading ? (
        <div className="p-8 text-center">
          <Loader2 className="animate-spin mx-auto text-gray-500 mb-2" />
          <span className="text-gray-400 text-sm">Loading history...</span>
        </div>
      ) : filteredRecords.length === 0 ? (
        <div className="p-8 text-center text-gray-500">
          No conversion records found
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-xs text-gray-500 border-b border-gray-700">
                <th className="p-3 w-10">
                  <input
                    type="checkbox"
                    checked={selectedIds.size === filteredRecords.length && filteredRecords.length > 0}
                    onChange={e => {
                      if (e.target.checked) {
                        setSelectedIds(new Set(filteredRecords.map(r => r.id)))
                      } else {
                        setSelectedIds(new Set())
                      }
                    }}
                    className="rounded border-gray-600"
                  />
                </th>
                <th className="p-3">File</th>
                <th className="p-3 cursor-pointer hover:text-white" onClick={() => handleSort('status')}>
                  <span className="flex items-center gap-1">
                    Status <SortIcon field="status" />
                  </span>
                </th>
                <th className="p-3 cursor-pointer hover:text-white" onClick={() => handleSort('pipeline_type')}>
                  <span className="flex items-center gap-1">
                    Pipeline <SortIcon field="pipeline_type" />
                  </span>
                </th>
                <th className="p-3 cursor-pointer hover:text-white" onClick={() => handleSort('duration')}>
                  <span className="flex items-center gap-1">
                    Duration <SortIcon field="duration" />
                  </span>
                </th>
                <th className="p-3 cursor-pointer hover:text-white" onClick={() => handleSort('created_at')}>
                  <span className="flex items-center gap-1">
                    Date <SortIcon field="created_at" />
                  </span>
                </th>
                <th className="p-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredRecords.map(record => {
                const StatusIcon = statusConfig[record.status].icon
                return (
                  <tr
                    key={record.id}
                    className={clsx(
                      'border-b border-gray-700/50 hover:bg-gray-750 cursor-pointer',
                      selectedIds.has(record.id) && 'bg-blue-900/20'
                    )}
                    onClick={() => onSelect?.(record)}
                  >
                    <td className="p-3" onClick={e => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={selectedIds.has(record.id)}
                        onChange={() => toggleSelection(record.id)}
                        className="rounded border-gray-600"
                      />
                    </td>
                    <td className="p-3">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={e => {
                            e.stopPropagation()
                            toggleFavoriteMutation.mutate({
                              id: record.id,
                              isFavorite: !record.isFavorite,
                            })
                          }}
                          className={clsx(
                            'p-1 rounded transition-colors',
                            record.isFavorite
                              ? 'text-yellow-400'
                              : 'text-gray-600 hover:text-yellow-400'
                          )}
                        >
                          {record.isFavorite ? <Star size={14} /> : <StarOff size={14} />}
                        </button>
                        <div>
                          <div className="text-sm font-medium truncate max-w-[200px]">
                            {record.originalFileName || record.input_file}
                          </div>
                          <div className="text-xs text-gray-500">
                            {record.targetVoice || record.profile_id}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="p-3">
                      <span className={clsx('flex items-center gap-1', statusConfig[record.status].color)}>
                        <StatusIcon size={14} className={record.status === 'processing' ? 'animate-spin' : ''} />
                        {statusConfig[record.status].label}
                      </span>
                    </td>
                    <td className="p-3">
                      <div className="flex flex-col gap-1">
                        {record.pipeline_type && (
                          <PipelineBadge pipeline={record.pipeline_type} />
                        )}
                        {record.adapter_type && record.adapter_type !== 'unified' && (
                          <AdapterBadge adapterType={record.adapter_type as 'hq' | 'nvfp4'} />
                        )}
                        {!record.pipeline_type && !record.adapter_type && (
                          <span className="text-xs text-gray-500">-</span>
                        )}
                      </div>
                    </td>
                    <td className="p-3 font-mono text-sm text-gray-400">
                      {formatDuration(record.duration)}
                    </td>
                    <td className="p-3 text-sm text-gray-400">
                      {formatDate(record.created_at)}
                    </td>
                    <td className="p-3" onClick={e => e.stopPropagation()}>
                      <div className="flex items-center justify-end gap-1">
                        {record.status === 'complete' && record.resultUrl && (
                          <>
                            <button
                              onClick={() => handlePlayToggle(record)}
                              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                              title={playingId === record.id ? 'Pause' : 'Play'}
                            >
                              {playingId === record.id ? <Pause size={14} /> : <Play size={14} />}
                            </button>
                            <a
                              href={record.resultUrl}
                              download
                              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                              title="Download"
                            >
                              <Download size={14} />
                            </a>
                          </>
                        )}
                        <button
                          onClick={() => {
                            if (confirm('Delete this record?')) {
                              deleteMutation.mutate(record.id)
                            }
                          }}
                          className="p-1.5 text-gray-400 hover:text-red-400 hover:bg-gray-700 rounded"
                          title="Delete"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Quality Metrics (shown for selected record) */}
      {selectedIds.size === 1 && (() => {
        const selectedRecord = filteredRecords.find(r => selectedIds.has(r.id))
        if (!selectedRecord) return null

        return (
          <div className="p-4 border-t border-gray-700 bg-gray-750">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <BarChart2 size={14} className="text-blue-400" />
                <span className="text-sm font-medium">Conversion Details</span>
              </div>
              <div className="flex items-center gap-2">
                {selectedRecord.pipeline_type && (
                  <PipelineBadge pipeline={selectedRecord.pipeline_type} />
                )}
                {selectedRecord.adapter_type && selectedRecord.adapter_type !== 'unified' && (
                  <AdapterBadge adapterType={selectedRecord.adapter_type as 'hq' | 'nvfp4'} />
                )}
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
              {selectedRecord.processing_time_seconds != null && (
                <div>
                  <div className="flex items-center gap-1 text-gray-500">
                    <Clock size={12} />
                    Processing Time
                  </div>
                  <div className="font-mono mt-1">
                    {selectedRecord.processing_time_seconds.toFixed(1)}s
                  </div>
                </div>
              )}
              {selectedRecord.rtf != null && (
                <div>
                  <div className="flex items-center gap-1 text-gray-500">
                    <BarChart2 size={12} />
                    Speed (RTF)
                  </div>
                  <div className={clsx(
                    'font-mono mt-1',
                    selectedRecord.rtf < 1 ? 'text-green-400' : 'text-yellow-400'
                  )}>
                    {selectedRecord.rtf < 1
                      ? `${(selectedRecord.rtf * 100).toFixed(0)}% RT`
                      : `${selectedRecord.rtf.toFixed(1)}x RT`
                    }
                  </div>
                </div>
              )}
              {selectedRecord.audio_duration_seconds != null && (
                <div>
                  <div className="text-gray-500">Audio Length</div>
                  <div className="font-mono mt-1">
                    {formatDuration(selectedRecord.audio_duration_seconds)}
                  </div>
                </div>
              )}
              {selectedRecord.duration != null && (
                <div>
                  <div className="text-gray-500">Output Duration</div>
                  <div className="font-mono mt-1">
                    {formatDuration(selectedRecord.duration)}
                  </div>
                </div>
              )}
            </div>
          </div>
        )
      })()}
    </div>
  )
}
