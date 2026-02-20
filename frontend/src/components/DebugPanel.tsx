import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Bug, Download, Trash2, Filter, ChevronDown, Search,
  AlertCircle, AlertTriangle, Info, Activity, Loader2, Play, Pause
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { apiService, wsManager } from '../services/api'
import clsx from 'clsx'

export type LogLevel = 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'

interface LogEntry {
  id: string
  timestamp: string
  level: LogLevel
  source: string
  message: string
  details?: Record<string, unknown>
}

interface DebugPanelProps {
  maxLogs?: number
  autoScroll?: boolean
}

const levelConfig: Record<LogLevel, { icon: typeof Info; color: string; bgColor: string }> = {
  DEBUG: { icon: Bug, color: 'text-gray-400', bgColor: 'bg-gray-800' },
  INFO: { icon: Info, color: 'text-blue-400', bgColor: 'bg-blue-900/20' },
  WARN: { icon: AlertTriangle, color: 'text-yellow-400', bgColor: 'bg-yellow-900/20' },
  ERROR: { icon: AlertCircle, color: 'text-red-400', bgColor: 'bg-red-900/20' },
}

const levelPriority: Record<LogLevel, number> = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3,
}

export function DebugPanel({ maxLogs = 500, autoScroll: initialAutoScroll = true }: DebugPanelProps) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [filterLevel, setFilterLevel] = useState<LogLevel>('INFO')
  const [searchQuery, setSearchQuery] = useState('')
  const [sourceFilter, setSourceFilter] = useState<string>('all')
  const [autoScroll, setAutoScroll] = useState(initialAutoScroll)
  const [isStreaming, setIsStreaming] = useState(true)
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const logContainerRef = useRef<HTMLDivElement>(null)

  // Fetch initial system diagnostics
  const { data: systemInfo, refetch: refetchSystem } = useQuery({
    queryKey: ['systemInfo'],
    queryFn: () => apiService.getSystemStatus(),
    refetchInterval: 30000,
  })

  const { data: gpuMetrics } = useQuery({
    queryKey: ['gpuMetrics'],
    queryFn: () => apiService.getGPUMetrics(),
    refetchInterval: 5000,
  })

  // Add log entry
  const addLog = useCallback((entry: Omit<LogEntry, 'id'>) => {
    setLogs(prev => {
      const newLog: LogEntry = {
        ...entry,
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      }
      const updated = [...prev, newLog]
      return updated.slice(-maxLogs)
    })
  }, [maxLogs])

  // Subscribe to WebSocket logs
  useEffect(() => {
    if (!isStreaming) return

    // System startup log
    addLog({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      source: 'frontend',
      message: 'Debug panel connected, streaming logs...',
    })

    // Subscribe to various events for logging
    const unsubConversion = wsManager.subscribe('conversion_progress', (event) => {
      addLog({
        timestamp: event.timestamp,
        level: 'INFO',
        source: 'conversion',
        message: `Conversion progress: ${JSON.stringify(event.data)}`,
        details: event.data as Record<string, unknown>,
      })
    })

    const unsubTraining = wsManager.subscribe('training_progress', (event) => {
      addLog({
        timestamp: event.timestamp,
        level: 'INFO',
        source: 'training',
        message: `Training progress: ${JSON.stringify(event.data)}`,
        details: event.data as Record<string, unknown>,
      })
    })

    const unsubError = wsManager.subscribe('conversion_error', (event) => {
      addLog({
        timestamp: event.timestamp,
        level: 'ERROR',
        source: 'conversion',
        message: `Conversion error: ${JSON.stringify(event.data)}`,
        details: event.data as Record<string, unknown>,
      })
    })

    const unsubTrainingError = wsManager.subscribe('training_error', (event) => {
      addLog({
        timestamp: event.timestamp,
        level: 'ERROR',
        source: 'training',
        message: `Training error: ${JSON.stringify(event.data)}`,
        details: event.data as Record<string, unknown>,
      })
    })

    return () => {
      unsubConversion()
      unsubTraining()
      unsubError()
      unsubTrainingError()
    }
  }, [isStreaming, addLog])

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  // Filter logs
  const filteredLogs = logs.filter(log => {
    // Level filter
    if (levelPriority[log.level] < levelPriority[filterLevel]) return false
    // Source filter
    if (sourceFilter !== 'all' && log.source !== sourceFilter) return false
    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      return (
        log.message.toLowerCase().includes(query) ||
        log.source.toLowerCase().includes(query)
      )
    }
    return true
  })

  // Get unique sources for filter dropdown
  const sources = Array.from(new Set(logs.map(l => l.source)))

  // Export diagnostics
  const exportDiagnostics = () => {
    const diagnostics = {
      exportedAt: new Date().toISOString(),
      system: systemInfo,
      gpu: gpuMetrics,
      logs: logs,
      browserInfo: {
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        language: navigator.language,
        screenWidth: window.screen.width,
        screenHeight: window.screen.height,
      },
    }

    const blob = new Blob([JSON.stringify(diagnostics, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `autovoice-diagnostics-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)

    addLog({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      source: 'frontend',
      message: 'Diagnostics exported',
    })
  }

  const clearLogs = () => {
    setLogs([])
    addLog({
      timestamp: new Date().toISOString(),
      level: 'INFO',
      source: 'frontend',
      message: 'Logs cleared',
    })
  }

  const toggleExpand = (id: string) => {
    setExpandedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const formatTimestamp = (ts: string) => {
    const date = new Date(ts)
    const time = date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
    const ms = date.getMilliseconds().toString().padStart(3, '0')
    return `${time}.${ms}`
  }

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Bug size={18} className="text-purple-400" />
            <h3 className="font-semibold">Debug Console</h3>
            <span className="text-xs text-gray-500">
              {filteredLogs.length} / {logs.length} entries
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsStreaming(!isStreaming)}
              className={clsx(
                'flex items-center gap-1 px-2 py-1 rounded text-xs',
                isStreaming ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-400'
              )}
            >
              {isStreaming ? <Pause size={12} /> : <Play size={12} />}
              {isStreaming ? 'Streaming' : 'Paused'}
            </button>
            <button
              onClick={() => setAutoScroll(!autoScroll)}
              className={clsx(
                'px-2 py-1 rounded text-xs',
                autoScroll ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-400'
              )}
            >
              Auto-scroll
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2">
          {/* Search */}
          <div className="flex-1 relative">
            <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-500" />
            <input
              type="text"
              placeholder="Search logs..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 bg-gray-700 border border-gray-600 rounded text-sm"
            />
          </div>

          {/* Level filter */}
          <div className="relative">
            <Filter size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-500" />
            <select
              value={filterLevel}
              onChange={e => setFilterLevel(e.target.value as LogLevel)}
              className="pl-8 pr-6 py-1.5 bg-gray-700 border border-gray-600 rounded text-sm appearance-none"
            >
              <option value="DEBUG">DEBUG+</option>
              <option value="INFO">INFO+</option>
              <option value="WARN">WARN+</option>
              <option value="ERROR">ERROR</option>
            </select>
            <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none" />
          </div>

          {/* Source filter */}
          <select
            value={sourceFilter}
            onChange={e => setSourceFilter(e.target.value)}
            className="px-3 py-1.5 bg-gray-700 border border-gray-600 rounded text-sm"
          >
            <option value="all">All Sources</option>
            {sources.map(source => (
              <option key={source} value={source}>{source}</option>
            ))}
          </select>

          {/* Actions */}
          <button
            onClick={exportDiagnostics}
            className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded text-sm"
            title="Export diagnostics"
          >
            <Download size={14} />
            Export
          </button>
          <button
            onClick={clearLogs}
            className="p-1.5 text-gray-400 hover:text-red-400 hover:bg-gray-700 rounded"
            title="Clear logs"
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      {/* System Status Bar */}
      <div className="px-4 py-2 bg-gray-750 border-b border-gray-700 flex items-center gap-4 text-xs">
        <div className="flex items-center gap-2">
          <Activity size={12} className="text-green-400" />
          <span className="text-gray-400">System:</span>
          <span>{systemInfo?.status ?? 'Unknown'}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-gray-400">GPU:</span>
          <span className={gpuMetrics?.available ? 'text-green-400' : 'text-red-400'}>
            {gpuMetrics?.available ? gpuMetrics.devices?.[0]?.name ?? 'Available' : 'Not Available'}
          </span>
        </div>
        {gpuMetrics?.available && gpuMetrics.devices?.[0] && (
          <>
            <div className="flex items-center gap-2">
              <span className="text-gray-400">Memory:</span>
              <span>
                {((gpuMetrics.devices[0].memory_used || 0) / 1024 / 1024 / 1024).toFixed(1)} /
                {((gpuMetrics.devices[0].memory_total || 0) / 1024 / 1024 / 1024).toFixed(1)} GB
              </span>
            </div>
            {gpuMetrics.devices[0].temperature_c != null && (
              <div className="flex items-center gap-2">
                <span className="text-gray-400">Temp:</span>
                <span className={gpuMetrics.devices[0].temperature_c > 80 ? 'text-red-400' : ''}>
                  {gpuMetrics.devices[0].temperature_c}°C
                </span>
              </div>
            )}
          </>
        )}
        <button
          onClick={() => refetchSystem()}
          className="ml-auto text-gray-500 hover:text-white"
        >
          <Loader2 size={12} />
        </button>
      </div>

      {/* Log Container */}
      <div
        ref={logContainerRef}
        className="flex-1 overflow-y-auto font-mono text-xs p-2 space-y-0.5"
      >
        {filteredLogs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            No logs matching filter criteria
          </div>
        ) : (
          filteredLogs.map(log => {
            const { icon: Icon, color, bgColor } = levelConfig[log.level]
            const isExpanded = expandedIds.has(log.id)

            return (
              <div
                key={log.id}
                className={clsx(
                  'px-2 py-1 rounded cursor-pointer hover:bg-gray-700/50 transition-colors',
                  bgColor
                )}
                onClick={() => log.details && toggleExpand(log.id)}
              >
                <div className="flex items-start gap-2">
                  <span className="text-gray-500 shrink-0">
                    {formatTimestamp(log.timestamp)}
                  </span>
                  <Icon size={12} className={clsx('mt-0.5 shrink-0', color)} />
                  <span className="text-gray-400 shrink-0 w-20 truncate">
                    [{log.source}]
                  </span>
                  <span className="flex-1 break-all">{log.message}</span>
                  {log.details && (
                    <ChevronDown
                      size={12}
                      className={clsx(
                        'text-gray-500 transition-transform shrink-0',
                        isExpanded && 'rotate-180'
                      )}
                    />
                  )}
                </div>
                {isExpanded && log.details && (
                  <pre className="mt-2 p-2 bg-gray-900 rounded text-gray-300 overflow-x-auto">
                    {JSON.stringify(log.details, null, 2)}
                  </pre>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
