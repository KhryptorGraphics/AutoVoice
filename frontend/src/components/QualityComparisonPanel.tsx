/**
 * Quality Comparison Panel - Compare voice conversion quality between adapters
 *
 * Shows side-by-side comparison of HQ vs nvfp4 adapter performance metrics.
 */
import { useState, useEffect } from 'react'
import { Sparkles, Zap, ArrowRight, CheckCircle, Loader2, AlertCircle } from 'lucide-react'
import { apiService, AdapterType, AdapterMetricsResponse, AdapterMetrics } from '../services/api'
import clsx from 'clsx'

interface QualityComparisonPanelProps {
  profileId: string
  onAdapterSelect?: (adapter: AdapterType) => void
}

interface ComparisonRow {
  label: string
  hq: string | number
  nvfp4: string | number
  winner: 'hq' | 'nvfp4' | 'tie'
  higherIsBetter: boolean
}

function getComparisonRows(metrics: Record<AdapterType, AdapterMetrics>): ComparisonRow[] {
  const hq = metrics.hq
  const nvfp4 = metrics.nvfp4

  if (!hq || !nvfp4) return []

  const rows: ComparisonRow[] = [
    {
      label: 'Training Epochs',
      hq: hq.epochs,
      nvfp4: nvfp4.epochs,
      winner: hq.epochs > nvfp4.epochs ? 'hq' : hq.epochs < nvfp4.epochs ? 'nvfp4' : 'tie',
      higherIsBetter: true,
    },
    {
      label: 'Final Loss',
      hq: hq.loss?.toFixed(4) ?? 'N/A',
      nvfp4: nvfp4.loss?.toFixed(4) ?? 'N/A',
      winner: (hq.loss ?? 1) < (nvfp4.loss ?? 1) ? 'hq' : (hq.loss ?? 1) > (nvfp4.loss ?? 1) ? 'nvfp4' : 'tie',
      higherIsBetter: false,
    },
    {
      label: 'Parameters',
      hq: hq.parameter_count_formatted,
      nvfp4: nvfp4.parameter_count_formatted,
      winner: hq.parameter_count > nvfp4.parameter_count ? 'hq' : 'nvfp4',
      higherIsBetter: true, // More params = more capacity
    },
    {
      label: 'Memory Usage',
      hq: `${hq.performance.memory_estimate_mb} MB`,
      nvfp4: `${nvfp4.performance.memory_estimate_mb} MB`,
      winner: hq.performance.memory_estimate_mb < nvfp4.performance.memory_estimate_mb ? 'hq' : 'nvfp4',
      higherIsBetter: false,
    },
    {
      label: 'Relative Quality',
      hq: hq.performance.relative_quality,
      nvfp4: nvfp4.performance.relative_quality,
      winner: hq.performance.relative_quality === 'Highest' ? 'hq' : 'nvfp4',
      higherIsBetter: true,
    },
    {
      label: 'Relative Speed',
      hq: hq.performance.relative_speed,
      nvfp4: nvfp4.performance.relative_speed,
      winner: nvfp4.performance.relative_speed === 'Fast' ? 'nvfp4' : 'hq',
      higherIsBetter: true,
    },
    {
      label: 'Precision',
      hq: hq.precision,
      nvfp4: nvfp4.precision,
      winner: 'tie', // Different precision for different use cases
      higherIsBetter: true,
    },
  ]

  return rows
}

export function QualityComparisonPanel({ profileId, onAdapterSelect }: QualityComparisonPanelProps) {
  const [metrics, setMetrics] = useState<AdapterMetricsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadMetrics = async () => {
      try {
        setLoading(true)
        const data = await apiService.getAdapterMetrics(profileId)
        setMetrics(data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load metrics')
      } finally {
        setLoading(false)
      }
    }

    if (profileId) {
      loadMetrics()
    }
  }, [profileId])

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 flex items-center justify-center">
        <Loader2 className="animate-spin text-gray-400 mr-2" size={24} />
        <span className="text-gray-400">Loading comparison data...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <AlertCircle className="mx-auto text-red-400 mb-2" size={32} />
        <p className="text-red-400">{error}</p>
      </div>
    )
  }

  if (!metrics || metrics.adapter_count < 2) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <AlertCircle className="mx-auto text-yellow-400 mb-2" size={32} />
        <p className="text-gray-400">
          Both HQ and nvfp4 adapters are needed for comparison.
        </p>
        <p className="text-sm text-gray-500 mt-2">
          Train both adapter types to see quality comparison.
        </p>
      </div>
    )
  }

  const comparisonRows = getComparisonRows(metrics.adapters)

  // Count wins
  const hqWins = comparisonRows.filter(r => r.winner === 'hq').length
  const nvfp4Wins = comparisonRows.filter(r => r.winner === 'nvfp4').length

  return (
    <div className="space-y-4">
      {/* Header with adapter icons */}
      <div className="grid grid-cols-3 gap-4 bg-gray-800 rounded-lg p-4">
        <div className="text-center">
          <div className="inline-flex items-center gap-2 px-3 py-2 bg-violet-600/20 border border-violet-500/50 rounded-lg">
            <Sparkles size={20} className="text-violet-400" />
            <span className="font-semibold text-violet-300">HQ</span>
          </div>
          <p className="text-xs text-gray-500 mt-2">High Quality</p>
        </div>

        <div className="flex items-center justify-center">
          <ArrowRight className="text-gray-600" />
          <span className="text-sm text-gray-500 mx-2">vs</span>
          <ArrowRight className="text-gray-600 rotate-180" />
        </div>

        <div className="text-center">
          <div className="inline-flex items-center gap-2 px-3 py-2 bg-yellow-600/20 border border-yellow-500/50 rounded-lg">
            <Zap size={20} className="text-yellow-400" />
            <span className="font-semibold text-yellow-300">nvfp4</span>
          </div>
          <p className="text-xs text-gray-500 mt-2">Fast (4-bit)</p>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-gray-800 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">Metric</th>
              <th className="px-4 py-3 text-center text-sm font-medium text-violet-400">HQ</th>
              <th className="px-4 py-3 text-center text-sm font-medium text-yellow-400">nvfp4</th>
              <th className="px-4 py-3 text-center text-sm font-medium text-gray-400">Winner</th>
            </tr>
          </thead>
          <tbody>
            {comparisonRows.map((row, i) => (
              <tr
                key={row.label}
                className={clsx(
                  'border-b border-gray-700/50',
                  i % 2 === 0 ? 'bg-gray-800' : 'bg-gray-800/50'
                )}
              >
                <td className="px-4 py-3 text-sm text-gray-300">{row.label}</td>
                <td className={clsx(
                  'px-4 py-3 text-center text-sm font-mono',
                  row.winner === 'hq' ? 'text-violet-300 font-semibold' : 'text-gray-400'
                )}>
                  {row.hq}
                </td>
                <td className={clsx(
                  'px-4 py-3 text-center text-sm font-mono',
                  row.winner === 'nvfp4' ? 'text-yellow-300 font-semibold' : 'text-gray-400'
                )}>
                  {row.nvfp4}
                </td>
                <td className="px-4 py-3 text-center">
                  {row.winner === 'hq' && (
                    <span className="inline-flex items-center gap-1 text-xs text-violet-400">
                      <CheckCircle size={14} />
                      HQ
                    </span>
                  )}
                  {row.winner === 'nvfp4' && (
                    <span className="inline-flex items-center gap-1 text-xs text-yellow-400">
                      <CheckCircle size={14} />
                      nvfp4
                    </span>
                  )}
                  {row.winner === 'tie' && (
                    <span className="text-xs text-gray-500">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-2 gap-4">
        <button
          onClick={() => onAdapterSelect?.('hq')}
          className={clsx(
            'p-4 rounded-lg border-2 transition-all text-left',
            hqWins > nvfp4Wins
              ? 'border-violet-500 bg-violet-500/10 hover:bg-violet-500/20'
              : 'border-gray-700 bg-gray-800 hover:border-gray-600'
          )}
        >
          <div className="flex items-center gap-2 mb-2">
            <Sparkles size={20} className="text-violet-400" />
            <span className="font-semibold text-violet-300">HQ Adapter</span>
            {hqWins > nvfp4Wins && (
              <span className="text-xs px-2 py-0.5 bg-violet-600 rounded text-white">Recommended</span>
            )}
          </div>
          <p className="text-sm text-gray-400">
            Best for song conversion and studio-quality output.
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Wins: {hqWins}/{comparisonRows.length} metrics
          </p>
        </button>

        <button
          onClick={() => onAdapterSelect?.('nvfp4')}
          className={clsx(
            'p-4 rounded-lg border-2 transition-all text-left',
            nvfp4Wins > hqWins
              ? 'border-yellow-500 bg-yellow-500/10 hover:bg-yellow-500/20'
              : 'border-gray-700 bg-gray-800 hover:border-gray-600'
          )}
        >
          <div className="flex items-center gap-2 mb-2">
            <Zap size={20} className="text-yellow-400" />
            <span className="font-semibold text-yellow-300">nvfp4 Adapter</span>
            {nvfp4Wins > hqWins && (
              <span className="text-xs px-2 py-0.5 bg-yellow-600 rounded text-white">Recommended</span>
            )}
          </div>
          <p className="text-sm text-gray-400">
            Best for live karaoke and real-time applications.
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Wins: {nvfp4Wins}/{comparisonRows.length} metrics
          </p>
        </button>
      </div>

      {/* Recommendation */}
      {metrics.recommended && (
        <div className="bg-gray-700/50 rounded-lg p-4 text-center">
          <p className="text-sm text-gray-300">
            <strong>Recommended for general use:</strong>{' '}
            <span className={metrics.recommended === 'hq' ? 'text-violet-400' : 'text-yellow-400'}>
              {metrics.recommended === 'hq' ? 'High Quality' : 'Fast (nvfp4)'}
            </span>
          </p>
        </div>
      )}
    </div>
  )
}
