/**
 * Quality Metrics Dashboard - Displays quality validation report data
 *
 * Task 5.3: Shows quality metrics, comparisons, and recommendations
 * across all voice profiles with adapters.
 */
import { useState, useEffect } from 'react'
import {
  BarChart3,
  Zap,
  Sparkles,
  Timer,
  RefreshCw,
  AlertCircle,
  Loader2,
  CheckCircle,
  Trophy,
} from 'lucide-react'
import { AdapterType } from '../services/api'
import clsx from 'clsx'

interface QualityMetrics {
  adapter_type: AdapterType
  profile_id: string
  inference_time_ms: number
  real_time_factor: number
  output_duration_sec: number
  sample_rate: number
  snr_db: number
  energy_ratio: number
  spectral_centroid_hz: number
  zero_crossing_rate: number
  adapter_params: number
  adapter_size_mb: number
}

interface ProfileReport {
  profile_id: string
  timestamp: string
  quality_winner: AdapterType | 'tie'
  speed_winner: AdapterType | 'tie'
  recommended: AdapterType
  notes: string[]
  hq?: QualityMetrics
  nvfp4?: QualityMetrics
}

interface QualityReport {
  generated_at: string
  total_profiles: number
  profiles: ProfileReport[]
  summary: {
    hq_quality_wins: number
    nvfp4_quality_wins: number
    hq_speed_wins: number
    nvfp4_speed_wins: number
    hq_recommended: number
    nvfp4_recommended: number
  }
}

interface QualityMetricsDashboardProps {
  onProfileSelect?: (profileId: string, adapter: AdapterType) => void
}

function MetricCard({
  label,
  value,
  icon: Icon,
  variant = 'default',
  subtext,
}: {
  label: string
  value: string | number
  icon: React.ElementType
  variant?: 'default' | 'hq' | 'nvfp4' | 'success' | 'warning'
  subtext?: string
}) {
  const variantClasses = {
    default: 'bg-gray-800 border-gray-700',
    hq: 'bg-violet-900/30 border-violet-500/50',
    nvfp4: 'bg-yellow-900/30 border-yellow-500/50',
    success: 'bg-green-900/30 border-green-500/50',
    warning: 'bg-amber-900/30 border-amber-500/50',
  }

  const iconClasses = {
    default: 'text-gray-400',
    hq: 'text-violet-400',
    nvfp4: 'text-yellow-400',
    success: 'text-green-400',
    warning: 'text-amber-400',
  }

  return (
    <div className={clsx('rounded-lg border p-4', variantClasses[variant])}>
      <div className="flex items-center gap-2 mb-2">
        <Icon size={16} className={iconClasses[variant]} />
        <span className="text-sm text-gray-400">{label}</span>
      </div>
      <div className="text-2xl font-semibold text-white">{value}</div>
      {subtext && <div className="text-xs text-gray-500 mt-1">{subtext}</div>}
    </div>
  )
}

function ProfileCard({
  report,
  onSelect,
}: {
  report: ProfileReport
  onSelect?: (profileId: string, adapter: AdapterType) => void
}) {
  const shortId = report.profile_id.slice(0, 8)

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-white">{shortId}...</h3>
          <p className="text-xs text-gray-500">{report.timestamp}</p>
        </div>
        <div className="flex items-center gap-2">
          {report.recommended === 'hq' ? (
            <span className="inline-flex items-center gap-1 px-2 py-1 bg-violet-600/30 text-violet-300 text-xs rounded">
              <Sparkles size={12} />
              HQ Recommended
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 px-2 py-1 bg-yellow-600/30 text-yellow-300 text-xs rounded">
              <Zap size={12} />
              nvfp4 Recommended
            </span>
          )}
        </div>
      </div>

      {/* Metrics comparison */}
      <div className="grid grid-cols-2 gap-4">
        {/* HQ metrics */}
        {report.hq && (
          <div
            className={clsx(
              'p-3 rounded border cursor-pointer transition-all',
              report.quality_winner === 'hq'
                ? 'border-violet-500 bg-violet-500/10'
                : 'border-gray-700 hover:border-gray-600'
            )}
            onClick={() => onSelect?.(report.profile_id, 'hq')}
          >
            <div className="flex items-center gap-2 mb-2">
              <Sparkles size={14} className="text-violet-400" />
              <span className="text-sm font-medium text-violet-300">HQ</span>
              {report.quality_winner === 'hq' && (
                <Trophy size={12} className="text-violet-400 ml-auto" />
              )}
            </div>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-500">Size:</span>
                <span className="text-gray-300">{report.hq.adapter_size_mb.toFixed(2)} MB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">RTF:</span>
                <span className="text-gray-300">{report.hq.real_time_factor.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">SNR:</span>
                <span className="text-gray-300">{report.hq.snr_db.toFixed(1)} dB</span>
              </div>
            </div>
          </div>
        )}

        {/* nvfp4 metrics */}
        {report.nvfp4 && (
          <div
            className={clsx(
              'p-3 rounded border cursor-pointer transition-all',
              report.speed_winner === 'nvfp4'
                ? 'border-yellow-500 bg-yellow-500/10'
                : 'border-gray-700 hover:border-gray-600'
            )}
            onClick={() => onSelect?.(report.profile_id, 'nvfp4')}
          >
            <div className="flex items-center gap-2 mb-2">
              <Zap size={14} className="text-yellow-400" />
              <span className="text-sm font-medium text-yellow-300">nvfp4</span>
              {report.speed_winner === 'nvfp4' && (
                <Trophy size={12} className="text-yellow-400 ml-auto" />
              )}
            </div>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-500">Size:</span>
                <span className="text-gray-300">{report.nvfp4.adapter_size_mb.toFixed(2)} MB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">RTF:</span>
                <span className="text-gray-300">{report.nvfp4.real_time_factor.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">SNR:</span>
                <span className="text-gray-300">{report.nvfp4.snr_db.toFixed(1)} dB</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Notes */}
      {report.notes.length > 0 && (
        <div className="text-xs text-gray-500 space-y-0.5">
          {report.notes.map((note, i) => (
            <p key={i}>• {note}</p>
          ))}
        </div>
      )}
    </div>
  )
}

export function QualityMetricsDashboard({ onProfileSelect }: QualityMetricsDashboardProps) {
  const [report, setReport] = useState<QualityReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadReport = async () => {
    try {
      setLoading(true)
      // In a real implementation, this would fetch from the API
      // For now, we fetch the static report file
      const response = await fetch('/reports/quality_validation.json')
      if (!response.ok) {
        throw new Error('Report not found')
      }
      const data = await response.json()
      setReport(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load report')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadReport()
  }, [])

  if (loading) {
    return (
      <div className="bg-gray-900 rounded-lg p-8 flex items-center justify-center">
        <Loader2 className="animate-spin text-gray-400 mr-2" size={24} />
        <span className="text-gray-400">Loading quality report...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-gray-900 rounded-lg p-6">
        <div className="flex items-center justify-center gap-2 text-amber-400 mb-4">
          <AlertCircle size={20} />
          <span>No quality report available</span>
        </div>
        <p className="text-sm text-gray-500 text-center mb-4">
          Run the quality validation script to generate a report:
        </p>
        <pre className="bg-gray-800 rounded p-3 text-xs text-gray-400 overflow-x-auto">
          python scripts/quality_validation.py --all-profiles --report
        </pre>
        <button
          onClick={loadReport}
          className="mt-4 w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded text-gray-300 transition-colors"
        >
          <RefreshCw size={16} />
          Retry
        </button>
      </div>
    )
  }

  if (!report) {
    return null
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <BarChart3 size={24} className="text-blue-400" />
            Quality Metrics Dashboard
          </h2>
          <p className="text-sm text-gray-500">
            Generated: {report.generated_at} • {report.total_profiles} profiles
          </p>
        </div>
        <button
          onClick={loadReport}
          className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-sm text-gray-300 transition-colors"
        >
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="HQ Quality Wins"
          value={report.summary.hq_quality_wins}
          icon={Sparkles}
          variant="hq"
          subtext={`${report.total_profiles - report.summary.hq_quality_wins} ties/losses`}
        />
        <MetricCard
          label="nvfp4 Speed Wins"
          value={report.summary.nvfp4_speed_wins}
          icon={Zap}
          variant="nvfp4"
          subtext={`${report.total_profiles - report.summary.nvfp4_speed_wins} ties/losses`}
        />
        <MetricCard
          label="HQ Recommended"
          value={report.summary.hq_recommended}
          icon={CheckCircle}
          variant={report.summary.hq_recommended > report.summary.nvfp4_recommended ? 'success' : 'default'}
        />
        <MetricCard
          label="nvfp4 Recommended"
          value={report.summary.nvfp4_recommended}
          icon={Timer}
          variant={report.summary.nvfp4_recommended > report.summary.hq_recommended ? 'success' : 'default'}
        />
      </div>

      {/* Recommendation summary */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Overall Recommendation</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center gap-3 p-3 rounded bg-violet-900/20 border border-violet-500/30">
            <Sparkles size={24} className="text-violet-400" />
            <div>
              <p className="font-medium text-violet-300">Use HQ for</p>
              <p className="text-sm text-gray-400">Studio recording, offline processing</p>
            </div>
          </div>
          <div className="flex items-center gap-3 p-3 rounded bg-yellow-900/20 border border-yellow-500/30">
            <Zap size={24} className="text-yellow-400" />
            <div>
              <p className="font-medium text-yellow-300">Use nvfp4 for</p>
              <p className="text-sm text-gray-400">Live karaoke, real-time streaming</p>
            </div>
          </div>
        </div>
      </div>

      {/* Profile cards */}
      <div>
        <h3 className="text-sm font-medium text-gray-400 mb-3">Profile Results</h3>
        <div className="grid gap-4">
          {report.profiles.map((profile) => (
            <ProfileCard
              key={profile.profile_id}
              report={profile}
              onSelect={onProfileSelect}
            />
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="text-xs text-gray-500 space-y-1">
        <p>
          <strong>RTF</strong> (Real-Time Factor): &lt; 1.0 = faster than real-time,
          &lt; 0.5 = suitable for live streaming
        </p>
        <p>
          <strong>SNR</strong> (Signal-to-Noise Ratio): Higher = cleaner audio output
        </p>
      </div>
    </div>
  )
}
