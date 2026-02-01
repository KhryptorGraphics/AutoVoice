/**
 * Quality Metrics Panel - Display voice conversion quality evaluation results
 *
 * Shows key metrics for evaluating voice conversion quality:
 * - Speaker Similarity (target: >0.85 cosine similarity)
 * - Pitch Accuracy (RMSE, correlation, mean error)
 * - Naturalness (spectral distortion, MOS estimate)
 * - Intelligibility (STOI, PESQ) when available
 */
import { useState, useEffect } from 'react'
import { Activity, Mic, Music, Headphones, CheckCircle, XCircle, AlertCircle, RefreshCw, Loader2 } from 'lucide-react'
import { QualityMetrics, apiService } from '../services/api'
import clsx from 'clsx'

interface QualityMetricsPanelProps {
  jobId?: string
  metrics?: QualityMetrics
  onRefresh?: () => void
  targetSimilarity?: number // Default: 0.85
}

interface GaugeProps {
  value: number
  max: number
  target?: number
  label: string
  format?: (v: number) => string
  colorScale?: 'default' | 'inverse' | 'centered'
}

function MetricGauge({ value, max, target, label, format = (v) => v.toFixed(2), colorScale = 'default' }: GaugeProps) {
  const percentage = Math.min((value / max) * 100, 100)
  const targetPercentage = target ? (target / max) * 100 : null
  const meetsTarget = target ? value >= target : true

  // Determine color based on scale type and value
  const getColor = () => {
    if (colorScale === 'inverse') {
      // Lower is better (e.g., RMSE, distortion)
      if (percentage < 30) return 'bg-green-500'
      if (percentage < 60) return 'bg-yellow-500'
      return 'bg-red-500'
    } else if (colorScale === 'centered') {
      // Closer to target is better
      const dist = Math.abs(percentage - (targetPercentage || 50))
      if (dist < 15) return 'bg-green-500'
      if (dist < 30) return 'bg-yellow-500'
      return 'bg-red-500'
    } else {
      // Higher is better (default)
      if (percentage > 85) return 'bg-green-500'
      if (percentage > 60) return 'bg-yellow-500'
      return 'bg-red-500'
    }
  }

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-gray-400">{label}</span>
        <span className={clsx(
          'font-mono',
          meetsTarget ? 'text-green-400' : target ? 'text-red-400' : 'text-gray-300'
        )}>
          {format(value)}
          {target && (
            <span className="text-xs text-gray-500 ml-1">
              (target: {format(target)})
            </span>
          )}
        </span>
      </div>
      <div className="relative h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={clsx('h-full transition-all duration-500 rounded-full', getColor())}
          style={{ width: `${percentage}%` }}
        />
        {targetPercentage && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white/50"
            style={{ left: `${targetPercentage}%` }}
          />
        )}
      </div>
    </div>
  )
}

function MetricCard({ title, icon: Icon, children, status }: {
  title: string
  icon: typeof Activity
  children: React.ReactNode
  status?: 'pass' | 'fail' | 'warning' | null
}) {
  const statusColors = {
    pass: 'border-green-500/30 bg-green-500/5',
    fail: 'border-red-500/30 bg-red-500/5',
    warning: 'border-yellow-500/30 bg-yellow-500/5',
  }

  const statusIcons = {
    pass: <CheckCircle size={16} className="text-green-400" />,
    fail: <XCircle size={16} className="text-red-400" />,
    warning: <AlertCircle size={16} className="text-yellow-400" />,
  }

  return (
    <div className={clsx(
      'bg-gray-800 rounded-lg p-4 border',
      status ? statusColors[status] : 'border-gray-700'
    )}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icon size={16} className="text-gray-400" />
          <h4 className="font-medium text-sm">{title}</h4>
        </div>
        {status && statusIcons[status]}
      </div>
      <div className="space-y-3">
        {children}
      </div>
    </div>
  )
}

export function QualityMetricsPanel({
  jobId,
  metrics: initialMetrics,
  onRefresh,
  targetSimilarity = 0.85,
}: QualityMetricsPanelProps) {
  const [metrics, setMetrics] = useState<QualityMetrics | null>(initialMetrics || null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchMetrics = async () => {
    if (!jobId) return
    setLoading(true)
    setError(null)
    try {
      const data = await apiService.getConversionMetrics(jobId)
      setMetrics(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load metrics')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (jobId && !initialMetrics) {
      fetchMetrics()
    }
  }, [jobId, initialMetrics])

  useEffect(() => {
    if (initialMetrics) {
      setMetrics(initialMetrics)
    }
  }, [initialMetrics])

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 flex items-center justify-center">
        <Loader2 className="animate-spin text-gray-400" size={32} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="text-center text-red-400">
          <AlertCircle className="mx-auto mb-2" size={32} />
          <p>{error}</p>
          {jobId && (
            <button
              onClick={fetchMetrics}
              className="mt-4 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
            >
              Retry
            </button>
          )}
        </div>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 text-center text-gray-500">
        <Activity className="mx-auto mb-2" size={32} />
        <p>No quality metrics available</p>
        <p className="text-xs mt-1">Metrics are generated after conversion completes</p>
      </div>
    )
  }

  // Determine overall status
  const speakerSimilarityPass = metrics.speaker_similarity.cosine_similarity >= targetSimilarity
  const pitchAccuracyGood = metrics.pitch_accuracy.correlation > 0.9
  const naturalnessGood = metrics.naturalness.mos_estimate >= 3.5

  const overallStatus = speakerSimilarityPass && pitchAccuracyGood && naturalnessGood
    ? 'pass'
    : speakerSimilarityPass
    ? 'warning'
    : 'fail'

  return (
    <div className="space-y-4">
      {/* Overall Summary */}
      <div className={clsx(
        'rounded-lg p-4 border flex items-center justify-between',
        overallStatus === 'pass' ? 'bg-green-500/10 border-green-500/30' :
        overallStatus === 'warning' ? 'bg-yellow-500/10 border-yellow-500/30' :
        'bg-red-500/10 border-red-500/30'
      )}>
        <div className="flex items-center gap-3">
          {overallStatus === 'pass' ? (
            <CheckCircle size={24} className="text-green-400" />
          ) : overallStatus === 'warning' ? (
            <AlertCircle size={24} className="text-yellow-400" />
          ) : (
            <XCircle size={24} className="text-red-400" />
          )}
          <div>
            <div className="font-semibold">
              {overallStatus === 'pass' ? 'Quality Target Met' :
               overallStatus === 'warning' ? 'Partial Quality' :
               'Below Quality Target'}
            </div>
            <div className="text-sm text-gray-400">
              Speaker similarity: {(metrics.speaker_similarity.cosine_similarity * 100).toFixed(1)}%
              {speakerSimilarityPass ? ' ✓' : ` (target: ${(targetSimilarity * 100).toFixed(0)}%)`}
            </div>
          </div>
        </div>
        {onRefresh && (
          <button
            onClick={() => { fetchMetrics(); onRefresh?.(); }}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg"
          >
            <RefreshCw size={16} />
          </button>
        )}
      </div>

      {/* Detailed Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Speaker Similarity */}
        <MetricCard
          title="Speaker Similarity"
          icon={Mic}
          status={speakerSimilarityPass ? 'pass' : 'fail'}
        >
          <MetricGauge
            value={metrics.speaker_similarity.cosine_similarity}
            max={1}
            target={targetSimilarity}
            label="Cosine Similarity"
            format={(v) => (v * 100).toFixed(1) + '%'}
          />
          <MetricGauge
            value={metrics.speaker_similarity.embedding_distance}
            max={2}
            label="Embedding Distance"
            format={(v) => v.toFixed(3)}
            colorScale="inverse"
          />
        </MetricCard>

        {/* Pitch Accuracy */}
        <MetricCard
          title="Pitch Accuracy"
          icon={Music}
          status={pitchAccuracyGood ? 'pass' : metrics.pitch_accuracy.correlation > 0.8 ? 'warning' : 'fail'}
        >
          <MetricGauge
            value={metrics.pitch_accuracy.correlation}
            max={1}
            target={0.9}
            label="Pitch Correlation"
            format={(v) => v.toFixed(3)}
          />
          <MetricGauge
            value={metrics.pitch_accuracy.rmse_hz}
            max={50}
            label="RMSE (Hz)"
            format={(v) => v.toFixed(1) + ' Hz'}
            colorScale="inverse"
          />
          <div className="flex justify-between text-xs">
            <span className="text-gray-500">Mean Error</span>
            <span className="text-gray-300 font-mono">
              {metrics.pitch_accuracy.mean_error_cents.toFixed(1)} cents
            </span>
          </div>
        </MetricCard>

        {/* Naturalness */}
        <MetricCard
          title="Naturalness"
          icon={Headphones}
          status={naturalnessGood ? 'pass' : metrics.naturalness.mos_estimate >= 3.0 ? 'warning' : 'fail'}
        >
          <MetricGauge
            value={metrics.naturalness.mos_estimate}
            max={5}
            target={3.5}
            label="MOS Estimate"
            format={(v) => v.toFixed(2)}
          />
          <MetricGauge
            value={metrics.naturalness.spectral_distortion}
            max={10}
            label="Spectral Distortion (dB)"
            format={(v) => v.toFixed(2) + ' dB'}
            colorScale="inverse"
          />
        </MetricCard>

        {/* Intelligibility (if available) */}
        {metrics.intelligibility && (
          <MetricCard
            title="Intelligibility"
            icon={Activity}
            status={metrics.intelligibility.stoi > 0.9 ? 'pass' : 'warning'}
          >
            <MetricGauge
              value={metrics.intelligibility.stoi}
              max={1}
              label="STOI"
              format={(v) => v.toFixed(3)}
            />
            <MetricGauge
              value={metrics.intelligibility.pesq}
              max={4.5}
              label="PESQ"
              format={(v) => v.toFixed(2)}
            />
          </MetricCard>
        )}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 text-xs text-gray-500 pt-2 border-t border-gray-700">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-green-500" />
          <span>Good</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-yellow-500" />
          <span>Acceptable</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-red-500" />
          <span>Needs Improvement</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-0.5 h-3 bg-white/50" />
          <span>Target</span>
        </div>
      </div>
    </div>
  )
}

// Compact summary badge for lists/cards
export function QualityScoreBadge({ similarity, target = 0.85 }: { similarity: number; target?: number }) {
  const passes = similarity >= target
  const percentage = (similarity * 100).toFixed(0)

  return (
    <span className={clsx(
      'inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium',
      passes ? 'bg-green-900/50 text-green-300' : 'bg-red-900/50 text-red-300'
    )}>
      {passes ? <CheckCircle size={12} /> : <XCircle size={12} />}
      {percentage}%
    </span>
  )
}

// Progress tracking for training - shows similarity improvement over epochs
export function SimilarityProgressChart({ history }: { history: { epoch: number; similarity: number }[] }) {
  if (history.length < 2) return null

  const target = 0.85
  const max = Math.max(...history.map(h => h.similarity), target) * 1.1
  const min = Math.min(...history.map(h => h.similarity)) * 0.9

  const points = history.map((h, i) => {
    const x = (i / (history.length - 1)) * 100
    const y = 100 - ((h.similarity - min) / (max - min)) * 100
    return `${x},${y}`
  }).join(' ')

  const targetY = 100 - ((target - min) / (max - min)) * 100

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-gray-400">Speaker Similarity Progress</span>
        <span className={clsx(
          'text-xs px-2 py-0.5 rounded',
          history[history.length - 1].similarity >= target
            ? 'bg-green-900/50 text-green-300'
            : 'bg-yellow-900/50 text-yellow-300'
        )}>
          Current: {(history[history.length - 1].similarity * 100).toFixed(1)}%
        </span>
      </div>
      <svg viewBox="0 0 100 100" className="w-full h-24" preserveAspectRatio="none">
        {/* Target line */}
        <line
          x1="0"
          y1={targetY}
          x2="100"
          y2={targetY}
          stroke="#22c55e"
          strokeWidth="1"
          strokeDasharray="4 4"
          opacity="0.5"
          vectorEffect="non-scaling-stroke"
        />
        {/* Progress line */}
        <polyline
          points={points}
          fill="none"
          stroke="#8b5cf6"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <div className="flex justify-between text-xs text-gray-500 mt-1">
        <span>Epoch 1</span>
        <span className="text-green-400">Target: 85%</span>
        <span>Epoch {history.length}</span>
      </div>
    </div>
  )
}
