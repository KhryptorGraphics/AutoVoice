import { Music, Users, Sparkles, Volume2, Loader2, AlertCircle } from 'lucide-react'
import { QualityMetrics } from '../services/api'

interface QualityMetricsDisplayProps {
  metrics: QualityMetrics | null
  targets?: {
    min_pitch_accuracy_correlation?: number
    max_pitch_accuracy_rmse_hz?: number
    min_speaker_similarity?: number
    max_spectral_distortion?: number
    min_stoi_score?: number
    min_pesq_score?: number
    min_mos_estimate?: number
  }
  isLoading?: boolean
  error?: string | null
}

type MetricStatus = 'success' | 'warning' | 'error'

export function QualityMetricsDisplay({ metrics, targets, isLoading, error }: QualityMetricsDisplayProps) {
  // Helper function to determine metric status based on thresholds
  const getMetricStatus = (
    value: number,
    goodThreshold: number,
    warningThreshold: number,
    higherIsBetter: boolean = true
  ): MetricStatus => {
    if (higherIsBetter) {
      if (value >= goodThreshold) return 'success'
      if (value >= warningThreshold) return 'warning'
      return 'error'
    } else {
      if (value <= goodThreshold) return 'success'
      if (value <= warningThreshold) return 'warning'
      return 'error'
    }
  }

  // Helper function to calculate progress percentage
  const getProgressPercentage = (value: number, min: number, max: number): number => {
    return Math.min(Math.max(((value - min) / (max - min)) * 100, 0), 100)
  }

  // Helper function to format metric values
  const formatMetricValue = (value: number, decimals: number = 2): string => {
    return value.toFixed(decimals)
  }

  // Helper function to get status color classes
  const getStatusColor = (status: MetricStatus) => {
    switch (status) {
      case 'success':
        return 'bg-green-500 text-green-900 border-green-600'
      case 'warning':
        return 'bg-yellow-500 text-yellow-900 border-yellow-600'
      case 'error':
        return 'bg-red-500 text-red-900 border-red-600'
    }
  }

  // Helper function to get progress bar color
  const getProgressColor = (status: MetricStatus) => {
    switch (status) {
      case 'success':
        return 'bg-green-600'
      case 'warning':
        return 'bg-yellow-600'
      case 'error':
        return 'bg-red-600'
    }
  }

  // Metric Card Component
  const MetricCard = ({
    icon: Icon,
    title,
    metrics: metricValues,
  }: {
    icon: any
    title: string
    metrics: Array<{
      name: string
      value: number
      unit: string
      status: MetricStatus
      targetText: string
      progress?: number
      tooltip?: string
    }>
  }) => (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Icon className="w-6 h-6 text-primary-600" />
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      </div>
      <div className="space-y-3">
        {metricValues.map((metric, idx) => (
          <div key={idx}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-gray-600" title={metric.tooltip}>
                {metric.name}
              </span>
              <span className="text-sm font-semibold">
                {formatMetricValue(metric.value)} {metric.unit}
              </span>
            </div>
            {metric.progress !== undefined && (
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden mb-1">
                <div
                  className={`h-full transition-all ${getProgressColor(metric.status)}`}
                  style={{ width: `${metric.progress}%` }}
                />
              </div>
            )}
            <div className="flex items-center justify-between">
              <span className={`text-xs px-2 py-0.5 rounded ${getStatusColor(metric.status)}`}>
                {metric.status === 'success' ? '✓' : metric.status === 'warning' ? '⚠' : '✗'}{' '}
                {metric.targetText}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )

  // Loading State
  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-center space-x-2 text-gray-600">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span>Calculating quality metrics...</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="bg-gray-200 rounded-lg h-40 animate-pulse" />
          ))}
        </div>
      </div>
    )
  }

  // Error State
  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="bg-red-50 border-2 border-red-500 rounded-lg p-4">
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-5 h-5 text-red-800 mt-0.5" />
            <div className="flex-1">
              <h4 className="font-semibold text-red-800">Failed to load quality metrics</h4>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // No Metrics Available
  if (!metrics) {
    // Only show informational panel when conversion completed (not loading, no error)
    if (!isLoading && !error) {
      return (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Quality Metrics</h2>
          <div className="bg-gray-50 border-2 border-gray-300 rounded-lg p-4">
            <p className="text-gray-600 text-center">
              No quality metrics are available for this conversion
            </p>
          </div>
        </div>
      )
    }
    return null
  }

  // Calculate metric statuses and prepare display data
  const pitchMetrics = metrics.pitch_accuracy
    ? [
        {
          name: 'Pitch RMSE',
          value: metrics.pitch_accuracy.rmse_hz,
          unit: 'Hz',
          status: getMetricStatus(metrics.pitch_accuracy.rmse_hz, targets?.max_pitch_accuracy_rmse_hz || 10, (targets?.max_pitch_accuracy_rmse_hz || 10) + 5, false),
          targetText: `Target: < ${targets?.max_pitch_accuracy_rmse_hz || 10} Hz`,
          progress: getProgressPercentage((targets?.max_pitch_accuracy_rmse_hz || 10)*2 - metrics.pitch_accuracy.rmse_hz, 0, (targets?.max_pitch_accuracy_rmse_hz || 10)*2),
          tooltip: 'Root Mean Square Error of pitch frequency',
        },
        {
          name: 'Pitch Correlation',
          value: metrics.pitch_accuracy.correlation,
          unit: '',
          status: getMetricStatus(metrics.pitch_accuracy.correlation, targets?.min_pitch_accuracy_correlation || 0.8, (targets?.min_pitch_accuracy_correlation || 0.8) - 0.1, true),
          targetText: `Target: ≥ ${targets?.min_pitch_accuracy_correlation || 0.8}`,
          progress: getProgressPercentage(metrics.pitch_accuracy.correlation, 0, 1),
          tooltip: 'Correlation between original and converted pitch contours',
        },
        {
          name: 'Mean Error',
          value: metrics.pitch_accuracy.mean_error_cents,
          unit: 'cents',
          status: getMetricStatus(Math.abs(metrics.pitch_accuracy.mean_error_cents), 50, 100, false),
          targetText: 'Target: < 50 cents',
          progress: getProgressPercentage(
            150 - Math.abs(metrics.pitch_accuracy.mean_error_cents),
            0,
            150
          ),
          tooltip: 'Average pitch deviation in cents (100 cents = 1 semitone)',
        },
      ]
    : []

  const speakerMetrics = metrics.speaker_similarity
    ? [
        {
          name: 'Cosine Similarity',
          value: metrics.speaker_similarity.cosine_similarity,
          unit: '',
          status: getMetricStatus(metrics.speaker_similarity.cosine_similarity, targets?.min_speaker_similarity || 0.85, (targets?.min_speaker_similarity || 0.85) - 0.1, true),
          targetText: `Target: ≥ ${targets?.min_speaker_similarity || 0.85}`,
          progress: getProgressPercentage(metrics.speaker_similarity.cosine_similarity, 0, 1),
          tooltip: 'Similarity between speaker embeddings (1.0 = identical)',
        },
        {
          name: 'Embedding Distance',
          value: metrics.speaker_similarity.embedding_distance,
          unit: '',
          status: getMetricStatus(metrics.speaker_similarity.embedding_distance, 0.3, 0.5, false),
          targetText: 'Target: < 0.3',
          progress: getProgressPercentage(1.0 - metrics.speaker_similarity.embedding_distance, 0, 1),
          tooltip: 'Euclidean distance between speaker embeddings (lower is better)',
        },
      ]
    : []

  const naturalnessMetrics = metrics.naturalness
    ? [
        {
          name: 'MOS Estimate',
          value: metrics.naturalness.mos_estimate,
          unit: '',
          status: getMetricStatus(metrics.naturalness.mos_estimate, targets?.min_mos_estimate || 4.0, (targets?.min_mos_estimate || 4.0) - 0.5, true),
          targetText: `Target: ≥ ${targets?.min_mos_estimate || 4.0}`,
          progress: getProgressPercentage(metrics.naturalness.mos_estimate, 1, 5),
          tooltip: 'Mean Opinion Score estimate (1-5 scale, 5 = excellent)',
        },
        {
          name: 'Spectral Distortion',
          value: metrics.naturalness.spectral_distortion,
          unit: 'dB',
          status: getMetricStatus(metrics.naturalness.spectral_distortion, targets?.max_spectral_distortion || 10, (targets?.max_spectral_distortion || 10) + 5, false),
          targetText: `Target: < ${targets?.max_spectral_distortion || 10} dB`,
          progress: getProgressPercentage((targets?.max_spectral_distortion || 10)*2.5 - metrics.naturalness.spectral_distortion, 0, (targets?.max_spectral_distortion || 10)*2.5),
          tooltip: 'Spectral distance between original and converted audio',
        },
      ]
    : []

  const intelligibilityMetrics = metrics.intelligibility
    ? [
        {
          name: 'STOI Score',
          value: metrics.intelligibility.stoi,
          unit: '',
          status: getMetricStatus(metrics.intelligibility.stoi, targets?.min_stoi_score || 0.9, (targets?.min_stoi_score || 0.9) - 0.1, true),
          targetText: `Target: ≥ ${targets?.min_stoi_score || 0.9}`,
          progress: getProgressPercentage(metrics.intelligibility.stoi, 0, 1),
          tooltip: 'Short-Time Objective Intelligibility (higher = more intelligible)',
        },
        {
          name: 'PESQ Score',
          value: metrics.intelligibility.pesq,
          unit: '',
          status: getMetricStatus(metrics.intelligibility.pesq, targets?.min_pesq_score || 2.0, (targets?.min_pesq_score || 2.0) - 0.5, true),
          targetText: `Target: ≥ ${targets?.min_pesq_score || 2.0}`,
          progress: getProgressPercentage(metrics.intelligibility.pesq, -0.5, 4.5),
          tooltip: 'Perceptual Evaluation of Speech Quality (-0.5 to 4.5 scale)',
        },
      ]
    : []

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-6">Quality Metrics</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {pitchMetrics.length > 0 && (
          <MetricCard icon={Music} title="Pitch Accuracy" metrics={pitchMetrics} />
        )}
        {speakerMetrics.length > 0 && (
          <MetricCard icon={Users} title="Speaker Similarity" metrics={speakerMetrics} />
        )}
        {naturalnessMetrics.length > 0 && (
          <MetricCard icon={Sparkles} title="Naturalness" metrics={naturalnessMetrics} />
        )}
        {intelligibilityMetrics.length > 0 && (
          <MetricCard icon={Volume2} title="Intelligibility" metrics={intelligibilityMetrics} />
        )}
      </div>
    </div>
  )
}
