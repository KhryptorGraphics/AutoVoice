import { TrendingUp, Volume2, Mic, Activity } from 'lucide-react'
import { QualityMetrics as QualityMetricsType } from '../services/api'

interface QualityMetricsProps {
  metrics: QualityMetricsType
  className?: string
}

export function QualityMetrics({ metrics, className = '' }: QualityMetricsProps) {
  return (
    <div className={`bg-white rounded-lg shadow-lg p-6 ${className}`}>
      <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
        <Activity className="w-5 h-5 text-purple-600" />
        <span>Quality Metrics</span>
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Pitch Accuracy */}
        {metrics.pitch_accuracy && (
          <MetricCard
            icon={<TrendingUp className="w-5 h-5" />}
            title="Pitch Accuracy"
            metrics={[
              {
                label: 'RMSE',
                value: `${metrics.pitch_accuracy.rmse_hz.toFixed(2)} Hz`,
                quality: getQualityLevel(metrics.pitch_accuracy.rmse_hz, 15, 5, true),
              },
              {
                label: 'Correlation',
                value: metrics.pitch_accuracy.correlation.toFixed(3),
                quality: getQualityLevel(metrics.pitch_accuracy.correlation, 0.8, 0.95),
              },
              {
                label: 'Mean Error',
                value: `${metrics.pitch_accuracy.mean_error_cents.toFixed(1)} cents`,
                quality: getQualityLevel(Math.abs(metrics.pitch_accuracy.mean_error_cents), 20, 5, true),
              },
            ]}
          />
        )}

        {/* Speaker Similarity */}
        {metrics.speaker_similarity && (
          <MetricCard
            icon={<Mic className="w-5 h-5" />}
            title="Speaker Similarity"
            metrics={[
              {
                label: 'Cosine Similarity',
                value: metrics.speaker_similarity.cosine_similarity.toFixed(3),
                quality: getQualityLevel(metrics.speaker_similarity.cosine_similarity, 0.7, 0.9),
              },
              {
                label: 'Embedding Distance',
                value: metrics.speaker_similarity.embedding_distance.toFixed(3),
                quality: getQualityLevel(metrics.speaker_similarity.embedding_distance, 0.5, 0.2, true),
              },
            ]}
          />
        )}

        {/* Naturalness */}
        {metrics.naturalness && (
          <MetricCard
            icon={<Volume2 className="w-5 h-5" />}
            title="Naturalness"
            metrics={[
              {
                label: 'Spectral Distortion',
                value: `${metrics.naturalness.spectral_distortion.toFixed(2)} dB`,
                quality: getQualityLevel(metrics.naturalness.spectral_distortion, 8, 4, true),
              },
              {
                label: 'MOS Estimate',
                value: `${metrics.naturalness.mos_estimate.toFixed(2)} / 5.0`,
                quality: getQualityLevel(metrics.naturalness.mos_estimate, 3.0, 4.0),
              },
            ]}
          />
        )}

        {/* Intelligibility */}
        {metrics.intelligibility && (
          <MetricCard
            icon={<Activity className="w-5 h-5" />}
            title="Intelligibility"
            metrics={[
              {
                label: 'STOI',
                value: metrics.intelligibility.stoi.toFixed(3),
                quality: getQualityLevel(metrics.intelligibility.stoi, 0.7, 0.9),
              },
              {
                label: 'PESQ',
                value: `${metrics.intelligibility.pesq.toFixed(2)} / 4.5`,
                quality: getQualityLevel(metrics.intelligibility.pesq, 2.5, 3.5),
              },
            ]}
          />
        )}
      </div>
    </div>
  )
}

interface MetricCardProps {
  icon: React.ReactNode
  title: string
  metrics: {
    label: string
    value: string
    quality: 'excellent' | 'good' | 'fair' | 'poor'
  }[]
}

function MetricCard({ icon, title, metrics }: MetricCardProps) {
  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex items-center space-x-2 mb-3">
        <div className="text-purple-600">{icon}</div>
        <h4 className="font-semibold text-gray-900">{title}</h4>
      </div>
      <div className="space-y-2">
        {metrics.map((metric, index) => (
          <div key={index} className="flex items-center justify-between">
            <span className="text-sm text-gray-600">{metric.label}:</span>
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium text-gray-900">{metric.value}</span>
              <QualityBadge quality={metric.quality} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function QualityBadge({ quality }: { quality: 'excellent' | 'good' | 'fair' | 'poor' }) {
  const colors = {
    excellent: 'bg-green-100 text-green-800',
    good: 'bg-blue-100 text-blue-800',
    fair: 'bg-yellow-100 text-yellow-800',
    poor: 'bg-red-100 text-red-800',
  }

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors[quality]}`}>
      {quality}
    </span>
  )
}

function getQualityLevel(
  value: number,
  fairThreshold: number,
  goodThreshold: number,
  inverse: boolean = false
): 'excellent' | 'good' | 'fair' | 'poor' {
  if (inverse) {
    if (value <= goodThreshold) return 'excellent'
    if (value <= fairThreshold) return 'good'
    if (value <= fairThreshold * 1.5) return 'fair'
    return 'poor'
  } else {
    if (value >= goodThreshold) return 'excellent'
    if (value >= fairThreshold) return 'good'
    if (value >= fairThreshold * 0.7) return 'fair'
    return 'poor'
  }
}

