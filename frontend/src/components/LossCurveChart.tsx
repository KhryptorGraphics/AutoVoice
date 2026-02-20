import { useMemo } from 'react'
import { TrendingDown } from 'lucide-react'
import { TrainingJob } from '../services/api'

interface LossCurveChartProps {
  job: TrainingJob
  height?: number
}

interface Point {
  x: number
  y: number
}

function formatLoss(value: number): string {
  if (value < 0.001) return value.toExponential(2)
  return value.toFixed(4)
}

export function LossCurveChart({ job, height = 200 }: LossCurveChartProps) {
  const { points, minLoss, maxLoss, improvement } = useMemo(() => {
    const lossCurve = job.results?.loss_curve || []
    if (lossCurve.length === 0) {
      return { points: [], minLoss: 0, maxLoss: 1, improvement: 0 }
    }

    const min = Math.min(...lossCurve)
    const max = Math.max(...lossCurve)
    const padding = (max - min) * 0.1 || 0.1 // 10% padding

    const pts: Point[] = lossCurve.map((loss, i) => ({
      x: (i / (lossCurve.length - 1 || 1)) * 100,
      y: ((max + padding - loss) / (max - min + 2 * padding)) * 100,
    }))

    const initial = job.results?.initial_loss || lossCurve[0]
    const final = job.results?.final_loss || lossCurve[lossCurve.length - 1]
    const imp = initial > 0 ? ((initial - final) / initial) * 100 : 0

    return {
      points: pts,
      minLoss: min,
      maxLoss: max,
      improvement: imp,
    }
  }, [job.results])

  const pathD = useMemo(() => {
    if (points.length < 2) return ''
    return points.reduce((path, point, i) => {
      if (i === 0) return `M ${point.x} ${point.y}`
      return `${path} L ${point.x} ${point.y}`
    }, '')
  }, [points])

  const hasData = points.length > 0

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingDown size={18} className="text-gray-400" />
          <h3 className="font-semibold">Training Loss</h3>
        </div>
        {hasData && improvement > 0 && (
          <span className="text-xs text-green-400 bg-green-900/30 px-2 py-0.5 rounded">
            -{improvement.toFixed(1)}% improvement
          </span>
        )}
      </div>

      {!hasData ? (
        <div className="flex items-center justify-center text-gray-500" style={{ height }}>
          {job.status === 'pending' ? 'Waiting to start...' :
           job.status === 'running' ? 'Collecting data...' :
           'No loss data available'}
        </div>
      ) : (
        <div className="relative" style={{ height }}>
          {/* Y-axis labels */}
          <div className="absolute left-0 top-0 bottom-0 w-12 flex flex-col justify-between text-xs text-gray-500 font-mono">
            <span>{formatLoss(maxLoss)}</span>
            <span>{formatLoss(minLoss)}</span>
          </div>

          {/* Chart area */}
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            className="absolute left-14 right-0 top-0 bottom-4"
            style={{ height: height - 16 }}
          >
            {/* Grid lines */}
            <defs>
              <pattern id="grid" width="10" height="20" patternUnits="userSpaceOnUse">
                <path d="M 10 0 L 0 0 0 20" fill="none" stroke="rgba(75, 85, 99, 0.3)" strokeWidth="0.5" />
              </pattern>
            </defs>
            <rect width="100" height="100" fill="url(#grid)" />

            {/* Loss curve */}
            <path
              d={pathD}
              fill="none"
              stroke="url(#lossGradient)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />

            {/* Gradient definition */}
            <defs>
              <linearGradient id="lossGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#ef4444" />
                <stop offset="50%" stopColor="#eab308" />
                <stop offset="100%" stopColor="#22c55e" />
              </linearGradient>
            </defs>

            {/* Start and end points */}
            {points.length > 0 && (
              <>
                <circle
                  cx={points[0].x}
                  cy={points[0].y}
                  r="3"
                  fill="#ef4444"
                  vectorEffect="non-scaling-stroke"
                />
                <circle
                  cx={points[points.length - 1].x}
                  cy={points[points.length - 1].y}
                  r="3"
                  fill="#22c55e"
                  vectorEffect="non-scaling-stroke"
                />
              </>
            )}
          </svg>

          {/* X-axis labels */}
          <div className="absolute left-14 right-0 bottom-0 flex justify-between text-xs text-gray-500">
            <span>0</span>
            <span>Epoch</span>
            <span>{job.results?.loss_curve?.length || 0}</span>
          </div>
        </div>
      )}

      {/* Stats row */}
      {hasData && (
        <div className="grid grid-cols-3 gap-4 pt-2 border-t border-gray-700">
          <div>
            <div className="text-xs text-gray-500">Initial Loss</div>
            <div className="font-mono text-red-400">
              {formatLoss(job.results?.initial_loss || 0)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Final Loss</div>
            <div className="font-mono text-green-400">
              {formatLoss(job.results?.final_loss || 0)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Data Points</div>
            <div className="font-mono text-gray-300">
              {job.results?.loss_curve?.length || 0}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Compact inline version for job cards
export function LossCurveMini({ lossCurve }: { lossCurve?: number[] }) {
  if (!lossCurve || lossCurve.length < 2) return null

  const min = Math.min(...lossCurve)
  const max = Math.max(...lossCurve)
  const range = max - min || 1

  const points = lossCurve.map((loss, i) => {
    const x = (i / (lossCurve.length - 1)) * 100
    const y = ((max - loss) / range) * 100
    return `${x},${y}`
  }).join(' ')

  return (
    <svg viewBox="0 0 100 100" className="w-24 h-8" preserveAspectRatio="none">
      <polyline
        points={points}
        fill="none"
        stroke="#22c55e"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  )
}
