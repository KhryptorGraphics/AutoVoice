import { CheckCircle2, Circle, Loader2, AlertCircle } from 'lucide-react'
import clsx from 'clsx'

export interface PipelineStage {
  id: string
  name: string
  progress: number
  status: 'pending' | 'processing' | 'complete' | 'error'
  message?: string
  duration?: number
}

interface ProgressDisplayProps {
  stages: PipelineStage[]
  overallProgress: number
  estimatedTimeRemaining?: number
}

export function ProgressDisplay({ stages, overallProgress, estimatedTimeRemaining }: ProgressDisplayProps) {
  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`
    const minutes = Math.floor(seconds / 60)
    const secs = Math.round(seconds % 60)
    return `${minutes}m ${secs}s`
  }

  const getStatusIcon = (status: PipelineStage['status']) => {
    switch (status) {
      case 'complete':
        return <CheckCircle2 className="w-5 h-5 text-green-500" />
      case 'processing':
        return <Loader2 className="w-5 h-5 text-primary-500 animate-spin" />
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />
      default:
        return <Circle className="w-5 h-5 text-gray-300" />
    }
  }

  return (
    <div className="space-y-6">
      {/* Overall Progress */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Processing Pipeline</h3>
          <div className="text-right">
            <p className="text-2xl font-bold text-primary-600">{Math.round(overallProgress)}%</p>
            {estimatedTimeRemaining !== undefined && estimatedTimeRemaining > 0 && (
              <p className="text-xs text-gray-500">~{formatTime(estimatedTimeRemaining)} remaining</p>
            )}
          </div>
        </div>
        
        <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-primary-500 to-accent-500 transition-all duration-500 ease-out"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
      </div>

      {/* Stage Details */}
      <div className="space-y-3">
        {stages.map((stage, index) => (
          <div
            key={stage.id}
            className={clsx(
              'p-4 rounded-lg border-2 transition-all',
              stage.status === 'processing' && 'border-primary-500 bg-primary-50/50',
              stage.status === 'complete' && 'border-green-500 bg-green-50/50',
              stage.status === 'error' && 'border-red-500 bg-red-50/50',
              stage.status === 'pending' && 'border-gray-200 bg-gray-50/50'
            )}
          >
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 mt-0.5">
                {getStatusIcon(stage.status)}
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <p className="text-sm font-semibold text-gray-900">
                      {index + 1}. {stage.name}
                    </p>
                    {stage.message && (
                      <p className="text-xs text-gray-600 mt-0.5">{stage.message}</p>
                    )}
                  </div>
                  <div className="text-right ml-4">
                    <p className={clsx(
                      'text-sm font-semibold',
                      stage.status === 'processing' && 'text-primary-600',
                      stage.status === 'complete' && 'text-green-600',
                      stage.status === 'error' && 'text-red-600',
                      stage.status === 'pending' && 'text-gray-400'
                    )}>
                      {stage.progress}%
                    </p>
                    {stage.duration !== undefined && stage.status === 'complete' && (
                      <p className="text-xs text-gray-500">{formatTime(stage.duration)}</p>
                    )}
                  </div>
                </div>

                {/* Stage Progress Bar */}
                {stage.status !== 'pending' && (
                  <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={clsx(
                        'h-full transition-all duration-300',
                        stage.status === 'processing' && 'bg-primary-500',
                        stage.status === 'complete' && 'bg-green-500',
                        stage.status === 'error' && 'bg-red-500'
                      )}
                      style={{ width: `${stage.progress}%` }}
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Processing Animation */}
      {overallProgress > 0 && overallProgress < 100 && (
        <div className="flex items-center justify-center space-x-2 text-sm text-gray-600">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span>Processing your song...</span>
        </div>
      )}

      {/* Completion Message */}
      {overallProgress === 100 && (
        <div className="flex items-center justify-center space-x-2 p-4 bg-green-50 border-2 border-green-500 rounded-lg">
          <CheckCircle2 className="w-5 h-5 text-green-600" />
          <span className="text-sm font-semibold text-green-900">
            Conversion complete! Your song is ready.
          </span>
        </div>
      )}
    </div>
  )
}

