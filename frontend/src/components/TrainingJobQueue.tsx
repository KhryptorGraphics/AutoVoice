import { useState, useEffect, useCallback } from 'react'
import { XCircle, RefreshCw, Clock, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { apiService, TrainingJob, wsManager } from '../services/api'
import clsx from 'clsx'

interface TrainingJobQueueProps {
  profileId?: string
  onJobSelect?: (job: TrainingJob) => void
}

const statusConfig: Record<TrainingJob['status'], { icon: typeof Clock; color: string; bg: string; label: string; animate?: boolean }> = {
  pending: { icon: Clock, color: 'text-yellow-400', bg: 'bg-yellow-500', label: 'Pending' },
  running: { icon: Loader2, color: 'text-blue-400', bg: 'bg-blue-500', label: 'Running', animate: true },
  completed: { icon: CheckCircle, color: 'text-green-400', bg: 'bg-green-500', label: 'Completed' },
  failed: { icon: AlertCircle, color: 'text-red-400', bg: 'bg-red-500', label: 'Failed' },
  cancelled: { icon: XCircle, color: 'text-gray-400', bg: 'bg-gray-500', label: 'Cancelled' },
}

function JobStatusBadge({ status, isPaused }: { status: TrainingJob['status']; isPaused?: boolean }) {
  if (status === 'running' && isPaused) {
    return (
      <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded text-amber-300 bg-amber-500/20">
        <Clock size={12} />
        Paused
      </span>
    )
  }

  const config = statusConfig[status]
  const Icon = config.icon

  return (
    <span className={clsx('inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded', config.color, 'bg-opacity-20', config.bg.replace('bg-', 'bg-') + '/20')}>
      <Icon size={12} className={config.animate ? 'animate-spin' : ''} />
      {config.label}
    </span>
  )
}

function JobProgressBar({ job }: { job: TrainingJob }) {
  const progress = job.status === 'completed' ? 100 : job.progress
  const config = statusConfig[job.status]

  return (
    <div className="space-y-1">
      <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={clsx('h-full transition-all duration-300', config.bg)}
          style={{ width: `${progress}%` }}
        />
      </div>
      {job.status === 'running' && (
        <div className="text-xs text-gray-500">{job.progress}% complete</div>
      )}
    </div>
  )
}

function JobCard({ job, onCancel, onSelect }: { job: TrainingJob; onCancel: (id: string) => void; onSelect?: (job: TrainingJob) => void }) {
  const [cancelling, setCancelling] = useState(false)

  const handleCancel = async (e: React.MouseEvent) => {
    e.stopPropagation()
    setCancelling(true)
    try {
      await onCancel(job.job_id)
    } finally {
      setCancelling(false)
    }
  }

  const canCancel = job.status === 'pending' || job.status === 'running'

  return (
    <div
      onClick={() => onSelect?.(job)}
      data-testid="training-job-card"
      className={clsx(
        'bg-gray-750 rounded-lg p-4 space-y-3 border border-gray-700',
        onSelect && 'cursor-pointer hover:border-gray-600 transition-colors'
      )}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="font-medium text-sm">Job {job.job_id.slice(0, 8)}</div>
          <div className="text-xs text-gray-500">
            {new Date(job.created_at).toLocaleString()}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <JobStatusBadge status={job.status} isPaused={job.is_paused} />
          {canCancel && (
            <button
              onClick={handleCancel}
              disabled={cancelling}
              className="p-1 text-gray-400 hover:text-red-400 hover:bg-gray-700 rounded transition-colors disabled:opacity-50"
              title="Cancel job"
            >
              {cancelling ? <Loader2 size={14} className="animate-spin" /> : <XCircle size={14} />}
            </button>
          )}
        </div>
      </div>

      <JobProgressBar job={job} />

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <span className="text-gray-500">Samples:</span>{' '}
          <span className="text-gray-300">{job.sample_ids.length}</span>
        </div>
        {job.started_at && (
          <div>
            <span className="text-gray-500">Started:</span>{' '}
            <span className="text-gray-300">
              {new Date(job.started_at).toLocaleTimeString()}
            </span>
          </div>
        )}
        {job.is_paused && (
          <div>
            <span className="text-gray-500">State:</span>{' '}
            <span className="text-amber-300">Paused</span>
          </div>
        )}
        {job.completed_at && (
          <div>
            <span className="text-gray-500">Completed:</span>{' '}
            <span className="text-gray-300">
              {new Date(job.completed_at).toLocaleTimeString()}
            </span>
          </div>
        )}
        {job.results?.final_loss !== undefined && (
          <div>
            <span className="text-gray-500">Final Loss:</span>{' '}
            <span className="text-gray-300">{job.results.final_loss.toFixed(4)}</span>
          </div>
        )}
      </div>

      {job.error && (
        <div className="text-xs text-red-400 bg-red-900/20 rounded p-2">
          {job.error}
        </div>
      )}
    </div>
  )
}

export function TrainingJobQueue({ profileId, onJobSelect }: TrainingJobQueueProps) {
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'active' | 'completed'>('all')

  const fetchJobs = useCallback(async () => {
    try {
      const data = await apiService.listTrainingJobs(profileId)
      setJobs(data)
    } catch (error) {
      console.error('Failed to fetch training jobs:', error)
    } finally {
      setLoading(false)
    }
  }, [profileId])

  useEffect(() => {
    fetchJobs()

    // Set up WebSocket listener for job updates
    wsManager.connect()
    const unsubProgress = wsManager.subscribe('training_progress', () => {
      fetchJobs() // Refresh on progress update
    })
    const unsubComplete = wsManager.subscribe('training_complete', () => {
      fetchJobs()
    })
    const unsubError = wsManager.subscribe('training_error', () => {
      fetchJobs()
    })
    const unsubPaused = wsManager.subscribe('training_paused', () => {
      fetchJobs()
    })
    const unsubResumed = wsManager.subscribe('training_resumed', () => {
      fetchJobs()
    })
    const unsubCancelled = wsManager.subscribe('training_cancelled', () => {
      fetchJobs()
    })

    return () => {
      unsubProgress()
      unsubComplete()
      unsubError()
      unsubPaused()
      unsubResumed()
      unsubCancelled()
    }
  }, [fetchJobs])

  const handleCancel = async (jobId: string) => {
    try {
      await apiService.cancelTrainingJob(jobId)
      fetchJobs()
    } catch (error) {
      console.error('Failed to cancel job:', error)
    }
  }

  const filteredJobs = jobs.filter(job => {
    if (filter === 'active') return job.status === 'pending' || job.status === 'running'
    if (filter === 'completed') return job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'
    return true
  })

  const activeCount = jobs.filter(j => j.status === 'pending' || j.status === 'running').length

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold">Training Jobs</h3>
          {activeCount > 0 && (
            <span className="px-2 py-0.5 text-xs bg-blue-600 rounded-full">
              {activeCount} active
            </span>
          )}
        </div>
        <button
          onClick={fetchJobs}
          className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
          title="Refresh"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-1 p-1 bg-gray-750 rounded-lg">
        {(['all', 'active', 'completed'] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={clsx(
              'flex-1 px-3 py-1.5 text-xs font-medium rounded transition-colors capitalize',
              filter === f
                ? 'bg-gray-700 text-white'
                : 'text-gray-400 hover:text-white'
            )}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Job list */}
      {loading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="animate-spin text-gray-400" size={24} />
        </div>
      ) : filteredJobs.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          {filter === 'all' ? 'No training jobs yet' : `No ${filter} jobs`}
        </div>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {filteredJobs.map(job => (
            <JobCard
              key={job.job_id}
              job={job}
              onCancel={handleCancel}
              onSelect={onJobSelect}
            />
          ))}
        </div>
      )}
    </div>
  )
}
