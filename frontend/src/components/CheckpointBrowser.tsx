import { useState } from 'react'
import {
  Archive, RotateCcw, GitCompare, Trash2, CheckCircle, Clock,
  Loader2, AlertCircle, Download, ChevronDown, Play
} from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService } from '../services/api'
import { ConfirmActionButton } from './ConfirmActionButton'
import { useToastContext } from '../contexts/ToastContext'
import clsx from 'clsx'

interface Checkpoint {
  id: string
  profile_id: string
  version: string
  created_at: string
  epochs_trained: number
  final_loss: number
  is_active: boolean
  file_size_mb: number
  training_samples: number
  notes?: string
}

interface CheckpointBrowserProps {
  profileId: string
  onRollback?: (checkpoint: Checkpoint) => void
  onCompare?: (checkpoints: [Checkpoint, Checkpoint]) => void
}

export function CheckpointBrowser({ profileId, onRollback, onCompare }: CheckpointBrowserProps) {
  const toast = useToastContext()
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [testingId, setTestingId] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const { data: checkpoints, isLoading, error } = useQuery({
    queryKey: ['checkpoints', profileId],
    queryFn: () => apiService.getCheckpoints(profileId),
  })

  const rollbackMutation = useMutation({
    mutationFn: (checkpointId: string) =>
      apiService.rollbackToCheckpoint(profileId, checkpointId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['checkpoints', profileId] })
      queryClient.invalidateQueries({ queryKey: ['voiceProfile', profileId] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (checkpointId: string) =>
      apiService.deleteCheckpoint(profileId, checkpointId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['checkpoints', profileId] })
    },
  })

  const toggleSelection = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else if (next.size < 2) {
        next.add(id)
      }
      return next
    })
  }

  const handleRollback = (checkpoint: Checkpoint) => {
    if (checkpoint.is_active) return
    rollbackMutation.mutate(checkpoint.id)
    onRollback?.(checkpoint)
  }

  const handleCompare = () => {
    if (selectedIds.size !== 2 || !checkpoints || !onCompare) return
    const selected = checkpoints.filter(c => selectedIds.has(c.id))
    if (selected.length === 2) {
      onCompare([selected[0], selected[1]])
    }
  }

  const handleDelete = (checkpoint: Checkpoint) => {
    if (checkpoint.is_active) {
      toast.error('Cannot delete the active checkpoint')
      return
    }
    deleteMutation.mutate(checkpoint.id)
  }

  const handleTest = async (checkpoint: Checkpoint) => {
    setTestingId(checkpoint.id)
    // Simulate A/B test - in real impl, this would call backend
    await new Promise(resolve => setTimeout(resolve, 2000))
    setTestingId(null)
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  const formatSize = (mb: number) => {
    if (mb < 1024) return `${mb.toFixed(1)} MB`
    return `${(mb / 1024).toFixed(2)} GB`
  }

  if (isLoading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-center gap-2 text-gray-400">
          <Loader2 className="animate-spin" size={20} />
          <span>Loading checkpoints...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 text-red-400">
          <AlertCircle size={20} />
          <span>Failed to load checkpoints</span>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Archive size={18} className="text-green-400" />
            <h3 className="font-semibold">Model Checkpoints</h3>
          </div>
          <span className="text-sm text-gray-500">
            {checkpoints?.length || 0} versions
          </span>
        </div>

        {selectedIds.size === 2 && onCompare && (
          <button
            onClick={handleCompare}
            className="mt-3 flex items-center gap-2 px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm w-full justify-center"
          >
            <GitCompare size={14} />
            Compare Selected Checkpoints
          </button>
        )}
      </div>

      {/* Checkpoint List */}
      {!checkpoints || checkpoints.length === 0 ? (
        <div className="p-8 text-center text-gray-500">
          <Archive size={32} className="mx-auto mb-2 opacity-50" />
          <p>No checkpoints available</p>
          <p className="text-sm">Checkpoints are created during training</p>
        </div>
      ) : (
        <div className="divide-y divide-gray-700">
          {checkpoints.map(checkpoint => (
            <div key={checkpoint.id}>
              <div
                className={clsx(
                  'p-4 cursor-pointer hover:bg-gray-750 transition-colors',
                  checkpoint.is_active && 'bg-green-900/20',
                  selectedIds.has(checkpoint.id) && 'bg-blue-900/20'
                )}
                onClick={() => setExpandedId(expandedId === checkpoint.id ? null : checkpoint.id)}
              >
                <div className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    checked={selectedIds.has(checkpoint.id)}
                    onChange={e => {
                      e.stopPropagation()
                      toggleSelection(checkpoint.id)
                    }}
                    className="rounded border-gray-600"
                  />

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-medium">{checkpoint.version}</span>
                      {checkpoint.is_active && (
                        <span className="flex items-center gap-1 text-xs text-green-400 bg-green-900/30 px-2 py-0.5 rounded">
                          <CheckCircle size={10} />
                          Active
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-gray-500 flex items-center gap-3 mt-1">
                      <span className="flex items-center gap-1">
                        <Clock size={12} />
                        {formatDate(checkpoint.created_at)}
                      </span>
                      <span>•</span>
                      <span>{checkpoint.epochs_trained} epochs</span>
                      <span>•</span>
                      <span>Loss: {checkpoint.final_loss.toFixed(4)}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-1">
                    {!checkpoint.is_active && (
                      <div onClick={e => e.stopPropagation()}>
                        <ConfirmActionButton
                          label={
                            rollbackMutation.isPending && rollbackMutation.variables === checkpoint.id
                              ? <Loader2 size={16} className="animate-spin" />
                              : <RotateCcw size={16} />
                          }
                          confirmMessage={`Roll back to version ${checkpoint.version}? This will replace the current model.`}
                          confirmLabel="Rollback"
                          onConfirm={() => handleRollback(checkpoint)}
                          disabled={rollbackMutation.isPending}
                          pending={rollbackMutation.isPending && rollbackMutation.variables === checkpoint.id}
                          variant="neutral"
                          className="p-2 text-gray-400 hover:text-green-400 hover:bg-gray-700"
                          testId={`rollback-checkpoint-${checkpoint.id}`}
                        />
                      </div>
                    )}
                    <button
                      onClick={e => {
                        e.stopPropagation()
                        handleTest(checkpoint)
                      }}
                      disabled={testingId === checkpoint.id}
                      className="p-2 text-gray-400 hover:text-blue-400 hover:bg-gray-700 rounded transition-colors"
                      title="Test this checkpoint"
                    >
                      {testingId === checkpoint.id ? (
                        <Loader2 size={16} className="animate-spin" />
                      ) : (
                        <Play size={16} />
                      )}
                    </button>
                    <span className={clsx('transition-transform', expandedId === checkpoint.id && 'rotate-180')}>
                      <ChevronDown size={16} className="text-gray-500" />
                    </span>
                  </div>
                </div>
              </div>

              {/* Expanded Details */}
              {expandedId === checkpoint.id && (
                <div className="px-4 pb-4 bg-gray-750">
                  <div className="grid grid-cols-2 gap-4 py-3 border-t border-gray-700 text-sm">
                    <div>
                      <div className="text-gray-500">Training Samples</div>
                      <div>{checkpoint.training_samples}</div>
                    </div>
                    <div>
                      <div className="text-gray-500">File Size</div>
                      <div>{formatSize(checkpoint.file_size_mb)}</div>
                    </div>
                    <div>
                      <div className="text-gray-500">Final Loss</div>
                      <div className="font-mono">{checkpoint.final_loss.toFixed(6)}</div>
                    </div>
                    <div>
                      <div className="text-gray-500">Epochs Trained</div>
                      <div>{checkpoint.epochs_trained}</div>
                    </div>
                  </div>

                  {checkpoint.notes && (
                    <div className="py-3 border-t border-gray-700">
                      <div className="text-gray-500 text-sm mb-1">Notes</div>
                      <p className="text-sm">{checkpoint.notes}</p>
                    </div>
                  )}

                  <div className="flex gap-2 pt-3 border-t border-gray-700">
                    <a
                      href={`/api/v1/profiles/${profileId}/checkpoints/${checkpoint.id}/download`}
                      className="flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
                    >
                      <Download size={14} />
                      Download
                    </a>
                    {!checkpoint.is_active && (
                      <>
                        <ConfirmActionButton
                          label={<span className="flex items-center gap-2"><RotateCcw size={14} /> Rollback</span>}
                          confirmMessage={`Roll back to version ${checkpoint.version}? This will replace the current model.`}
                          confirmLabel="Rollback"
                          onConfirm={() => handleRollback(checkpoint)}
                          disabled={rollbackMutation.isPending}
                          pending={rollbackMutation.isPending && rollbackMutation.variables === checkpoint.id}
                          variant="neutral"
                          className="flex items-center gap-2 px-3 py-2 bg-green-600 hover:bg-green-700 rounded text-sm"
                          testId={`rollback-checkpoint-expanded-${checkpoint.id}`}
                        />
                        <ConfirmActionButton
                          label={<span className="flex items-center gap-2"><Trash2 size={14} /> Delete</span>}
                          confirmMessage={`Delete checkpoint ${checkpoint.version}? This cannot be undone.`}
                          confirmLabel="Delete"
                          onConfirm={() => handleDelete(checkpoint)}
                          disabled={deleteMutation.isPending}
                          pending={deleteMutation.isPending && deleteMutation.variables === checkpoint.id}
                          className="flex items-center gap-2 px-3 py-2 bg-red-600 hover:bg-red-700 rounded text-sm ml-auto"
                          testId={`delete-checkpoint-${checkpoint.id}`}
                        />
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Mutation Errors */}
      {(rollbackMutation.error || deleteMutation.error) && (
        <div className="p-4 border-t border-gray-700">
          <div className="flex items-center gap-2 text-red-400 text-sm">
            <AlertCircle size={14} />
            {(rollbackMutation.error as Error)?.message || (deleteMutation.error as Error)?.message}
          </div>
        </div>
      )}
    </div>
  )
}
