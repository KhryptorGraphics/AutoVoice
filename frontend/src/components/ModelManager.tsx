import { Box, Upload, RefreshCw, Loader2, AlertCircle, CheckCircle, HardDrive } from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService } from '../services/api'
import { useToastContext } from '../contexts/ToastContext'
import { ConfirmActionButton } from './ConfirmActionButton'
import { StatusBanner } from './StatusBanner'

interface ModelManagerProps {
  compact?: boolean
}

const MODEL_TYPES = [
  { id: 'encoder', label: 'Content Encoder', description: 'HuBERT/ContentVec for feature extraction' },
  { id: 'vocoder', label: 'Vocoder', description: 'HiFi-GAN/BigVGAN for audio synthesis' },
  { id: 'pitch', label: 'Pitch Extractor', description: 'RMVPE/CREPE for F0 estimation' },
  { id: 'separator', label: 'Vocal Separator', description: 'Demucs for stem separation' },
  { id: 'svc', label: 'SVC Decoder', description: 'Voice conversion decoder' },
]

function formatMemory(mb: number | undefined): string {
  if (!mb) return 'N/A'
  if (mb < 1024) return `${mb.toFixed(0)} MB`
  return `${(mb / 1024).toFixed(2)} GB`
}

function formatDate(dateStr: string | undefined): string {
  if (!dateStr) return 'N/A'
  return new Date(dateStr).toLocaleString()
}

export function ModelManager({ compact = false }: ModelManagerProps) {
  const queryClient = useQueryClient()
  const toast = useToastContext()

  const { data: models, isLoading, error, refetch } = useQuery({
    queryKey: ['loadedModels'],
    queryFn: () => apiService.getLoadedModels(),
    refetchInterval: 10000, // Refresh every 10s
  })

  const loadMutation = useMutation({
    mutationFn: ({ type, path }: { type: string; path?: string }) =>
      apiService.loadModel(type, path),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['loadedModels'] })
      toast.success('Model loaded')
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : 'Failed to load model')
    },
  })

  const unloadMutation = useMutation({
    mutationFn: (type: string) => apiService.unloadModel(type),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['loadedModels'] })
      toast.success('Model unloaded')
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : 'Failed to unload model')
    },
  })

  const totalMemory = models?.reduce((sum, m) => sum + (m.memory_mb || 0), 0) || 0
  const loadedTypes = new Set(models?.map(m => m.type) || [])

  const handleLoad = (type: string) => {
    loadMutation.mutate({ type })
  }

  const handleUnload = (type: string) => {
    unloadMutation.mutate(type)
  }

  if (isLoading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4" />
        <div className="space-y-3">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-16 bg-gray-700 rounded" />
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 text-red-400">
          <AlertCircle size={20} />
          <div>
            <div className="font-medium">Failed to load models</div>
            <div className="text-sm text-gray-500">{(error as Error)?.message}</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Box size={18} className="text-blue-400" />
          <h3 className="font-semibold">Model Manager</h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">
            {models?.length || 0} loaded · {formatMemory(totalMemory)}
          </span>
          <button
            onClick={() => refetch()}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
            title="Refresh"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Loaded Models */}
      {models && models.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm text-gray-400">Loaded Models</h4>
          {models.map(model => (
            <div
              key={model.type}
              className="flex items-center justify-between p-3 bg-gray-750 rounded-lg"
            >
              <div className="flex items-center gap-3">
                <CheckCircle size={16} className="text-green-400" />
                <div>
                  <div className="font-medium text-sm">{model.name || model.type}</div>
                  <div className="text-xs text-gray-500">
                    {model.type} · {formatMemory(model.memory_mb)}
                    {model.loaded_at && ` · Loaded ${formatDate(model.loaded_at)}`}
                  </div>
                </div>
              </div>
              <ConfirmActionButton
                label="Unload"
                confirmLabel="Unload model"
                confirmMessage={`Unload ${model.name || model.type} and free its GPU memory reservation?`}
                onConfirm={async () => handleUnload(model.type)}
                pending={unloadMutation.isPending && unloadMutation.variables === model.type}
                className="px-2 py-1 text-xs"
                testId={`unload-model-${model.type}`}
              />
            </div>
          ))}
        </div>
      )}

      {/* Available Models to Load */}
      {!compact && (
        <div className="space-y-2">
          <h4 className="text-sm text-gray-400">Available Models</h4>
          <div className="grid grid-cols-1 gap-2">
            {MODEL_TYPES.filter(t => !loadedTypes.has(t.id)).map(type => (
              <div
                key={type.id}
                className="flex items-center justify-between p-3 bg-gray-750 rounded-lg border border-gray-700"
              >
                <div>
                  <div className="font-medium text-sm">{type.label}</div>
                  <div className="text-xs text-gray-500">{type.description}</div>
                </div>
                <button
                  onClick={() => handleLoad(type.id)}
                  disabled={loadMutation.isPending}
                  className="flex items-center gap-1 px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-700 rounded transition-colors disabled:opacity-50"
                >
                  {loadMutation.isPending && loadMutation.variables?.type === type.id ? (
                    <Loader2 size={12} className="animate-spin" />
                  ) : (
                    <Upload size={12} />
                  )}
                  Load
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Memory Summary */}
      <div className="pt-2 border-t border-gray-700">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2 text-gray-400">
            <HardDrive size={14} />
            Total GPU Memory Used
          </div>
          <span className="font-mono">{formatMemory(totalMemory)}</span>
        </div>
      </div>

      {/* Error Display */}
      {(loadMutation.error || unloadMutation.error) && (
        <StatusBanner
          tone="danger"
          title="Model action failed"
          message={(loadMutation.error as Error)?.message || (unloadMutation.error as Error)?.message}
          compact
        />
      )}
    </div>
  )
}
