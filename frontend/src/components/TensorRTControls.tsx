import { useState } from 'react'
import { Zap, RefreshCw, Loader2, AlertCircle, CheckCircle, Clock, Settings } from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService } from '../services/api'
import clsx from 'clsx'

interface TensorRTControlsProps {
  compact?: boolean
}

type Precision = 'fp32' | 'fp16' | 'int8'

const PRECISION_OPTIONS: { value: Precision; label: string; description: string; color: string }[] = [
  {
    value: 'fp32',
    label: 'FP32',
    description: 'Full precision - highest accuracy, largest memory',
    color: 'text-blue-400',
  },
  {
    value: 'fp16',
    label: 'FP16',
    description: 'Half precision - good balance of speed and accuracy',
    color: 'text-green-400',
  },
  {
    value: 'int8',
    label: 'INT8',
    description: 'Quantized - fastest inference, may reduce quality',
    color: 'text-yellow-400',
  },
]

const MODEL_TARGETS = [
  { id: 'encoder', label: 'Content Encoder', description: 'HuBERT/ContentVec feature extraction' },
  { id: 'vocoder', label: 'Vocoder', description: 'HiFi-GAN/BigVGAN audio synthesis' },
  { id: 'pitch', label: 'Pitch Extractor', description: 'RMVPE/CREPE F0 estimation' },
  { id: 'svc', label: 'SVC Decoder', description: 'Voice conversion decoder' },
]

function formatDate(dateStr: string | undefined): string {
  if (!dateStr) return 'Never'
  return new Date(dateStr).toLocaleString()
}

export function TensorRTControls({ compact = false }: TensorRTControlsProps) {
  const [selectedPrecision, setSelectedPrecision] = useState<Precision>('fp16')
  const [selectedModels, setSelectedModels] = useState<string[]>(['encoder', 'vocoder'])
  const [showAdvanced, setShowAdvanced] = useState(false)
  const queryClient = useQueryClient()

  const { data: status, isLoading, error, refetch } = useQuery({
    queryKey: ['tensorrtStatus'],
    queryFn: () => apiService.getTensorRTStatus(),
    refetchInterval: (query) => query.state.data?.build_in_progress ? 2000 : 30000,
  })

  const buildMutation = useMutation({
    mutationFn: ({ models, precision }: { models: string[]; precision: Precision }) =>
      apiService.buildTensorRTEngines(models, precision),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tensorrtStatus'] })
    },
  })

  const handleModelToggle = (modelId: string) => {
    setSelectedModels(prev =>
      prev.includes(modelId)
        ? prev.filter(m => m !== modelId)
        : [...prev, modelId]
    )
  }

  const handleBuild = () => {
    if (selectedModels.length === 0) return
    buildMutation.mutate({ models: selectedModels, precision: selectedPrecision })
  }

  if (isLoading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4" />
        <div className="space-y-3">
          {[1, 2].map(i => (
            <div key={i} className="h-12 bg-gray-700 rounded" />
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
            <div className="font-medium">Failed to load TensorRT status</div>
            <div className="text-sm text-gray-500">{(error as Error)?.message}</div>
          </div>
        </div>
      </div>
    )
  }

  const isAvailable = status?.available ?? false
  const isBuildInProgress = status?.build_in_progress || buildMutation.isPending

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap size={18} className="text-yellow-400" />
          <h3 className="font-semibold">TensorRT Optimization</h3>
        </div>
        <div className="flex items-center gap-2">
          {isAvailable ? (
            <span className="flex items-center gap-1 text-xs text-green-400">
              <CheckCircle size={12} />
              Available
            </span>
          ) : (
            <span className="flex items-center gap-1 text-xs text-gray-500">
              <AlertCircle size={12} />
              Unavailable
            </span>
          )}
          <button
            onClick={() => refetch()}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
            title="Refresh"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {!isAvailable ? (
        <div className="text-center py-4">
          <Zap size={32} className="mx-auto text-gray-500 mb-2" />
          <div className="text-sm text-gray-400">TensorRT not available</div>
          <div className="text-xs text-gray-500 mt-1">
            Install TensorRT and rebuild to enable GPU optimization
          </div>
        </div>
      ) : (
        <>
          {/* Precision Selector */}
          <div className="space-y-2">
            <label className="text-sm text-gray-400">Precision Mode</label>
            <div className="grid grid-cols-3 gap-2">
              {PRECISION_OPTIONS.map(opt => (
                <button
                  key={opt.value}
                  onClick={() => setSelectedPrecision(opt.value)}
                  disabled={isBuildInProgress}
                  className={clsx(
                    'p-3 rounded-lg border text-center transition-all',
                    selectedPrecision === opt.value
                      ? 'border-blue-500 bg-blue-500/20'
                      : 'border-gray-700 hover:border-gray-600 bg-gray-750',
                    isBuildInProgress && 'opacity-50 cursor-not-allowed'
                  )}
                >
                  <div className={clsx('font-mono font-bold', opt.color)}>{opt.label}</div>
                  {!compact && (
                    <div className="text-xs text-gray-500 mt-1">{opt.description.split(' - ')[0]}</div>
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Model Selection */}
          {!compact && (
            <div className="space-y-2">
              <label className="text-sm text-gray-400">Models to Optimize</label>
              <div className="space-y-2">
                {MODEL_TARGETS.map(model => (
                  <label
                    key={model.id}
                    className={clsx(
                      'flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all',
                      selectedModels.includes(model.id)
                        ? 'border-blue-500 bg-blue-500/10'
                        : 'border-gray-700 hover:border-gray-600',
                      isBuildInProgress && 'opacity-50 cursor-not-allowed'
                    )}
                  >
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(model.id)}
                      onChange={() => handleModelToggle(model.id)}
                      disabled={isBuildInProgress}
                      className="w-4 h-4 rounded border-gray-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-gray-800"
                    />
                    <div className="flex-1">
                      <div className="font-medium text-sm">{model.label}</div>
                      <div className="text-xs text-gray-500">{model.description}</div>
                    </div>
                    {status?.engines?.find(e => e.name === model.id) && (
                      <span className="text-xs text-green-400 flex items-center gap-1">
                        <CheckCircle size={12} />
                        Built
                      </span>
                    )}
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* Built Engines Status */}
          {status?.engines && status.engines.length > 0 && (
            <div className="space-y-2">
              <label className="text-sm text-gray-400">Built Engines</label>
              <div className="bg-gray-750 rounded-lg p-3 space-y-2">
                {status.engines.map(engine => (
                  <div key={engine.name} className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <CheckCircle size={14} className="text-green-400" />
                      <span className="capitalize">{engine.name}</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-gray-500">
                      <span className={clsx(
                        'font-mono',
                        engine.precision === 'fp32' ? 'text-blue-400' :
                        engine.precision === 'fp16' ? 'text-green-400' : 'text-yellow-400'
                      )}>
                        {engine.precision.toUpperCase()}
                      </span>
                      {engine.built_at && (
                        <span className="flex items-center gap-1">
                          <Clock size={10} />
                          {formatDate(engine.built_at)}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Advanced Settings */}
          {!compact && (
            <div className="border-t border-gray-700 pt-3">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
              >
                <Settings size={14} />
                Advanced Settings
                <span className={clsx('transition-transform', showAdvanced && 'rotate-180')}>▼</span>
              </button>

              {showAdvanced && (
                <div className="mt-3 space-y-3 bg-gray-750 rounded-lg p-3">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Workspace Size</span>
                    <select className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm">
                      <option value="1024">1 GB</option>
                      <option value="2048" selected>2 GB</option>
                      <option value="4096">4 GB</option>
                    </select>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Min Batch Size</span>
                    <input
                      type="number"
                      defaultValue={1}
                      min={1}
                      max={32}
                      className="w-20 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm text-right"
                    />
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Max Batch Size</span>
                    <input
                      type="number"
                      defaultValue={8}
                      min={1}
                      max={32}
                      className="w-20 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm text-right"
                    />
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">Use DLA Core</span>
                    <input
                      type="checkbox"
                      className="w-4 h-4 rounded border-gray-600 text-blue-500"
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Build Button */}
          <button
            onClick={handleBuild}
            disabled={isBuildInProgress || selectedModels.length === 0}
            className={clsx(
              'w-full flex items-center justify-center gap-2 py-3 rounded-lg font-medium transition-all',
              isBuildInProgress || selectedModels.length === 0
                ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-yellow-600 hover:bg-yellow-700 text-white'
            )}
          >
            {isBuildInProgress ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                Building Engines...
              </>
            ) : (
              <>
                <Zap size={18} />
                Build TensorRT Engines ({selectedModels.length})
              </>
            )}
          </button>

          {/* Error Display */}
          {buildMutation.error && (
            <div className="flex items-center gap-2 p-2 bg-red-900/30 border border-red-800 rounded text-sm text-red-400">
              <AlertCircle size={14} />
              {(buildMutation.error as Error)?.message}
            </div>
          )}

          {/* Success Message */}
          {buildMutation.isSuccess && (
            <div className="flex items-center gap-2 p-2 bg-green-900/30 border border-green-800 rounded text-sm text-green-400">
              <CheckCircle size={14} />
              TensorRT engines built successfully
            </div>
          )}
        </>
      )}

      {/* Version Info */}
      {status?.version && (
        <div className="pt-2 border-t border-gray-700 text-xs text-gray-500">
          TensorRT version: {status.version}
        </div>
      )}
    </div>
  )
}
