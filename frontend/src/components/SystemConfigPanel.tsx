import { useState } from 'react'
import {
  Settings, Download, Upload, Loader2, ChevronDown, ChevronUp
} from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  apiService,
  TrainingConfig,
  ConversionConfig,
  SeparationConfig,
  PitchConfig,
  AudioRouterConfig,
  DEFAULT_TRAINING_CONFIG,
  DEFAULT_CONVERSION_CONFIG,
  DEFAULT_SEPARATION_CONFIG,
  DEFAULT_PITCH_CONFIG,
  DEFAULT_AUDIO_ROUTER_CONFIG,
} from '../services/api'
import clsx from 'clsx'
import { STORAGE_KEYS, usePersistedState } from '../hooks/usePersistedState'
import { useToastContext } from '../contexts/ToastContext'
import { ConfirmActionButton } from './ConfirmActionButton'
import { StatusBanner } from './StatusBanner'

interface SystemConfigPanelProps {
  onConfigChange?: () => void
}

interface FullConfig {
  training: TrainingConfig
  conversion: ConversionConfig
  separation: SeparationConfig
  pitch: PitchConfig
  audioRouter: AudioRouterConfig
  ui: UIConfig
}

interface UIConfig {
  theme: 'dark' | 'light'
  compactMode: boolean
  autoRefreshInterval: number
  showAdvancedControls: boolean
  defaultQualityPreset: string
}

const DEFAULT_UI_CONFIG: UIConfig = {
  theme: 'dark',
  compactMode: false,
  autoRefreshInterval: 5000,
  showAdvancedControls: false,
  defaultQualityPreset: 'balanced',
}

export function SystemConfigPanel({ onConfigChange }: SystemConfigPanelProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['ui']))
  const [hasChanges, setHasChanges] = useState(false)
  const [importError, setImportError] = useState<string | null>(null)
  const queryClient = useQueryClient()
  const toast = useToastContext()

  // Load UI config from localStorage
  const [uiConfig, setUIConfig] = usePersistedState<UIConfig>(
    STORAGE_KEYS.UI_CONFIG,
    DEFAULT_UI_CONFIG
  )

  // Fetch backend configs
  const { data: separationConfig, isLoading: loadingSeparation } = useQuery({
    queryKey: ['separationConfig'],
    queryFn: () => apiService.getSeparationConfig(),
  })

  const { data: pitchConfig, isLoading: loadingPitch } = useQuery({
    queryKey: ['pitchConfig'],
    queryFn: () => apiService.getPitchConfig(),
  })

  const { data: audioRouterConfig, isLoading: loadingRouter } = useQuery({
    queryKey: ['audioRouterConfig'],
    queryFn: () => apiService.getAudioRouterConfig(),
  })

  // Mutations for updating configs
  const updateSeparationMutation = useMutation({
    mutationFn: (config: Partial<SeparationConfig>) => apiService.updateSeparationConfig(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['separationConfig'] })
      onConfigChange?.()
      toast.success('Separation config updated')
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : 'Failed to update separation config')
    },
  })

  const updatePitchMutation = useMutation({
    mutationFn: (config: Partial<PitchConfig>) => apiService.updatePitchConfig(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pitchConfig'] })
      onConfigChange?.()
      toast.success('Pitch config updated')
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : 'Failed to update pitch config')
    },
  })

  const updateAudioRouterMutation = useMutation({
    mutationFn: (config: Partial<AudioRouterConfig>) => apiService.updateAudioRouterConfig(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['audioRouterConfig'] })
      onConfigChange?.()
      toast.success('Audio router config updated')
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : 'Failed to update audio router config')
    },
  })

  const isLoading = loadingSeparation || loadingPitch || loadingRouter
  const isSaving = updateSeparationMutation.isPending || updatePitchMutation.isPending || updateAudioRouterMutation.isPending

  const toggleSection = (section: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev)
      if (next.has(section)) {
        next.delete(section)
      } else {
        next.add(section)
      }
      return next
    })
  }

  // Save UI config to localStorage
  const saveUIConfig = (updates: Partial<UIConfig>) => {
    const newConfig = { ...uiConfig, ...updates }
    setUIConfig(newConfig)
    setHasChanges(true)
    onConfigChange?.()
  }

  // Export all configuration
  const exportConfig = () => {
    const fullConfig: FullConfig = {
      training: DEFAULT_TRAINING_CONFIG,
      conversion: DEFAULT_CONVERSION_CONFIG,
      separation: separationConfig || DEFAULT_SEPARATION_CONFIG,
      pitch: pitchConfig || DEFAULT_PITCH_CONFIG,
      audioRouter: audioRouterConfig || DEFAULT_AUDIO_ROUTER_CONFIG,
      ui: uiConfig,
    }

    const blob = new Blob([JSON.stringify(fullConfig, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `autovoice-config-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Import configuration
  const importConfig = () => {
    setImportError(null)
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return

      try {
        const text = await file.text()
        const imported = JSON.parse(text) as Partial<FullConfig>

        // Apply imported configs
        if (imported.separation) {
          await updateSeparationMutation.mutateAsync(imported.separation)
        }
        if (imported.pitch) {
          await updatePitchMutation.mutateAsync(imported.pitch)
        }
        if (imported.audioRouter) {
          await updateAudioRouterMutation.mutateAsync(imported.audioRouter)
        }
        if (imported.ui) {
          saveUIConfig(imported.ui)
        }

        setHasChanges(false)
        toast.success('System configuration imported')
      } catch (err) {
        const message = (err as Error).message || 'Failed to import configuration'
        setImportError(message)
        toast.error(message)
      }
    }
    input.click()
  }

  // Reset all to defaults
  const resetToDefaults = async () => {
    try {
      await Promise.all([
        updateSeparationMutation.mutateAsync(DEFAULT_SEPARATION_CONFIG),
        updatePitchMutation.mutateAsync(DEFAULT_PITCH_CONFIG),
        updateAudioRouterMutation.mutateAsync(DEFAULT_AUDIO_ROUTER_CONFIG),
      ])
      saveUIConfig(DEFAULT_UI_CONFIG)
      setHasChanges(false)
      toast.success('System configuration reset to defaults')
    } catch (err) {
      console.error('Failed to reset config:', err)
      toast.error(err instanceof Error ? err.message : 'Failed to reset configuration')
    }
  }

  const SectionHeader = ({ id, title }: { id: string; title: string }) => (
    <button
      onClick={() => toggleSection(id)}
      className="w-full flex items-center justify-between p-3 bg-gray-750 hover:bg-gray-700 rounded-lg transition-colors"
    >
      <span className="font-medium">{title}</span>
      {expandedSections.has(id) ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
    </button>
  )

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Settings size={18} className="text-blue-400" />
            <h3 className="font-semibold">System Configuration</h3>
          </div>
          {hasChanges && (
            <span className="text-xs text-yellow-400">Unsaved changes</span>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={exportConfig}
            disabled={isLoading}
            className="flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
          >
            <Download size={14} />
            Export
          </button>
          <button
            onClick={importConfig}
            disabled={isSaving}
            className="flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
          >
            <Upload size={14} />
            Import
          </button>
          <div className="ml-auto">
            <ConfirmActionButton
              label="Reset all"
              confirmLabel="Reset settings"
              confirmMessage="Reset separation, pitch, audio router, and UI defaults back to the canonical baseline?"
              onConfirm={resetToDefaults}
              pending={isSaving}
              testId="system-config-reset"
            />
          </div>
        </div>

        {importError && (
          <div className="mt-3">
            <StatusBanner
              tone="danger"
              title="Config import failed"
              message={importError}
              compact
            />
          </div>
        )}
      </div>

      {/* Config Sections */}
      <div className="p-4 space-y-3">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="animate-spin text-gray-500" />
          </div>
        ) : (
          <>
            {/* UI Settings */}
            <div>
              <SectionHeader id="ui" title="UI Settings" />
              {expandedSections.has('ui') && (
                <div className="mt-2 p-3 bg-gray-750 rounded-lg space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm">Compact Mode</label>
                    <button
                      onClick={() => saveUIConfig({ compactMode: !uiConfig.compactMode })}
                      className={clsx(
                        'w-10 h-5 rounded-full transition-colors relative',
                        uiConfig.compactMode ? 'bg-blue-600' : 'bg-gray-600'
                      )}
                    >
                      <div
                        className={clsx(
                          'absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform',
                          uiConfig.compactMode ? 'left-5' : 'left-0.5'
                        )}
                      />
                    </button>
                  </div>

                  <div className="flex items-center justify-between">
                    <label className="text-sm">Show Advanced Controls</label>
                    <button
                      onClick={() => saveUIConfig({ showAdvancedControls: !uiConfig.showAdvancedControls })}
                      className={clsx(
                        'w-10 h-5 rounded-full transition-colors relative',
                        uiConfig.showAdvancedControls ? 'bg-blue-600' : 'bg-gray-600'
                      )}
                    >
                      <div
                        className={clsx(
                          'absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform',
                          uiConfig.showAdvancedControls ? 'left-5' : 'left-0.5'
                        )}
                      />
                    </button>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Auto-Refresh Interval</label>
                    <select
                      value={uiConfig.autoRefreshInterval}
                      onChange={e => saveUIConfig({ autoRefreshInterval: parseInt(e.target.value) })}
                      className="mt-1 w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                    >
                      <option value={1000}>1 second</option>
                      <option value={2000}>2 seconds</option>
                      <option value={5000}>5 seconds</option>
                      <option value={10000}>10 seconds</option>
                      <option value={30000}>30 seconds</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Default Quality Preset</label>
                    <select
                      value={uiConfig.defaultQualityPreset}
                      onChange={e => saveUIConfig({ defaultQualityPreset: e.target.value })}
                      className="mt-1 w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                    >
                      <option value="draft">Draft (Fastest)</option>
                      <option value="fast">Fast</option>
                      <option value="balanced">Balanced</option>
                      <option value="high">High Quality</option>
                      <option value="studio">Studio</option>
                    </select>
                  </div>
                </div>
              )}
            </div>

            {/* Separation Settings */}
            <div>
              <SectionHeader id="separation" title="Separation Settings" />
              {expandedSections.has('separation') && separationConfig && (
                <div className="mt-2 p-3 bg-gray-750 rounded-lg space-y-4">
                  <div>
                    <label className="text-sm text-gray-400">Model</label>
                    <select
                      value={separationConfig.model}
                      onChange={e => updateSeparationMutation.mutate({ model: e.target.value as SeparationConfig['model'] })}
                      className="mt-1 w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                    >
                      <option value="htdemucs">HTDemucs</option>
                      <option value="htdemucs_ft">HTDemucs Fine-tuned</option>
                      <option value="mdx_extra">MDX Extra</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Shifts (quality vs speed)</label>
                    <input
                      type="range"
                      min={0}
                      max={5}
                      value={separationConfig.shifts}
                      onChange={e => updateSeparationMutation.mutate({ shifts: parseInt(e.target.value) })}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Fast (0)</span>
                      <span>{separationConfig.shifts}</span>
                      <span>Quality (5)</span>
                    </div>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Overlap</label>
                    <input
                      type="range"
                      min={0}
                      max={0.9}
                      step={0.05}
                      value={separationConfig.overlap}
                      onChange={e => updateSeparationMutation.mutate({ overlap: parseFloat(e.target.value) })}
                      className="mt-1 w-full"
                    />
                    <div className="text-xs text-gray-500 text-right">
                      {(separationConfig.overlap * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Pitch Settings */}
            <div>
              <SectionHeader id="pitch" title="Pitch Extraction" />
              {expandedSections.has('pitch') && pitchConfig && (
                <div className="mt-2 p-3 bg-gray-750 rounded-lg space-y-4">
                  <div>
                    <label className="text-sm text-gray-400">Method</label>
                    <select
                      value={pitchConfig.method}
                      onChange={e => updatePitchMutation.mutate({ method: e.target.value as PitchConfig['method'] })}
                      className="mt-1 w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                    >
                      <option value="rmvpe">RMVPE (Recommended)</option>
                      <option value="crepe">CREPE</option>
                      <option value="harvest">Harvest</option>
                      <option value="dio">DIO</option>
                    </select>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm text-gray-400">F0 Min (Hz)</label>
                      <input
                        type="number"
                        value={pitchConfig.f0_min}
                        onChange={e => updatePitchMutation.mutate({ f0_min: parseInt(e.target.value) })}
                        className="mt-1 w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">F0 Max (Hz)</label>
                      <input
                        type="number"
                        value={pitchConfig.f0_max}
                        onChange={e => updatePitchMutation.mutate({ f0_max: parseInt(e.target.value) })}
                        className="mt-1 w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                      />
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <label className="text-sm">Use GPU</label>
                    <button
                      onClick={() => updatePitchMutation.mutate({ use_gpu: !pitchConfig.use_gpu })}
                      className={clsx(
                        'w-10 h-5 rounded-full transition-colors relative',
                        pitchConfig.use_gpu ? 'bg-green-600' : 'bg-gray-600'
                      )}
                    >
                      <div
                        className={clsx(
                          'absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform',
                          pitchConfig.use_gpu ? 'left-5' : 'left-0.5'
                        )}
                      />
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Audio Router Settings */}
            <div>
              <SectionHeader id="audio" title="Audio Router" />
              {expandedSections.has('audio') && audioRouterConfig && (
                <div className="mt-2 p-3 bg-gray-750 rounded-lg space-y-4">
                  <div>
                    <label className="text-sm text-gray-400">Sample Rate</label>
                    <select
                      value={audioRouterConfig.sample_rate}
                      onChange={e => updateAudioRouterMutation.mutate({ sample_rate: parseInt(e.target.value) })}
                      className="mt-1 w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                    >
                      <option value={16000}>16 kHz</option>
                      <option value={22050}>22.05 kHz</option>
                      <option value={24000}>24 kHz</option>
                      <option value={44100}>44.1 kHz</option>
                      <option value={48000}>48 kHz</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Voice Gain</label>
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.1}
                      value={audioRouterConfig.voice_gain}
                      onChange={e => updateAudioRouterMutation.mutate({ voice_gain: parseFloat(e.target.value) })}
                      className="mt-1 w-full"
                    />
                    <div className="text-xs text-gray-500 text-right">
                      {(audioRouterConfig.voice_gain * 100).toFixed(0)}%
                    </div>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Instrumental Gain</label>
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.1}
                      value={audioRouterConfig.instrumental_gain}
                      onChange={e => updateAudioRouterMutation.mutate({ instrumental_gain: parseFloat(e.target.value) })}
                      className="mt-1 w-full"
                    />
                    <div className="text-xs text-gray-500 text-right">
                      {(audioRouterConfig.instrumental_gain * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Save Status */}
      {isSaving && (
        <div className="p-3 border-t border-gray-700 flex items-center gap-2 text-sm text-gray-400">
          <Loader2 size={14} className="animate-spin" />
          Saving changes...
        </div>
      )}
    </div>
  )
}
