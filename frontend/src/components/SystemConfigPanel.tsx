import { useState, type ChangeEvent } from 'react'
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
  // ponytail: range inputs used to fire one PATCH (and one success toast) per
  // step of a drag — hold the dragged value locally and commit once on release.
  const [sliderDraft, setSliderDraft] = useState<Record<string, number>>({})
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

  const { data: appSettings } = useQuery({
    queryKey: ['appSettings'],
    queryFn: () => apiService.getAppSettings(),
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

  const updateAppSettingsMutation = useMutation({
    mutationFn: (settings: Parameters<typeof apiService.updateAppSettings>[0]) =>
      apiService.updateAppSettings(settings),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['appSettings'] })
      onConfigChange?.()
      toast.success('Multi-speaker settings updated (applies to the next conversion)')
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : 'Failed to update multi-speaker settings')
    },
  })

  const isLoading = loadingSeparation || loadingPitch || loadingRouter
  const isSaving = updateSeparationMutation.isPending || updatePitchMutation.isPending || updateAudioRouterMutation.isPending

  const dragValue = (key: string, serverValue: number) => sliderDraft[key] ?? serverValue
  const onDrag = (key: string) => (event: ChangeEvent<HTMLInputElement>) =>
    setSliderDraft(prev => ({ ...prev, [key]: parseFloat(event.target.value) }))
  const onCommit = (key: string, apply: (value: number) => void) => () => {
    const value = sliderDraft[key]
    if (value !== undefined) apply(value)
  }

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

        setSliderDraft({})
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
      setSliderDraft({})
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
          <span className="text-xs text-gray-500">Changes save as you make them</span>
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
                  <p className="text-xs text-amber-300/80" data-testid="ui-config-scope-note">
                    Stored in this browser and included in Export, but no screen reads these yet —
                    changing them will not alter the app until they are wired up.
                  </p>
                  <div className="flex items-center justify-between">
                    <label className="text-sm">Compact Mode</label>
                    <button
                      aria-label="Compact Mode"
                      aria-pressed={uiConfig.compactMode}
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
                      aria-label="Show Advanced Controls"
                      aria-pressed={uiConfig.showAdvancedControls}
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
                      value={dragValue('shifts', separationConfig.shifts)}
                      onChange={onDrag('shifts')}
                      onPointerUp={onCommit('shifts', v => updateSeparationMutation.mutate({ shifts: Math.round(v) }))}
                      onKeyUp={onCommit('shifts', v => updateSeparationMutation.mutate({ shifts: Math.round(v) }))}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Fast (0)</span>
                      <span>{dragValue('shifts', separationConfig.shifts)}</span>
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
                      value={dragValue('overlap', separationConfig.overlap)}
                      onChange={onDrag('overlap')}
                      onPointerUp={onCommit('overlap', v => updateSeparationMutation.mutate({ overlap: v }))}
                      onKeyUp={onCommit('overlap', v => updateSeparationMutation.mutate({ overlap: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="text-xs text-gray-500 text-right">
                      {(dragValue('overlap', separationConfig.overlap) * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Multi-Speaker Conversion */}
            <div>
              <SectionHeader id="multiSpeaker" title="Multi-Speaker Conversion" />
              {expandedSections.has('multiSpeaker') && appSettings && (
                <div className="mt-2 p-3 bg-gray-750 rounded-lg space-y-4" data-testid="multi-speaker-settings">
                  <div>
                    <label className="text-sm text-gray-400">Lead/backing separator</label>
                    <select
                      value={appSettings.multi_speaker_separator ?? 'diarization'}
                      onChange={e => updateAppSettingsMutation.mutate({
                        multi_speaker_separator: e.target.value as 'diarization' | 'karaoke_model',
                      })}
                      className="mt-1 w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                      data-testid="multi-speaker-separator-select"
                    >
                      <option value="karaoke_model">Karaoke model (splits simultaneous harmonies)</option>
                      <option value="diarization">Diarization spans (turn-taking only)</option>
                    </select>
                    <p className="mt-1 text-xs text-gray-500">
                      The karaoke model pulls harmony doubles out from under the lead; it falls back to
                      diarization automatically when its bridge is unavailable.
                    </p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Converted-backing loudness</label>
                    <input
                      type="range"
                      min={0.5}
                      max={2}
                      step={0.05}
                      value={dragValue('backingGain', appSettings.multi_speaker_backing_gain ?? 1.0)}
                      onChange={onDrag('backingGain')}
                      onPointerUp={onCommit('backingGain', v => updateAppSettingsMutation.mutate({ multi_speaker_backing_gain: v }))}
                      onKeyUp={onCommit('backingGain', v => updateAppSettingsMutation.mutate({ multi_speaker_backing_gain: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Quieter (0.5×)</span>
                      <span>{dragValue('backingGain', appSettings.multi_speaker_backing_gain ?? 1.0).toFixed(2)}×</span>
                      <span>Louder (2×)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      1.0× matches the original backing stem&apos;s loudness.
                    </p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Unconverted-backing loudness</label>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={dragValue('keptBackingGain', appSettings.multi_speaker_kept_backing_gain ?? 1.0)}
                      onChange={onDrag('keptBackingGain')}
                      onPointerUp={onCommit('keptBackingGain', v => updateAppSettingsMutation.mutate({ multi_speaker_kept_backing_gain: v }))}
                      onKeyUp={onCommit('keptBackingGain', v => updateAppSettingsMutation.mutate({ multi_speaker_kept_backing_gain: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Silent (0)</span>
                      <span>{dragValue('keptBackingGain', appSettings.multi_speaker_kept_backing_gain ?? 1.0).toFixed(2)}×</span>
                      <span>Original level (1.0)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      When a harmony line fails to convert, its unmatched separation residue is mixed in at
                      this fraction of its original level. 1.0 matches today&apos;s behaviour; lower it if
                      untouched backing reads as buzzy or too loud next to the converted lead.
                    </p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Harmony fullness (harmonics kept)</label>
                    <input
                      type="range"
                      min={4}
                      max={64}
                      step={2}
                      value={dragValue('lineHarmonics', appSettings.multi_speaker_line_harmonics ?? 24)}
                      onChange={onDrag('lineHarmonics')}
                      onPointerUp={onCommit('lineHarmonics', v => updateAppSettingsMutation.mutate({ multi_speaker_line_harmonics: v }))}
                      onKeyUp={onCommit('lineHarmonics', v => updateAppSettingsMutation.mutate({ multi_speaker_line_harmonics: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Isolated (4)</span>
                      <span>{Math.round(dragValue('lineHarmonics', appSettings.multi_speaker_line_harmonics ?? 24))}</span>
                      <span>Full (64)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      How much of each harmony line is captured before it is re-sung. Low keeps lines
                      cleanly separated but thin and quiet; high gives fuller, louder harmonies but
                      lines start sharing upper harmonics and bleeding into each other.
                    </p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Lead-bleed cancellation</label>
                    <select
                      value={appSettings.multi_speaker_bleed_suppression ?? 'off'}
                      onChange={e => updateAppSettingsMutation.mutate({
                        multi_speaker_bleed_suppression: e.target.value as 'off' | 'ls',
                      })}
                      className="mt-1 w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                      data-testid="bleed-suppression-select"
                    >
                      <option value="off">Off (original behaviour)</option>
                      <option value="ls">On — coherence-throttled cancellation</option>
                    </select>
                    <p className="mt-1 text-xs text-gray-500">
                      The separator builds the backing stem by subtracting its lead estimate from the
                      mix, so whatever lead it misses stays in the backing phase-aligned with the lead.
                      A real backing singer is never phase-aligned, so cancelling only the coherent
                      part removes the leaked lead while leaving harmonies intact — even harmonies
                      that share partials with the lead.
                    </p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Bleed cancellation ceiling (dB)</label>
                    <input
                      type="range"
                      min={0}
                      max={24}
                      step={1}
                      value={dragValue('bleedMaxDb', appSettings.multi_speaker_bleed_max_db ?? 12)}
                      onChange={onDrag('bleedMaxDb')}
                      onPointerUp={onCommit('bleedMaxDb', v => updateAppSettingsMutation.mutate({ multi_speaker_bleed_max_db: v }))}
                      onKeyUp={onCommit('bleedMaxDb', v => updateAppSettingsMutation.mutate({ multi_speaker_bleed_max_db: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Gentle (0)</span>
                      <span>{Math.round(dragValue('bleedMaxDb', appSettings.multi_speaker_bleed_max_db ?? 12))} dB</span>
                      <span>Aggressive (24)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      Most any single frequency may be attenuated. This is the guard against carving
                      into a harmony that shares partials with the lead; raise it only if audible
                      lead survives in the backing.
                    </p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Bleed estimate ceiling</label>
                    <input
                      type="range"
                      min={0.1}
                      max={1}
                      step={0.05}
                      value={dragValue('bleedHMax', appSettings.multi_speaker_bleed_h_max ?? 0.7)}
                      onChange={onDrag('bleedHMax')}
                      onPointerUp={onCommit('bleedHMax', v => updateAppSettingsMutation.mutate({ multi_speaker_bleed_h_max: v }))}
                      onKeyUp={onCommit('bleedHMax', v => updateAppSettingsMutation.mutate({ multi_speaker_bleed_h_max: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Conservative (0.1)</span>
                      <span>{dragValue('bleedHMax', appSettings.multi_speaker_bleed_h_max ?? 0.7).toFixed(2)}</span>
                      <span>Trusting (1.0)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      Caps how much leakage the estimator is allowed to believe it found. If a harmony
                      sings under the lead almost continuously the estimate can read high; this stops
                      that turning into a large subtraction.
                    </p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Fold unison doubles into the lead (semitones)</label>
                    <input
                      type="range"
                      min={0}
                      max={4}
                      step={0.25}
                      value={dragValue('unisonSemi', appSettings.multi_speaker_unison_semitones ?? 1.0)}
                      onChange={onDrag('unisonSemi')}
                      onPointerUp={onCommit('unisonSemi', v => updateAppSettingsMutation.mutate({ multi_speaker_unison_semitones: v }))}
                      onKeyUp={onCommit('unisonSemi', v => updateAppSettingsMutation.mutate({ multi_speaker_unison_semitones: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Off (0)</span>
                      <span>{dragValue('unisonSemi', appSettings.multi_speaker_unison_semitones ?? 1.0).toFixed(2)}</span>
                      <span>Wide (4)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      A double-tracked lead lands in the backing stem and would otherwise be converted
                      as its own singer, then summed against the separately-converted lead — two takes
                      of the same phrase that do not line up, which is what makes the lead and the
                      background singers smear together. Lines this close to the lead&apos;s pitch are
                      folded back into it and converted once. 0 disables.
                    </p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Unison decision threshold (share of notes)</label>
                    <input
                      type="range"
                      min={0.1}
                      max={1}
                      step={0.05}
                      value={dragValue('unisonFrac', appSettings.multi_speaker_unison_note_frac ?? 0.5)}
                      onChange={onDrag('unisonFrac')}
                      onPointerUp={onCommit('unisonFrac', v => updateAppSettingsMutation.mutate({ multi_speaker_unison_note_frac: v }))}
                      onKeyUp={onCommit('unisonFrac', v => updateAppSettingsMutation.mutate({ multi_speaker_unison_note_frac: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Fold readily (0.1)</span>
                      <span>{dragValue('unisonFrac', appSettings.multi_speaker_unison_note_frac ?? 0.5).toFixed(2)}</span>
                      <span>Only exact doubles (1.0)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      What share of a line&apos;s notes must sit at the lead&apos;s pitch before it counts
                      as a double rather than a harmony. Too low and a real harmony that crosses the
                      lead gets absorbed into it.
                    </p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Harmony-line detection strictness</label>
                    <input
                      type="range"
                      min={0.5}
                      max={10}
                      step={0.1}
                      value={dragValue('lineConcMin', appSettings.multi_speaker_line_concentration_min ?? 1.2)}
                      onChange={onDrag('lineConcMin')}
                      onPointerUp={onCommit('lineConcMin', v => updateAppSettingsMutation.mutate({ multi_speaker_line_concentration_min: v }))}
                      onKeyUp={onCommit('lineConcMin', v => updateAppSettingsMutation.mutate({ multi_speaker_line_concentration_min: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Convert more (0.5)</span>
                      <span>{dragValue('lineConcMin', appSettings.multi_speaker_line_concentration_min ?? 1.2).toFixed(1)}x</span>
                      <span>Stricter (10)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      How much more concentrated a line's energy must be than pure noise before it
                      counts as a real harmony. 1.0 means "no better than noise", so anything at or
                      below that is texture the filter merely made sound tonal. Lower converts more
                      marginal lines; too low and noise gets re-sung as a phantom voice.
                    </p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Harmony note attack (ms)</label>
                    <input
                      type="range"
                      min={0}
                      max={120}
                      step={5}
                      value={dragValue('lineOnsetMs', appSettings.multi_speaker_line_onset_ms ?? 30)}
                      onChange={onDrag('lineOnsetMs')}
                      onPointerUp={onCommit('lineOnsetMs', v => updateAppSettingsMutation.mutate({ multi_speaker_line_onset_ms: v }))}
                      onKeyUp={onCommit('lineOnsetMs', v => updateAppSettingsMutation.mutate({ multi_speaker_line_onset_ms: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Off (0)</span>
                      <span>{Math.round(dragValue('lineOnsetMs', appSettings.multi_speaker_line_onset_ms ?? 30))} ms</span>
                      <span>120 ms</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      Note attacks are broadband, so a purely harmonic filter cannot pass them and
                      harmonies smear into one legato line. This passes the first few milliseconds of
                      each note whole. 0 restores the old attack-less behaviour.
                    </p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Harmony-line conversion strictness</label>
                    <input
                      type="range"
                      min={0.3}
                      max={0.95}
                      step={0.05}
                      value={dragValue('backingVoicedMin', appSettings.multi_speaker_backing_voiced_min ?? 0.65)}
                      onChange={onDrag('backingVoicedMin')}
                      onPointerUp={onCommit('backingVoicedMin', v => updateAppSettingsMutation.mutate({ multi_speaker_backing_voiced_min: v }))}
                      onKeyUp={onCommit('backingVoicedMin', v => updateAppSettingsMutation.mutate({ multi_speaker_backing_voiced_min: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Convert more (0.3)</span>
                      <span>{dragValue('backingVoicedMin', appSettings.multi_speaker_backing_voiced_min ?? 0.65).toFixed(2)}</span>
                      <span>Convert less (0.95)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      Minimum voicing a harmony line needs before it is re-sung in the target voice; lines
                      below the gate stay original. Lower = more lines converted (riskier).
                    </p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Karaoke leak guard</label>
                    <input
                      type="range"
                      min={0.3}
                      max={0.95}
                      step={0.05}
                      value={dragValue('karaokeLeakMin', appSettings.multi_speaker_karaoke_leak_voiced_min ?? 0.65)}
                      onChange={onDrag('karaokeLeakMin')}
                      onPointerUp={onCommit('karaokeLeakMin', v => updateAppSettingsMutation.mutate({ multi_speaker_karaoke_leak_voiced_min: v }))}
                      onKeyUp={onCommit('karaokeLeakMin', v => updateAppSettingsMutation.mutate({ multi_speaker_karaoke_leak_voiced_min: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Cautious (0.3)</span>
                      <span>{dragValue('karaokeLeakMin', appSettings.multi_speaker_karaoke_leak_voiced_min ?? 0.65).toFixed(2)}</span>
                      <span>Trusting (0.95)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      When the karaoke split&apos;s backing stem sounds this lead-like, the split is rejected
                      as a leak and diarization is used instead. Solo covers with strong self-harmony doubles
                      can trip the guard — raise toward 0.85 to accept the split on such tracks.
                    </p>
                  </div>

                  <div>
                    <label className="flex items-center gap-2 text-sm text-gray-400">
                      <input
                        type="checkbox"
                        checked={appSettings.multi_speaker_convert_backing ?? false}
                        onChange={e => updateAppSettingsMutation.mutate({ multi_speaker_convert_backing: e.target.checked })}
                        data-testid="convert-backing-checkbox"
                      />
                      Convert backing vocals
                    </label>
                    <p className="mt-1 text-xs text-gray-500">Off by default. When off, harmony and backing lines stay in the original singer&apos;s voice.</p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Treat backing stack as one doubled voice</label>
                    <input
                      type="range"
                      min={0.3}
                      max={0.99}
                      step={0.01}
                      value={dragValue('backingWholeMin', appSettings.multi_speaker_backing_whole_voiced_min ?? 0.7)}
                      onChange={onDrag('backingWholeMin')}
                      onPointerUp={onCommit('backingWholeMin', v => updateAppSettingsMutation.mutate({ multi_speaker_backing_whole_voiced_min: v }))}
                      onKeyUp={onCommit('backingWholeMin', v => updateAppSettingsMutation.mutate({ multi_speaker_backing_whole_voiced_min: v }))}
                      className="mt-1 w-full"
                      data-testid="backingWholeMin-slider"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Decompose (0.3)</span>
                      <span data-testid="backingWholeMin-value">{dragValue('backingWholeMin', appSettings.multi_speaker_backing_whole_voiced_min ?? 0.7).toFixed(2)}</span>
                      <span>Convert whole (0.99)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">Above this voiced fraction the backing stack is converted in one pass — faster and seamless, but a mono-F0 engine flattens genuine harmony. Raise it to force per-line decomposition.</p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Lead vs backing threshold</label>
                    <input
                      type="range"
                      min={0.3}
                      max={0.95}
                      step={0.05}
                      value={dragValue('mergeVoicedMin', appSettings.multi_speaker_merge_voiced_min ?? 0.65)}
                      onChange={onDrag('mergeVoicedMin')}
                      onPointerUp={onCommit('mergeVoicedMin', v => updateAppSettingsMutation.mutate({ multi_speaker_merge_voiced_min: v }))}
                      onKeyUp={onCommit('mergeVoicedMin', v => updateAppSettingsMutation.mutate({ multi_speaker_merge_voiced_min: v }))}
                      className="mt-1 w-full"
                      data-testid="mergeVoicedMin-slider"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>More leads (0.3)</span>
                      <span data-testid="mergeVoicedMin-value">{dragValue('mergeVoicedMin', appSettings.multi_speaker_merge_voiced_min ?? 0.65).toFixed(2)}</span>
                      <span>More backing (0.95)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">Decides whether each non-primary singer is treated as a lead or as backing.</p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Minimum line coverage</label>
                    <input
                      type="range"
                      min={0.1}
                      max={1.0}
                      step={0.05}
                      value={dragValue('minCoverage', appSettings.multi_speaker_min_coverage ?? 0.9)}
                      onChange={onDrag('minCoverage')}
                      onPointerUp={onCommit('minCoverage', v => updateAppSettingsMutation.mutate({ multi_speaker_min_coverage: v }))}
                      onKeyUp={onCommit('minCoverage', v => updateAppSettingsMutation.mutate({ multi_speaker_min_coverage: v }))}
                      className="mt-1 w-full"
                      data-testid="minCoverage-slider"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Lenient (0.1)</span>
                      <span data-testid="minCoverage-value">{dragValue('minCoverage', appSettings.multi_speaker_min_coverage ?? 0.9).toFixed(2)}</span>
                      <span>Strict (1.0)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">How much of a detected line must be covered before it is accepted.</p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Minimum segment length</label>
                    <input
                      type="range"
                      min={0.1}
                      max={30.0}
                      step={0.1}
                      value={dragValue('minSegmentS', appSettings.multi_speaker_min_segment_s ?? 2.0)}
                      onChange={onDrag('minSegmentS')}
                      onPointerUp={onCommit('minSegmentS', v => updateAppSettingsMutation.mutate({ multi_speaker_min_segment_s: v }))}
                      onKeyUp={onCommit('minSegmentS', v => updateAppSettingsMutation.mutate({ multi_speaker_min_segment_s: v }))}
                      className="mt-1 w-full"
                      data-testid="minSegmentS-slider"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>0.1 s</span>
                      <span data-testid="minSegmentS-value">{dragValue('minSegmentS', appSettings.multi_speaker_min_segment_s ?? 2.0).toFixed(1)}</span>
                      <span>30 s</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">Shorter detected segments than this are discarded.</p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Minimum backing ratio</label>
                    <input
                      type="range"
                      min={0.0}
                      max={1.0}
                      step={0.01}
                      value={dragValue('minBackingRatio', appSettings.multi_speaker_min_backing_ratio ?? 0.01)}
                      onChange={onDrag('minBackingRatio')}
                      onPointerUp={onCommit('minBackingRatio', v => updateAppSettingsMutation.mutate({ multi_speaker_min_backing_ratio: v }))}
                      onKeyUp={onCommit('minBackingRatio', v => updateAppSettingsMutation.mutate({ multi_speaker_min_backing_ratio: v }))}
                      className="mt-1 w-full"
                      data-testid="minBackingRatio-slider"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>0.0</span>
                      <span data-testid="minBackingRatio-value">{dragValue('minBackingRatio', appSettings.multi_speaker_min_backing_ratio ?? 0.01).toFixed(2)}</span>
                      <span>1.0</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">Backing stems quieter than this fraction of the lead are ignored.</p>
                  </div>
                </div>
              )}
            </div>

            {/* Fork HQ vocal lane */}
            <div>
              <SectionHeader id="forkhq" title="Fork HQ Vocal Lane" />
              {expandedSections.has('forkhq') && appSettings && (
                <div className="mt-2 p-3 bg-gray-750 rounded-lg space-y-4">

                  <div>
                    <label className="flex items-center gap-2 text-sm text-gray-400">
                      <input
                        type="checkbox"
                        checked={appSettings.fork_hq_match_source_bandwidth ?? true}
                        onChange={e => updateAppSettingsMutation.mutate({ fork_hq_match_source_bandwidth: e.target.checked })}
                        data-testid="match-source-bandwidth-checkbox"
                      />
                      Match source bandwidth
                    </label>
                    <p className="mt-1 text-xs text-gray-500">The decoder is full-band; a separated stem off a lossy encode usually is not. Measured on a reference song the render carried +25 dB more 16–22 kHz energy than the source had. On by default; a no-op on a genuinely full-band source.</p>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Converted vocal stereo width</label>
                    <input
                      type="range"
                      min={0.0}
                      max={1.0}
                      step={0.01}
                      value={dragValue('forkHqStereoWidth', appSettings.fork_hq_stereo_width ?? 0.0)}
                      onChange={onDrag('forkHqStereoWidth')}
                      onPointerUp={onCommit('forkHqStereoWidth', v => updateAppSettingsMutation.mutate({ fork_hq_stereo_width: v }))}
                      onKeyUp={onCommit('forkHqStereoWidth', v => updateAppSettingsMutation.mutate({ fork_hq_stereo_width: v }))}
                      className="mt-1 w-full"
                      data-testid="forkHqStereoWidth-slider"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Centred (0.0)</span>
                      <span data-testid="forkHqStereoWidth-value">{dragValue('forkHqStereoWidth', appSettings.fork_hq_stereo_width ?? 0.0).toFixed(2)}</span>
                      <span>Wide (1.0)</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">0.0 keeps the historical hard-centred vocal. Costs two extra conversion passes. Above ~0.5 the side channel reads as a second, phasey take rather than width.</p>
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
                      aria-label="Use GPU for pitch extraction"
                      aria-pressed={pitchConfig.use_gpu}
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
                      value={dragValue('voiceGain', audioRouterConfig.voice_gain)}
                      onChange={onDrag('voiceGain')}
                      onPointerUp={onCommit('voiceGain', v => updateAudioRouterMutation.mutate({ voice_gain: v }))}
                      onKeyUp={onCommit('voiceGain', v => updateAudioRouterMutation.mutate({ voice_gain: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="text-xs text-gray-500 text-right">
                      {(dragValue('voiceGain', audioRouterConfig.voice_gain) * 100).toFixed(0)}%
                    </div>
                  </div>

                  <div>
                    <label className="text-sm text-gray-400">Instrumental Gain</label>
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.1}
                      value={dragValue('instrumentalGain', audioRouterConfig.instrumental_gain)}
                      onChange={onDrag('instrumentalGain')}
                      onPointerUp={onCommit('instrumentalGain', v => updateAudioRouterMutation.mutate({ instrumental_gain: v }))}
                      onKeyUp={onCommit('instrumentalGain', v => updateAudioRouterMutation.mutate({ instrumental_gain: v }))}
                      className="mt-1 w-full"
                    />
                    <div className="text-xs text-gray-500 text-right">
                      {(dragValue('instrumentalGain', audioRouterConfig.instrumental_gain) * 100).toFixed(0)}%
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
