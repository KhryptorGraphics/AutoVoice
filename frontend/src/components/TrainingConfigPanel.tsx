import { useState, useEffect } from 'react'
import { Settings, Info, ChevronDown, ChevronUp } from 'lucide-react'
import { TrainingConfig, DEFAULT_TRAINING_CONFIG } from '../services/api'
import clsx from 'clsx'

interface TrainingConfigPanelProps {
  config: TrainingConfig
  onChange: (config: TrainingConfig) => void
  disabled?: boolean
}

interface SliderInputProps {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  unit?: string
  tooltip?: string
  disabled?: boolean
}

function SliderInput({ label, value, onChange, min, max, step, unit, tooltip, disabled }: SliderInputProps) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center">
        <label className="text-sm text-gray-400 flex items-center gap-1">
          {label}
          {tooltip && (
            <span title={tooltip} className="cursor-help">
              <Info size={12} className="text-gray-500" />
            </span>
          )}
        </label>
        <span className="text-sm font-mono">
          {value}{unit}
        </span>
      </div>
      <input
        type="range"
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed accent-blue-500"
      />
    </div>
  )
}

interface NumberInputProps {
  label: string
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  scientific?: boolean
  tooltip?: string
  disabled?: boolean
}

function NumberInput({ label, value, onChange, min, max, step, scientific, tooltip, disabled }: NumberInputProps) {
  const [inputValue, setInputValue] = useState(scientific ? value.toExponential() : String(value))

  useEffect(() => {
    setInputValue(scientific ? value.toExponential() : String(value))
  }, [value, scientific])

  const handleBlur = () => {
    const parsed = parseFloat(inputValue)
    if (!isNaN(parsed)) {
      onChange(parsed)
    } else {
      setInputValue(scientific ? value.toExponential() : String(value))
    }
  }

  return (
    <div className="space-y-1">
      <label className="text-sm text-gray-400 flex items-center gap-1">
        {label}
        {tooltip && (
          <span title={tooltip} className="cursor-help">
            <Info size={12} className="text-gray-500" />
          </span>
        )}
      </label>
      <input
        type="text"
        value={inputValue}
        onChange={e => setInputValue(e.target.value)}
        onBlur={handleBlur}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm font-mono focus:outline-none focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
      />
    </div>
  )
}

interface ToggleProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  tooltip?: string
  disabled?: boolean
}

function Toggle({ label, checked, onChange, tooltip, disabled }: ToggleProps) {
  return (
    <div className="flex items-center justify-between">
      <label className="text-sm text-gray-400 flex items-center gap-1">
        {label}
        {tooltip && (
          <span title={tooltip} className="cursor-help">
            <Info size={12} className="text-gray-500" />
          </span>
        )}
      </label>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        disabled={disabled}
        className={clsx(
          'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
          checked ? 'bg-blue-600' : 'bg-gray-600',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      >
        <span
          className={clsx(
            'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
            checked ? 'translate-x-6' : 'translate-x-1'
          )}
        />
      </button>
    </div>
  )
}

export function TrainingConfigPanel({ config, onChange, disabled }: TrainingConfigPanelProps) {
  const [expanded, setExpanded] = useState(false)

  const update = <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => {
    onChange({ ...config, [key]: value })
  }

  const resetToDefaults = () => {
    onChange(DEFAULT_TRAINING_CONFIG)
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Settings size={18} className="text-gray-400" />
          <h3 className="font-semibold">Training Configuration</h3>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-sm text-gray-400 hover:text-white"
        >
          {expanded ? 'Collapse' : 'Expand'}
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {/* Training Mode Selector */}
      <div className="space-y-2">
        <label className="text-sm text-gray-400 flex items-center gap-1">
          Training Mode
          <span title="LoRA: Fast fine-tuning (~1MB). Full: Train from scratch for higher quality (~184MB, needs 1+ hour of audio)." className="cursor-help">
            <Info size={12} className="text-gray-500" />
          </span>
        </label>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => {
              onChange({
                ...config,
                training_mode: 'lora',
                epochs: config.epochs > 200 ? 100 : config.epochs,
                learning_rate: 1e-4,
              })
            }}
            disabled={disabled}
            className={clsx(
              'p-3 rounded-lg border-2 transition-all text-left',
              config.training_mode === 'lora'
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-gray-600 hover:border-gray-500',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
          >
            <div className="font-medium">LoRA Fine-tune</div>
            <div className="text-xs text-gray-400 mt-1">Fast • ~1MB • 1-10 min audio</div>
          </button>
          <button
            onClick={() => {
              onChange({
                ...config,
                training_mode: 'full',
                epochs: config.epochs < 200 ? 500 : config.epochs,
                learning_rate: 5e-5,
              })
            }}
            disabled={disabled}
            className={clsx(
              'p-3 rounded-lg border-2 transition-all text-left',
              config.training_mode === 'full'
                ? 'border-purple-500 bg-purple-500/10'
                : 'border-gray-600 hover:border-gray-500',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
          >
            <div className="font-medium">Full Training</div>
            <div className="text-xs text-gray-400 mt-1">Higher quality • ~184MB • 1+ hour audio</div>
          </button>
        </div>
      </div>

      {/* Always visible: key parameters */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {config.training_mode === 'lora' && (
          <SliderInput
            label="LoRA Rank"
            value={config.lora_rank}
            onChange={v => update('lora_rank', v)}
            min={1}
            max={64}
            step={1}
            tooltip="Higher rank = more capacity but slower training (4-16 typical)"
            disabled={disabled}
          />
        )}
        <SliderInput
          label="Epochs"
          value={config.epochs}
          onChange={v => update('epochs', v)}
          min={1}
          max={config.training_mode === 'full' ? 1000 : 200}
          step={config.training_mode === 'full' ? 10 : 1}
          tooltip={config.training_mode === 'full'
            ? "Full training typically needs 300-500+ epochs"
            : "Number of training passes through the data"}
          disabled={disabled}
        />
      </div>

      {/* Expandable advanced settings */}
      {expanded && (
        <div className="space-y-6 pt-4 border-t border-gray-700">
          {/* LoRA Settings (only show for LoRA mode) */}
          {config.training_mode === 'lora' && (
          <div>
            <h4 className="text-sm font-medium text-gray-300 mb-3">LoRA Parameters</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <NumberInput
                label="LoRA Alpha"
                value={config.lora_alpha}
                onChange={v => update('lora_alpha', v)}
                min={1}
                tooltip="Scaling factor for LoRA weights (typically 2x rank)"
                disabled={disabled}
              />
              <SliderInput
                label="LoRA Dropout"
                value={config.lora_dropout}
                onChange={v => update('lora_dropout', v)}
                min={0}
                max={0.5}
                step={0.05}
                tooltip="Dropout rate for LoRA layers (0.1-0.2 typical)"
                disabled={disabled}
              />
            </div>
            <div className="mt-3">
              <label className="text-sm text-gray-400 mb-1 block">Target Modules</label>
              <div className="flex flex-wrap gap-2">
                {['q_proj', 'v_proj', 'k_proj', 'o_proj', 'content_encoder'].map(module => (
                  <button
                    key={module}
                    onClick={() => {
                      const modules = config.lora_target_modules.includes(module)
                        ? config.lora_target_modules.filter(m => m !== module)
                        : [...config.lora_target_modules, module]
                      update('lora_target_modules', modules)
                    }}
                    disabled={disabled}
                    className={clsx(
                      'px-2 py-1 text-xs rounded transition-colors',
                      config.lora_target_modules.includes(module)
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-400 hover:bg-gray-600',
                      disabled && 'opacity-50 cursor-not-allowed'
                    )}
                  >
                    {module}
                  </button>
                ))}
              </div>
            </div>
          </div>
          )}

          {/* Training Parameters */}
          <div>
            <h4 className="text-sm font-medium text-gray-300 mb-3">Training Parameters</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <NumberInput
                label="Learning Rate"
                value={config.learning_rate}
                onChange={v => update('learning_rate', v)}
                scientific
                tooltip="Optimizer learning rate (1e-4 to 1e-6 typical)"
                disabled={disabled}
              />
              <div>
                <label className="text-sm text-gray-400 mb-1 block">Batch Size</label>
                <select
                  value={config.batch_size}
                  onChange={e => update('batch_size', Number(e.target.value))}
                  disabled={disabled}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
                >
                  {[1, 2, 4, 8, 16, 32].map(size => (
                    <option key={size} value={size}>{size}</option>
                  ))}
                </select>
              </div>
              <NumberInput
                label="Warmup Steps"
                value={config.warmup_steps}
                onChange={v => update('warmup_steps', Math.round(v))}
                min={0}
                tooltip="Number of warmup steps for learning rate scheduler"
                disabled={disabled}
              />
              <NumberInput
                label="Max Gradient Norm"
                value={config.max_grad_norm}
                onChange={v => update('max_grad_norm', v)}
                min={0.1}
                tooltip="Maximum gradient norm for clipping (prevents exploding gradients)"
                disabled={disabled}
              />
            </div>
          </div>

          {/* EWC Settings */}
          <div>
            <h4 className="text-sm font-medium text-gray-300 mb-3">Memory Preservation (EWC)</h4>
            <div className="space-y-4">
              <Toggle
                label="Enable EWC"
                checked={config.use_ewc}
                onChange={v => update('use_ewc', v)}
                tooltip="Elastic Weight Consolidation prevents forgetting prior voice characteristics"
                disabled={disabled}
              />
              {config.use_ewc && (
                <NumberInput
                  label="EWC Lambda"
                  value={config.ewc_lambda}
                  onChange={v => update('ewc_lambda', v)}
                  min={0}
                  scientific
                  tooltip="Strength of EWC regularization (higher = more memory preservation)"
                  disabled={disabled}
                />
              )}
            </div>
          </div>

          {/* Prior Preservation */}
          <div>
            <h4 className="text-sm font-medium text-gray-300 mb-3">Prior Preservation</h4>
            <div className="space-y-4">
              <Toggle
                label="Enable Prior Preservation"
                checked={config.use_prior_preservation}
                onChange={v => update('use_prior_preservation', v)}
                tooltip="Generate prior samples to maintain general voice quality"
                disabled={disabled}
              />
              {config.use_prior_preservation && (
                <SliderInput
                  label="Prior Loss Weight"
                  value={config.prior_loss_weight}
                  onChange={v => update('prior_loss_weight', v)}
                  min={0}
                  max={1}
                  step={0.1}
                  tooltip="Weight of prior preservation loss (0.5 typical)"
                  disabled={disabled}
                />
              )}
            </div>
          </div>

          {/* Reset Button */}
          <div className="pt-2">
            <button
              onClick={resetToDefaults}
              disabled={disabled}
              className="text-sm text-gray-400 hover:text-white disabled:opacity-50"
            >
              Reset to Defaults
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
