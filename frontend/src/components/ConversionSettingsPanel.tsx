import { useState } from 'react'
import { Sliders, ChevronDown, ChevronUp, RotateCcw } from 'lucide-react'

import {
  ConversionConfig,
  EMPTY_REGISTER_PROFILES,
  QUALITY_PRESETS,
  QualityPreset,
  RegisterProfiles,
  VoiceProfile,
} from '../services/api'

interface ConversionSettingsPanelProps {
  config: ConversionConfig
  onChange: (patch: Partial<ConversionConfig>) => void
  disabled?: boolean
  /** Trained target profiles offered as per-register duet voices. */
  registerProfileOptions?: VoiceProfile[]
  /** The request's main profile. It occupies whichever slot you put it in. */
  leadProfileId?: string | null
}

const PRESET_ORDER: QualityPreset[] = ['draft', 'fast', 'balanced', 'high', 'studio']

const VOCAL_REGISTERS: { key: keyof RegisterProfiles; label: string }[] = [
  { key: 'female', label: 'Female singer' },
  { key: 'male', label: 'Male singer' },
]

const otherRegister = (key: keyof RegisterProfiles): keyof RegisterProfiles =>
  key === 'female' ? 'male' : 'female'

// Maps the (enable_multi_speaker, convert_backing) pair to one select:
// auto = server default; preserve = per-speaker, harmonies kept original;
// convert = per-speaker AND harmonies converted (experimental); single = one voice
type BackingMode = 'auto' | 'preserve' | 'convert' | 'single'
const backingModeFromConfig = (ems: boolean | null, cb: boolean | null): BackingMode =>
  ems == null ? 'auto' : !ems ? 'single' : cb ? 'convert' : 'preserve'
const backingModeToConfig = (mode: BackingMode): { enable_multi_speaker: boolean | null; convert_backing: boolean | null } => {
  switch (mode) {
    case 'auto': return { enable_multi_speaker: null, convert_backing: null }
    case 'preserve': return { enable_multi_speaker: true, convert_backing: null }
    case 'convert': return { enable_multi_speaker: true, convert_backing: true }
    case 'single': return { enable_multi_speaker: false, convert_backing: null }
  }
}

function formatPitch(pitch: number): string {
  if (pitch === 0) return 'no shift'
  return `${pitch > 0 ? '+' : ''}${pitch} st`
}

function VolumeSlider({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  disabled?: boolean
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center">
        <label className="text-sm text-gray-400">{label}</label>
        <span className="text-sm font-mono">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={0}
        max={2}
        step={0.05}
        disabled={disabled}
        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed accent-blue-500"
      />
    </div>
  )
}

export function ConversionSettingsPanel({
  config,
  onChange,
  disabled,
  registerProfileOptions = [],
  leadProfileId,
}: ConversionSettingsPanelProps) {
  const [expanded, setExpanded] = useState(true)
  // A preset saved before this setting existed has no register_profiles.
  const registers = config.register_profiles ?? EMPTY_REGISTER_PROFILES

  // The register bit comes from which slot holds the main profile, so two
  // non-lead models leave nothing to say which singer is which and the
  // pipeline refuses the render. Keep that state unreachable: once one slot
  // holds a non-lead model, the other offers only the lead or no conversion.
  const optionsFor = (key: keyof RegisterProfiles) => {
    const other = registers[otherRegister(key)]
    if (!other || other === leadProfileId) return registerProfileOptions
    return registerProfileOptions.filter(
      (profile) => profile.profile_id === leadProfileId || profile.profile_id === registers[key]
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4" data-testid="conversion-settings-panel">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sliders size={18} className="text-gray-400" />
          <h3 className="font-semibold">Conversion Settings</h3>
        </div>
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-sm text-gray-400 hover:text-white"
        >
          {expanded ? 'Collapse' : 'Expand'}
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {expanded && (
        <div className="space-y-4">
          <div className="space-y-1">
            <label className="text-sm text-gray-400">Quality preset</label>
            <select
              value={config.preset}
              onChange={(e) => onChange({ preset: e.target.value as QualityPreset })}
              disabled={disabled}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
            >
              {PRESET_ORDER.map((key) => {
                const preset = QUALITY_PRESETS[key]
                return (
                  <option key={key} value={key}>
                    {preset.label} — {preset.detail}
                  </option>
                )
              })}
            </select>
          </div>

          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <label className="text-sm text-gray-400">Pitch shift</label>
              <span className="flex items-center gap-2">
                <span className="text-sm font-mono">{formatPitch(config.pitch_shift)}</span>
                <button
                  type="button"
                  onClick={() => onChange({ pitch_shift: 0 })}
                  disabled={disabled || config.pitch_shift === 0}
                  title="Reset to no shift"
                  className="text-gray-500 hover:text-white disabled:opacity-30 disabled:hover:text-gray-500"
                >
                  <RotateCcw size={14} />
                </button>
              </span>
            </div>
            <input
              type="range"
              value={config.pitch_shift}
              onChange={(e) => onChange({ pitch_shift: Number(e.target.value) })}
              min={-12}
              max={12}
              step={0.5}
              disabled={disabled}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed accent-blue-500"
            />
          </div>

          <VolumeSlider
            label="Vocal volume"
            value={config.vocal_volume}
            onChange={(value) => onChange({ vocal_volume: value })}
            disabled={disabled}
          />
          <VolumeSlider
            label="Instrumental volume"
            value={config.instrumental_volume}
            onChange={(value) => onChange({ instrumental_volume: value })}
            disabled={disabled}
          />

          <label className="flex items-center gap-3 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={config.return_stems}
              onChange={(e) => onChange({ return_stems: e.target.checked })}
              disabled={disabled}
              className="h-4 w-4 accent-blue-500 disabled:opacity-40"
            />
            Keep separated stems (vocals + instrumental downloads)
          </label>

          <div className="space-y-1">
            <label className="text-sm text-gray-400">Backing vocals</label>
            <select
              value={backingModeFromConfig(config.enable_multi_speaker, config.convert_backing)}
              onChange={(e) => onChange(backingModeToConfig(e.target.value as BackingMode))}
              disabled={disabled}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
            >
              <option value="auto">Auto (server default)</option>
              <option value="preserve">Preserve backing vocals</option>
              <option value="convert">Convert backing too (experimental)</option>
              <option value="single">Convert everything as one voice</option>
            </select>
            <p className="text-xs text-gray-500">
              Preserve: converts only the lead singer and keeps backing-vocal harmonies original (best for
              tracks with harmonies; fork-backed profiles only). Convert backing too: additionally re-sings
              each harmony line in the target voice (experimental; falls back to preserving when unsure).
            </p>
          </div>

          <div className="space-y-1">
            <label className="text-sm text-gray-400">Keep original singers</label>
            <input
              type="text"
              value={config.preserve_speakers}
              onChange={(e) => onChange({ preserve_speakers: e.target.value })}
              disabled={disabled}
              placeholder="e.g. 1:23-1:40 (time where they sing) or SPEAKER_02"
              data-testid="preserve-speakers-input"
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50 placeholder:text-gray-500"
            />
            <p className="text-xs text-gray-500">
              When the target artist already sings on this track (duets, features), give a time range where
              they sing solo — that singer is kept original instead of being re-converted. Comma-separate
              multiple ranges. Cluster ids from a previous result&apos;s speaker list also work.
            </p>
          </div>

          <div className="space-y-2" data-testid="register-profiles">
            <label className="text-sm text-gray-400">Duet voices</label>
            <div className="grid grid-cols-2 gap-2">
              {VOCAL_REGISTERS.map(({ key, label }) => (
                <div key={key} className="space-y-1">
                  <label className="text-xs text-gray-500" htmlFor={`register-${key}`}>
                    {label}
                  </label>
                  <select
                    id={`register-${key}`}
                    value={registers[key] ?? ''}
                    onChange={(e) =>
                      onChange({
                        register_profiles: { ...registers, [key]: e.target.value || null },
                      })
                    }
                    disabled={disabled}
                    data-testid={`register-profile-${key}`}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
                  >
                    <option value="">Keep original voice</option>
                    {optionsFor(key).map((profile) => (
                      <option key={profile.profile_id} value={profile.profile_id}>
                        {profile.name || profile.profile_id}
                        {profile.profile_id === leadProfileId ? ' (main profile)' : ''}
                      </option>
                    ))}
                  </select>
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-500">
              For duets. Put the main profile in the register it sings, and the other singer&apos;s
              model in the other register. A register left on &ldquo;Keep original voice&rdquo; is not
              converted at all — that singer stays as recorded. Assigning one model and leaving the
              other register alone works too: the main profile takes the register you did not fill.
              Leave both alone for a single-singer song. Nothing here is auto-detected from pitch,
              which is why the slots decide who is who.
              {registerProfileOptions.length === 0 &&
                ' No fork-backed profiles are trained yet, so there is nothing to assign.'}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
