import { AlertTriangle } from 'lucide-react'

import { RegisterAssignment, RegisterRoutingRow } from '../services/api'

interface RegisterRoutingPanelProps {
  rows?: RegisterRoutingRow[]
  /** What the operator assigned. Present even when nothing routed. */
  assignment?: RegisterAssignment
  /** Resolves a profile id to a display name; an id alone is not legible. */
  profileName: (profileId: string) => string
}

const REASON_LABEL: Record<string, string> = {
  voice_category: 'brightness gate',
  embedding: 'voice embedding',
}

// ponytail: the gate that collapses on dense mixes is worth distinguishing
// from the one added to cover it, and nothing else about `reason` matters here.
const REASON_HINT: Record<string, string> = {
  voice_category:
    'Flagged by the brightness gate, which measures the mix and loses accuracy on dense arrangements. A split that sounds wrong on this row is worth checking first.',
  embedding:
    'Flagged by the voice-embedding pass, which was added because the brightness gate collapses on dense mixes.',
}

function formatSeconds(seconds: number): string {
  return `${seconds.toFixed(1)}s`
}

function assignedRegisters(assignment?: RegisterAssignment): string[] {
  if (!assignment) return []
  return (['female', 'male'] as const).filter((key) => assignment[key])
}

export function RegisterRoutingPanel({
  rows = [],
  assignment,
  profileName,
}: RegisterRoutingPanelProps) {
  const assigned = assignedRegisters(assignment)
  if (rows.length === 0 && assigned.length === 0) return null

  const converted = rows.filter((row) => row.action === 'convert')
  const kept = rows.filter((row) => row.action !== 'convert')
  const nothingRouted = assigned.length > 0 && converted.length === 0
  const disagreements = rows.filter(
    (row) => row.measured_register && row.register && row.measured_register !== row.register
  )

  return (
    <div
      className="mt-3 w-full space-y-2 rounded-lg border border-gray-700 p-3"
      data-testid="register-routing-panel"
    >
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-200">Duet routing</h4>
        {assignment?.lead_register ? (
          <span className="text-xs text-gray-500">
            main profile sang the {assignment.lead_register} register
          </span>
        ) : null}
      </div>

      {nothingRouted && (
        <div
          className="flex gap-2 rounded border border-amber-600/60 bg-amber-500/10 p-2 text-xs text-amber-100"
          data-testid="register-routing-noop"
        >
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          <span>
            You assigned a model to the {assigned.join(' and ')} register
            {assigned.length > 1 ? 's' : ''}, but no part of this song routed to it. Nothing was
            re-sung through that model and the whole vocal is in the main profile&apos;s voice. This
            happens when no foreign voice was detected, which is common on a dense mix.
          </span>
        </div>
      )}

      {converted.length > 0 && (
        <ul className="space-y-1 text-xs text-gray-300" data-testid="register-routing-converted">
          {converted.map((row) => (
            <li key={`${row.group}-${row.reason}`} className="flex flex-wrap items-baseline gap-x-2">
              <span className="font-medium text-green-300">
                {formatSeconds(row.spans_s)} → {row.register} → {profileName(row.profile_id || '')}
              </span>
              <span className="text-gray-500" title={REASON_HINT[row.reason]}>
                {REASON_LABEL[row.reason] || row.reason}
                {row.median_f0_hz != null ? ` · median ${Math.round(row.median_f0_hz)} Hz` : ''}
              </span>
            </li>
          ))}
        </ul>
      )}

      {kept.length > 0 && (
        <ul className="space-y-1 text-xs text-gray-400" data-testid="register-routing-kept">
          {kept.map((row) => (
            <li key={`${row.group}-${row.reason}`} className="flex flex-wrap items-baseline gap-x-2">
              <span className="text-amber-300">
                {formatSeconds(row.spans_s)} kept original
                {row.register ? ` (${row.register} register)` : ' (register unknown)'}
              </span>
              <span className="text-gray-500" title={REASON_HINT[row.reason]}>
                {REASON_LABEL[row.reason] || row.reason}
                {row.median_f0_hz != null ? ` · median ${Math.round(row.median_f0_hz)} Hz` : ''}
              </span>
            </li>
          ))}
        </ul>
      )}

      {disagreements.length > 0 && (
        <p className="text-xs text-amber-200/80" data-testid="register-routing-disagreement">
          On {disagreements.length} {disagreements.length === 1 ? 'part' : 'parts'} the measured
          pitch leans{' '}
          {disagreements.map((row) => `${row.measured_register} where you assigned ${row.register}`)
            .join('; ')}
          . Treat that as a weak hint only — the two registers overlap by about 118 Hz on this
          project&apos;s own training data, so the measurement gets it wrong often enough that it is
          recorded and never acted on.
        </p>
      )}

      <p className="text-xs text-gray-500">
        Registers are not detected from the audio. Each part above was routed by which slot you put
        the main profile in. Listen to the seconds listed to check the split.
      </p>
    </div>
  )
}
