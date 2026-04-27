import { useState } from 'react'
import type { ReactNode } from 'react'
import clsx from 'clsx'
import { AlertTriangle, Loader2, X } from 'lucide-react'

interface ConfirmActionButtonProps {
  label: ReactNode
  confirmLabel?: ReactNode
  confirmMessage: string
  onConfirm: () => Promise<void> | void
  disabled?: boolean
  pending?: boolean
  variant?: 'danger' | 'neutral'
  className?: string
  testId?: string
}

export function ConfirmActionButton({
  label,
  confirmLabel = 'Confirm',
  confirmMessage,
  onConfirm,
  disabled = false,
  pending = false,
  variant = 'danger',
  className,
  testId,
}: ConfirmActionButtonProps) {
  const [confirming, setConfirming] = useState(false)

  const buttonClasses = clsx(
    'rounded-lg px-3 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50',
    variant === 'danger'
      ? 'bg-rose-500/15 text-rose-200 hover:bg-rose-500/25'
      : 'bg-gray-700 text-gray-100 hover:bg-gray-600',
    className
  )

  if (!confirming) {
    return (
      <button
        type="button"
        data-testid={testId}
        disabled={disabled || pending}
        onClick={() => setConfirming(true)}
        className={buttonClasses}
      >
        {pending ? <Loader2 className="h-4 w-4 animate-spin" /> : label}
      </button>
    )
  }

  return (
      <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3" data-testid={testId ? `${testId}-confirm` : undefined}>
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-200" />
        <div className="min-w-0 flex-1 text-sm text-amber-100">
          <div className="font-medium">Confirm action</div>
          <p className="mt-1 text-xs text-amber-200/90">{confirmMessage}</p>
        </div>
      </div>
      <div className="mt-3 flex items-center gap-2">
        <button
          type="button"
          data-testid={testId ? `${testId}-accept` : undefined}
          disabled={pending}
          onClick={async () => {
            await onConfirm()
            setConfirming(false)
          }}
          className="inline-flex items-center gap-2 rounded-lg bg-amber-400 px-3 py-2 text-sm font-medium text-gray-950 hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {pending ? <Loader2 className="h-4 w-4 animate-spin" /> : confirmLabel}
        </button>
        <button
          type="button"
          data-testid={testId ? `${testId}-cancel` : undefined}
          disabled={pending}
          onClick={() => setConfirming(false)}
          className="inline-flex items-center gap-2 rounded-lg border border-gray-600 px-3 py-2 text-sm text-gray-200 hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <X className="h-4 w-4" />
          Cancel
        </button>
      </div>
    </div>
  )
}
