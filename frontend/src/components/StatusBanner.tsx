import type { ReactNode } from 'react'
import clsx from 'clsx'
import { AlertCircle, AlertTriangle, CheckCircle2, Info } from 'lucide-react'

type StatusTone = 'info' | 'success' | 'warning' | 'danger'

interface StatusBannerProps {
  tone: StatusTone
  title: string
  message?: string
  details?: string[]
  actions?: ReactNode
  testId?: string
  compact?: boolean
}

const TONE_STYLES: Record<StatusTone, { wrapper: string; title: string; copy: string; icon: typeof Info }> = {
  info: {
    wrapper: 'border-sky-500/40 bg-sky-500/10',
    title: 'text-sky-100',
    copy: 'text-sky-200/90',
    icon: Info,
  },
  success: {
    wrapper: 'border-emerald-500/40 bg-emerald-500/10',
    title: 'text-emerald-100',
    copy: 'text-emerald-200/90',
    icon: CheckCircle2,
  },
  warning: {
    wrapper: 'border-amber-500/40 bg-amber-500/10',
    title: 'text-amber-100',
    copy: 'text-amber-200/90',
    icon: AlertTriangle,
  },
  danger: {
    wrapper: 'border-rose-500/40 bg-rose-500/10',
    title: 'text-rose-100',
    copy: 'text-rose-200/90',
    icon: AlertCircle,
  },
}

export function StatusBanner({
  tone,
  title,
  message,
  details,
  actions,
  testId,
  compact = false,
}: StatusBannerProps) {
  const style = TONE_STYLES[tone]
  const Icon = style.icon

  return (
    <div
      data-testid={testId}
      className={clsx(
        'rounded-xl border p-4',
        style.wrapper,
        compact ? 'space-y-2' : 'space-y-3'
      )}
    >
      <div className="flex items-start gap-3">
        <Icon className={clsx('mt-0.5 shrink-0', compact ? 'h-4 w-4' : 'h-5 w-5', style.title)} />
        <div className="min-w-0 flex-1">
          <div className={clsx('font-semibold', style.title, compact ? 'text-sm' : 'text-sm')}>
            {title}
          </div>
          {message && (
            <p className={clsx('mt-1', style.copy, compact ? 'text-xs' : 'text-sm')}>
              {message}
            </p>
          )}
        </div>
      </div>

      {details && details.length > 0 && (
        <ul className={clsx('list-disc pl-9', style.copy, compact ? 'text-xs space-y-1' : 'text-sm space-y-1')}>
          {details.map((detail) => (
            <li key={detail}>{detail}</li>
          ))}
        </ul>
      )}

      {actions && <div className="pl-8">{actions}</div>}
    </div>
  )
}
