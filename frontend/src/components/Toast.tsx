import { useEffect } from 'react'
import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-react'
import clsx from 'clsx'

export type ToastVariant = 'success' | 'error' | 'warning' | 'info'

export interface ToastProps {
  id: string
  message: string
  variant?: ToastVariant
  duration?: number
  onClose?: (id: string) => void
  autoClose?: boolean
  showCloseButton?: boolean
}

const VARIANT_STYLES: Record<ToastVariant, {
  container: string
  icon: string
  iconBg: string
}> = {
  success: {
    container: 'border-green-600/50 bg-green-900/20',
    icon: 'text-green-400',
    iconBg: 'bg-green-900/30',
  },
  error: {
    container: 'border-red-600/50 bg-red-900/20',
    icon: 'text-red-400',
    iconBg: 'bg-red-900/30',
  },
  warning: {
    container: 'border-yellow-600/50 bg-yellow-900/20',
    icon: 'text-yellow-400',
    iconBg: 'bg-yellow-900/30',
  },
  info: {
    container: 'border-blue-600/50 bg-blue-900/20',
    icon: 'text-blue-400',
    iconBg: 'bg-blue-900/30',
  },
}

const VARIANT_ICONS: Record<ToastVariant, React.ComponentType<{ size?: number; className?: string }>> = {
  success: CheckCircle,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
}

export function Toast({
  id,
  message,
  variant = 'info',
  duration = 5000,
  onClose,
  autoClose = true,
  showCloseButton = true,
}: ToastProps) {
  const Icon = VARIANT_ICONS[variant]
  const styles = VARIANT_STYLES[variant]

  useEffect(() => {
    if (!autoClose || !onClose) return

    const timer = setTimeout(() => {
      onClose(id)
    }, duration)

    return () => clearTimeout(timer)
  }, [id, duration, autoClose, onClose])

  const handleClose = () => {
    onClose?.(id)
  }

  return (
    <div
      className={clsx(
        'flex items-start gap-3 p-4 rounded-lg border backdrop-blur-sm',
        'shadow-lg transform transition-all duration-300 ease-out',
        'animate-slide-in-right',
        styles.container
      )}
      role="alert"
      aria-live={variant === 'error' ? 'assertive' : 'polite'}
    >
      <div className={clsx(
        'flex items-center justify-center w-8 h-8 rounded-full shrink-0',
        styles.iconBg
      )}>
        <Icon size={18} className={styles.icon} />
      </div>

      <div className="flex-1 min-w-0 pt-0.5">
        <p className="text-sm text-gray-100 break-words">{message}</p>
      </div>

      {showCloseButton && (
        <button
          onClick={handleClose}
          className="shrink-0 p-1 text-gray-400 hover:text-gray-200 transition-colors rounded hover:bg-gray-700/50"
          aria-label="Close notification"
        >
          <X size={16} />
        </button>
      )}
    </div>
  )
}

export interface ToastContainerProps {
  children: React.ReactNode
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center'
}

const POSITION_STYLES: Record<NonNullable<ToastContainerProps['position']>, string> = {
  'top-right': 'top-4 right-4',
  'top-left': 'top-4 left-4',
  'bottom-right': 'bottom-4 right-4',
  'bottom-left': 'bottom-4 left-4',
  'top-center': 'top-4 left-1/2 -translate-x-1/2',
  'bottom-center': 'bottom-4 left-1/2 -translate-x-1/2',
}

export function ToastContainer({ children, position = 'top-right' }: ToastContainerProps) {
  return (
    <div
      className={clsx(
        'fixed z-50 flex flex-col gap-3 max-w-md w-full pointer-events-none',
        POSITION_STYLES[position]
      )}
      aria-live="polite"
      aria-atomic="false"
    >
      <div className="flex flex-col gap-3 pointer-events-auto">
        {children}
      </div>
    </div>
  )
}
