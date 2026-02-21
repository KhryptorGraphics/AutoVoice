import { useToastContext } from '../contexts/ToastContext'

/**
 * Options for toast notifications
 */
export interface ToastOptions {
  /** Duration in milliseconds before toast auto-closes (default: 5000) */
  duration?: number
  /** Whether the toast should auto-close (default: true) */
  autoClose?: boolean
  /** Whether to show the close button (default: true) */
  showCloseButton?: boolean
  /** Optional action button (future expansion) */
  action?: {
    label: string
    onClick: () => void
  }
}

/**
 * Toast notification methods
 */
export interface ToastMethods {
  /** Show a success toast */
  success: (message: string, options?: ToastOptions) => string
  /** Show an error toast */
  error: (message: string, options?: ToastOptions) => string
  /** Show a warning toast */
  warning: (message: string, options?: ToastOptions) => string
  /** Show an info toast */
  info: (message: string, options?: ToastOptions) => string
}

/**
 * Hook for displaying toast notifications
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const toast = useToast()
 *
 *   const handleSave = async () => {
 *     try {
 *       await saveData()
 *       toast.success('Data saved successfully!')
 *     } catch (error) {
 *       toast.error('Failed to save data')
 *     }
 *   }
 *
 *   return <button onClick={handleSave}>Save</button>
 * }
 * ```
 *
 * @example
 * ```tsx
 * // With custom duration
 * toast.success('Profile created!', { duration: 3000 })
 *
 * // Persistent toast (no auto-close)
 * toast.error('Critical error occurred', { autoClose: false })
 *
 * // With action button (future expansion)
 * toast.info('New update available', {
 *   action: { label: 'Update', onClick: () => updateApp() }
 * })
 * ```
 *
 * @returns Object with toast methods: success, error, warning, info
 * @throws Error if used outside ToastProvider
 */
export function useToast(): ToastMethods {
  const context = useToastContext()

  return {
    success: context.success,
    error: context.error,
    warning: context.warning,
    info: context.info,
  }
}
