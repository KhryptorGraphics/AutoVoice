import clsx from 'clsx'

export interface ToastContainerProps {
  children: React.ReactNode
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center'
}

/**
 * Position styles for toast container
 * Maps position prop to Tailwind CSS classes for fixed positioning
 */
const POSITION_STYLES: Record<NonNullable<ToastContainerProps['position']>, string> = {
  'top-right': 'top-4 right-4',
  'top-left': 'top-4 left-4',
  'bottom-right': 'bottom-4 right-4',
  'bottom-left': 'bottom-4 left-4',
  'top-center': 'top-4 left-1/2 -translate-x-1/2',
  'bottom-center': 'bottom-4 left-1/2 -translate-x-1/2',
}

/**
 * ToastContainer - Container component for managing multiple toast notifications
 *
 * Positions toasts in a fixed location on screen (default: top-right) and stacks
 * them vertically with proper spacing. Uses high z-index to appear above other content.
 *
 * @example
 * ```tsx
 * <ToastContainer position="top-right">
 *   <Toast id="1" message="Success!" variant="success" />
 *   <Toast id="2" message="Error occurred" variant="error" />
 * </ToastContainer>
 * ```
 */
export function ToastContainer({ children, position = 'top-right' }: ToastContainerProps) {
  return (
    <div
      className={clsx(
        // Fixed positioning with high z-index to appear above all content
        'fixed z-50',
        // Flexbox layout for vertical stacking
        'flex flex-col gap-3',
        // Max width for readability, full width on mobile
        'max-w-md w-full px-4',
        // Pointer events handling - container doesn't block clicks but toasts do
        'pointer-events-none',
        // Position based on prop
        POSITION_STYLES[position]
      )}
      aria-live="polite"
      aria-atomic="false"
      role="region"
      aria-label="Notifications"
    >
      {/* Inner wrapper with pointer-events-auto so toasts can be clicked/closed */}
      <div className="flex flex-col gap-3 pointer-events-auto">
        {children}
      </div>
    </div>
  )
}
