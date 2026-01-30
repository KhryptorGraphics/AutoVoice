import clsx from 'clsx'
import { prefersReducedMotion } from '../utils/accessibility'

interface SkeletonProps {
  className?: string
  variant?: 'text' | 'circular' | 'rectangular'
  width?: number | string
  height?: number | string
  animation?: boolean
}

export function Skeleton({
  className,
  variant = 'text',
  width,
  height,
  animation = true,
}: SkeletonProps) {
  const useAnimation = animation && !prefersReducedMotion()

  const baseClasses = clsx(
    'bg-gray-700',
    useAnimation && 'animate-pulse',
    variant === 'text' && 'h-4 rounded',
    variant === 'circular' && 'rounded-full',
    variant === 'rectangular' && 'rounded-lg',
    className
  )

  return (
    <div
      className={baseClasses}
      style={{
        width: width ?? (variant === 'circular' ? 40 : '100%'),
        height: height ?? (variant === 'circular' ? 40 : variant === 'rectangular' ? 100 : undefined),
      }}
      aria-hidden="true"
    />
  )
}

/**
 * Skeleton for a card component
 */
export function CardSkeleton({ className }: { className?: string }) {
  return (
    <div className={clsx('bg-gray-800 rounded-lg p-4 space-y-3', className)}>
      <div className="flex items-center gap-3">
        <Skeleton variant="circular" width={40} height={40} />
        <div className="flex-1 space-y-2">
          <Skeleton width="60%" />
          <Skeleton width="40%" />
        </div>
      </div>
      <Skeleton variant="rectangular" height={80} />
      <div className="flex gap-2">
        <Skeleton width={80} height={32} className="rounded" />
        <Skeleton width={80} height={32} className="rounded" />
      </div>
    </div>
  )
}

/**
 * Skeleton for a table row
 */
export function TableRowSkeleton({ columns = 5 }: { columns?: number }) {
  return (
    <tr className="border-b border-gray-700">
      {Array.from({ length: columns }).map((_, i) => (
        <td key={i} className="p-3">
          <Skeleton width={`${Math.random() * 40 + 40}%`} />
        </td>
      ))}
    </tr>
  )
}

/**
 * Skeleton for a form field
 */
export function FormFieldSkeleton() {
  return (
    <div className="space-y-2">
      <Skeleton width={100} height={16} />
      <Skeleton height={40} className="rounded" />
    </div>
  )
}

/**
 * Skeleton for a chart
 */
export function ChartSkeleton({ height = 200 }: { height?: number }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <Skeleton width={150} height={20} className="mb-4" />
      <Skeleton variant="rectangular" height={height} />
    </div>
  )
}

/**
 * Skeleton for the GPU monitor
 */
export function GPUMonitorSkeleton() {
  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center gap-2">
        <Skeleton variant="circular" width={24} height={24} />
        <Skeleton width={120} />
      </div>
      <div className="grid grid-cols-3 gap-4">
        <FormFieldSkeleton />
        <FormFieldSkeleton />
        <FormFieldSkeleton />
      </div>
      <ChartSkeleton height={100} />
    </div>
  )
}

/**
 * Skeleton for voice profile card
 */
export function ProfileCardSkeleton() {
  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-3">
      <div className="flex items-center gap-3">
        <Skeleton variant="circular" width={48} height={48} />
        <div className="flex-1 space-y-2">
          <Skeleton width="50%" height={18} />
          <Skeleton width="30%" height={14} />
        </div>
      </div>
      <div className="flex gap-2 pt-2">
        <Skeleton width={60} height={28} className="rounded" />
        <Skeleton width={60} height={28} className="rounded" />
        <Skeleton width={60} height={28} className="rounded" />
      </div>
    </div>
  )
}
