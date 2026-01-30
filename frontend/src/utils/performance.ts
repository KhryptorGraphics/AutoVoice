/**
 * Performance monitoring utilities
 */

/**
 * Measure component render time
 */
export function measureRender(componentName: string): () => void {
  const start = performance.now()
  return () => {
    const duration = performance.now() - start
    if (duration > 16) {
      // Log slow renders (> 1 frame at 60fps)
      console.warn(`[Performance] Slow render in ${componentName}: ${duration.toFixed(2)}ms`)
    }
  }
}

/**
 * Track Core Web Vitals
 */
export interface WebVitals {
  lcp?: number // Largest Contentful Paint
  fid?: number // First Input Delay
  cls?: number // Cumulative Layout Shift
  fcp?: number // First Contentful Paint
  ttfb?: number // Time to First Byte
}

export function observeWebVitals(callback: (vitals: WebVitals) => void): void {
  const vitals: WebVitals = {}

  // Largest Contentful Paint
  if ('PerformanceObserver' in window) {
    try {
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        const lastEntry = entries[entries.length - 1]
        vitals.lcp = lastEntry.startTime
        callback({ ...vitals })
      })
      lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true })
    } catch {
      // LCP not supported
    }

    // First Input Delay
    try {
      const fidObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries()
        const firstEntry = entries[0] as PerformanceEventTiming
        vitals.fid = firstEntry.processingStart - firstEntry.startTime
        callback({ ...vitals })
      })
      fidObserver.observe({ type: 'first-input', buffered: true })
    } catch {
      // FID not supported
    }

    // Cumulative Layout Shift
    try {
      let clsValue = 0
      const clsObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (!(entry as LayoutShift).hadRecentInput) {
            clsValue += (entry as LayoutShift).value
          }
        }
        vitals.cls = clsValue
        callback({ ...vitals })
      })
      clsObserver.observe({ type: 'layout-shift', buffered: true })
    } catch {
      // CLS not supported
    }
  }

  // First Contentful Paint (from Navigation Timing)
  try {
    const paintEntries = performance.getEntriesByType('paint')
    const fcpEntry = paintEntries.find(e => e.name === 'first-contentful-paint')
    if (fcpEntry) {
      vitals.fcp = fcpEntry.startTime
      callback({ ...vitals })
    }
  } catch {
    // Paint timing not supported
  }

  // Time to First Byte (from Navigation Timing)
  try {
    const navEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
    if (navEntry) {
      vitals.ttfb = navEntry.responseStart - navEntry.requestStart
      callback({ ...vitals })
    }
  } catch {
    // Navigation timing not supported
  }
}

/**
 * Performance budget checker
 */
export interface PerformanceBudget {
  lcp?: number // ms
  fid?: number // ms
  cls?: number
  bundleSize?: number // bytes
  imageSize?: number // bytes
}

export const DEFAULT_BUDGET: PerformanceBudget = {
  lcp: 2500, // Good LCP is < 2.5s
  fid: 100, // Good FID is < 100ms
  cls: 0.1, // Good CLS is < 0.1
  bundleSize: 500 * 1024, // 500KB
  imageSize: 100 * 1024, // 100KB per image
}

export function checkBudget(vitals: WebVitals, budget = DEFAULT_BUDGET): string[] {
  const violations: string[] = []

  if (vitals.lcp && budget.lcp && vitals.lcp > budget.lcp) {
    violations.push(`LCP ${vitals.lcp.toFixed(0)}ms exceeds budget of ${budget.lcp}ms`)
  }
  if (vitals.fid && budget.fid && vitals.fid > budget.fid) {
    violations.push(`FID ${vitals.fid.toFixed(0)}ms exceeds budget of ${budget.fid}ms`)
  }
  if (vitals.cls && budget.cls && vitals.cls > budget.cls) {
    violations.push(`CLS ${vitals.cls.toFixed(3)} exceeds budget of ${budget.cls}`)
  }

  return violations
}

/**
 * Debounce function for performance optimization
 */
export function debounce<T extends (...args: Parameters<T>) => ReturnType<T>>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: number | undefined

  return (...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    timeoutId = window.setTimeout(() => {
      fn(...args)
    }, delay)
  }
}

/**
 * Throttle function for performance optimization
 */
export function throttle<T extends (...args: Parameters<T>) => ReturnType<T>>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      fn(...args)
      inThrottle = true
      setTimeout(() => {
        inThrottle = false
      }, limit)
    }
  }
}

/**
 * Request idle callback with fallback
 */
export function requestIdleCallback(
  callback: () => void,
  options?: { timeout?: number }
): number {
  if ('requestIdleCallback' in window) {
    return (window as Window & { requestIdleCallback: (cb: () => void, opts?: { timeout?: number }) => number }).requestIdleCallback(callback, options)
  }
  return setTimeout(callback, options?.timeout ?? 1) as unknown as number
}

/**
 * Cancel idle callback with fallback
 */
export function cancelIdleCallback(id: number): void {
  if ('cancelIdleCallback' in window) {
    (window as Window & { cancelIdleCallback: (id: number) => void }).cancelIdleCallback(id)
  } else {
    clearTimeout(id)
  }
}

// Type definitions for Layout Shift
interface LayoutShift extends PerformanceEntry {
  hadRecentInput: boolean
  value: number
}
