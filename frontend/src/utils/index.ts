// Accessibility utilities
export {
  announceToScreenReader,
  createFocusTrap,
  prefersReducedMotion,
  getContrastRatio,
  meetsContrastAA,
  generateAriaId,
  Keys,
  handleListNavigation,
} from './accessibility'

// Performance utilities
export {
  measureRender,
  observeWebVitals,
  checkBudget,
  debounce,
  throttle,
  requestIdleCallback,
  cancelIdleCallback,
  DEFAULT_BUDGET,
} from './performance'
export type { WebVitals, PerformanceBudget } from './performance'
