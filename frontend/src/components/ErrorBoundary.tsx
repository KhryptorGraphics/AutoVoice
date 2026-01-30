import { Component, ErrorInfo, ReactNode } from 'react'
import { AlertTriangle, RefreshCw, Home, Bug } from 'lucide-react'

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
  resetKeys?: unknown[]
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    }
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ errorInfo })
    this.props.onError?.(error, errorInfo)

    // Log to console in development
    if (import.meta.env.DEV) {
      console.error('Error caught by boundary:', error, errorInfo)
    }
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps) {
    // Reset error state if resetKeys change
    if (this.state.hasError && this.props.resetKeys) {
      const keysChanged = this.props.resetKeys.some(
        (key, index) => key !== prevProps.resetKeys?.[index]
      )
      if (keysChanged) {
        this.resetError()
      }
    }
  }

  resetError = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <ErrorFallback
          error={this.state.error}
          errorInfo={this.state.errorInfo}
          onRetry={this.resetError}
        />
      )
    }

    return this.props.children
  }
}

interface ErrorFallbackProps {
  error: Error | null
  errorInfo: ErrorInfo | null
  onRetry?: () => void
}

export function ErrorFallback({ error, errorInfo, onRetry }: ErrorFallbackProps) {
  const handleReload = () => {
    window.location.reload()
  }

  const handleGoHome = () => {
    window.location.href = '/'
  }

  const exportErrorReport = () => {
    const report = {
      timestamp: new Date().toISOString(),
      error: {
        message: error?.message,
        name: error?.name,
        stack: error?.stack,
      },
      componentStack: errorInfo?.componentStack,
      userAgent: navigator.userAgent,
      url: window.location.href,
    }

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `autovoice-error-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="min-h-[400px] flex items-center justify-center p-8">
      <div className="max-w-md w-full bg-gray-800 rounded-lg p-6 text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-900/30 flex items-center justify-center">
          <AlertTriangle size={32} className="text-red-400" />
        </div>

        <h2 className="text-xl font-semibold mb-2">Something went wrong</h2>
        <p className="text-gray-400 text-sm mb-4">
          An unexpected error occurred. You can try again or return to the home page.
        </p>

        {error && (
          <div className="mb-4 p-3 bg-gray-900 rounded text-left">
            <div className="text-xs text-gray-500 mb-1">Error message:</div>
            <div className="text-sm text-red-400 font-mono break-all">
              {error.message}
            </div>
          </div>
        )}

        <div className="flex flex-col gap-2">
          {onRetry && (
            <button
              onClick={onRetry}
              className="flex items-center justify-center gap-2 w-full py-2 bg-blue-600 hover:bg-blue-700 rounded transition-colors"
            >
              <RefreshCw size={16} />
              Try Again
            </button>
          )}

          <button
            onClick={handleReload}
            className="flex items-center justify-center gap-2 w-full py-2 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >
            <RefreshCw size={16} />
            Reload Page
          </button>

          <button
            onClick={handleGoHome}
            className="flex items-center justify-center gap-2 w-full py-2 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >
            <Home size={16} />
            Go to Home
          </button>

          <button
            onClick={exportErrorReport}
            className="flex items-center justify-center gap-2 w-full py-2 text-gray-400 hover:text-white transition-colors text-sm"
          >
            <Bug size={14} />
            Export Error Report
          </button>
        </div>
      </div>
    </div>
  )
}

/**
 * Higher-order component to wrap a component with error boundary
 */
export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ErrorBoundaryProps, 'children'>
) {
  return function WithErrorBoundary(props: P) {
    return (
      <ErrorBoundary {...errorBoundaryProps}>
        <WrappedComponent {...props} />
      </ErrorBoundary>
    )
  }
}
