import React, { createContext, useContext, useState, useCallback } from 'react'
import { Toast, ToastVariant } from '../components/Toast'
import { ToastContainer } from '../components/ToastContainer'

interface ToastOptions {
  duration?: number
  autoClose?: boolean
  showCloseButton?: boolean
}

interface ToastItem {
  id: string
  message: string
  variant: ToastVariant
  duration: number
  autoClose: boolean
  showCloseButton: boolean
}

interface ToastContextValue {
  toasts: ToastItem[]
  addToast: (message: string, variant: ToastVariant, options?: ToastOptions) => string
  removeToast: (id: string) => void
  success: (message: string, options?: ToastOptions) => string
  error: (message: string, options?: ToastOptions) => string
  warning: (message: string, options?: ToastOptions) => string
  info: (message: string, options?: ToastOptions) => string
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined)

export function useToastContext() {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error('useToastContext must be used within ToastProvider')
  }
  return context
}

interface ToastProviderProps {
  children: React.ReactNode
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center'
}

export function ToastProvider({ children, position = 'top-right' }: ToastProviderProps) {
  const [toasts, setToasts] = useState<ToastItem[]>([])

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id))
  }, [])

  const addToast = useCallback(
    (message: string, variant: ToastVariant, options?: ToastOptions): string => {
      const id = `toast-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
      const toast: ToastItem = {
        id,
        message,
        variant,
        duration: options?.duration ?? 5000,
        autoClose: options?.autoClose ?? true,
        showCloseButton: options?.showCloseButton ?? true,
      }
      setToasts((prev) => [...prev, toast])
      return id
    },
    []
  )

  const success = useCallback(
    (message: string, options?: ToastOptions) => addToast(message, 'success', options),
    [addToast]
  )

  const error = useCallback(
    (message: string, options?: ToastOptions) => addToast(message, 'error', options),
    [addToast]
  )

  const warning = useCallback(
    (message: string, options?: ToastOptions) => addToast(message, 'warning', options),
    [addToast]
  )

  const info = useCallback(
    (message: string, options?: ToastOptions) => addToast(message, 'info', options),
    [addToast]
  )

  const value: ToastContextValue = {
    toasts,
    addToast,
    removeToast,
    success,
    error,
    warning,
    info,
  }

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastContainer position={position}>
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            id={toast.id}
            message={toast.message}
            variant={toast.variant}
            duration={toast.duration}
            autoClose={toast.autoClose}
            showCloseButton={toast.showCloseButton}
            onClose={removeToast}
          />
        ))}
      </ToastContainer>
    </ToastContext.Provider>
  )
}
