import { useState, useEffect, useCallback } from 'react'

/**
 * Hook for persisting state to localStorage with automatic sync
 */
export function usePersistedState<T>(
  key: string,
  defaultValue: T,
  options?: {
    serialize?: (value: T) => string
    deserialize?: (stored: string) => T
    syncAcrossTabs?: boolean
  }
): [T, (value: T | ((prev: T) => T)) => void, () => void] {
  const {
    serialize = JSON.stringify,
    deserialize = JSON.parse,
    syncAcrossTabs = true,
  } = options ?? {}

  // Initialize state from localStorage
  const [state, setState] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key)
      if (stored !== null) {
        return deserialize(stored)
      }
    } catch (e) {
      console.warn(`Failed to load persisted state for key "${key}":`, e)
    }
    return defaultValue
  })

  // Persist state changes to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(key, serialize(state))
    } catch (e) {
      console.warn(`Failed to persist state for key "${key}":`, e)
    }
  }, [key, state, serialize])

  // Sync across tabs
  useEffect(() => {
    if (!syncAcrossTabs) return

    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== null) {
        try {
          setState(deserialize(e.newValue))
        } catch (err) {
          console.warn(`Failed to sync state for key "${key}":`, err)
        }
      }
    }

    window.addEventListener('storage', handleStorageChange)
    return () => window.removeEventListener('storage', handleStorageChange)
  }, [key, deserialize, syncAcrossTabs])

  // Reset to default value
  const reset = useCallback(() => {
    setState(defaultValue)
    localStorage.removeItem(key)
  }, [key, defaultValue])

  return [state, setState, reset]
}

/**
 * Storage keys used by the application
 */
export const STORAGE_KEYS = {
  UI_CONFIG: 'autovoice_ui_config',
  NOTIFICATIONS: 'autovoice_notifications',
  PIPELINE_PREFERENCE: 'autovoice_preferred_pipeline',
  CONVERSION_SETTINGS: 'autovoice_conversion_settings',
  TRAINING_SETTINGS: 'autovoice_training_settings',
  AUDIO_SETTINGS: 'autovoice_audio_settings',
  RECENT_PROFILES: 'autovoice_recent_profiles',
  VIEW_PREFERENCES: 'autovoice_view_preferences',
  DEBUG_SETTINGS: 'autovoice_debug_settings',
} as const

/**
 * Explicit boundary for client-side persistence.
 *
 * These keys are local-only browser preferences. Product state such as
 * profiles, training jobs, conversion history, checkpoints, stems, and audio
 * routing live on the server and must be fetched through the API.
 */
export const STORAGE_BOUNDARIES = {
  UI_CONFIG: 'local-only UI presentation preferences',
  NOTIFICATIONS: 'local-only browser notification and webhook preferences',
  PIPELINE_PREFERENCE: 'local-only preferred default conversion pipeline',
  CONVERSION_SETTINGS: 'local-only last-used conversion form selections',
  TRAINING_SETTINGS: 'reserved local-only training form defaults',
  AUDIO_SETTINGS: 'reserved local-only client audio preferences',
  RECENT_PROFILES: 'local-only quick-access profile history',
  VIEW_PREFERENCES: 'local-only layout and browsing preferences',
  DEBUG_SETTINGS: 'local-only diagnostics presentation preferences',
} as const

/**
 * Default values for persisted settings
 */
export const DEFAULT_VIEW_PREFERENCES = {
  sidebarCollapsed: false,
  defaultTab: 'convert',
  historyPageSize: 20,
  showQualityMetrics: true,
  showGPUMonitor: true,
  compactMode: false,
}

export const DEFAULT_DEBUG_SETTINGS = {
  logLevel: 'INFO' as const,
  maxLogs: 500,
  autoScroll: true,
  showTimestamps: true,
  streamLogs: true,
}

/**
 * Hook for managing view preferences
 */
export function useViewPreferences() {
  return usePersistedState(STORAGE_KEYS.VIEW_PREFERENCES, DEFAULT_VIEW_PREFERENCES)
}

/**
 * Hook for managing debug settings
 */
export function useDebugSettings() {
  return usePersistedState(STORAGE_KEYS.DEBUG_SETTINGS, DEFAULT_DEBUG_SETTINGS)
}

/**
 * Hook for managing recent profile IDs (for quick access)
 */
export function useRecentProfiles(maxRecent = 5) {
  const [profiles, setProfiles, reset] = usePersistedState<string[]>(
    STORAGE_KEYS.RECENT_PROFILES,
    []
  )

  const addProfile = useCallback((profileId: string) => {
    setProfiles(prev => {
      const filtered = prev.filter(id => id !== profileId)
      return [profileId, ...filtered].slice(0, maxRecent)
    })
  }, [setProfiles, maxRecent])

  const removeProfile = useCallback((profileId: string) => {
    setProfiles(prev => prev.filter(id => id !== profileId))
  }, [setProfiles])

  return { profiles, addProfile, removeProfile, reset }
}

/**
 * Utility to clear all persisted data
 */
export function clearAllPersistedData() {
  Object.values(STORAGE_KEYS).forEach(key => {
    localStorage.removeItem(key)
  })
}

/**
 * Utility to export all persisted data
 */
export function exportPersistedData(): Record<string, unknown> {
  const data: Record<string, unknown> = {}
  Object.entries(STORAGE_KEYS).forEach(([name, key]) => {
    try {
      const stored = localStorage.getItem(key)
      if (stored) {
        data[name] = JSON.parse(stored)
      }
    } catch {
      // Skip unparseable data
    }
  })
  return data
}

/**
 * Utility to import persisted data
 */
export function importPersistedData(data: Record<string, unknown>) {
  Object.entries(STORAGE_KEYS).forEach(([name, key]) => {
    if (name in data) {
      try {
        localStorage.setItem(key, JSON.stringify(data[name]))
      } catch (e) {
        console.warn(`Failed to import ${name}:`, e)
      }
    }
  })
}
