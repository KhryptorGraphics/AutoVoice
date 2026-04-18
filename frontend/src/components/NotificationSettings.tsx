import { useState, useEffect } from 'react'
import {
  Bell, Webhook, Plus, Trash2, Check, X,
  Loader2, Volume2, VolumeX, TestTube, Edit2
} from 'lucide-react'
import clsx from 'clsx'
import { useToastContext } from '../contexts/ToastContext'
import { STORAGE_KEYS, usePersistedState } from '../hooks/usePersistedState'

interface WebhookConfig {
  id: string
  url: string
  name: string
  enabled: boolean
  events: NotificationEvent[]
  secret?: string
}

type NotificationEvent =
  | 'conversion_complete'
  | 'conversion_error'
  | 'training_complete'
  | 'training_error'
  | 'gpu_warning'
  | 'model_loaded'

interface NotificationConfig {
  browserNotifications: boolean
  soundEnabled: boolean
  soundVolume: number
  webhooks: WebhookConfig[]
  enabledEvents: NotificationEvent[]
}

const NOTIFICATION_EVENTS: { key: NotificationEvent; label: string; description: string }[] = [
  { key: 'conversion_complete', label: 'Conversion Complete', description: 'When a voice conversion finishes' },
  { key: 'conversion_error', label: 'Conversion Error', description: 'When a conversion fails' },
  { key: 'training_complete', label: 'Training Complete', description: 'When model training finishes' },
  { key: 'training_error', label: 'Training Error', description: 'When training fails' },
  { key: 'gpu_warning', label: 'GPU Warning', description: 'High temperature or memory usage' },
  { key: 'model_loaded', label: 'Model Loaded', description: 'When a model is loaded/unloaded' },
]

const STORAGE_KEY = STORAGE_KEYS.NOTIFICATIONS

const DEFAULT_CONFIG: NotificationConfig = {
  browserNotifications: false,
  soundEnabled: true,
  soundVolume: 0.5,
  webhooks: [],
  enabledEvents: ['conversion_complete', 'conversion_error', 'training_complete', 'training_error'],
}

export function NotificationSettings() {
  const toast = useToastContext()
  const [config, setConfig] = usePersistedState<NotificationConfig>(
    STORAGE_KEY,
    DEFAULT_CONFIG
  )

  const [showAddWebhook, setShowAddWebhook] = useState(false)
  const [newWebhook, setNewWebhook] = useState({ url: '', name: '' })
  const [editingWebhookId, setEditingWebhookId] = useState<string | null>(null)
  const [testingWebhookId, setTestingWebhookId] = useState<string | null>(null)
  const [permissionStatus, setPermissionStatus] = useState<NotificationPermission | null>(null)

  // Check notification permission on mount
  useEffect(() => {
    if ('Notification' in window) {
      setPermissionStatus(Notification.permission)
    }
  }, [])

  // Save config changes
  const updateConfig = (updates: Partial<NotificationConfig>) => {
    const newConfig = { ...config, ...updates }
    setConfig(newConfig)
  }

  // Request browser notification permission
  const requestNotificationPermission = async () => {
    if (!('Notification' in window)) {
      toast.error('This browser does not support notifications')
      return
    }

    const permission = await Notification.requestPermission()
    setPermissionStatus(permission)
    if (permission === 'granted') {
      updateConfig({ browserNotifications: true })
      new Notification('AutoVoice Notifications Enabled', {
        body: 'You will now receive notifications for important events.',
        icon: '/favicon.ico',
      })
    }
  }

  // Toggle event
  const toggleEvent = (event: NotificationEvent) => {
    const newEvents = config.enabledEvents.includes(event)
      ? config.enabledEvents.filter(e => e !== event)
      : [...config.enabledEvents, event]
    updateConfig({ enabledEvents: newEvents })
  }

  // Add webhook
  const addWebhook = () => {
    if (!newWebhook.url.trim()) return

    const webhook: WebhookConfig = {
      id: `webhook-${Date.now()}`,
      url: newWebhook.url.trim(),
      name: newWebhook.name.trim() || 'Unnamed Webhook',
      enabled: true,
      events: [...config.enabledEvents],
    }

    updateConfig({ webhooks: [...config.webhooks, webhook] })
    setNewWebhook({ url: '', name: '' })
    setShowAddWebhook(false)
  }

  // Delete webhook
  const deleteWebhook = (id: string) => {
    if (confirm('Delete this webhook?')) {
      updateConfig({ webhooks: config.webhooks.filter(w => w.id !== id) })
      toast.success('Webhook deleted successfully')
    }
  }

  // Toggle webhook
  const toggleWebhook = (id: string) => {
    updateConfig({
      webhooks: config.webhooks.map(w =>
        w.id === id ? { ...w, enabled: !w.enabled } : w
      ),
    })
  }

  // Toggle webhook event
  const toggleWebhookEvent = (webhookId: string, event: NotificationEvent) => {
    updateConfig({
      webhooks: config.webhooks.map(w => {
        if (w.id !== webhookId) return w
        const newEvents = w.events.includes(event)
          ? w.events.filter(e => e !== event)
          : [...w.events, event]
        return { ...w, events: newEvents }
      }),
    })
  }

  // Test webhook
  const testWebhook = async (webhook: WebhookConfig) => {
    setTestingWebhookId(webhook.id)
    try {
      const response = await fetch(webhook.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          event: 'test',
          timestamp: new Date().toISOString(),
          message: 'This is a test notification from AutoVoice',
        }),
      })
      if (response.ok) {
        toast.success('Webhook test successful!')
      } else {
        toast.error(`Webhook test failed: ${response.status} ${response.statusText}`)
      }
    } catch (err) {
      toast.error(`Webhook test failed: ${(err as Error).message}`)
    } finally {
      setTestingWebhookId(null)
    }
  }

  // Play test sound
  const playTestSound = () => {
    const audio = new Audio('/notification.mp3')
    audio.volume = config.soundVolume
    audio.play().catch(() => {
      // Fallback to Web Audio API beep
      const ctx = new AudioContext()
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.connect(gain)
      gain.connect(ctx.destination)
      gain.gain.value = config.soundVolume * 0.3
      osc.frequency.value = 880
      osc.start()
      osc.stop(ctx.currentTime + 0.15)
    })
  }

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <Bell size={18} className="text-yellow-400" />
          <h3 className="font-semibold">Notification Settings</h3>
        </div>
      </div>

      <div className="p-4 space-y-6">
        {/* Browser Notifications */}
        <div>
          <h4 className="text-sm font-medium mb-3">Browser Notifications</h4>
          <div className="bg-gray-750 rounded-lg p-3 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm">Desktop Notifications</div>
                <div className="text-xs text-gray-500">
                  {permissionStatus === 'granted'
                    ? 'Notifications enabled'
                    : permissionStatus === 'denied'
                    ? 'Notifications blocked in browser'
                    : 'Click to enable'}
                </div>
              </div>
              {permissionStatus === 'granted' ? (
                <button
                  onClick={() => updateConfig({ browserNotifications: !config.browserNotifications })}
                  className={clsx(
                    'w-10 h-5 rounded-full transition-colors relative',
                    config.browserNotifications ? 'bg-green-600' : 'bg-gray-600'
                  )}
                >
                  <div
                    className={clsx(
                      'absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform',
                      config.browserNotifications ? 'left-5' : 'left-0.5'
                    )}
                  />
                </button>
              ) : (
                <button
                  onClick={requestNotificationPermission}
                  disabled={permissionStatus === 'denied'}
                  className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 rounded text-sm"
                >
                  {permissionStatus === 'denied' ? 'Blocked' : 'Enable'}
                </button>
              )}
            </div>

            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm flex items-center gap-2">
                  {config.soundEnabled ? <Volume2 size={14} /> : <VolumeX size={14} />}
                  Sound Alerts
                </div>
                <div className="text-xs text-gray-500">Play sound on notifications</div>
              </div>
              <button
                onClick={() => updateConfig({ soundEnabled: !config.soundEnabled })}
                className={clsx(
                  'w-10 h-5 rounded-full transition-colors relative',
                  config.soundEnabled ? 'bg-green-600' : 'bg-gray-600'
                )}
              >
                <div
                  className={clsx(
                    'absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform',
                    config.soundEnabled ? 'left-5' : 'left-0.5'
                  )}
                />
              </button>
            </div>

            {config.soundEnabled && (
              <div className="flex items-center gap-3">
                <span className="text-sm text-gray-400">Volume</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.1}
                  value={config.soundVolume}
                  onChange={e => updateConfig({ soundVolume: parseFloat(e.target.value) })}
                  className="flex-1"
                />
                <span className="text-sm text-gray-500 w-8">
                  {Math.round(config.soundVolume * 100)}%
                </span>
                <button
                  onClick={playTestSound}
                  className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs"
                >
                  Test
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Event Toggles */}
        <div>
          <h4 className="text-sm font-medium mb-3">Notification Events</h4>
          <div className="bg-gray-750 rounded-lg p-3 space-y-2">
            {NOTIFICATION_EVENTS.map(event => (
              <div
                key={event.key}
                className="flex items-center justify-between py-2 border-b border-gray-700 last:border-0"
              >
                <div>
                  <div className="text-sm">{event.label}</div>
                  <div className="text-xs text-gray-500">{event.description}</div>
                </div>
                <button
                  onClick={() => toggleEvent(event.key)}
                  className={clsx(
                    'w-10 h-5 rounded-full transition-colors relative',
                    config.enabledEvents.includes(event.key) ? 'bg-blue-600' : 'bg-gray-600'
                  )}
                >
                  <div
                    className={clsx(
                      'absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform',
                      config.enabledEvents.includes(event.key) ? 'left-5' : 'left-0.5'
                    )}
                  />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Webhooks */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Webhook size={14} />
              Webhooks
            </h4>
            <button
              onClick={() => setShowAddWebhook(true)}
              className="flex items-center gap-1 px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs"
            >
              <Plus size={12} />
              Add Webhook
            </button>
          </div>

          <div className="space-y-2">
            {config.webhooks.length === 0 && !showAddWebhook ? (
              <div className="bg-gray-750 rounded-lg p-4 text-center text-gray-500 text-sm">
                No webhooks configured
              </div>
            ) : (
              <>
                {config.webhooks.map(webhook => (
                  <div
                    key={webhook.id}
                    className={clsx(
                      'bg-gray-750 rounded-lg p-3',
                      !webhook.enabled && 'opacity-60'
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <button
                        onClick={() => toggleWebhook(webhook.id)}
                        className={clsx(
                          'mt-1 w-8 h-4 rounded-full transition-colors relative shrink-0',
                          webhook.enabled ? 'bg-green-600' : 'bg-gray-600'
                        )}
                      >
                        <div
                          className={clsx(
                            'absolute top-0.5 w-3 h-3 bg-white rounded-full transition-transform',
                            webhook.enabled ? 'left-4' : 'left-0.5'
                          )}
                        />
                      </button>

                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm">{webhook.name}</div>
                        <div className="text-xs text-gray-500 truncate">{webhook.url}</div>

                        {editingWebhookId === webhook.id ? (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {NOTIFICATION_EVENTS.map(event => (
                              <button
                                key={event.key}
                                onClick={() => toggleWebhookEvent(webhook.id, event.key)}
                                className={clsx(
                                  'px-2 py-0.5 rounded text-xs transition-colors',
                                  webhook.events.includes(event.key)
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-700 text-gray-400'
                                )}
                              >
                                {event.label}
                              </button>
                            ))}
                          </div>
                        ) : (
                          <div className="mt-1 text-xs text-gray-500">
                            {webhook.events.length} events
                          </div>
                        )}
                      </div>

                      <div className="flex items-center gap-1 shrink-0">
                        <button
                          onClick={() => testWebhook(webhook)}
                          disabled={testingWebhookId === webhook.id}
                          className="p-1.5 text-gray-400 hover:text-blue-400 hover:bg-gray-700 rounded"
                          title="Test webhook"
                        >
                          {testingWebhookId === webhook.id ? (
                            <Loader2 size={14} className="animate-spin" />
                          ) : (
                            <TestTube size={14} />
                          )}
                        </button>
                        <button
                          onClick={() => setEditingWebhookId(
                            editingWebhookId === webhook.id ? null : webhook.id
                          )}
                          className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
                          title="Edit events"
                        >
                          <Edit2 size={14} />
                        </button>
                        <button
                          onClick={() => deleteWebhook(webhook.id)}
                          className="p-1.5 text-gray-400 hover:text-red-400 hover:bg-gray-700 rounded"
                          title="Delete webhook"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}

                {showAddWebhook && (
                  <div className="bg-gray-750 rounded-lg p-3 space-y-3">
                    <input
                      type="text"
                      placeholder="Webhook name (optional)"
                      value={newWebhook.name}
                      onChange={e => setNewWebhook(prev => ({ ...prev, name: e.target.value }))}
                      className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                    />
                    <input
                      type="url"
                      placeholder="https://example.com/webhook"
                      value={newWebhook.url}
                      onChange={e => setNewWebhook(prev => ({ ...prev, url: e.target.value }))}
                      className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={addWebhook}
                        disabled={!newWebhook.url.trim()}
                        className="flex-1 flex items-center justify-center gap-2 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 rounded text-sm"
                      >
                        <Check size={14} />
                        Add
                      </button>
                      <button
                        onClick={() => {
                          setShowAddWebhook(false)
                          setNewWebhook({ url: '', name: '' })
                        }}
                        className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
