import { useState, useEffect } from 'react'
import { Settings, Save, RotateCcw } from 'lucide-react'

interface AppSettings {
  defaultQuality: string
  autoDownload: boolean
  preservePitch: boolean
  preserveVibrato: boolean
  preserveExpression: boolean
  denoiseInput: boolean
  enhanceOutput: boolean
  maxConcurrentJobs: number
  theme: 'light' | 'dark' | 'auto'
}

const DEFAULT_SETTINGS: AppSettings = {
  defaultQuality: 'balanced',
  autoDownload: false,
  preservePitch: true,
  preserveVibrato: true,
  preserveExpression: true,
  denoiseInput: false,
  enhanceOutput: false,
  maxConcurrentJobs: 3,
  theme: 'light',
}

export function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    const savedSettings = localStorage.getItem('appSettings')
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings))
    }
  }, [])

  const handleSave = () => {
    localStorage.setItem('appSettings', JSON.stringify(settings))
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const handleReset = () => {
    if (confirm('Reset all settings to defaults?')) {
      setSettings(DEFAULT_SETTINGS)
      localStorage.removeItem('appSettings')
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center space-x-3">
          <Settings className="w-8 h-8 text-primary-600" />
          <span>Settings</span>
        </h1>
        <p className="text-gray-600 mt-2">Configure your conversion preferences</p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6 space-y-8">
        {/* Conversion Settings */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Conversion Defaults</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Default Quality
              </label>
              <select
                value={settings.defaultQuality}
                onChange={(e) => setSettings({ ...settings, defaultQuality: e.target.value })}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="draft">Draft (Fastest)</option>
                <option value="fast">Fast</option>
                <option value="balanced">Balanced</option>
                <option value="high">High Quality</option>
                <option value="studio">Studio (Best)</option>
              </select>
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">Auto-download results</label>
              <input
                type="checkbox"
                checked={settings.autoDownload}
                onChange={(e) => setSettings({ ...settings, autoDownload: e.target.checked })}
                className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
              />
            </div>
          </div>
        </div>

        {/* Audio Processing */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Audio Processing</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-700">Preserve Original Pitch</label>
                <p className="text-xs text-gray-500">Keep the original singer's pitch</p>
              </div>
              <input
                type="checkbox"
                checked={settings.preservePitch}
                onChange={(e) => setSettings({ ...settings, preservePitch: e.target.checked })}
                className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-700">Preserve Vibrato</label>
                <p className="text-xs text-gray-500">Maintain vibrato characteristics</p>
              </div>
              <input
                type="checkbox"
                checked={settings.preserveVibrato}
                onChange={(e) => setSettings({ ...settings, preserveVibrato: e.target.checked })}
                className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-700">Preserve Expression</label>
                <p className="text-xs text-gray-500">Keep emotional dynamics</p>
              </div>
              <input
                type="checkbox"
                checked={settings.preserveExpression}
                onChange={(e) =>
                  setSettings({ ...settings, preserveExpression: e.target.checked })
                }
                className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-700">Denoise Input</label>
                <p className="text-xs text-gray-500">Remove background noise</p>
              </div>
              <input
                type="checkbox"
                checked={settings.denoiseInput}
                onChange={(e) => setSettings({ ...settings, denoiseInput: e.target.checked })}
                className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-700">Enhance Output</label>
                <p className="text-xs text-gray-500">Apply audio enhancement</p>
              </div>
              <input
                type="checkbox"
                checked={settings.enhanceOutput}
                onChange={(e) => setSettings({ ...settings, enhanceOutput: e.target.checked })}
                className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
              />
            </div>
          </div>
        </div>

        {/* Performance */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Performance</h2>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Max Concurrent Jobs: {settings.maxConcurrentJobs}
            </label>
            <input
              type="range"
              min="1"
              max="10"
              value={settings.maxConcurrentJobs}
              onChange={(e) =>
                setSettings({ ...settings, maxConcurrentJobs: parseInt(e.target.value) })
              }
              className="w-full"
            />
            <p className="text-xs text-gray-500 mt-1">
              Number of conversions to process simultaneously
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex space-x-4 pt-4 border-t">
          <button
            onClick={handleSave}
            className="flex-1 bg-primary-600 hover:bg-primary-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center space-x-2"
          >
            <Save className="w-5 h-5" />
            <span>{saved ? 'Saved!' : 'Save Settings'}</span>
          </button>
          <button
            onClick={handleReset}
            className="bg-gray-600 hover:bg-gray-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center space-x-2"
          >
            <RotateCcw className="w-5 h-5" />
            <span>Reset</span>
          </button>
        </div>
      </div>
    </div>
  )
}

