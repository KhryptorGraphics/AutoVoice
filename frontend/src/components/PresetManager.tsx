import { useState } from 'react'
import {
  Bookmark, Plus, Download, Upload, Edit2, Check, X,
  Loader2, AlertCircle, CheckCircle, MoreVertical
} from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService, UserPreset, ConversionConfig } from '../services/api'
import { useToastContext } from '../contexts/ToastContext'
import { ConfirmActionButton } from './ConfirmActionButton'
import { StatusBanner } from './StatusBanner'

interface PresetManagerProps {
  currentConfig: Partial<ConversionConfig>
  onLoadPreset: (config: Partial<ConversionConfig>) => void
  compact?: boolean
}

export function PresetManager({ currentConfig, onLoadPreset, compact = false }: PresetManagerProps) {
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [newPresetName, setNewPresetName] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState('')
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null)
  const [importError, setImportError] = useState<string | null>(null)
  const queryClient = useQueryClient()
  const toast = useToastContext()

  const { data: presets, isLoading } = useQuery({
    queryKey: ['presets'],
    queryFn: () => apiService.listPresets(),
  })

  const saveMutation = useMutation({
    mutationFn: (name: string) => apiService.savePreset(name, currentConfig),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['presets'] })
      setShowSaveDialog(false)
      setNewPresetName('')
      toast.success('Preset saved')
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : 'Failed to save preset')
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiService.deletePreset(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['presets'] })
      toast.success('Preset deleted')
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : 'Failed to delete preset')
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) =>
      apiService.updatePreset(id, { name }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['presets'] })
      setEditingId(null)
      toast.success('Preset renamed')
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : 'Failed to rename preset')
    },
  })

  const handleSave = () => {
    if (!newPresetName.trim()) return
    saveMutation.mutate(newPresetName.trim())
  }

  const handleLoad = (preset: UserPreset) => {
    onLoadPreset(preset.config)
    setMenuOpenId(null)
    toast.info(`Loaded preset ${preset.name}`)
  }

  const handleDelete = (id: string) => {
    setMenuOpenId(null)
    deleteMutation.mutate(id)
  }

  const handleExport = () => {
    if (!presets) return
    const data = JSON.stringify(presets, null, 2)
    const blob = new Blob([data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `autovoice-presets-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleImport = () => {
    setImportError(null)
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return

      try {
        const text = await file.text()
        const imported = JSON.parse(text) as UserPreset[]
        for (const preset of imported) {
          await apiService.savePreset(preset.name, preset.config)
        }
        queryClient.invalidateQueries({ queryKey: ['presets'] })
        toast.success(`Imported ${imported.length} preset${imported.length === 1 ? '' : 's'}`)
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to import presets. Invalid file format.'
        setImportError(message)
        toast.error(message)
      }
    }
    input.click()
  }

  const startEditing = (preset: UserPreset) => {
    setEditingId(preset.id)
    setEditName(preset.name)
    setMenuOpenId(null)
  }

  const saveEdit = () => {
    if (!editingId || !editName.trim()) return
    updateMutation.mutate({ id: editingId, name: editName.trim() })
  }

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <div className="flex-1">
          <label htmlFor="preset-select-compact" className="sr-only">
            Load preset
          </label>
          <select
            id="preset-select-compact"
            onChange={(e) => {
              const preset = presets?.find((p: UserPreset) => p.id === e.target.value)
              if (preset) handleLoad(preset)
            }}
            className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
            defaultValue=""
          >
            <option value="" disabled>Load preset...</option>
            {presets?.map((preset: UserPreset) => (
              <option key={preset.id} value={preset.id}>{preset.name}</option>
            ))}
          </select>
        </div>
        <button
          onClick={() => setShowSaveDialog(true)}
          className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded"
          title="Save as preset"
          aria-label="Save as preset"
        >
          <Plus size={14} />
        </button>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bookmark size={18} className="text-yellow-400" />
          <h3 className="font-semibold">Presets</h3>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={handleImport}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
            title="Import presets"
          >
            <Upload size={14} />
          </button>
          <button
            onClick={handleExport}
            disabled={!presets?.length}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded disabled:opacity-50"
            title="Export presets"
          >
            <Download size={14} />
          </button>
        </div>
      </div>

      {importError && (
        <StatusBanner
          tone="danger"
          title="Preset import failed"
          message={importError}
          compact
        />
      )}

      {/* Preset List */}
      {isLoading ? (
        <div className="flex items-center justify-center py-4">
          <Loader2 className="animate-spin text-gray-500" />
        </div>
      ) : presets && presets.length > 0 ? (
        <div className="space-y-1 max-h-48 overflow-y-auto">
          {presets.map((preset: UserPreset) => (
            <div
              key={preset.id}
              className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-750 group"
            >
              {editingId === preset.id ? (
                <>
                  <input
                    type="text"
                    value={editName}
                    onChange={e => setEditName(e.target.value)}
                    className="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
                    autoFocus
                    onKeyDown={e => {
                      if (e.key === 'Enter') saveEdit()
                      if (e.key === 'Escape') setEditingId(null)
                    }}
                    aria-label="Preset name"
                  />
                  <button
                    onClick={saveEdit}
                    className="p-1 text-green-400 hover:bg-gray-700 rounded"
                  >
                    <Check size={14} />
                  </button>
                  <button
                    onClick={() => setEditingId(null)}
                    className="p-1 text-gray-400 hover:bg-gray-700 rounded"
                  >
                    <X size={14} />
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => handleLoad(preset)}
                    className="flex-1 text-left text-sm hover:text-white transition-colors"
                  >
                    <div>{preset.name}</div>
                    <div className="text-xs text-gray-500">
                      {(preset.config.pipeline_type as string) || 'quality'}
                    </div>
                  </button>
                  <span className="text-xs text-gray-500">
                    {new Date(preset.created_at).toLocaleDateString()}
                  </span>
                  <div className="relative">
                    <button
                      onClick={() => setMenuOpenId(menuOpenId === preset.id ? null : preset.id)}
                      className="p-1 text-gray-500 hover:text-white rounded opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <MoreVertical size={14} />
                    </button>
                    {menuOpenId === preset.id && (
                      <div className="absolute right-0 top-full mt-1 bg-gray-700 rounded-lg shadow-lg z-10 py-1 min-w-[120px]">
                        <button
                          onClick={() => handleLoad(preset)}
                          className="w-full px-3 py-1.5 text-left text-sm hover:bg-gray-600 flex items-center gap-2"
                        >
                          <CheckCircle size={12} /> Load
                        </button>
                        <button
                          onClick={() => startEditing(preset)}
                          className="w-full px-3 py-1.5 text-left text-sm hover:bg-gray-600 flex items-center gap-2"
                        >
                          <Edit2 size={12} /> Rename
                        </button>
                        <div className="px-2 py-1">
                          <ConfirmActionButton
                            label="Delete"
                            confirmLabel="Delete preset"
                            confirmMessage={`Remove ${preset.name} from saved presets?`}
                            onConfirm={async () => handleDelete(preset.id)}
                            pending={deleteMutation.isPending && deleteMutation.variables === preset.id}
                            className="w-full justify-center px-3 py-1.5 text-left text-sm"
                            testId={`preset-delete-${preset.id}`}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-4 text-gray-500 text-sm">
          No presets saved yet
        </div>
      )}

      {/* Save New Preset */}
      {showSaveDialog ? (
        <div className="bg-gray-750 rounded-lg p-3 space-y-3">
          <label className="text-sm text-gray-400">Save Current Settings</label>
          <input
            type="text"
            value={newPresetName}
            onChange={e => setNewPresetName(e.target.value)}
            placeholder="Preset name..."
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"
            autoFocus
            onKeyDown={e => {
              if (e.key === 'Enter') handleSave()
              if (e.key === 'Escape') setShowSaveDialog(false)
            }}
          />
          <div className="flex gap-2">
            <button
              onClick={handleSave}
              disabled={!newPresetName.trim() || saveMutation.isPending}
              className="flex-1 flex items-center justify-center gap-2 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm disabled:opacity-50"
            >
              {saveMutation.isPending ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Check size={14} />
              )}
              Save
            </button>
            <button
              onClick={() => setShowSaveDialog(false)}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button
          onClick={() => setShowSaveDialog(true)}
          className="w-full flex items-center justify-center gap-2 py-2 border border-dashed border-gray-600 hover:border-gray-500 rounded-lg text-sm text-gray-400 hover:text-white transition-colors"
        >
          <Plus size={16} />
          Save Current Settings as Preset
        </button>
      )}

      {/* Error Display */}
      {(saveMutation.error || deleteMutation.error) && (
        <div className="flex items-center gap-2 p-2 bg-red-900/30 border border-red-800 rounded text-sm text-red-400">
          <AlertCircle size={14} />
          {(saveMutation.error as Error)?.message || (deleteMutation.error as Error)?.message}
        </div>
      )}
    </div>
  )
}
