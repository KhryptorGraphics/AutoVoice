import { useCallback, useEffect, useMemo, useState } from 'react'
import { CheckCircle, Loader2, Mic, Music, RefreshCw, Upload, Users, Wand2 } from 'lucide-react'
import clsx from 'clsx'

import { AdapterSelector, AdapterBadge } from '../components/AdapterSelector'
import { ConversionHistoryTable } from '../components/ConversionHistoryTable'
import { LiveTrainingMonitor } from '../components/LiveTrainingMonitor'
import { PresetManager } from '../components/PresetManager'
import { TrainingConfigPanel } from '../components/TrainingConfigPanel'
import { PipelineBadge, PipelineSelector, getPreferredPipeline, isOfflinePipeline, type PipelineType } from '../components/PipelineSelector'
import { useToastContext } from '../contexts/ToastContext'
import {
  apiService,
  type AdapterType,
  type ConversionConfig,
  type ConversionRecord,
  type ConversionWorkflow,
  type ConversionWorkflowReviewItem,
  DEFAULT_CONVERSION_CONFIG,
  DEFAULT_TRAINING_CONFIG,
  type OfflinePipelineType,
  type TrainingConfig,
  type TrainingJob,
  type TrainingSample,
  type VoiceProfile,
} from '../services/api'

const WORKFLOW_STEPS = [
  { key: 'uploaded', label: 'Uploaded' },
  { key: 'separating_artist_song', label: 'Split Artist Song' },
  { key: 'analyzing_user_vocals', label: 'Analyze User Vocals' },
  { key: 'diarizing_artist_song', label: 'Diarize Artist Song' },
  { key: 'matching_profiles', label: 'Resolve Profiles' },
  { key: 'awaiting_review', label: 'Review Needed' },
  { key: 'ready_for_training', label: 'Ready for Training' },
  { key: 'training_in_progress', label: 'Training' },
  { key: 'ready_for_conversion', label: 'Ready for Conversion' },
] as const

function formatWorkflowStage(stage: string): string {
  const found = WORKFLOW_STEPS.find((step) => step.key === stage)
  if (found) return found.label
  return stage.replace(/_/g, ' ')
}

function WorkflowProgressRail({ workflow }: { workflow: ConversionWorkflow | null }) {
  const currentIndex = workflow
    ? Math.max(WORKFLOW_STEPS.findIndex((step) => step.key === workflow.stage), 0)
    : -1

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
      {WORKFLOW_STEPS.map((step, index) => {
        const isActive = workflow?.stage === step.key
        const isComplete = currentIndex > index || workflow?.status === 'ready_for_conversion'
        return (
          <div
            key={step.key}
            className={clsx(
              'rounded-lg border p-3 text-sm',
              isActive
                ? 'border-blue-500 bg-blue-500/10 text-blue-100'
                : isComplete
                  ? 'border-green-600 bg-green-500/10 text-green-100'
                  : 'border-gray-700 bg-gray-800 text-gray-400'
            )}
          >
            <div className="font-medium">{step.label}</div>
            {workflow?.stage === step.key && (
              <div className="mt-1 text-xs text-gray-300">{workflow.progress}%</div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function FileDropField({
  label,
  help,
  files,
  multiple = false,
  accept = 'audio/*',
  inputId,
  onChange,
}: {
  label: string
  help: string
  files: File[]
  multiple?: boolean
  accept?: string
  inputId: string
  onChange: (files: File[]) => void
}) {
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="mb-3">
        <h2 className="text-lg font-semibold">{label}</h2>
        <p className="text-sm text-gray-400 mt-1">{help}</p>
      </div>
      <div
        onDrop={(event) => {
          event.preventDefault()
          const nextFiles = Array.from(event.dataTransfer.files).filter((file) => file.type.startsWith('audio/'))
          if (nextFiles.length > 0) {
            onChange(multiple ? nextFiles : [nextFiles[0]])
          }
        }}
        onDragOver={(event) => event.preventDefault()}
        className={clsx(
          'border-2 border-dashed rounded-lg p-6 text-center transition',
          files.length > 0 ? 'border-green-500 bg-green-500/10' : 'border-gray-600 hover:border-gray-500'
        )}
      >
        {files.length > 0 ? (
          <div className="space-y-2">
            <CheckCircle className="mx-auto h-10 w-10 text-green-500" />
            {files.map((file) => (
              <div key={`${file.name}-${file.size}`} className="text-sm text-gray-200">
                {file.name} <span className="text-gray-400">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
              </div>
            ))}
            <button
              type="button"
              className="text-red-400 hover:text-red-300 text-sm"
              onClick={() => onChange([])}
            >
              Clear
            </button>
          </div>
        ) : (
          <>
            <Upload className="mx-auto h-12 w-12 text-gray-500 mb-3" />
            <p className="text-gray-400 mb-3">
              {multiple ? 'Drop audio files here or click to upload' : 'Drop an audio file here or click to upload'}
            </p>
            <input
              id={inputId}
              type="file"
              accept={accept}
              className="hidden"
              multiple={multiple}
              onChange={(event) => onChange(Array.from(event.target.files || []))}
            />
            <label
              htmlFor={inputId}
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded cursor-pointer"
            >
              <Upload size={16} />
              Select {multiple ? 'Files' : 'File'}
            </label>
          </>
        )}
      </div>
    </div>
  )
}

export function ConversionWorkflowPage() {
  const toast = useToastContext()
  const [profiles, setProfiles] = useState<VoiceProfile[]>([])
  const [artistSong, setArtistSong] = useState<File | null>(null)
  const [userVocalFiles, setUserVocalFiles] = useState<File[]>([])
  const [targetProfileOverride, setTargetProfileOverride] = useState<string>('')
  const [dominantSourceProfileOverride, setDominantSourceProfileOverride] = useState<string>('')
  const [workflow, setWorkflow] = useState<ConversionWorkflow | null>(null)
  const [submittingWorkflow, setSubmittingWorkflow] = useState(false)
  const [workflowError, setWorkflowError] = useState<string | null>(null)
  const [targetProfileDetail, setTargetProfileDetail] = useState<(VoiceProfile & { training_history: TrainingJob[] }) | null>(null)
  const [targetSamples, setTargetSamples] = useState<TrainingSample[]>([])
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG)
  const [startingTraining, setStartingTraining] = useState(false)
  const [selectedAdapter, setSelectedAdapter] = useState<AdapterType | null>(null)
  const [config, setConfig] = useState<ConversionConfig>(() => {
    const preferred = getPreferredPipeline()
    return {
      ...DEFAULT_CONVERSION_CONFIG,
      pipeline_type: preferred && isOfflinePipeline(preferred) ? preferred : 'quality_seedvc',
    }
  })
  const [pipelineStatus, setPipelineStatus] = useState<Record<string, { loaded: boolean; memory_gb?: number; latency_target_ms?: number }>>({})
  const [conversionStatus, setConversionStatus] = useState<ConversionRecord | null>(null)
  const [isConverting, setIsConverting] = useState(false)
  const [reviewActions, setReviewActions] = useState<Record<string, { resolution: 'use_suggested' | 'use_existing' | 'create_new'; profile_id?: string; name?: string }>>({})
  const [lastSubmissionKey, setLastSubmissionKey] = useState<string | null>(null)

  const targetProfiles = useMemo(
    () => profiles.filter((profile) => profile.profile_role !== 'source_artist'),
    [profiles]
  )
  const sourceProfiles = useMemo(
    () => profiles.filter((profile) => profile.profile_role === 'source_artist'),
    [profiles]
  )

  const submissionKey = useMemo(() => {
    if (!artistSong || userVocalFiles.length === 0) return null
    return JSON.stringify({
      artist: [artistSong.name, artistSong.size, artistSong.lastModified],
      user: userVocalFiles.map((file) => [file.name, file.size, file.lastModified]),
      targetProfileOverride,
      dominantSourceProfileOverride,
    })
  }, [artistSong, userVocalFiles, targetProfileOverride, dominantSourceProfileOverride])

  const loadProfiles = useCallback(async () => {
    try {
      const nextProfiles = await apiService.listProfiles()
      setProfiles(nextProfiles)
    } catch (error) {
      console.error('Failed to load profiles:', error)
    }
  }, [])

  useEffect(() => {
    void loadProfiles()
  }, [loadProfiles])

  useEffect(() => {
    let cancelled = false
    const loadPipelinePreferences = async () => {
      try {
        const [settings, status] = await Promise.all([
          apiService.getAppSettings(),
          apiService.getPipelineStatus(),
        ])
        if (cancelled) return
        if (isOfflinePipeline(settings.preferred_offline_pipeline)) {
          setConfig((prev) => ({ ...prev, pipeline_type: settings.preferred_offline_pipeline }))
        }
        setPipelineStatus(status.pipelines || {})
      } catch (error) {
        console.error('Failed to load conversion preferences:', error)
      }
    }
    void loadPipelinePreferences()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!workflow?.workflow_id) return
    let timeoutId: number | undefined
    let cancelled = false

    const poll = async () => {
      try {
        const latest = await apiService.getConversionWorkflow(workflow.workflow_id)
        if (cancelled) return
        setWorkflow(latest)
        if (latest.status === 'processing' || latest.status === 'queued' || latest.status === 'training_in_progress') {
          timeoutId = window.setTimeout(poll, 1500)
        } else {
          timeoutId = window.setTimeout(poll, 3000)
        }
      } catch (error) {
        console.error('Failed to poll conversion workflow:', error)
      }
    }

    void poll()
    return () => {
      cancelled = true
      if (timeoutId) window.clearTimeout(timeoutId)
    }
  }, [workflow?.workflow_id])

  useEffect(() => {
    if (!submissionKey || submittingWorkflow) return
    if (submissionKey === lastSubmissionKey) return

    const startWorkflow = async () => {
      if (!artistSong || userVocalFiles.length === 0) return
      setSubmittingWorkflow(true)
      setWorkflowError(null)
      setConversionStatus(null)
      try {
        const created = await apiService.createConversionWorkflow(artistSong, userVocalFiles, {
          target_profile_id: targetProfileOverride || null,
          dominant_source_profile_id: dominantSourceProfileOverride || null,
        })
        setWorkflow(created)
        setLastSubmissionKey(submissionKey)
        toast.success('Conversion intake workflow started')
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to start conversion workflow'
        setWorkflowError(message)
        toast.error(message)
      } finally {
        setSubmittingWorkflow(false)
      }
    }

    void startWorkflow()
  }, [
    artistSong,
    dominantSourceProfileOverride,
    lastSubmissionKey,
    submissionKey,
    submittingWorkflow,
    targetProfileOverride,
    toast,
    userVocalFiles,
  ])

  useEffect(() => {
    const targetProfileId = workflow?.resolved_target_profile_id
    if (!targetProfileId) {
      setTargetProfileDetail(null)
      setTargetSamples([])
      return
    }

    let cancelled = false
    const loadTargetProfile = async () => {
      try {
        const [detail, samples] = await Promise.all([
          apiService.getProfileDetails(targetProfileId),
          apiService.listSamples(targetProfileId),
        ])
        if (cancelled) return
        setTargetProfileDetail(detail)
        setTargetSamples(samples)
      } catch (error) {
        console.error('Failed to load workflow target profile:', error)
      }
    }

    void loadTargetProfile()
    return () => {
      cancelled = true
    }
  }, [workflow?.resolved_target_profile_id])

  useEffect(() => {
    if (targetProfileDetail?.active_model_type === 'full_model' && selectedAdapter) {
      setSelectedAdapter(null)
    }
  }, [selectedAdapter, targetProfileDetail?.active_model_type])

  const allowFullTraining = Boolean(targetProfileDetail?.full_model_eligible)
  const allowContinueLora = Boolean(targetProfileDetail?.has_adapter_model)
  const allowContinueFull = Boolean(targetProfileDetail?.has_full_model)
  const remainingMinutes = targetProfileDetail?.full_model_remaining_minutes ?? ((targetProfileDetail?.full_model_remaining_seconds ?? 0) / 60)
  const continuationHint = trainingConfig.training_mode === 'full'
    ? (
      allowContinueFull
        ? 'Continue training will reuse the latest full-model checkpoint or artifact.'
        : 'Continue training becomes available after this profile has a trained full model.'
    )
    : (
      allowContinueLora
        ? 'Continue training will reuse the latest LoRA checkpoint or artifact.'
        : 'Continue training becomes available after this profile has a trained LoRA adapter.'
    )

  const handleResolveReview = async (item: ConversionWorkflowReviewItem) => {
    if (!workflow) return
    const action = reviewActions[item.review_id] ?? {
      resolution: item.suggested_match ? 'use_suggested' as const : 'create_new' as const,
      profile_id: item.suggested_match?.profile_id,
      name: item.candidate.name,
    }
    try {
      const updated = await apiService.resolveConversionWorkflowMatch(workflow.workflow_id, {
        review_id: item.review_id,
        resolution: action.resolution,
        profile_id: action.profile_id,
        name: action.name,
      })
      setWorkflow(updated)
      void loadProfiles()
      toast.success('Review item resolved')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to resolve workflow match'
      toast.error(message)
    }
  }

  const handleStartTraining = async () => {
    if (!workflow?.resolved_target_profile_id) return
    if (targetSamples.length === 0) {
      toast.error('No samples are attached to the resolved target profile yet.')
      return
    }
    setStartingTraining(true)
    try {
      const sampleIds = targetSamples.map((sample) => sample.id)
      const job = await apiService.createTrainingJob(workflow.resolved_target_profile_id, sampleIds, trainingConfig)
      const updated = await apiService.attachConversionWorkflowTrainingJob(workflow.workflow_id, job.job_id)
      setWorkflow(updated)
      toast.success('Training started')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to start training'
      toast.error(message)
    } finally {
      setStartingTraining(false)
    }
  }

  const handleConvert = async () => {
    if (!workflow) return
    setIsConverting(true)
    try {
      const result = await apiService.convertWorkflow(workflow.workflow_id, {
        preset: config.preset,
        vocal_volume: config.vocal_volume,
        instrumental_volume: config.instrumental_volume,
        pitch_shift: config.pitch_shift,
        pipeline_type: config.pipeline_type,
        adapter_type: targetProfileDetail?.active_model_type === 'full_model' ? undefined : (selectedAdapter || undefined),
        return_stems: config.return_stems,
      })
      const pollStatus = async () => {
        const status = await apiService.getConversionStatus(result.job_id)
        setConversionStatus({ ...status, id: result.job_id })
        if (status.status === 'queued' || status.status === 'processing' || status.status === 'in_progress') {
          window.setTimeout(pollStatus, 1000)
          return
        }
        if (status.status === 'complete' || status.status === 'completed') {
          toast.success('Conversion completed successfully')
        } else if (status.status === 'error' || status.status === 'failed') {
          toast.error(status.error || 'Conversion failed')
        }
        setIsConverting(false)
      }
      void pollStatus()
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to start conversion'
      toast.error(message)
      setIsConverting(false)
    }
  }

  const triggerDownload = useCallback((blob: Blob, downloadName: string) => {
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = downloadName
    anchor.click()
    URL.revokeObjectURL(url)
  }, [])

  const handleDownload = async () => {
    if (!conversionStatus) return
    const blob = await apiService.downloadResult(conversionStatus.id)
    triggerDownload(blob, `converted_${artistSong?.name || 'song.wav'}`)
  }

  const handleDownloadStem = async (variant: 'vocals' | 'instrumental') => {
    if (!conversionStatus) return
    const blob = await apiService.downloadConversionAsset(conversionStatus.id, variant)
    const baseName = artistSong?.name?.replace(/\.[^/.]+$/, '') || 'song'
    triggerDownload(blob, `${baseName}_${variant}.wav`)
  }

  const handleReassemble = async () => {
    if (!conversionStatus) return
    const blob = await apiService.reassembleConversion(conversionStatus.id)
    const baseName = artistSong?.name?.replace(/\.[^/.]+$/, '') || 'song'
    triggerDownload(blob, `${baseName}_reassembled.wav`)
  }

  const handlePipelineChange = useCallback((nextPipeline: PipelineType) => {
    if (!isOfflinePipeline(nextPipeline)) return
    setConfig((prev) => ({ ...prev, pipeline_type: nextPipeline }))
    void apiService.updateAppSettings({ preferred_offline_pipeline: nextPipeline }).catch((error) => {
      console.error('Failed to persist pipeline preference:', error)
    })
  }, [])

  const handlePresetLoad = useCallback((presetConfig: Partial<ConversionConfig>) => {
    setConfig((prev) => ({ ...prev, ...presetConfig }))
    if (presetConfig.pipeline_type && isOfflinePipeline(presetConfig.pipeline_type)) {
      void apiService.updateAppSettings({ preferred_offline_pipeline: presetConfig.pipeline_type }).catch((error) => {
        console.error('Failed to persist preset pipeline preference:', error)
      })
    }
  }, [])

  const handleHistorySelect = useCallback((record: ConversionRecord) => {
    if (record.adapter_type && record.adapter_type !== 'unified') {
      setSelectedAdapter(record.adapter_type)
    }
    if (record.pipeline_type && isOfflinePipeline(record.pipeline_type)) {
      setConfig((prev) => ({
        ...prev,
        pipeline_type: record.pipeline_type as OfflinePipelineType,
        preset: (record.preset as ConversionConfig['preset']) || prev.preset,
      }))
    }
    toast.info(`Loaded settings from ${record.originalFileName || record.input_file}`)
  }, [toast])

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Voice Conversion</h1>
        <p className="text-gray-400 mt-2">
          Upload an artist song and your own vocal clips. AutoVoice will split the song, diarize every singer,
          match or create profiles automatically, then hand the resolved target voice into the same granular
          training and conversion flow used elsewhere in the app.
        </p>
      </div>

      {workflowError && (
        <div className="rounded-lg border border-red-500 bg-red-500/10 p-4 text-red-200">
          {workflowError}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="space-y-4">
          <FileDropField
            label="1. Artist Song"
            help="Upload the source song/performance. This gets separated into vocals and instrumental, then diarized to resolve every detected singer."
            files={artistSong ? [artistSong] : []}
            inputId="artist-song-upload"
            onChange={(files) => {
              setArtistSong(files[0] || null)
              setWorkflow(null)
              setLastSubmissionKey(null)
            }}
          />
          <div className="bg-gray-800 rounded-lg p-4">
            <label className="block text-sm text-gray-400 mb-2">Optional dominant artist override</label>
            <select
              value={dominantSourceProfileOverride}
              onChange={(event) => {
                setDominantSourceProfileOverride(event.target.value)
                setWorkflow(null)
                setLastSubmissionKey(null)
              }}
              className="w-full p-3 bg-gray-700 rounded-lg"
            >
              <option value="">Auto-match dominant singer</option>
              {sourceProfiles.map((profile) => (
                <option key={profile.profile_id} value={profile.profile_id}>
                  {profile.name || profile.profile_id}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="space-y-4">
          <FileDropField
            label="2. User Vocals"
            help="Upload one or more user vocal clips. These attach to an existing target-user profile when the match is clear, or create a new one automatically."
            files={userVocalFiles}
            inputId="user-vocals-upload"
            multiple={true}
            onChange={(files) => {
              setUserVocalFiles(files)
              setWorkflow(null)
              setLastSubmissionKey(null)
            }}
          />
          <div className="bg-gray-800 rounded-lg p-4">
            <label className="block text-sm text-gray-400 mb-2">Optional target-user override</label>
            <select
              value={targetProfileOverride}
              onChange={(event) => {
                setTargetProfileOverride(event.target.value)
                setWorkflow(null)
                setLastSubmissionKey(null)
              }}
              className="w-full p-3 bg-gray-700 rounded-lg"
            >
              <option value="">Auto-match target user</option>
              {targetProfiles.map((profile) => (
                <option key={profile.profile_id} value={profile.profile_id}>
                  {profile.name || profile.profile_id}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Wand2 size={18} />
              Workflow Status
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Uploading both assets starts the workflow automatically. Current stage:{' '}
              <span className="text-white">{workflow ? formatWorkflowStage(workflow.stage) : 'waiting for uploads'}</span>
            </p>
          </div>
          {submittingWorkflow && (
            <div className="flex items-center gap-2 text-blue-300">
              <Loader2 size={16} className="animate-spin" />
              Starting workflow
            </div>
          )}
        </div>
        <WorkflowProgressRail workflow={workflow} />
        {workflow && (
          <div className="text-sm text-gray-400">
            Status: <span className="text-white">{workflow.status}</span>
            {' · '}
            Progress: <span className="text-white">{workflow.progress}%</span>
          </div>
        )}
      </div>

      {workflow?.review_items?.length ? (
        <div className="bg-yellow-500/10 border border-yellow-600 rounded-lg p-6 space-y-4">
          <h2 className="text-lg font-semibold text-yellow-100 flex items-center gap-2">
            <Users size={18} />
            Review Required
          </h2>
          {workflow.review_items.map((item) => {
            const action = reviewActions[item.review_id] ?? {
              resolution: item.suggested_match ? 'use_suggested' as const : 'create_new' as const,
              profile_id: item.suggested_match?.profile_id,
              name: item.candidate.name || '',
            }
            const compatibleProfiles = item.role === 'source_artist' ? sourceProfiles : targetProfiles
            return (
              <div key={item.review_id} className="rounded-lg border border-yellow-700 bg-yellow-500/5 p-4 space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full bg-yellow-500/20 px-2 py-1 text-xs text-yellow-100">{item.role}</span>
                  <span className="text-sm text-yellow-50">{item.reason.replace(/_/g, ' ')}</span>
                </div>
                {item.suggested_match && (
                  <div className="text-sm text-yellow-100">
                    Suggested match: <strong>{item.suggested_match.name}</strong> ({Math.round(item.suggested_match.similarity * 100)}% similarity)
                  </div>
                )}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <button
                    type="button"
                    onClick={() => setReviewActions((prev) => ({
                      ...prev,
                      [item.review_id]: { ...action, resolution: 'use_suggested', profile_id: item.suggested_match?.profile_id },
                    }))}
                    disabled={!item.suggested_match}
                    className={clsx(
                      'rounded-lg border px-3 py-2 text-sm',
                      action.resolution === 'use_suggested' ? 'border-blue-500 bg-blue-500/10 text-blue-100' : 'border-gray-700 text-gray-300',
                      !item.suggested_match && 'opacity-50 cursor-not-allowed'
                    )}
                  >
                    Accept Suggested
                  </button>
                  <button
                    type="button"
                    onClick={() => setReviewActions((prev) => ({
                      ...prev,
                      [item.review_id]: { ...action, resolution: 'use_existing' },
                    }))}
                    className={clsx(
                      'rounded-lg border px-3 py-2 text-sm',
                      action.resolution === 'use_existing' ? 'border-blue-500 bg-blue-500/10 text-blue-100' : 'border-gray-700 text-gray-300'
                    )}
                  >
                    Choose Existing
                  </button>
                  <button
                    type="button"
                    onClick={() => setReviewActions((prev) => ({
                      ...prev,
                      [item.review_id]: { ...action, resolution: 'create_new' },
                    }))}
                    className={clsx(
                      'rounded-lg border px-3 py-2 text-sm',
                      action.resolution === 'create_new' ? 'border-blue-500 bg-blue-500/10 text-blue-100' : 'border-gray-700 text-gray-300'
                    )}
                  >
                    Create New
                  </button>
                </div>
                {action.resolution === 'use_existing' && (
                  <select
                    value={action.profile_id || ''}
                    onChange={(event) => setReviewActions((prev) => ({
                      ...prev,
                      [item.review_id]: { ...action, profile_id: event.target.value },
                    }))}
                    className="w-full p-3 bg-gray-900 border border-gray-700 rounded-lg"
                  >
                    <option value="">Choose existing profile...</option>
                    {compatibleProfiles.map((profile) => (
                      <option key={profile.profile_id} value={profile.profile_id}>
                        {profile.name || profile.profile_id}
                      </option>
                    ))}
                  </select>
                )}
                {action.resolution === 'create_new' && (
                  <input
                    type="text"
                    value={action.name || ''}
                    onChange={(event) => setReviewActions((prev) => ({
                      ...prev,
                      [item.review_id]: { ...action, name: event.target.value },
                    }))}
                    placeholder="New profile name"
                    className="w-full p-3 bg-gray-900 border border-gray-700 rounded-lg"
                  />
                )}
                <button
                  type="button"
                  onClick={() => void handleResolveReview(item)}
                  className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg text-sm font-medium"
                >
                  Resolve Match
                </button>
              </div>
            )
          })}
        </div>
      ) : null}

      {workflow && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="bg-gray-800 rounded-lg p-6 space-y-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Music size={18} />
              Resolved Artist Profiles
            </h2>
            {workflow.resolved_source_profiles.length === 0 ? (
              <p className="text-sm text-gray-400">No artist profiles resolved yet.</p>
            ) : (
              workflow.resolved_source_profiles.map((profile) => (
                <div key={`${profile.profile_id}-${profile.speaker_id}`} className="rounded-lg border border-gray-700 p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">{profile.name}</div>
                      <div className="text-sm text-gray-400">
                        {profile.speaker_id || 'Resolved artist'} · {profile.duration_seconds?.toFixed(1) || '0.0'}s
                      </div>
                    </div>
                    <span className="rounded-full border border-gray-600 px-2 py-1 text-xs text-gray-300">
                      {profile.status || 'matched'}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="bg-gray-800 rounded-lg p-6 space-y-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Mic size={18} />
              Resolved Target Profile
            </h2>
            {workflow.resolved_target_profile ? (
              <div className="rounded-lg border border-emerald-700 bg-emerald-500/10 p-4 space-y-2">
                <div className="font-medium text-emerald-100">{workflow.resolved_target_profile.name}</div>
                <div className="text-sm text-gray-300">
                  {workflow.resolved_target_profile.sample_count || 0} samples · {workflow.resolved_target_profile.clean_vocal_minutes?.toFixed(1) || '0.0'} min clean vocals
                </div>
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <span className="rounded-full border border-gray-600 px-2 py-1 text-xs text-gray-200">
                    {workflow.resolved_target_profile.active_model_type || 'base'}
                  </span>
                  {workflow.resolved_target_profile.has_trained_model ? (
                    <span className="rounded-full bg-green-500/20 px-2 py-1 text-xs text-green-200">Trained</span>
                  ) : (
                    <span className="rounded-full bg-yellow-500/20 px-2 py-1 text-xs text-yellow-100">Needs training</span>
                  )}
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-400">The workflow has not resolved a target user profile yet.</p>
            )}
          </div>
        </div>
      )}

      {workflow?.resolved_target_profile_id && targetProfileDetail && (
        <div className="bg-gray-800 rounded-lg p-6 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold">Training</h2>
              <p className="text-sm text-gray-400 mt-1">
                Training stays manual. Uploads auto-attach samples and resolve the target profile, then you choose whether to train a LoRA or full model from scratch or continue the existing artifact.
              </p>
            </div>
            {workflow.training_readiness.ready ? (
              <span className="rounded-full bg-green-500/20 px-3 py-1 text-sm text-green-200">
                Ready for training
              </span>
            ) : (
              <span className="rounded-full bg-yellow-500/20 px-3 py-1 text-sm text-yellow-100">
                {workflow.training_readiness.reason.replace(/_/g, ' ')}
              </span>
            )}
          </div>

          <TrainingConfigPanel
            config={trainingConfig}
            onChange={setTrainingConfig}
            disabled={startingTraining || !workflow.training_readiness.ready}
            allowFullTraining={allowFullTraining}
            allowContinueLora={allowContinueLora}
            allowContinueFull={allowContinueFull}
            fullTrainingHint={
              allowFullTraining
                ? 'Full-model training is unlocked for this target profile.'
                : `Full-model training unlocks after 30 minutes of clean user vocals. ${Math.max(remainingMinutes, 0).toFixed(1)} minutes remaining.`
            }
            continuationHint={continuationHint}
          />

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => void handleStartTraining()}
              disabled={!workflow.training_readiness.ready || startingTraining}
              className={clsx(
                'px-4 py-3 rounded-lg font-medium',
                !workflow.training_readiness.ready || startingTraining
                  ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              )}
            >
              {startingTraining ? (
                <span className="inline-flex items-center gap-2">
                  <Loader2 size={16} className="animate-spin" />
                  Starting training
                </span>
              ) : (
                'Start Training'
              )}
            </button>
            {workflow.current_training_job_id && (
              <span className="text-sm text-gray-400">
                Active job: {workflow.current_training_job_id}
              </span>
            )}
          </div>

          {workflow.current_training_job_id && (
            <LiveTrainingMonitor
              jobId={workflow.current_training_job_id}
              profileId={workflow.resolved_target_profile_id}
              onComplete={() => {
                void loadProfiles()
                if (workflow.workflow_id) {
                  void apiService.getConversionWorkflow(workflow.workflow_id).then(setWorkflow).catch(console.error)
                }
              }}
            />
          )}
        </div>
      )}

      {workflow?.resolved_target_profile_id && (
        <div className="bg-gray-800 rounded-lg p-6 space-y-6">
          <div>
            <h2 className="text-lg font-semibold">Conversion</h2>
            <p className="text-sm text-gray-400 mt-1">
              Once the target profile is trained, conversion reuses the stored artist song from the workflow and hands the target profile directly into the existing offline conversion engine.
            </p>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <div className="space-y-4">
              {targetProfileDetail?.active_model_type !== 'full_model' && workflow.resolved_target_profile_id && (
                <div>
                  <h3 className="font-medium mb-3">Adapter</h3>
                  <AdapterSelector
                    profileId={workflow.resolved_target_profile_id}
                    value={selectedAdapter}
                    onChange={setSelectedAdapter}
                    showMetrics={true}
                  />
                </div>
              )}
              <div>
                <h3 className="font-medium mb-3">Pipeline</h3>
                <PipelineSelector
                  value={config.pipeline_type}
                  onChange={handlePipelineChange}
                  context="offline"
                  showDescription={true}
                  statusByPipeline={pipelineStatus}
                />
              </div>
            </div>

            <div className="space-y-4">
              <PresetManager currentConfig={config} onLoadPreset={handlePresetLoad} />
              <div className="rounded-lg border border-gray-700 p-4 space-y-2 text-sm text-gray-300">
                <div>Training readiness: <strong>{workflow.training_readiness.reason.replace(/_/g, ' ')}</strong></div>
                <div>Conversion readiness: <strong>{workflow.conversion_readiness.reason.replace(/_/g, ' ')}</strong></div>
                {conversionStatus?.pipeline_type && (
                  <div className="flex items-center gap-2">
                    Last conversion pipeline
                    <PipelineBadge pipeline={conversionStatus.pipeline_type as PipelineType} />
                    {conversionStatus.adapter_type && conversionStatus.adapter_type !== 'unified' ? (
                      <AdapterBadge adapterType={conversionStatus.adapter_type} />
                    ) : null}
                  </div>
                )}
              </div>
            </div>
          </div>

          {conversionStatus && (conversionStatus.status === 'complete' || conversionStatus.status === 'completed') && (
            <div className="rounded-lg border border-green-600 bg-green-500/10 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-green-200">
                  <CheckCircle size={18} />
                  Conversion complete
                </div>
                <div className="flex flex-wrap gap-2">
                  {conversionStatus.stem_urls?.vocals && (
                    <button onClick={() => void handleDownloadStem('vocals')} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded">
                      Download Vocals
                    </button>
                  )}
                  {conversionStatus.stem_urls?.instrumental && (
                    <button onClick={() => void handleDownloadStem('instrumental')} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded">
                      Download Instrumental
                    </button>
                  )}
                  {conversionStatus.reassemble_url && (
                    <button onClick={() => void handleReassemble()} className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded">
                      Reassemble
                    </button>
                  )}
                  <button onClick={() => void handleDownload()} className="px-3 py-2 bg-green-600 hover:bg-green-700 rounded">
                    Download Mix
                  </button>
                </div>
              </div>
            </div>
          )}

          <button
            type="button"
            onClick={() => void handleConvert()}
            disabled={!workflow.conversion_readiness.ready || isConverting}
            className={clsx(
              'w-full flex items-center justify-center gap-3 px-6 py-4 rounded-lg text-lg font-semibold transition',
              !workflow.conversion_readiness.ready || isConverting
                ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            )}
          >
            {isConverting ? (
              <>
                <Loader2 size={20} className="animate-spin" />
                Converting
              </>
            ) : (
              <>
                <Music size={20} />
                Convert Workflow Song
              </>
            )}
          </button>
        </div>
      )}

      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Conversion History</h2>
          <button
            type="button"
            onClick={() => {
              setArtistSong(null)
              setUserVocalFiles([])
              setWorkflow(null)
              setLastSubmissionKey(null)
              setTargetProfileOverride('')
              setDominantSourceProfileOverride('')
              setConversionStatus(null)
            }}
            className="inline-flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
          >
            <RefreshCw size={14} />
            Reset Workflow
          </button>
        </div>
        <ConversionHistoryTable onSelect={handleHistorySelect} />
      </div>
    </div>
  )
}
