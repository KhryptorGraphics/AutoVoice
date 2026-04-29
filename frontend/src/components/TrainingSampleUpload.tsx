import { useState, useCallback } from 'react';
import { Upload, Loader2, Users, CheckCircle, AlertTriangle } from 'lucide-react';
import { apiService, TrainingSample } from '../services/api';
import { DiarizationTimeline } from './DiarizationTimeline';

interface DiarizationSegment {
  start: number;
  end: number;
  speaker_id: string;
  confidence: number;
  duration: number;
}

interface DiarizationResult {
  diarization_id: string;
  audio_duration: number;
  num_speakers: number;
  segments: DiarizationSegment[];
  speaker_durations: Record<string, number>;
}

interface TrainingSampleUploadProps {
  profileId: string;
  profileName?: string;
  onSampleAdded: (sample: TrainingSample) => void;
}

type UploadStage = 'idle' | 'uploading' | 'diarizing' | 'review' | 'complete';

export function TrainingSampleUpload({
  profileId,
  profileName,
  onSampleAdded,
}: TrainingSampleUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [stage, setStage] = useState<UploadStage>('idle');
  const [error, setError] = useState<string | null>(null);
  const [diarizationResult, setDiarizationResult] = useState<DiarizationResult | null>(null);
  const [selectedSpeaker, setSelectedSpeaker] = useState<string | null>(null);
  const [enableDiarization, setEnableDiarization] = useState(true);
  const [isFiltering, setIsFiltering] = useState(false);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setAudioUrl(URL.createObjectURL(selectedFile));
    setDiarizationResult(null);
    setSelectedSpeaker(null);
    setError(null);
    setStage('idle');
  }, []);

  const handleUploadWithDiarization = async () => {
    if (!file) return;

    setError(null);

    if (enableDiarization) {
      // First run diarization to check for multiple speakers
      setStage('diarizing');
      try {
        const result = await apiService.diarizeAudio(file);
        setDiarizationResult(result);

        if (result.num_speakers > 1) {
          // Multiple speakers detected - show review UI
          setStage('review');
        } else {
          // Single speaker - upload directly
          await uploadSample();
        }
      } catch (err) {
        console.error('Diarization failed:', err);
        // If diarization fails, fall back to direct upload
        await uploadSample();
      }
    } else {
      // Direct upload without diarization
      await uploadSample();
    }
  };

  const uploadSample = async () => {
    if (!file) return;

    setStage('uploading');
    try {
      const sample = await apiService.uploadSample(profileId, file);
      setStage('complete');
      onSampleAdded(sample);
      resetState();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload sample');
      setStage('idle');
    }
  };

  const handleFilterAndUpload = async () => {
    if (!file || !diarizationResult || !selectedSpeaker) return;

    setIsFiltering(true);
    setError(null);

    try {
      // Backend speaker extraction for this flow is not wired yet; keep the UI truthful.
      const sample = await apiService.uploadSample(profileId, file);

      setStage('complete');
      onSampleAdded(sample);
      resetState();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload sample for review');
      setIsFiltering(false);
    }
  };

  const handleSkipFilter = async () => {
    // Upload without filtering even though multiple speakers detected
    await uploadSample();
  };

  const resetState = () => {
    setFile(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioUrl(null);
    setDiarizationResult(null);
    setSelectedSpeaker(null);
    setStage('idle');
    setIsFiltering(false);
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Render based on stage
  if (stage === 'review' && diarizationResult) {
    return (
      <div className="bg-zinc-800 rounded-lg p-6 space-y-4">
        <div className="flex items-center gap-2 text-yellow-400">
          <AlertTriangle size={20} />
          <h3 className="text-lg font-semibold">Multiple Speakers Detected</h3>
        </div>

        <p className="text-zinc-400 text-sm">
          This audio contains {diarizationResult.num_speakers} different speakers.
          Select the intended speaker for {profileName ? `"${profileName}"` : 'this profile'} review notes,
          then upload the original audio.
          Automatic speaker-only extraction is not available in this upload flow yet.
        </p>

        {/* Timeline preview */}
        <DiarizationTimeline
          segments={diarizationResult.segments}
          audioDuration={diarizationResult.audio_duration}
          audioUrl={audioUrl || undefined}
        />

        {/* Speaker selection */}
        <div className="space-y-2">
          <label className="text-sm text-zinc-400">Select speaker for training:</label>
          <div className="flex flex-wrap gap-2">
            {Object.entries(diarizationResult.speaker_durations).map(([speakerId, duration]) => {
              const percentage = (duration / diarizationResult.audio_duration) * 100;
              const isSelected = selectedSpeaker === speakerId;

              return (
                <button
                  key={speakerId}
                  onClick={() => setSelectedSpeaker(speakerId)}
                  className={`px-4 py-2 rounded-lg border-2 transition-colors ${
                    isSelected
                      ? 'border-blue-500 bg-blue-500/20 text-white'
                      : 'border-zinc-600 bg-zinc-700 text-zinc-300 hover:border-zinc-500'
                  }`}
                >
                  <div className="font-medium">{speakerId}</div>
                  <div className="text-xs opacity-75">
                    {formatDuration(duration)} ({percentage.toFixed(0)}%)
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {error && (
          <div className="p-3 bg-red-900/50 border border-red-700 rounded text-red-200 text-sm">
            {error}
          </div>
        )}

        {/* Action buttons */}
        <div className="flex gap-3">
          <button
            onClick={handleFilterAndUpload}
            disabled={!selectedSpeaker || isFiltering}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white font-medium"
          >
            {isFiltering ? (
              <Loader2 className="animate-spin" size={18} />
            ) : null}
            Upload for Manual Review
          </button>
          <button
            onClick={handleSkipFilter}
            disabled={isFiltering}
            className="px-4 py-3 bg-zinc-600 hover:bg-zinc-500 rounded-lg text-white"
          >
            Upload All
          </button>
          <button
            onClick={resetState}
            disabled={isFiltering}
            className="px-4 py-3 bg-zinc-700 hover:bg-zinc-600 rounded-lg text-zinc-300"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  if (stage === 'complete') {
    return (
      <div className="bg-zinc-800 rounded-lg p-6 text-center">
        <CheckCircle className="mx-auto text-green-400 mb-2" size={48} />
        <h3 className="text-lg font-semibold text-white mb-1">Sample Added</h3>
        <p className="text-zinc-400 text-sm">The training sample has been uploaded successfully.</p>
      </div>
    );
  }

  return (
    <div className="bg-zinc-800 rounded-lg p-6 space-y-4">
      <h3 className="text-lg font-semibold text-white">Upload Training Sample</h3>

      {/* File selection */}
      <label className="block">
        <span className="sr-only">Select training sample audio file</span>
        <div className="border-2 border-dashed border-zinc-600 hover:border-blue-500 rounded-lg p-6 text-center cursor-pointer transition-colors">
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileSelect}
            className="hidden"
            disabled={stage !== 'idle'}
          />
          {file ? (
            <div>
              <p className="text-white font-medium">{file.name}</p>
              <p className="text-zinc-400 text-sm mt-1">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          ) : (
            <div>
              <Upload className="mx-auto text-zinc-500 mb-2" size={32} />
              <p className="text-zinc-300">Click to select audio file</p>
              <p className="text-zinc-500 text-sm mt-1">WAV, MP3, FLAC (10-60 seconds recommended)</p>
            </div>
          )}
        </div>
      </label>

      {/* Diarization option */}
      <label className="flex items-center gap-3 p-3 bg-zinc-700 rounded-lg cursor-pointer">
        <input
          type="checkbox"
          checked={enableDiarization}
          onChange={(e) => setEnableDiarization(e.target.checked)}
          className="w-4 h-4 rounded border-zinc-500 text-blue-600 focus:ring-blue-500 focus:ring-offset-zinc-800"
        />
        <div className="flex-1">
          <div className="flex items-center gap-2 text-white">
            <Users size={16} />
            <span className="font-medium">Detect multiple speakers</span>
          </div>
          <p className="text-zinc-400 text-xs mt-0.5">
            Analyze audio for speaker review before upload. This does not automatically remove other singers yet.
          </p>
        </div>
      </label>

      {error && (
        <div className="p-3 bg-red-900/50 border border-red-700 rounded text-red-200 text-sm">
          {error}
        </div>
      )}

      {/* Upload button */}
      <button
        onClick={handleUploadWithDiarization}
        disabled={!file || stage !== 'idle'}
        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-colors"
      >
        {stage === 'diarizing' ? (
          <>
            <Loader2 className="animate-spin" size={18} />
            Analyzing speakers...
          </>
        ) : stage === 'uploading' ? (
          <>
            <Loader2 className="animate-spin" size={18} />
            Uploading...
          </>
        ) : (
          <>
            <Upload size={18} />
            Upload Sample
          </>
        )}
      </button>
    </div>
  );
}

export default TrainingSampleUpload;
