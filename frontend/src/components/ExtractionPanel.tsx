/**
 * ExtractionPanel - Display and manage speaker extraction jobs
 */
import React, { useState, useEffect, useRef } from 'react';

interface DetectedSpeaker {
  speaker_id: string;
  duration_sec: number;
  segments: number;
  is_primary: boolean;
}

interface ExtractionJob {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  artist_name: string;
  tracks_processed: number;
  tracks_total: number;
  speakers_detected: DetectedSpeaker[];
  error?: string;
  started_at?: string;
  completed_at?: string;
}

interface ExtractionPanelProps {
  artistName?: string;
  onExtractionComplete?: (job: ExtractionJob) => void;
  onSpeakerSelect?: (speaker: DetectedSpeaker) => void;
}

const ExtractionPanel: React.FC<ExtractionPanelProps> = ({
  artistName,
  onExtractionComplete,
  onSpeakerSelect,
}) => {
  const [job, setJob] = useState<ExtractionJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedArtist, setSelectedArtist] = useState(artistName || '');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const pollInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // Available artists (could be fetched from API)
  const artists = [
    { value: 'conor_maynard', label: 'Conor Maynard' },
    { value: 'william_singe', label: 'William Singe' },
  ];

  // Start extraction job
  const startExtraction = async () => {
    if (!selectedArtist) return;

    setLoading(true);
    setError(null);
    setJob(null);

    try {
      const response = await fetch('/api/v1/speakers/extraction/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ artist_name: selectedArtist }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to start extraction');
      }

      setJob({
        job_id: data.job_id,
        status: 'running',
        progress: 0,
        artist_name: selectedArtist,
        tracks_processed: 0,
        tracks_total: 0,
        speakers_detected: [],
      });

      // Start polling for status
      startPolling(data.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start extraction');
    } finally {
      setLoading(false);
    }
  };

  // Poll for job status
  const startPolling = (jobId: string) => {
    if (pollInterval.current) {
      clearInterval(pollInterval.current);
    }

    pollInterval.current = setInterval(async () => {
      try {
        const response = await fetch(`/api/v1/speakers/extraction/status/${jobId}`);
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || 'Failed to get status');
        }

        setJob(data);

        // Stop polling when complete or failed
        if (data.status === 'completed' || data.status === 'failed') {
          if (pollInterval.current) {
            clearInterval(pollInterval.current);
            pollInterval.current = null;
          }

          if (data.status === 'completed') {
            onExtractionComplete?.(data);
          }
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 2000);
  };

  // Preview speaker audio
  const previewSpeaker = async (speakerId: string) => {
    try {
      // Find cluster ID for this speaker
      const response = await fetch('/api/v1/speakers/clusters');
      const data = await response.json();

      const cluster = data.clusters?.find((c: any) =>
        c.name.toLowerCase().includes(speakerId.toLowerCase())
      );

      if (cluster) {
        const sampleResponse = await fetch(
          `/api/v1/speakers/clusters/${cluster.id}/sample?max_duration=10`
        );

        if (!sampleResponse.ok) {
          throw new Error('Failed to load sample');
        }

        const blob = await sampleResponse.blob();
        const url = URL.createObjectURL(blob);

        if (previewUrl) {
          URL.revokeObjectURL(previewUrl);
        }

        setPreviewUrl(url);
        if (audioRef.current) {
          audioRef.current.src = url;
          audioRef.current.play();
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to preview');
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollInterval.current) {
        clearInterval(pollInterval.current);
      }
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-400';
      case 'running':
        return 'text-blue-400';
      case 'failed':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h2 className="text-xl font-semibold text-white mb-4">
        Speaker Extraction
      </h2>

      {/* Artist Selection */}
      <div className="mb-4">
        <label className="block text-sm text-gray-400 mb-2">
          Select Artist
        </label>
        <div className="flex gap-2">
          <select
            value={selectedArtist}
            onChange={(e) => setSelectedArtist(e.target.value)}
            disabled={job?.status === 'running'}
            className="flex-1 bg-gray-700 text-white rounded px-3 py-2"
          >
            <option value="">Choose an artist...</option>
            {artists.map((artist) => (
              <option key={artist.value} value={artist.value}>
                {artist.label}
              </option>
            ))}
          </select>
          <button
            onClick={startExtraction}
            disabled={!selectedArtist || loading || job?.status === 'running'}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Starting...' : 'Extract Speakers'}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-900/30 border border-red-700 rounded text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Job Status */}
      {job && (
        <div className="space-y-4">
          {/* Status Header */}
          <div className="flex justify-between items-center">
            <div>
              <span className={`font-medium ${getStatusColor(job.status)}`}>
                {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
              </span>
              <span className="text-gray-400 ml-2">
                {job.artist_name}
              </span>
            </div>
            {job.status === 'running' && (
              <span className="text-sm text-gray-400">
                {job.tracks_processed} / {job.tracks_total} tracks
              </span>
            )}
          </div>

          {/* Progress Bar */}
          {job.status === 'running' && (
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${job.progress}%` }}
              />
            </div>
          )}

          {/* Error Message */}
          {job.error && (
            <div className="p-3 bg-red-900/30 border border-red-700 rounded text-red-300 text-sm">
              {job.error}
            </div>
          )}

          {/* Detected Speakers */}
          {job.speakers_detected.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-300 mb-2">
                Detected Speakers ({job.speakers_detected.length})
              </h3>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {job.speakers_detected.map((speaker) => (
                  <div
                    key={speaker.speaker_id}
                    onClick={() => onSpeakerSelect?.(speaker)}
                    className={`bg-gray-700 rounded p-3 cursor-pointer hover:bg-gray-600 ${
                      speaker.is_primary ? 'border-l-4 border-green-500' : ''
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <span className="text-white font-medium">
                          {speaker.speaker_id}
                        </span>
                        {speaker.is_primary && (
                          <span className="ml-2 text-xs text-green-400">
                            Primary
                          </span>
                        )}
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          previewSpeaker(speaker.speaker_id);
                        }}
                        className="text-blue-400 hover:text-blue-300 text-sm"
                      >
                        ▶ Preview
                      </button>
                    </div>
                    <div className="text-sm text-gray-400 mt-1">
                      {formatDuration(speaker.duration_sec)} •{' '}
                      {speaker.segments} segments
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Completion Summary */}
          {job.status === 'completed' && (
            <div className="p-3 bg-green-900/30 border border-green-700 rounded">
              <p className="text-green-300 text-sm">
                Extraction complete! Found {job.speakers_detected.length} speaker
                {job.speakers_detected.length !== 1 ? 's' : ''} across{' '}
                {job.tracks_processed} tracks.
              </p>
              {job.completed_at && (
                <p className="text-gray-400 text-xs mt-1">
                  Completed at {new Date(job.completed_at).toLocaleTimeString()}
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Hidden Audio Player */}
      <audio ref={audioRef} className="hidden" />

      {/* Instructions */}
      {!job && (
        <div className="text-gray-400 text-sm mt-4">
          <p>
            Speaker extraction analyzes audio tracks to identify and separate
            different voices. Select an artist to extract speakers from their
            tracks.
          </p>
          <ul className="mt-2 list-disc list-inside space-y-1">
            <li>Extracts speaker embeddings using WavLM</li>
            <li>Clusters similar voices across tracks</li>
            <li>Identifies primary vs featured artists</li>
            <li>Generates isolated audio samples</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default ExtractionPanel;
