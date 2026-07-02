/**
 * ExtractionPanel - Display and manage speaker extraction jobs
 */
import React, { useState, useEffect, useRef } from 'react';
import { apiService } from '../services/api';
import type { DetectedSpeaker, SpeakerExtractionJob } from '../services/api';

interface ExtractionPanelProps {
  artistName?: string;
  onExtractionComplete?: (job: SpeakerExtractionJob) => void;
  onSpeakerSelect?: (speaker: DetectedSpeaker) => void;
}

const ExtractionPanel: React.FC<ExtractionPanelProps> = ({
  artistName,
  onExtractionComplete,
  onSpeakerSelect,
}) => {
  const [job, setJob] = useState<SpeakerExtractionJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedArtist, setSelectedArtist] = useState(artistName || '');
  const [knownArtists, setKnownArtists] = useState<string[]>([]);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const pollInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // Sync artist input when parent filter changes
  useEffect(() => {
    if (artistName) {
      setSelectedArtist(artistName);
    }
  }, [artistName]);

  // Best-effort suggestions for the artist datalist
  useEffect(() => {
    apiService
      .listFeaturedArtists()
      .then((data) => setKnownArtists(data.artists.map((a) => a.name)))
      .catch(() => {
        // ponytail: suggestions only — free-text input works without them
      });
  }, []);

  // Start extraction job
  const startExtraction = async () => {
    if (!selectedArtist.trim()) return;

    setLoading(true);
    setError(null);
    setJob(null);

    try {
      const data = await apiService.runSpeakerExtraction(selectedArtist.trim());

      setJob({
        job_id: data.job_id,
        status: (data.status === 'queued'
          ? 'pending'
          : data.status) as SpeakerExtractionJob['status'],
        progress: 0,
        artist_name: selectedArtist.trim(),
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
        const data = await apiService.getSpeakerExtractionStatus(jobId);

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
      const data = await apiService.listSpeakerClusters();

      const cluster = data.clusters?.find((c) =>
        c.name.toLowerCase().includes(speakerId.toLowerCase())
      );

      if (cluster) {
        const blob = await apiService.fetchSpeakerClusterSample(cluster.id, 10);
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
          Artist
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            list="extraction-artist-options"
            value={selectedArtist}
            onChange={(e) => setSelectedArtist(e.target.value)}
            disabled={job?.status === 'running'}
            placeholder="Enter artist name..."
            data-testid="extraction-artist-input"
            className="flex-1 bg-gray-700 text-white rounded px-3 py-2"
          />
          <datalist id="extraction-artist-options">
            {knownArtists.map((name) => (
              <option key={name} value={name} />
            ))}
          </datalist>
          <button
            onClick={startExtraction}
            disabled={!selectedArtist.trim() || loading || job?.status === 'running'}
            data-testid="run-extraction-button"
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
            different voices. Enter an artist to extract speakers from their
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
