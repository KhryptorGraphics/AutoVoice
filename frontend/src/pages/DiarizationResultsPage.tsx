import React, { useState, useCallback } from 'react';
import { DiarizationTimeline } from '../components/DiarizationTimeline';
import { SpeakerAssignmentPanel } from '../components/SpeakerAssignmentPanel';
import { apiService as api } from '../services/api';

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

interface VoiceProfile {
  profile_id: string;
  name: string;
  has_embedding?: boolean;
}

export function DiarizationResultsPage() {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [diarizationResult, setDiarizationResult] = useState<DiarizationResult | null>(null);
  const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSegmentIndex, setSelectedSegmentIndex] = useState<number | undefined>();
  const [suggestedArtists, setSuggestedArtists] = useState<string[]>([]);

  // Load profiles
  const loadProfiles = useCallback(async () => {
    try {
      const profiles = await api.listProfiles();
      setProfiles(profiles.map(p => ({
        profile_id: p.profile_id,
        name: p.name || p.profile_id,
        has_embedding: false  // Will be updated when we add embedding check
      })));
    } catch (err) {
      console.error('Failed to load profiles:', err);
    }
  }, []);

  // Initial load
  React.useEffect(() => {
    loadProfiles();
  }, [loadProfiles]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setAudioFile(file);
    setAudioUrl(URL.createObjectURL(file));
    setDiarizationResult(null);
    setError(null);

    // Try to extract artist names from filename
    const filename = file.name.replace(/\.[^/.]+$/, '');
    const artists = extractArtistsFromFilename(filename);
    setSuggestedArtists(artists);
  };

  const extractArtistsFromFilename = (filename: string): string[] => {
    const artists: string[] = [];

    // Common patterns: "Artist - Song ft. Other Artist"
    const ftMatch = filename.match(/(?:ft\.?|feat\.?|featuring)\s*([^()\[\]]+)/i);
    if (ftMatch) {
      artists.push(ftMatch[1].trim());
    }

    // Pattern: "Artist1 vs Artist2"
    const vsMatch = filename.match(/(.+?)\s+vs\.?\s+(.+)/i);
    if (vsMatch) {
      artists.push(vsMatch[1].trim(), vsMatch[2].trim());
    }

    // Pattern: "Artist1 & Artist2 - Song"
    const ampMatch = filename.match(/^([^-]+?)\s*&\s*([^-]+)/i);
    if (ampMatch && !vsMatch) {
      artists.push(ampMatch[1].trim(), ampMatch[2].trim());
    }

    return [...new Set(artists)].filter(a => a.length > 0);
  };

  const handleDiarize = async () => {
    if (!audioFile) return;

    setIsProcessing(true);
    setError(null);

    try {
      const result = await api.diarizeAudio(audioFile);
      setDiarizationResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run diarization');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSegmentClick = (index: number) => {
    setSelectedSegmentIndex(index);
  };

  const handleAssignSegment = async (segmentIndex: number, profileId: string) => {
    if (!diarizationResult) return;

    await api.assignDiarizationSegment(
      diarizationResult.diarization_id,
      segmentIndex,
      profileId
    );
  };

  const handleCreateProfile = async (speakerId: string, name: string) => {
    if (!diarizationResult) return;

    await api.autoCreateProfileFromDiarization(
      diarizationResult.diarization_id,
      speakerId,
      name,
      undefined,
      true,
      {
        profileRole: 'source_artist',
      }
    );

    // Refresh profiles
    await loadProfiles();
  };

  return (
    <div className="min-h-screen bg-zinc-900 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Speaker Diarization</h1>
          <p className="text-zinc-400">
            Upload audio to detect and separate different speakers. Assign segments to voice profiles for training.
          </p>
        </div>

        {/* Upload Section */}
        <div className="bg-zinc-800 rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold text-white mb-4">Upload Audio</h2>

          <div className="flex items-center gap-4">
            <label className="flex-1">
              <div className="border-2 border-dashed border-zinc-600 hover:border-blue-500 rounded-lg p-8 text-center cursor-pointer transition-colors">
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                {audioFile ? (
                  <div>
                    <p className="text-white font-medium">{audioFile.name}</p>
                    <p className="text-zinc-400 text-sm mt-1">
                      {(audioFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                ) : (
                  <div>
                    <svg className="w-12 h-12 mx-auto text-zinc-500 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <p className="text-zinc-300">Click to upload or drag and drop</p>
                    <p className="text-zinc-500 text-sm mt-1">WAV, MP3, FLAC, etc.</p>
                  </div>
                )}
              </div>
            </label>

            <button
              onClick={handleDiarize}
              disabled={!audioFile || isProcessing}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-colors"
            >
              {isProcessing ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Processing...
                </span>
              ) : (
                'Run Diarization'
              )}
            </button>
          </div>

          {error && (
            <div className="mt-4 p-4 bg-red-900/50 border border-red-700 rounded-lg text-red-200">
              {error}
            </div>
          )}
        </div>

        {/* Results Section */}
        {diarizationResult && (
          <div className="space-y-6">
            {/* Summary */}
            <div className="bg-zinc-800 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Results Summary</h2>
              <div className="grid grid-cols-3 gap-6">
                <div className="bg-zinc-700 rounded-lg p-4 text-center">
                  <p className="text-3xl font-bold text-blue-400">{diarizationResult.num_speakers}</p>
                  <p className="text-zinc-400 text-sm mt-1">Speakers Detected</p>
                </div>
                <div className="bg-zinc-700 rounded-lg p-4 text-center">
                  <p className="text-3xl font-bold text-green-400">{diarizationResult.segments.length}</p>
                  <p className="text-zinc-400 text-sm mt-1">Total Segments</p>
                </div>
                <div className="bg-zinc-700 rounded-lg p-4 text-center">
                  <p className="text-3xl font-bold text-purple-400">
                    {Math.floor(diarizationResult.audio_duration / 60)}:{String(Math.floor(diarizationResult.audio_duration % 60)).padStart(2, '0')}
                  </p>
                  <p className="text-zinc-400 text-sm mt-1">Audio Duration</p>
                </div>
              </div>

              {/* Speaker breakdown */}
              <div className="mt-6">
                <h3 className="text-lg font-medium text-white mb-3">Speaker Breakdown</h3>
                <div className="space-y-2">
                  {Object.entries(diarizationResult.speaker_durations).map(([speakerId, duration]) => {
                    const percentage = (duration / diarizationResult.audio_duration) * 100;
                    return (
                      <div key={speakerId} className="flex items-center gap-4">
                        <span className="text-zinc-300 w-24">{speakerId}</span>
                        <div className="flex-1 bg-zinc-600 rounded-full h-3">
                          <div
                            className="bg-blue-500 rounded-full h-3 transition-all"
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                        <span className="text-zinc-400 text-sm w-20 text-right">
                          {percentage.toFixed(1)}% ({Math.floor(duration)}s)
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Timeline */}
            <DiarizationTimeline
              segments={diarizationResult.segments}
              audioDuration={diarizationResult.audio_duration}
              audioUrl={audioUrl || undefined}
              selectedSegmentIndex={selectedSegmentIndex}
              onSegmentClick={handleSegmentClick}
            />

            {/* Assignment Panel */}
            <SpeakerAssignmentPanel
              segments={diarizationResult.segments}
              diarizationId={diarizationResult.diarization_id}
              profiles={profiles}
              suggestedArtists={suggestedArtists}
              onAssignSegment={handleAssignSegment}
              onCreateProfile={handleCreateProfile}
              onRefreshProfiles={loadProfiles}
            />
          </div>
        )}

        {/* Help text when no result */}
        {!diarizationResult && !isProcessing && (
          <div className="bg-zinc-800 rounded-lg p-8 text-center">
            <svg className="w-16 h-16 mx-auto text-zinc-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            <h3 className="text-xl font-semibold text-white mb-2">No Audio Loaded</h3>
            <p className="text-zinc-400 max-w-md mx-auto">
              Upload an audio file to detect different speakers. This is useful for:
            </p>
            <ul className="text-zinc-400 mt-4 space-y-1">
              <li>Separating vocals in duets or collaborations</li>
              <li>Filtering training data to specific artists</li>
              <li>Creating profiles for featured artists automatically</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default DiarizationResultsPage;
