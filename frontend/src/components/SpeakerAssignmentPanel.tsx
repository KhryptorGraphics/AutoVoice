import { useState } from 'react';

interface DiarizationSegment {
  start: number;
  end: number;
  speaker_id: string;
  confidence: number;
  duration: number;
}

interface VoiceProfile {
  profile_id: string;
  name: string;
  has_embedding?: boolean;
}

interface SpeakerAssignmentPanelProps {
  segments: DiarizationSegment[];
  diarizationId: string;
  profiles: VoiceProfile[];
  suggestedArtists?: string[];
  onAssignSegment: (segmentIndex: number, profileId: string) => Promise<void>;
  onCreateProfile: (speakerId: string, name: string) => Promise<void>;
  onRefreshProfiles?: () => void;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function SpeakerAssignmentPanel({
  segments,
  diarizationId: _diarizationId,
  profiles,
  suggestedArtists = [],
  onAssignSegment,
  onCreateProfile,
  onRefreshProfiles,
}: SpeakerAssignmentPanelProps) {
  // Note: diarizationId is passed for context but assignment is handled by parent via onAssignSegment
  void _diarizationId;
  const [selectedSegmentIndex, setSelectedSegmentIndex] = useState<number | null>(null);
  const [selectedProfileId, setSelectedProfileId] = useState<string>('');
  const [newProfileName, setNewProfileName] = useState<string>('');
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [assignments, setAssignments] = useState<Map<number, string>>(new Map());

  // Get unique speakers
  const speakerIds = [...new Set(segments.map(s => s.speaker_id))];

  const handleAssign = async () => {
    if (selectedSegmentIndex === null || !selectedProfileId) return;

    setIsLoading(true);
    setError(null);

    try {
      await onAssignSegment(selectedSegmentIndex, selectedProfileId);

      // Track assignment locally
      const newAssignments = new Map(assignments);
      newAssignments.set(selectedSegmentIndex, selectedProfileId);
      setAssignments(newAssignments);

      // Reset selection
      setSelectedSegmentIndex(null);
      setSelectedProfileId('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to assign segment');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateProfile = async () => {
    if (!newProfileName.trim()) return;

    const speakerId = selectedSegmentIndex !== null
      ? segments[selectedSegmentIndex].speaker_id
      : speakerIds[0];

    setIsLoading(true);
    setError(null);

    try {
      await onCreateProfile(speakerId, newProfileName.trim());
      setNewProfileName('');
      setShowCreateForm(false);
      onRefreshProfiles?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create profile');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestedArtistClick = (artist: string) => {
    setNewProfileName(artist);
    setShowCreateForm(true);
  };

  // Group segments by speaker
  const segmentsBySpeaker = speakerIds.map(speakerId => ({
    speakerId,
    segments: segments
      .map((seg, idx) => ({ ...seg, index: idx }))
      .filter(seg => seg.speaker_id === speakerId),
    totalDuration: segments
      .filter(s => s.speaker_id === speakerId)
      .reduce((sum, s) => sum + s.duration, 0),
  }));

  return (
    <div className="bg-zinc-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Speaker Assignment</h3>
        {onRefreshProfiles && (
          <button
            onClick={onRefreshProfiles}
            className="text-sm text-blue-400 hover:text-blue-300"
          >
            Refresh Profiles
          </button>
        )}
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-900/50 border border-red-700 rounded text-red-200 text-sm">
          {error}
        </div>
      )}

      {/* Suggested Artists from YouTube metadata */}
      {suggestedArtists.length > 0 && (
        <div className="mb-4 p-3 bg-zinc-700 rounded">
          <p className="text-sm text-zinc-400 mb-2">Detected Artists (from title/description):</p>
          <div className="flex flex-wrap gap-2">
            {suggestedArtists.map((artist) => (
              <button
                key={artist}
                onClick={() => handleSuggestedArtistClick(artist)}
                className="px-3 py-1 bg-blue-600/50 hover:bg-blue-600 rounded-full text-sm text-white transition-colors"
              >
                + {artist}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Speaker list with assignment options */}
      <div className="space-y-4">
        {segmentsBySpeaker.map(({ speakerId, segments: speakerSegments, totalDuration }) => (
          <div key={speakerId} className="p-3 bg-zinc-700 rounded">
            <div className="flex items-center justify-between mb-2">
              <div>
                <span className="font-medium text-white">{speakerId}</span>
                <span className="ml-2 text-sm text-zinc-400">
                  {speakerSegments.length} segments ({formatTime(totalDuration)})
                </span>
              </div>
            </div>

            {/* Segment chips */}
            <div className="flex flex-wrap gap-2 mb-3">
              {speakerSegments.map((segment) => {
                const isAssigned = assignments.has(segment.index);
                const assignedProfile = isAssigned
                  ? profiles.find(p => p.profile_id === assignments.get(segment.index))
                  : null;

                return (
                  <button
                    key={segment.index}
                    onClick={() => setSelectedSegmentIndex(
                      selectedSegmentIndex === segment.index ? null : segment.index
                    )}
                    className={`px-2 py-1 rounded text-xs transition-colors ${
                      selectedSegmentIndex === segment.index
                        ? 'bg-blue-600 text-white'
                        : isAssigned
                        ? 'bg-green-700 text-green-100'
                        : 'bg-zinc-600 text-zinc-300 hover:bg-zinc-500'
                    }`}
                    title={isAssigned ? `Assigned to: ${assignedProfile?.name}` : undefined}
                  >
                    {formatTime(segment.start)}-{formatTime(segment.end)}
                    {isAssigned && ' ✓'}
                  </button>
                );
              })}
            </div>

            {/* Quick assign all segments for this speaker */}
            <div className="flex items-center gap-2">
              <select
                className="flex-1 bg-zinc-600 text-white rounded px-3 py-1.5 text-sm"
                onChange={(e) => {
                  // For quick assignment, we'd assign all segments of this speaker
                  // This is a simplified version - in production, you might batch these
                  if (e.target.value) {
                    setSelectedProfileId(e.target.value);
                    // Select first unassigned segment of this speaker
                    const firstUnassigned = speakerSegments.find(s => !assignments.has(s.index));
                    if (firstUnassigned) {
                      setSelectedSegmentIndex(firstUnassigned.index);
                    }
                  }
                }}
                defaultValue=""
              >
                <option value="">Assign to profile...</option>
                {profiles.map((profile) => (
                  <option key={profile.profile_id} value={profile.profile_id}>
                    {profile.name} {profile.has_embedding ? '(has embedding)' : ''}
                  </option>
                ))}
              </select>
              <button
                onClick={() => {
                  setNewProfileName(speakerId.replace('speaker_', 'Speaker '));
                  setShowCreateForm(true);
                }}
                className="px-3 py-1.5 bg-green-600 hover:bg-green-700 rounded text-sm text-white"
              >
                New Profile
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Assignment action */}
      {selectedSegmentIndex !== null && selectedProfileId && (
        <div className="mt-4 p-3 bg-blue-900/30 border border-blue-700 rounded">
          <p className="text-sm text-blue-200 mb-2">
            Assign segment {selectedSegmentIndex + 1} ({formatTime(segments[selectedSegmentIndex].start)} - {formatTime(segments[selectedSegmentIndex].end)}) to {profiles.find(p => p.profile_id === selectedProfileId)?.name}?
          </p>
          <div className="flex gap-2">
            <button
              onClick={handleAssign}
              disabled={isLoading}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded text-white text-sm"
            >
              {isLoading ? 'Assigning...' : 'Confirm Assignment'}
            </button>
            <button
              onClick={() => {
                setSelectedSegmentIndex(null);
                setSelectedProfileId('');
              }}
              className="px-4 py-2 bg-zinc-600 hover:bg-zinc-500 rounded text-white text-sm"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Create profile form */}
      {showCreateForm && (
        <div className="mt-4 p-3 bg-green-900/30 border border-green-700 rounded">
          <p className="text-sm text-green-200 mb-2">Create new profile from speaker:</p>
          <div className="flex gap-2">
            <input
              type="text"
              value={newProfileName}
              onChange={(e) => setNewProfileName(e.target.value)}
              placeholder="Profile name"
              className="flex-1 bg-zinc-700 text-white rounded px-3 py-2 text-sm"
              autoFocus
            />
            <button
              onClick={handleCreateProfile}
              disabled={isLoading || !newProfileName.trim()}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:opacity-50 rounded text-white text-sm"
            >
              {isLoading ? 'Creating...' : 'Create'}
            </button>
            <button
              onClick={() => {
                setShowCreateForm(false);
                setNewProfileName('');
              }}
              className="px-4 py-2 bg-zinc-600 hover:bg-zinc-500 rounded text-white text-sm"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Summary */}
      <div className="mt-4 pt-4 border-t border-zinc-700">
        <div className="flex justify-between text-sm text-zinc-400">
          <span>Total speakers: {speakerIds.length}</span>
          <span>Assigned: {assignments.size} / {segments.length} segments</span>
        </div>
      </div>
    </div>
  );
}

export default SpeakerAssignmentPanel;
