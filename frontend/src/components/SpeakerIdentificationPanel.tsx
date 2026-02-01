/**
 * SpeakerIdentificationPanel - Manage speaker clusters across tracks
 */
import React, { useState, useEffect, useRef } from 'react';

interface ClusterMember {
  track_id: string;
  speaker_id: string;
  duration_sec: number;
  is_primary: boolean;
  confidence: number | null;
  track_title: string | null;
  artist_name: string;
}

interface Cluster {
  id: string;
  name: string;
  is_verified: boolean;
  member_count: number;
  total_duration_sec: number | null;
}

interface SpeakerIdentificationPanelProps {
  onClusterSelect?: (cluster: Cluster) => void;
  onRunIdentification?: () => void;
}

const SpeakerIdentificationPanel: React.FC<SpeakerIdentificationPanelProps> = ({
  onClusterSelect,
  onRunIdentification,
}) => {
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<Cluster | null>(null);
  const [members, setMembers] = useState<ClusterMember[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editingName, setEditingName] = useState<string | null>(null);
  const [newName, setNewName] = useState('');
  const [running, setRunning] = useState(false);
  const [mergeMode, setMergeMode] = useState(false);
  const [mergeTarget, setMergeTarget] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Load clusters
  const loadClusters = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/speakers/clusters');
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to load clusters');
      }

      setClusters(data.clusters || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load clusters');
    } finally {
      setLoading(false);
    }
  };

  // Load cluster details
  const loadClusterDetails = async (clusterId: string) => {
    try {
      const response = await fetch(`/api/v1/speakers/clusters/${clusterId}`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to load cluster details');
      }

      setMembers(data.members || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load details');
    }
  };

  // Run speaker identification
  const runIdentification = async () => {
    setRunning(true);
    setError(null);
    try {
      const response = await fetch('/api/v1/speakers/identify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          threshold: 0.85,
          min_duration: 30,
        }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Identification failed');
      }

      // Reload clusters
      await loadClusters();
      onRunIdentification?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Identification failed');
    } finally {
      setRunning(false);
    }
  };

  // Update cluster name
  const updateName = async (clusterId: string) => {
    if (!newName.trim()) return;

    try {
      const response = await fetch(`/api/v1/speakers/clusters/${clusterId}/name`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newName.trim(), is_verified: true }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to update name');
      }

      // Reload clusters
      await loadClusters();
      setEditingName(null);
      setNewName('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update name');
    }
  };

  // Merge clusters
  const mergeClusters = async (targetId: string, sourceId: string) => {
    try {
      const response = await fetch('/api/v1/speakers/clusters/merge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_id: targetId, source_id: sourceId }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to merge clusters');
      }

      // Reload and reset
      await loadClusters();
      setMergeMode(false);
      setMergeTarget(null);
      setSelectedCluster(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to merge');
    }
  };

  // Play sample
  const playSample = async (clusterId: string) => {
    try {
      const response = await fetch(
        `/api/v1/speakers/clusters/${clusterId}/sample?max_duration=10`
      );
      if (!response.ok) {
        throw new Error('Failed to load sample');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);

      if (audioRef.current) {
        audioRef.current.src = url;
        audioRef.current.play();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to play sample');
    }
  };

  useEffect(() => {
    loadClusters();
  }, []);

  useEffect(() => {
    if (selectedCluster) {
      loadClusterDetails(selectedCluster.id);
      onClusterSelect?.(selectedCluster);
    }
  }, [selectedCluster]);

  const formatDuration = (seconds: number | null) => {
    if (!seconds) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-white">Speaker Clusters</h2>
        <div className="flex gap-2">
          {mergeMode ? (
            <button
              onClick={() => {
                setMergeMode(false);
                setMergeTarget(null);
              }}
              className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-500"
            >
              Cancel Merge
            </button>
          ) : (
            <button
              onClick={() => setMergeMode(true)}
              className="px-3 py-1 bg-yellow-600 text-white rounded text-sm hover:bg-yellow-700"
            >
              Merge Clusters
            </button>
          )}
          <button
            onClick={runIdentification}
            disabled={running}
            className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            {running ? 'Running...' : 'Run Identification'}
          </button>
        </div>
      </div>

      {/* Merge Mode Instructions */}
      {mergeMode && (
        <div className="mb-4 p-3 bg-yellow-900/30 border border-yellow-700 rounded text-yellow-300 text-sm">
          {mergeTarget
            ? 'Now select the cluster to merge INTO the first selection'
            : 'Select the cluster you want to KEEP'}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="text-red-400 text-sm mb-4 p-2 bg-red-900/20 rounded">
          {error}
        </div>
      )}

      {/* Audio player (hidden) */}
      <audio ref={audioRef} className="hidden" />

      {/* Loading */}
      {loading && (
        <div className="text-gray-400 text-center py-8">Loading clusters...</div>
      )}

      {/* Two-column layout */}
      {!loading && (
        <div className="grid grid-cols-2 gap-4">
          {/* Cluster List */}
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {clusters.length === 0 ? (
              <div className="text-gray-400 text-center py-8">
                No clusters found. Run identification to detect speakers.
              </div>
            ) : (
              clusters.map((cluster) => (
                <div
                  key={cluster.id}
                  onClick={() => {
                    if (mergeMode) {
                      if (!mergeTarget) {
                        setMergeTarget(cluster.id);
                      } else if (mergeTarget !== cluster.id) {
                        mergeClusters(mergeTarget, cluster.id);
                      }
                    } else {
                      setSelectedCluster(cluster);
                    }
                  }}
                  className={`p-3 rounded cursor-pointer ${
                    selectedCluster?.id === cluster.id
                      ? 'bg-blue-600'
                      : mergeTarget === cluster.id
                      ? 'bg-yellow-600'
                      : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      {editingName === cluster.id ? (
                        <div className="flex gap-2">
                          <input
                            type="text"
                            value={newName}
                            onChange={(e) => setNewName(e.target.value)}
                            onClick={(e) => e.stopPropagation()}
                            className="bg-gray-600 text-white rounded px-2 py-1 text-sm"
                            placeholder="Enter name..."
                            autoFocus
                          />
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              updateName(cluster.id);
                            }}
                            className="text-green-400 hover:text-green-300"
                          >
                            ✓
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setEditingName(null);
                            }}
                            className="text-red-400 hover:text-red-300"
                          >
                            ✕
                          </button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2">
                          <span className="text-white font-medium">
                            {cluster.name}
                          </span>
                          {cluster.is_verified && (
                            <span className="text-green-400 text-xs">✓</span>
                          )}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setEditingName(cluster.id);
                              setNewName(cluster.name);
                            }}
                            className="text-gray-400 hover:text-white text-xs"
                          >
                            ✏️
                          </button>
                        </div>
                      )}
                      <p className="text-sm text-gray-400">
                        {cluster.member_count} tracks •{' '}
                        {formatDuration(cluster.total_duration_sec)}
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        playSample(cluster.id);
                      }}
                      className="text-blue-400 hover:text-blue-300"
                    >
                      ▶
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Selected Cluster Details */}
          <div className="bg-gray-700 rounded p-4">
            {selectedCluster ? (
              <>
                <h3 className="text-lg font-semibold text-white mb-4">
                  {selectedCluster.name}
                </h3>

                <div className="space-y-2 max-h-72 overflow-y-auto">
                  {members.map((member, i) => (
                    <div key={i} className="bg-gray-600 rounded p-2 text-sm">
                      <div className="text-white">
                        {member.track_title || member.track_id}
                      </div>
                      <div className="text-gray-400">
                        {member.speaker_id} •{' '}
                        {formatDuration(member.duration_sec)}
                        {member.is_primary && (
                          <span className="ml-2 text-green-400">(Primary)</span>
                        )}
                        {member.confidence && (
                          <span className="ml-2 text-yellow-400">
                            {(member.confidence * 100).toFixed(0)}% confidence
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="text-gray-400 text-center py-8">
                Select a cluster to view details
              </div>
            )}
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="mt-4 pt-4 border-t border-gray-700 text-sm text-gray-400">
        {clusters.length} speaker clusters •{' '}
        {clusters.filter((c) => c.is_verified).length} verified
      </div>
    </div>
  );
};

export default SpeakerIdentificationPanel;
