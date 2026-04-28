/**
 * TrackListPanel - Display tracks with YouTube metadata and featured artists
 */
import React, { useState, useEffect, useCallback } from 'react';

interface Track {
  id: string;
  title: string | null;
  channel: string | null;
  artist_name: string;
  duration_sec: number | null;
  featured_artists: string[];
  vocals_path: string | null;
}

interface TrackListPanelProps {
  artistFilter?: string;
  onTrackSelect?: (track: Track) => void;
}

const TrackListPanel: React.FC<TrackListPanelProps> = ({
  artistFilter,
  onTrackSelect,
}) => {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState({
    artist: artistFilter || '',
    hasFeatured: false,
    search: '',
  });
  const [fetchingMetadata, setFetchingMetadata] = useState(false);

  // Load tracks
  const loadTracks = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (filter.artist) params.append('artist', filter.artist);
      if (filter.hasFeatured) params.append('has_featured', 'true');

      const response = await fetch(`/api/v1/speakers/tracks?${params}`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to load tracks');
      }

      setTracks(data.tracks || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load tracks');
    } finally {
      setLoading(false);
    }
  }, [filter.artist, filter.hasFeatured]);

  // Fetch YouTube metadata for all tracks
  const fetchMetadata = async () => {
    setFetchingMetadata(true);
    try {
      const response = await fetch('/api/v1/speakers/tracks/fetch-metadata', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ artist_name: filter.artist || undefined }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch metadata');
      }

      // Reload tracks after fetching
      await loadTracks();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch metadata');
    } finally {
      setFetchingMetadata(false);
    }
  };

  useEffect(() => {
    loadTracks();
  }, [loadTracks]);

  // Filter tracks by search
  const filteredTracks = tracks.filter((track) => {
    if (!filter.search) return true;
    const searchLower = filter.search.toLowerCase();
    return (
      track.id.toLowerCase().includes(searchLower) ||
      track.title?.toLowerCase().includes(searchLower) ||
      track.featured_artists.some((a) => a.toLowerCase().includes(searchLower))
    );
  });

  const formatDuration = (seconds: number | null) => {
    if (!seconds) return '--:--';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-white">Tracks</h2>
        <button
          onClick={fetchMetadata}
          disabled={fetchingMetadata}
          className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
        >
          {fetchingMetadata ? 'Fetching...' : 'Fetch YouTube Metadata'}
        </button>
      </div>

      {/* Filters */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <label className="block text-sm text-gray-400 mb-1">Artist</label>
          <select
            value={filter.artist}
            onChange={(e) => setFilter({ ...filter, artist: e.target.value })}
            className="w-full bg-gray-700 text-white rounded px-3 py-2 text-sm"
          >
            <option value="">All Artists</option>
            <option value="conor_maynard">Conor Maynard</option>
            <option value="william_singe">William Singe</option>
          </select>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-1">Search</label>
          <input
            type="text"
            value={filter.search}
            onChange={(e) => setFilter({ ...filter, search: e.target.value })}
            placeholder="Search titles, IDs..."
            className="w-full bg-gray-700 text-white rounded px-3 py-2 text-sm"
          />
        </div>

        <div className="flex items-end">
          <label className="flex items-center text-sm text-gray-400">
            <input
              type="checkbox"
              checked={filter.hasFeatured}
              onChange={(e) =>
                setFilter({ ...filter, hasFeatured: e.target.checked })
              }
              className="mr-2"
            />
            Has Featured Artists
          </label>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="text-red-400 text-sm mb-4 p-2 bg-red-900/20 rounded">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="text-gray-400 text-center py-8">Loading tracks...</div>
      )}

      {/* Track List */}
      {!loading && (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredTracks.length === 0 ? (
            <div className="text-gray-400 text-center py-8">No tracks found</div>
          ) : (
            filteredTracks.map((track) => (
              <div
                key={track.id}
                onClick={() => onTrackSelect?.(track)}
                className="bg-gray-700 rounded p-3 hover:bg-gray-600 cursor-pointer"
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1 min-w-0">
                    <p className="text-white font-medium truncate">
                      {track.title || track.id}
                    </p>
                    <p className="text-sm text-gray-400 truncate">
                      {track.channel || track.artist_name} •{' '}
                      {formatDuration(track.duration_sec)}
                    </p>
                  </div>
                  <div className="ml-4 text-xs text-gray-500">{track.id}</div>
                </div>

                {/* Featured Artists */}
                {track.featured_artists.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {track.featured_artists.map((artist, i) => (
                      <span
                        key={i}
                        className="px-2 py-0.5 bg-purple-600/30 text-purple-300 rounded-full text-xs"
                      >
                        ft. {artist}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {/* Stats */}
      <div className="mt-4 pt-4 border-t border-gray-700 text-sm text-gray-400">
        Showing {filteredTracks.length} of {tracks.length} tracks
        {filter.hasFeatured && ` with featured artists`}
      </div>
    </div>
  );
};

export default TrackListPanel;
