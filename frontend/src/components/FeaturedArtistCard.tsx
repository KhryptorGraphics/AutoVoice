/**
 * FeaturedArtistCard - Display featured artist info with training option
 */
import React from 'react';

interface FeaturedArtist {
  name: string;
  track_count: number;
  track_ids: string;
  total_duration_sec?: number;
}

interface FeaturedArtistCardProps {
  artist: FeaturedArtist;
  onTrainLoRA?: (artistName: string) => void;
  onViewTracks?: (trackIds: string[]) => void;
}

const FeaturedArtistCard: React.FC<FeaturedArtistCardProps> = ({
  artist,
  onTrainLoRA,
  onViewTracks,
}) => {
  const trackIds = artist.track_ids?.split(',') || [];

  const formatDuration = (seconds?: number) => {
    if (!seconds) return 'Unknown';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  return (
    <div className="bg-gray-700 rounded-lg p-4 hover:bg-gray-600 transition-colors">
      <div className="flex justify-between items-start mb-3">
        <div>
          <h3 className="text-lg font-semibold text-white">{artist.name}</h3>
          <p className="text-sm text-gray-400">
            {artist.track_count} track{artist.track_count !== 1 ? 's' : ''}
            {artist.total_duration_sec && (
              <span> • {formatDuration(artist.total_duration_sec)}</span>
            )}
          </p>
        </div>
        <span className="px-2 py-1 bg-purple-600/30 text-purple-300 rounded-full text-xs">
          Featured Artist
        </span>
      </div>

      {/* Track IDs Preview */}
      {trackIds.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-gray-500 mb-1">Appears in:</p>
          <div className="flex flex-wrap gap-1">
            {trackIds.slice(0, 5).map((id) => (
              <span
                key={id}
                className="px-1.5 py-0.5 bg-gray-600 text-gray-300 rounded text-xs"
              >
                {id}
              </span>
            ))}
            {trackIds.length > 5 && (
              <span className="px-1.5 py-0.5 bg-gray-600 text-gray-300 rounded text-xs">
                +{trackIds.length - 5} more
              </span>
            )}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={() => onViewTracks?.(trackIds)}
          className="flex-1 px-3 py-2 bg-gray-600 text-white rounded text-sm hover:bg-gray-500"
        >
          View Tracks
        </button>
        <button
          onClick={() => onTrainLoRA?.(artist.name)}
          className="flex-1 px-3 py-2 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
          title={`Train LoRA adapter for ${artist.name}`}
        >
          Train LoRA
        </button>
      </div>
    </div>
  );
};

export default FeaturedArtistCard;
