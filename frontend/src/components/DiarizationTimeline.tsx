import React, { useRef, useState, useEffect } from 'react';

interface DiarizationSegment {
  start: number;
  end: number;
  speaker_id: string;
  confidence: number;
  duration: number;
}

interface DiarizationTimelineProps {
  segments: DiarizationSegment[];
  audioDuration: number;
  audioUrl?: string;
  selectedSegmentIndex?: number;
  onSegmentClick?: (index: number, segment: DiarizationSegment) => void;
  onSegmentHover?: (index: number | null, segment: DiarizationSegment | null) => void;
}

const SPEAKER_COLORS = [
  'bg-blue-500',
  'bg-green-500',
  'bg-purple-500',
  'bg-orange-500',
  'bg-pink-500',
  'bg-cyan-500',
  'bg-yellow-500',
  'bg-red-500',
];

const SPEAKER_BORDER_COLORS = [
  'border-blue-600',
  'border-green-600',
  'border-purple-600',
  'border-orange-600',
  'border-pink-600',
  'border-cyan-600',
  'border-yellow-600',
  'border-red-600',
];

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function DiarizationTimeline({
  segments,
  audioDuration,
  audioUrl,
  selectedSegmentIndex,
  onSegmentClick,
  onSegmentHover,
}: DiarizationTimelineProps) {
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hoveredSegment, setHoveredSegment] = useState<number | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  // Get unique speaker IDs and assign colors
  const speakerIds = [...new Set(segments.map(s => s.speaker_id))];
  const speakerColorMap = new Map(
    speakerIds.map((id, index) => [id, index % SPEAKER_COLORS.length])
  );

  // Update current time during playback
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
    };
  }, []);

  const handleSegmentClick = (index: number, segment: DiarizationSegment) => {
    // Seek audio to segment start
    if (audioRef.current) {
      audioRef.current.currentTime = segment.start;
      audioRef.current.play();
    }
    onSegmentClick?.(index, segment);
  };

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!timelineRef.current || !audioRef.current) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * audioDuration;
    audioRef.current.currentTime = newTime;
  };

  const handleSegmentMouseEnter = (index: number, segment: DiarizationSegment) => {
    setHoveredSegment(index);
    onSegmentHover?.(index, segment);
  };

  const handleSegmentMouseLeave = () => {
    setHoveredSegment(null);
    onSegmentHover?.(null, null);
  };

  const togglePlayPause = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
  };

  // Calculate playhead position
  const playheadPosition = (currentTime / audioDuration) * 100;

  return (
    <div className="bg-zinc-800 rounded-lg p-4">
      {/* Audio element (hidden) */}
      {audioUrl && <audio ref={audioRef} src={audioUrl} preload="metadata" />}

      {/* Header with controls */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Speaker Timeline</h3>
        <div className="flex items-center gap-4">
          {audioUrl && (
            <button
              onClick={togglePlayPause}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm flex items-center gap-2"
            >
              {isPlaying ? (
                <>
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  Pause
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                  </svg>
                  Play
                </>
              )}
            </button>
          )}
          <span className="text-zinc-400 text-sm">
            {formatTime(currentTime)} / {formatTime(audioDuration)}
          </span>
        </div>
      </div>

      {/* Timeline */}
      <div
        ref={timelineRef}
        className="relative h-16 bg-zinc-900 rounded cursor-pointer mb-4"
        onClick={handleTimelineClick}
      >
        {/* Segments */}
        {segments.map((segment, index) => {
          const left = (segment.start / audioDuration) * 100;
          const width = (segment.duration / audioDuration) * 100;
          const colorIndex = speakerColorMap.get(segment.speaker_id) || 0;
          const isSelected = selectedSegmentIndex === index;
          const isHovered = hoveredSegment === index;

          return (
            <div
              key={`${segment.speaker_id}-${segment.start}`}
              className={`absolute top-2 bottom-2 rounded transition-all cursor-pointer
                ${SPEAKER_COLORS[colorIndex]}
                ${isSelected ? `ring-2 ring-white ${SPEAKER_BORDER_COLORS[colorIndex]}` : ''}
                ${isHovered ? 'brightness-125 scale-y-110' : 'hover:brightness-110'}
              `}
              style={{
                left: `${left}%`,
                width: `${Math.max(width, 0.5)}%`,
                opacity: 0.6 + segment.confidence * 0.4,
              }}
              onClick={(e) => {
                e.stopPropagation();
                handleSegmentClick(index, segment);
              }}
              onMouseEnter={() => handleSegmentMouseEnter(index, segment)}
              onMouseLeave={handleSegmentMouseLeave}
              title={`${segment.speaker_id}: ${formatTime(segment.start)} - ${formatTime(segment.end)}`}
            />
          );
        })}

        {/* Playhead */}
        {audioUrl && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-10 pointer-events-none"
            style={{ left: `${playheadPosition}%` }}
          >
            <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-red-500 rounded-full" />
          </div>
        )}

        {/* Time markers */}
        <div className="absolute bottom-0 left-0 right-0 h-4 flex justify-between px-1 text-xs text-zinc-500">
          <span>0:00</span>
          <span>{formatTime(audioDuration / 4)}</span>
          <span>{formatTime(audioDuration / 2)}</span>
          <span>{formatTime((audioDuration * 3) / 4)}</span>
          <span>{formatTime(audioDuration)}</span>
        </div>
      </div>

      {/* Speaker legend */}
      <div className="flex flex-wrap gap-4">
        {speakerIds.map((speakerId) => {
          const colorIndex = speakerColorMap.get(speakerId) || 0;
          const speakerSegments = segments.filter(s => s.speaker_id === speakerId);
          const totalDuration = speakerSegments.reduce((sum, s) => sum + s.duration, 0);

          return (
            <div key={speakerId} className="flex items-center gap-2">
              <div className={`w-4 h-4 rounded ${SPEAKER_COLORS[colorIndex]}`} />
              <span className="text-sm text-zinc-300">
                {speakerId} ({formatTime(totalDuration)})
              </span>
            </div>
          );
        })}
      </div>

      {/* Hovered segment info */}
      {hoveredSegment !== null && segments[hoveredSegment] && (
        <div className="mt-4 p-3 bg-zinc-700 rounded text-sm">
          <div className="flex justify-between text-zinc-300">
            <span>Speaker: <span className="text-white font-medium">{segments[hoveredSegment].speaker_id}</span></span>
            <span>Confidence: <span className="text-white font-medium">{(segments[hoveredSegment].confidence * 100).toFixed(0)}%</span></span>
          </div>
          <div className="flex justify-between text-zinc-400 mt-1">
            <span>{formatTime(segments[hoveredSegment].start)} - {formatTime(segments[hoveredSegment].end)}</span>
            <span>Duration: {segments[hoveredSegment].duration.toFixed(1)}s</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default DiarizationTimeline;
