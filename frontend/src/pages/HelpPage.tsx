/**
 * Help page with user documentation for the Live Karaoke feature.
 * Task 8.5: Create user documentation / help page
 */
export default function HelpPage() {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Live Karaoke Help</h1>

      {/* Quick Start */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Quick Start</h2>
        <ol className="list-decimal list-inside space-y-2 text-gray-700">
          <li>Upload a song (MP3, WAV, FLAC, M4A, or OGG)</li>
          <li>Wait for vocal separation to complete</li>
          <li>Select a voice model (or extract from the song)</li>
          <li>Configure your audio devices</li>
          <li>Click "Start Performing" and sing along!</li>
        </ol>
      </section>

      {/* How It Works */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">How It Works</h2>
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-gray-700 mb-4">
            The Live Karaoke system uses advanced AI to transform your voice in real-time:
          </p>
          <ul className="list-disc list-inside space-y-2 text-gray-700">
            <li>
              <strong>Vocal Separation:</strong> AI separates the vocals from the instrumental
              track using MelBandRoFormer technology.
            </li>
            <li>
              <strong>Voice Conversion:</strong> Your voice is converted to match the original
              artist's voice characteristics while preserving your pitch and timing.
            </li>
            <li>
              <strong>Dual Output:</strong> The converted voice is mixed with instrumentals for
              speakers, while the original song plays in headphones so you can follow along.
            </li>
          </ul>
        </div>
        <div className="mt-4 bg-green-50 p-4 rounded-lg">
          <h3 className="font-semibold text-lg mb-2">Browser Sing-Along Recording</h3>
          <p className="text-gray-700 mb-3">
            When you open AutoVoice from another computer on the local network,
            that browser can use its own headphones and microphone to record
            training takes. Use the browser device controls on the Karaoke page,
            not the server audio-device controls, for this workflow.
          </p>
          <ul className="list-disc list-inside text-gray-700 space-y-2">
            <li>Serve AutoVoice over HTTPS on LAN so the browser can access the mic.</li>
            <li>Select the browser&apos;s headset mic and headphones before recording.</li>
            <li>Preview each take before attaching it to the target voice profile.</li>
            <li>If output selection is unsupported, the browser uses its system default output.</li>
          </ul>
        </div>
      </section>

      {/* Uploading Songs */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Uploading Songs</h2>
        <div className="space-y-4">
          <div>
            <h3 className="font-semibold text-lg">Supported Formats</h3>
            <p className="text-gray-700">WAV, MP3, FLAC, M4A, OGG, AAC</p>
          </div>
          <div>
            <h3 className="font-semibold text-lg">Limits</h3>
            <ul className="list-disc list-inside text-gray-700">
              <li>Maximum file size: 100 MB</li>
              <li>Maximum duration: 10 minutes</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-lg">Tips</h3>
            <ul className="list-disc list-inside text-gray-700">
              <li>Higher quality audio files produce better separation results</li>
              <li>Songs with clear vocals separate better than heavily processed tracks</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Voice Models */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Voice Models</h2>
        <div className="space-y-4">
          <div>
            <h3 className="font-semibold text-lg">Pre-trained Models</h3>
            <p className="text-gray-700">
              Select from available pre-trained voice models to convert your voice
              to sound like different artists.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-lg">Extract from Song</h3>
            <p className="text-gray-700">
              After separation, you can extract a voice model from the original
              artist's vocals. This creates a custom model that captures the
              artist's voice characteristics from that specific song.
            </p>
          </div>
        </div>
      </section>

      {/* Audio Configuration */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Audio Configuration</h2>
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="font-semibold text-lg mb-2">Dual Output Setup</h3>
          <p className="text-gray-700 mb-4">
            For the best karaoke experience, use two separate audio outputs:
          </p>
          <ul className="list-disc list-inside text-gray-700 space-y-2">
            <li>
              <strong>Speakers (Audience):</strong> Plays your converted voice
              mixed with the instrumental track
            </li>
            <li>
              <strong>Headphones (Performer):</strong> Plays the original song
              so you can hear the melody and lyrics to follow along
            </li>
          </ul>
        </div>
      </section>

      {/* Microphone Tips */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Microphone Tips</h2>
        <ul className="list-disc list-inside space-y-2 text-gray-700">
          <li>Use a quality USB or XLR microphone for best results</li>
          <li>Position the microphone 6-12 inches from your mouth</li>
          <li>Reduce background noise in your environment</li>
          <li>Use headphones to prevent audio feedback</li>
          <li>Check your input level meter - aim for green/yellow, avoid red</li>
        </ul>
      </section>

      {/* Latency */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Latency & Performance</h2>
        <div className="space-y-4">
          <p className="text-gray-700">
            The system targets less than 50ms latency for natural-feeling performance.
            Actual latency depends on:
          </p>
          <ul className="list-disc list-inside text-gray-700">
            <li>Network connection speed and stability</li>
            <li>Server GPU availability</li>
            <li>Audio buffer size settings</li>
          </ul>
          <p className="text-gray-700">
            The latency indicator in the performance view shows real-time processing delay.
          </p>
        </div>
      </section>

      {/* Troubleshooting */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Troubleshooting</h2>
        <div className="space-y-4">
          <div className="border-l-4 border-yellow-400 pl-4">
            <h3 className="font-semibold">No audio input detected</h3>
            <p className="text-gray-700">
              Check that your browser has microphone permission and the correct
              input device is selected.
            </p>
          </div>
          <div className="border-l-4 border-yellow-400 pl-4">
            <h3 className="font-semibold">LAN browser cannot access the mic</h3>
            <p className="text-gray-700">
              Browser microphone and output-device APIs require HTTPS for LAN
              clients. Use localhost for same-machine testing or configure HTTPS
              before recording from another computer.
            </p>
          </div>
          <div className="border-l-4 border-yellow-400 pl-4">
            <h3 className="font-semibold">High latency or dropouts</h3>
            <p className="text-gray-700">
              Try using a wired network connection instead of WiFi. Close other
              browser tabs and applications.
            </p>
          </div>
          <div className="border-l-4 border-yellow-400 pl-4">
            <h3 className="font-semibold">Separation takes too long</h3>
            <p className="text-gray-700">
              Longer songs take more time to separate. A 3-minute song typically
              takes 15-30 seconds to process.
            </p>
          </div>
          <div className="border-l-4 border-yellow-400 pl-4">
            <h3 className="font-semibold">Voice conversion sounds unnatural</h3>
            <p className="text-gray-700">
              Try extracting a voice model from the specific song you're singing.
              Sing in a similar pitch range to the original artist for best results.
            </p>
          </div>
        </div>
      </section>

      {/* API Documentation Link */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">For Developers</h2>
        <p className="text-gray-700">
          The Karaoke API is available at <code className="bg-gray-100 px-2 py-1 rounded">/api/v1/karaoke/</code>.
          See the API health status at{' '}
          <a
            href="/api/v1/karaoke/health"
            className="text-blue-600 hover:underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            /api/v1/karaoke/health
          </a>
          .
        </p>
      </section>
    </div>
  );
}
