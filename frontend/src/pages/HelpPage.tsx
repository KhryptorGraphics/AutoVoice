/**
 * Help page with user documentation.
 * Part 1: end-to-end guide for making a converted song.
 * Part 2: Live Karaoke feature (Task 8.5).
 */
export default function HelpPage() {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">AutoVoice Help</h1>

      {/* Anchor navigation */}
      <nav className="mb-8 flex flex-wrap items-center gap-3 text-sm">
        <span className="text-gray-500">Jump to:</span>
        <a href="#making-a-converted-song" className="text-blue-600 hover:underline">Making a Converted Song</a>
        <span className="text-gray-400">·</span>
        <a href="#live-karaoke" className="text-blue-600 hover:underline">Live Karaoke</a>
      </nav>

      {/* ===================== Part 1: Making a Converted Song ===================== */}
      <section id="making-a-converted-song" className="mb-12 scroll-mt-6">
        <h2 className="text-2xl font-bold mb-2 pb-2 border-b border-gray-300">Making a Converted Song</h2>
        <p className="text-gray-700 mb-6">
          Convert any song into a target voice in five steps — from creating a profile to
          downloading the finished mix.
        </p>

        <div className="space-y-6">
          {/* Step 1 */}
          <div>
            <h3 className="font-semibold text-lg mb-1">Step 1 · Create a target profile</h3>
            <p className="text-gray-700">
              Go to <a href="/profiles" className="text-blue-600 hover:underline">Profiles</a> and
              click <strong>Create</strong>. Choose the role that fits: a <strong>target user</strong> profile
              is the voice you want to convert <em>into</em>, while a <strong>source artist</strong> profile is
              a reference for the original singer.
            </p>
          </div>

          {/* Step 2 */}
          <div>
            <h3 className="font-semibold text-lg mb-1">Step 2 · Add training samples</h3>
            <p className="text-gray-700">
              Upload clean vocals to the target profile — each sample must be at least
              <strong> 3 seconds</strong> (shorter clips are rejected). Use <strong>Add Song</strong> to drop
              in a full song and have it auto-split into samples. Once a profile accumulates
              <strong> 30 minutes</strong> of clean vocals, full-model training unlocks (LoRA training is
              available immediately).
            </p>
          </div>

          {/* Step 3 */}
          <div>
            <h3 className="font-semibold text-lg mb-1">Step 3 · Train the model</h3>
            <p className="text-gray-700 mb-2">
              Open the profile's <strong>Train</strong> tab and pick an architecture:
            </p>
            <ul className="list-disc list-inside text-gray-700 space-y-1 mb-2">
              <li><strong>Diffusion</strong> — the default, balanced quality.</li>
              <li><strong>so-vits-svc-fork</strong> — best quality, recommended.</li>
            </ul>
            <p className="text-gray-700">
              Choose <strong>LoRA</strong> (fast fine-tune) or <strong>Full</strong> (higher quality, unlocks
              at 30 minutes), then start training and watch progress under <strong>Training Jobs</strong>. When
              a fork-backed model is ready, the profile shows a{' '}
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-600 text-white align-middle">Fork HQ</span>{' '}
              badge.
            </p>
          </div>

          {/* Step 4 */}
          <div>
            <h3 className="font-semibold text-lg mb-1">Step 4 · Convert a song</h3>
            <p className="text-gray-700 mb-3">
              On the <a href="/" className="text-blue-600 hover:underline">Convert</a> page, upload the
              artist's song (and optionally your own vocals). If multiple singers are detected, review and
              confirm them when prompted. Then dial in <strong>Conversion Settings</strong>:
            </p>
            <div className="overflow-x-auto mb-3">
              <table className="text-sm text-gray-700 border border-gray-200 rounded-lg">
                <thead>
                  <tr className="bg-gray-50 text-left">
                    <th className="px-3 py-2 font-semibold">Quality preset</th>
                    <th className="px-3 py-2 font-semibold">Steps</th>
                    <th className="px-3 py-2 font-semibold">Denoise</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t border-gray-200"><td className="px-3 py-2">Draft (Fast)</td><td className="px-3 py-2">10</td><td className="px-3 py-2">0.3</td></tr>
                  <tr className="border-t border-gray-200"><td className="px-3 py-2">Fast</td><td className="px-3 py-2">20</td><td className="px-3 py-2">0.5</td></tr>
                  <tr className="border-t border-gray-200"><td className="px-3 py-2">Balanced</td><td className="px-3 py-2">50</td><td className="px-3 py-2">0.7</td></tr>
                  <tr className="border-t border-gray-200"><td className="px-3 py-2">High Quality</td><td className="px-3 py-2">100</td><td className="px-3 py-2">0.8</td></tr>
                  <tr className="border-t border-gray-200"><td className="px-3 py-2">Studio</td><td className="px-3 py-2">200</td><td className="px-3 py-2">0.9</td></tr>
                </tbody>
              </table>
            </div>
            <ul className="list-disc list-inside text-gray-700 space-y-1">
              <li><strong>Pitch shift</strong> — up to ±12 semitones.</li>
              <li><strong>Vocal / instrumental volume</strong> — balance the converted voice against the backing track.</li>
              <li><strong>Keep stems</strong> — also save the separated vocal and instrumental tracks.</li>
              <li>
                <strong>Backing vocals</strong> — <em>Preserve backing vocals</em> converts only the lead and
                keeps harmonies original (recommended for tracks with backing vocals; fork-backed profiles).
                <em>Convert backing too (experimental)</em> additionally re-sings each harmony line in the
                target voice: harmonies are split into individual lines and converted one by one, falling
                back to preserving any line that cannot be cleanly isolated.
              </li>
              <li>
                <strong>Keep original singers</strong> — for duets and features where the target artist
                already sings on the track. Enter a time range where they sing (e.g. <code>0:00-0:12</code>)
                and that singer&apos;s parts are kept verbatim instead of being re-converted — the range is
                matched to whichever detected singer owns it, so all of their sections across the song are
                preserved. Comma-separate multiple ranges; cluster ids from a result&apos;s speaker badge
                (e.g. <code>SPEAKER_02</code>) also work.
              </li>
            </ul>
          </div>

          {/* Step 5 */}
          <div>
            <h3 className="font-semibold text-lg mb-1">Step 5 · Review results</h3>
            <p className="text-gray-700">
              Play the result inline and download the <strong>Mix</strong>, <strong>Vocals</strong>,
              <strong> Instrumental</strong>, or a <strong>Reassemble</strong>. Result badges call out
              <strong> Stereo HQ</strong>, the <strong>Fork engine</strong>, and a backing-vocal summary. Past
              runs live under <a href="/history" className="text-blue-600 hover:underline">History</a> (with
              play/compare), and objective scores under{' '}
              <a href="/quality" className="text-blue-600 hover:underline">Quality</a>.
            </p>
          </div>
        </div>

        {/* Troubleshooting subsection */}
        <div className="mt-8">
          <h3 className="text-xl font-semibold mb-3">Troubleshooting conversions</h3>
          <div className="space-y-4">
            <div className="border-l-4 border-yellow-400 pl-4">
              <h4 className="font-semibold">Sample too short (under 3 s)</h4>
              <p className="text-gray-700">
                Samples under 3 seconds are rejected. Trim to a longer clean phrase, or use Add Song to
                auto-split a full track into valid samples.
              </p>
            </div>
            <div className="border-l-4 border-yellow-400 pl-4">
              <h4 className="font-semibold">Profile is not trainable</h4>
              <p className="text-gray-700">
                Only target user profiles train. A source artist profile is a reference only — create a
                target user profile to train a voice.
              </p>
            </div>
            <div className="border-l-4 border-yellow-400 pl-4">
              <h4 className="font-semibold">Conversion sounds muffled</h4>
              <p className="text-gray-700">
                Prefer a fork-backed profile — look for the <strong>Fork HQ</strong> badge. The
                so-vits-svc-fork HQ lane (stereo, native 44.1 kHz) gives the clearest results.
              </p>
            </div>
            <div className="border-l-4 border-yellow-400 pl-4">
              <h4 className="font-semibold">Backing vocals sound original</h4>
              <p className="text-gray-700">
                That is intentional when <em>Preserve backing vocals</em> is on — only the lead is converted.
                Turn it off to convert the entire vocal.
              </p>
            </div>
            <div className="border-l-4 border-yellow-400 pl-4">
              <h4 className="font-semibold">The target artist&apos;s own parts got re-converted</h4>
              <p className="text-gray-700">
                On duets/features the system cannot tell that a singer already <em>is</em> the target voice.
                Use <strong>Keep original singers</strong> with a time range where that artist sings
                (e.g. <code>0:00-0:12</code>) and re-run — their parts are then kept verbatim. The result
                card&apos;s speaker badge shows which detected singers were converted, merged, or preserved.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ===================== Part 2: Live Karaoke ===================== */}
      <section id="live-karaoke" className="scroll-mt-6">
        <h2 className="text-2xl font-bold mb-6 pb-2 border-b border-gray-300">Live Karaoke</h2>

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
      </section>
    </div>
  );
}
