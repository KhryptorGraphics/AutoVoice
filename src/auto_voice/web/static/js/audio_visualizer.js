// Audio Visualizer - Real-time waveform and spectrum visualization

let analyser = null;
let dataArray = null;
let animationId = null;
let visualizerCanvas = null;
let visualizerCtx = null;

// Start audio visualizer
window.startAudioVisualizer = function(stream) {
    if (!audioContext) {
        console.error('Audio context not initialized');
        return;
    }

    visualizerCanvas = document.getElementById('waveform-canvas');
    if (!visualizerCanvas) return;

    visualizerCtx = visualizerCanvas.getContext('2d');

    // Set canvas size
    visualizerCanvas.width = visualizerCanvas.offsetWidth;
    visualizerCanvas.height = 200;

    // Create analyser node
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    const bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);

    // Connect audio source to analyser
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    // Start visualization
    drawWaveform();
};

// Stop audio visualizer
window.stopAudioVisualizer = function() {
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }

    if (visualizerCtx) {
        visualizerCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
    }
};

// Draw waveform visualization
function drawWaveform() {
    animationId = requestAnimationFrame(drawWaveform);

    if (!analyser || !dataArray || !visualizerCtx) return;

    analyser.getByteTimeDomainData(dataArray);

    const width = visualizerCanvas.width;
    const height = visualizerCanvas.height;

    // Clear canvas with gradient background
    const gradient = visualizerCtx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, 'rgba(102, 126, 234, 0.05)');
    gradient.addColorStop(1, 'rgba(118, 75, 162, 0.05)');
    visualizerCtx.fillStyle = gradient;
    visualizerCtx.fillRect(0, 0, width, height);

    // Draw waveform
    visualizerCtx.lineWidth = 2;
    visualizerCtx.strokeStyle = '#667eea';
    visualizerCtx.beginPath();

    const sliceWidth = width / dataArray.length;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * height / 2;

        if (i === 0) {
            visualizerCtx.moveTo(x, y);
        } else {
            visualizerCtx.lineTo(x, y);
        }

        x += sliceWidth;
    }

    visualizerCtx.lineTo(width, height / 2);
    visualizerCtx.stroke();

    // Draw center line
    visualizerCtx.strokeStyle = 'rgba(160, 174, 192, 0.3)';
    visualizerCtx.lineWidth = 1;
    visualizerCtx.beginPath();
    visualizerCtx.moveTo(0, height / 2);
    visualizerCtx.lineTo(width, height / 2);
    visualizerCtx.stroke();
}

// Draw frequency spectrum
function drawSpectrum() {
    animationId = requestAnimationFrame(drawSpectrum);

    if (!analyser || !dataArray || !visualizerCtx) return;

    analyser.getByteFrequencyData(dataArray);

    const width = visualizerCanvas.width;
    const height = visualizerCanvas.height;

    // Clear canvas
    visualizerCtx.fillStyle = 'rgba(15, 20, 25, 0.95)';
    visualizerCtx.fillRect(0, 0, width, height);

    // Draw frequency bars
    const barWidth = (width / dataArray.length) * 2.5;
    let barHeight;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
        barHeight = (dataArray[i] / 255) * height;

        // Create gradient for bars
        const gradient = visualizerCtx.createLinearGradient(0, height - barHeight, 0, height);
        gradient.addColorStop(0, '#667eea');
        gradient.addColorStop(0.5, '#764ba2');
        gradient.addColorStop(1, '#667eea');

        visualizerCtx.fillStyle = gradient;
        visualizerCtx.fillRect(x, height - barHeight, barWidth, barHeight);

        x += barWidth + 1;
    }
}

// Draw spectrogram (for results display)
window.drawAdvancedSpectrogram = function(canvas, spectrogramData) {
    if (!canvas || !spectrogramData) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#0f1419';
    ctx.fillRect(0, 0, width, height);

    const timeSteps = spectrogramData.length;
    const freqBins = spectrogramData[0].length;
    const pixelWidth = width / timeSteps;
    const pixelHeight = height / freqBins;

    // Draw spectrogram
    for (let t = 0; t < timeSteps; t++) {
        for (let f = 0; f < freqBins; f++) {
            const value = spectrogramData[t][f];
            const intensity = Math.min(255, value * 255);

            // Color mapping (purple to blue gradient)
            const r = intensity * 0.4;
            const g = intensity * 0.3;
            const b = intensity * 0.9;

            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(
                t * pixelWidth,
                height - (f + 1) * pixelHeight,
                pixelWidth,
                pixelHeight
            );
        }
    }

    // Add grid
    ctx.strokeStyle = 'rgba(160, 174, 192, 0.1)';
    ctx.lineWidth = 0.5;

    // Vertical lines (time)
    for (let i = 0; i <= 10; i++) {
        const x = (width / 10) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }

    // Horizontal lines (frequency)
    for (let i = 0; i <= 5; i++) {
        const y = (height / 5) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }

    // Add labels
    ctx.fillStyle = '#a0aec0';
    ctx.font = '10px monospace';
    ctx.fillText('0 Hz', 5, height - 5);
    ctx.fillText('8000 Hz', 5, 15);
    ctx.fillText('Time →', width - 40, height - 5);
};

// Draw pitch contour
window.drawPitchContour = function(canvas, pitchData) {
    if (!canvas || !pitchData) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#0f1419';
    ctx.fillRect(0, 0, width, height);

    // Find min and max pitch for scaling
    const validPitches = pitchData.filter(p => p > 0);
    const minPitch = Math.min(...validPitches);
    const maxPitch = Math.max(...validPitches);
    const pitchRange = maxPitch - minPitch;

    // Draw pitch curve
    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 2;
    ctx.beginPath();

    let firstPoint = true;
    for (let i = 0; i < pitchData.length; i++) {
        const x = (i / pitchData.length) * width;

        if (pitchData[i] > 0) {
            const y = height - ((pitchData[i] - minPitch) / pitchRange) * height;

            if (firstPoint) {
                ctx.moveTo(x, y);
                firstPoint = false;
            } else {
                ctx.lineTo(x, y);
            }
        }
    }
    ctx.stroke();

    // Draw reference lines
    ctx.strokeStyle = 'rgba(160, 174, 192, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    // Draw average pitch line
    const avgPitch = validPitches.reduce((a, b) => a + b, 0) / validPitches.length;
    const avgY = height - ((avgPitch - minPitch) / pitchRange) * height;

    ctx.beginPath();
    ctx.moveTo(0, avgY);
    ctx.lineTo(width, avgY);
    ctx.stroke();

    // Add label
    ctx.setLineDash([]);
    ctx.fillStyle = '#a0aec0';
    ctx.font = '10px monospace';
    ctx.fillText(`Avg: ${avgPitch.toFixed(1)} Hz`, 5, avgY - 5);
};

// Draw formants
window.drawFormants = function(canvas, formantData) {
    if (!canvas || !formantData) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#0f1419';
    ctx.fillRect(0, 0, width, height);

    const colors = ['#667eea', '#764ba2', '#f6ad55', '#48bb78'];
    const formantCount = formantData[0].length;

    // Draw each formant track
    for (let f = 0; f < formantCount; f++) {
        ctx.strokeStyle = colors[f % colors.length];
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let t = 0; t < formantData.length; t++) {
            const x = (t / formantData.length) * width;
            const y = height - (formantData[t][f] / 5000) * height; // Assuming max 5000 Hz

            if (t === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }

    // Add legend
    ctx.fillStyle = '#a0aec0';
    ctx.font = '10px monospace';
    for (let f = 0; f < formantCount; f++) {
        ctx.fillStyle = colors[f % colors.length];
        ctx.fillText(`F${f + 1}`, 10 + f * 30, 20);
    }
};

// Initialize visualizer controls if on voice synthesis page
document.addEventListener('DOMContentLoaded', function() {
    const vizCanvas = document.getElementById('waveform-canvas');
    if (vizCanvas) {
        // Set initial canvas size
        vizCanvas.width = vizCanvas.offsetWidth;
        vizCanvas.height = 200;

        // Draw placeholder
        const ctx = vizCanvas.getContext('2d');
        ctx.fillStyle = 'rgba(102, 126, 234, 0.05)';
        ctx.fillRect(0, 0, vizCanvas.width, vizCanvas.height);

        ctx.strokeStyle = 'rgba(160, 174, 192, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, vizCanvas.height / 2);
        ctx.lineTo(vizCanvas.width, vizCanvas.height / 2);
        ctx.stroke();

        ctx.fillStyle = '#a0aec0';
        ctx.font = '12px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Ready to record', vizCanvas.width / 2, vizCanvas.height / 2 - 10);
    }
});