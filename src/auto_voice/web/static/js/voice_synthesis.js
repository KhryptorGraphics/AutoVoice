// Voice Synthesis JavaScript - Audio processing and control

let audioContext = null;
let mediaRecorder = null;
let audioChunks = [];
let recordingTimer = null;
let recordingStartTime = null;
let currentAudioBlob = null;
let isProcessing = false;

// Initialize voice synthesis page
document.addEventListener('DOMContentLoaded', function() {
    initializeAudioContext();
    setupInputControls();
    setupProcessingOptions();
    setupExportButtons();

    // Register WebSocket handlers
    if (window.wsManager) {
        window.wsManager.on('audio_progress', handleProcessingProgress);
        window.wsManager.on('processing_complete', handleProcessingComplete);
    }
});

// Initialize Web Audio API context
function initializeAudioContext() {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    } catch (error) {
        console.error('Web Audio API not supported:', error);
        window.AutoVoiceUtils.showNotification('Your browser does not support audio recording', 'error');
    }
}

// Setup input type controls
function setupInputControls() {
    const inputRadios = document.querySelectorAll('input[name="input-type"]');

    inputRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            // Hide all control sections
            document.getElementById('recording-controls').style.display = 'none';
            document.getElementById('upload-controls').style.display = 'none';
            document.getElementById('text-controls').style.display = 'none';

            // Show selected section
            switch (this.value) {
                case 'record':
                    document.getElementById('recording-controls').style.display = 'block';
                    setupRecordingControls();
                    break;
                case 'upload':
                    document.getElementById('upload-controls').style.display = 'block';
                    setupUploadControls();
                    break;
                case 'text':
                    document.getElementById('text-controls').style.display = 'block';
                    setupTextControls();
                    break;
            }
        });
    });

    // Trigger default selection
    document.getElementById('record-audio').dispatchEvent(new Event('change'));
}

// Setup recording controls
function setupRecordingControls() {
    const recordBtn = document.getElementById('record-btn');

    recordBtn.addEventListener('click', async function() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            await startRecording();
        }
    });
}

// Start audio recording
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = function(event) {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = function() {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            currentAudioBlob = audioBlob;
            enableProcessButton();

            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();

        // Update UI
        const recordBtn = document.getElementById('record-btn');
        recordBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
        recordBtn.classList.add('recording');

        // Start timer
        startRecordingTimer();

        // Start visualizer
        if (window.startAudioVisualizer) {
            window.startAudioVisualizer(stream);
        }

        window.AutoVoiceUtils.showNotification('Recording started', 'success');
    } catch (error) {
        console.error('Failed to start recording:', error);
        window.AutoVoiceUtils.showNotification('Failed to access microphone', 'error');
    }
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();

        // Update UI
        const recordBtn = document.getElementById('record-btn');
        recordBtn.innerHTML = '<i class="fas fa-circle"></i> Start Recording';
        recordBtn.classList.remove('recording');

        // Stop timer
        stopRecordingTimer();

        // Stop visualizer
        if (window.stopAudioVisualizer) {
            window.stopAudioVisualizer();
        }

        window.AutoVoiceUtils.showNotification('Recording stopped', 'success');
    }
}

// Recording timer functions
function startRecordingTimer() {
    recordingStartTime = Date.now();
    document.getElementById('recording-timer').style.display = 'flex';

    recordingTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        document.getElementById('timer-display').textContent =
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }, 100);
}

function stopRecordingTimer() {
    if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
    }
    document.getElementById('recording-timer').style.display = 'none';
}

// Setup file upload controls
function setupUploadControls() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('audio-file-input');
    const removeBtn = document.getElementById('remove-file');

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    removeBtn.addEventListener('click', () => {
        currentAudioBlob = null;
        fileInput.value = '';
        document.getElementById('file-info').style.display = 'none';
        document.getElementById('upload-area').style.display = 'block';
        disableProcessButton();
    });
}

// Handle file upload
function handleFileUpload(file) {
    if (!file.type.startsWith('audio/')) {
        window.AutoVoiceUtils.showNotification('Please upload an audio file', 'error');
        return;
    }

    currentAudioBlob = file;

    // Update UI
    document.getElementById('upload-area').style.display = 'none';
    document.getElementById('file-info').style.display = 'flex';
    document.getElementById('file-name').textContent = file.name;

    enableProcessButton();
    window.AutoVoiceUtils.showNotification(`File uploaded: ${file.name}`, 'success');
}

// Setup text-to-speech controls
function setupTextControls() {
    const textInput = document.getElementById('text-to-synthesize');
    const rateSlider = document.getElementById('speaking-rate');
    const rateValue = document.getElementById('rate-value');

    textInput.addEventListener('input', () => {
        if (textInput.value.trim().length > 0) {
            enableProcessButton();
        } else {
            disableProcessButton();
        }
    });

    rateSlider.addEventListener('input', () => {
        rateValue.textContent = `${rateSlider.value}x`;
    });
}

// Setup processing options
function setupProcessingOptions() {
    // VAD threshold slider
    const vadSlider = document.getElementById('vad-threshold');
    const vadValue = document.getElementById('vad-value');

    vadSlider.addEventListener('input', () => {
        vadValue.textContent = vadSlider.value;
    });

    // Process button
    const processBtn = document.getElementById('process-btn');
    processBtn.addEventListener('click', processAudio);
}

// Enable/disable process button
function enableProcessButton() {
    document.getElementById('process-btn').disabled = false;
}

function disableProcessButton() {
    document.getElementById('process-btn').disabled = true;
}

// Process audio
async function processAudio() {
    if (isProcessing) return;

    isProcessing = true;
    document.getElementById('process-btn').disabled = true;
    document.getElementById('processing-status').style.display = 'flex';

    const inputType = document.querySelector('input[name="input-type"]:checked').value;
    const formData = new FormData();

    // Add input based on type
    if (inputType === 'record' || inputType === 'upload') {
        if (!currentAudioBlob) {
            window.AutoVoiceUtils.showNotification('No audio to process', 'error');
            resetProcessingUI();
            return;
        }
        formData.append('audio', currentAudioBlob);
    } else if (inputType === 'text') {
        const text = document.getElementById('text-to-synthesize').value;
        if (!text.trim()) {
            window.AutoVoiceUtils.showNotification('Please enter text to synthesize', 'error');
            resetProcessingUI();
            return;
        }
        formData.append('text', text);
        formData.append('voice_model', document.getElementById('voice-model').value);
        formData.append('speaking_rate', document.getElementById('speaking-rate').value);
    }

    // Add processing options
    const options = {
        enable_pitch: document.getElementById('enable-pitch').checked,
        pitch_min: document.getElementById('pitch-min').value,
        pitch_max: document.getElementById('pitch-max').value,
        enable_vad: document.getElementById('enable-vad').checked,
        vad_threshold: document.getElementById('vad-threshold').value,
        enable_denoise: document.getElementById('enable-denoise').checked,
        denoise_strength: document.getElementById('denoise-strength').value,
        enable_enhance: document.getElementById('enable-enhance').checked,
        enhance_model: document.getElementById('enhance-model').value,
    };

    formData.append('options', JSON.stringify(options));

    try {
        const response = await fetch(`${window.location.origin}/api/v1/audio/process`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Processing failed: ${response.statusText}`);
        }

        const result = await response.json();
        handleProcessingComplete(result);
    } catch (error) {
        console.error('Processing error:', error);
        window.AutoVoiceUtils.showNotification('Processing failed: ' + error.message, 'error');
        resetProcessingUI();
    }
}

// Handle processing progress updates
function handleProcessingProgress(progress) {
    const statusText = document.getElementById('status-text');
    if (statusText) {
        statusText.textContent = `${progress.stage} - ${progress.percent}%`;
    }
}

// Handle processing completion
function handleProcessingComplete(result) {
    isProcessing = false;
    resetProcessingUI();

    // Show results section
    document.getElementById('results-section').style.display = 'block';

    // Load processed audio
    if (result.audio_url) {
        const audioPlayer = document.getElementById('result-audio');
        audioPlayer.src = result.audio_url;
    }

    // Update metrics
    updateMetrics(result.metrics);

    // Draw visualizations
    if (result.spectrogram) {
        drawSpectrogram(result.spectrogram);
    }

    window.AutoVoiceUtils.showNotification('Processing complete!', 'success');

    // Scroll to results
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

// Reset processing UI
function resetProcessingUI() {
    isProcessing = false;
    document.getElementById('process-btn').disabled = false;
    document.getElementById('processing-status').style.display = 'none';
}

// Update metrics display
function updateMetrics(metrics) {
    if (metrics.avg_pitch) {
        document.getElementById('avg-pitch').textContent = `${metrics.avg_pitch.toFixed(1)} Hz`;
    }
    if (metrics.speech_rate) {
        document.getElementById('speech-rate').textContent = `${metrics.speech_rate} words/min`;
    }
    if (metrics.snr) {
        document.getElementById('snr').textContent = `${metrics.snr.toFixed(1)} dB`;
    }
    if (metrics.processing_time) {
        document.getElementById('proc-time').textContent = `${metrics.processing_time.toFixed(0)} ms`;
    }
    if (metrics.gpu_memory_used) {
        document.getElementById('gpu-mem-used').textContent = `${metrics.gpu_memory_used} MB`;
    }
    if (metrics.quality_score) {
        document.getElementById('quality-score').textContent = `${metrics.quality_score}/100`;
    }
}

// Draw spectrogram
function drawSpectrogram(spectrogramData) {
    const canvas = document.getElementById('spectrogram-canvas');
    const ctx = canvas.getContext('2d');

    // Implementation would depend on spectrogram data format
    // This is a placeholder for actual spectrogram rendering
    ctx.fillStyle = '#667eea';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Setup visualization controls
document.addEventListener('DOMContentLoaded', function() {
    const vizButtons = document.querySelectorAll('.viz-btn');

    vizButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            vizButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            const vizType = this.dataset.viz;
            // Switch visualization based on type
            // Implementation would load different visualizations
        });
    });
});

// Setup export buttons
function setupExportButtons() {
    const exportButtons = document.querySelectorAll('.export-btn');

    exportButtons.forEach(btn => {
        btn.addEventListener('click', async function() {
            const format = this.dataset.format;
            await exportAudio(format);
        });
    });

    // Download button
    const downloadBtn = document.getElementById('download-audio');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            const audioPlayer = document.getElementById('result-audio');
            if (audioPlayer.src) {
                const a = document.createElement('a');
                a.href = audioPlayer.src;
                a.download = 'processed_audio.wav';
                a.click();
            }
        });
    }

    // Compare button
    const compareBtn = document.getElementById('compare-audio');
    if (compareBtn) {
        compareBtn.addEventListener('click', () => {
            // Implementation for A/B comparison
            window.AutoVoiceUtils.showNotification('Comparison feature coming soon', 'info');
        });
    }
}

// Export audio in different formats
async function exportAudio(format) {
    try {
        const response = await window.AutoVoiceUtils.apiRequest(`/v1/audio/export?format=${format}`, {
            method: 'POST',
            body: JSON.stringify({ audio_id: 'current' })
        });

        if (response.url) {
            const a = document.createElement('a');
            a.href = response.url;
            a.download = `audio_export.${format}`;
            a.click();

            window.AutoVoiceUtils.showNotification(`Exported as ${format.toUpperCase()}`, 'success');
        }
    } catch (error) {
        console.error('Export failed:', error);
        window.AutoVoiceUtils.showNotification('Export failed', 'error');
    }
}

// Add custom styles
const style = document.createElement('style');
style.textContent = `
    .recording {
        background: #f56565 !important;
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }

    .recording-timer {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1rem;
        font-size: 1.2rem;
    }

    .recording-dot {
        width: 12px;
        height: 12px;
        background: #f56565;
        border-radius: 50%;
        animation: pulse 1s infinite;
    }

    .dragover {
        border-color: var(--primary-color) !important;
        background: rgba(102, 126, 234, 0.1) !important;
    }

    .file-info {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background: var(--darker-bg);
        border-radius: 8px;
    }

    .viz-btn {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s;
    }

    .viz-btn:hover {
        background: var(--darker-bg);
    }

    .viz-btn.active {
        background: var(--gradient);
        color: white;
        border-color: transparent;
    }

    .export-btn {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        transition: all 0.3s;
    }

    .export-btn:hover {
        background: var(--darker-bg);
        transform: translateY(-2px);
    }
`;
document.head.appendChild(style);