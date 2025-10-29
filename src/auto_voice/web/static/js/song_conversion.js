/**
 * Song Conversion Module
 * Handles singing voice conversion with real-time progress tracking
 */

import {
    validateAudioFile,
    encodeAudioToBase64,
    decodeBase64ToAudio,
    createAudioPlayer,
    createProgressBar,
    estimateTimeRemaining,
    formatDuration,
    formatFileSize,
    showToast,
    downloadAudio,
    getAudioMetadata
} from './audio_utils.js';
import socket from './websocket.js';

// Conversion state
let currentConversionId = null;
let conversionStartTime = null;
let progressBar = null;

/**
 * Convert song using HTTP POST (fallback without progress tracking)
 * @param {File} songFile - Song audio file
 * @param {string} profileId - Target voice profile ID
 * @param {Object} options - Conversion options
 * @returns {Promise<Object>} Conversion result
 */
export async function convertSongHTTP(songFile, profileId, options = {}) {
    const {
        vocalVolume = 1.0,
        instrumentalVolume = 0.9,
        returnStems = false
    } = options;

    // Validate song file
    const validation = await validateAudioFile(songFile, {
        maxSize: 100 * 1024 * 1024 // 100MB
    });

    if (!validation.valid) {
        throw new Error(validation.error);
    }

    // Create form data
    const formData = new FormData();
    formData.append('song', songFile);
    // COMMENT 2 FIX: Use 'profile_id' for HTTP POST (WebSocket uses 'target_profile_id')
    formData.append('profile_id', profileId);
    formData.append('vocal_volume', vocalVolume.toString());
    formData.append('instrumental_volume', instrumentalVolume.toString());
    formData.append('return_stems', returnStems.toString());

    // Show indeterminate progress
    showToast('Converting song... This may take 10-60 seconds.', 'info', 2000);

    try {
        const response = await fetch('/api/v1/convert/song', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        const result = await response.json();
        return result;

    } catch (error) {
        throw error;
    }
}

/**
 * Update conversion progress UI
 * @param {Object} progressData - Progress data from WebSocket
 */
export function updateConversionProgress(progressData) {
    const { progress, stage, timestamp } = progressData;

    // Update progress bar
    if (progressBar) {
        progressBar.setProgress(progress);
        progressBar.setStage(stage);
    }

    // Update time estimate
    if (conversionStartTime) {
        const elapsed = (Date.now() - conversionStartTime) / 1000;
        const remaining = estimateTimeRemaining(progress, elapsed);

        const timeDisplay = document.getElementById('time-remaining');
        if (timeDisplay) {
            timeDisplay.textContent = remaining;
        }
    }
}

/**
 * Display conversion results
 * @param {Object} result - Conversion result
 * @param {string} containerId - Container element ID
 */
export function displayConversionResults(result, containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }

    // Decode audio
    const audioBlob = decodeBase64ToAudio(result.audio, 'audio/wav');

    container.innerHTML = `
        <div class="conversion-results">
            <h3>Conversion Complete</h3>

            <div class="audio-player-section">
                <h4>Converted Song</h4>
                <div id="${containerId}-main"></div>
                <button class="btn btn-download" onclick="window.downloadConvertedAudio()">
                    Download WAV
                </button>
            </div>

            <div class="metadata-section">
                <h4>Details</h4>
                <div class="metadata-grid">
                    <div class="metadata-item">
                        <span class="label">Duration:</span>
                        <span class="value">${formatDuration(result.duration)}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="label">Sample Rate:</span>
                        <span class="value">${result.sample_rate} Hz</span>
                    </div>
                    ${result.metadata && result.metadata.processing_time ? `
                    <div class="metadata-item">
                        <span class="label">Processing Time:</span>
                        <span class="value">${result.metadata.processing_time.toFixed(1)}s</span>
                    </div>
                    ` : ''}
                    ${result.metadata && result.metadata.f0_stats ? `
                    <div class="metadata-item">
                        <span class="label">Pitch Range:</span>
                        <span class="value">${Math.round(result.metadata.f0_stats.min_f0)}-${Math.round(result.metadata.f0_stats.max_f0)} Hz</span>
                    </div>
                    ` : ''}
                </div>
            </div>

            ${result.stems ? `
            <div class="stems-section">
                <h4>Separated Stems</h4>
                <div class="stem-players">
                    <div class="stem-player">
                        <h5>Vocals</h5>
                        <div id="${containerId}-vocals"></div>
                    </div>
                    <div class="stem-player">
                        <h5>Instrumental</h5>
                        <div id="${containerId}-instrumental"></div>
                    </div>
                </div>
            </div>
            ` : ''}

            <div class="actions">
                <button class="btn btn-primary" onclick="window.convertAnotherSong()">
                    Convert Another Song
                </button>
            </div>
        </div>
    `;

    // Store for download
    window.currentAudioBlob = audioBlob;

    // COMMENT 6 FIX: Create audio player for main converted song
    createAudioPlayer(audioBlob, `${containerId}-main`);

    // Create stem players if available
    if (result.stems) {
        const vocalsBlob = decodeBase64ToAudio(result.stems.vocals, 'audio/wav');
        const instrumentalBlob = decodeBase64ToAudio(result.stems.instrumental, 'audio/wav');

        createAudioPlayer(vocalsBlob, `${containerId}-vocals`);
        createAudioPlayer(instrumentalBlob, `${containerId}-instrumental`);

        window.currentVocalsBlob = vocalsBlob;
        window.currentInstrumentalBlob = instrumentalBlob;
    }

    showToast('Song conversion complete!', 'success');
}

/**
 * Cancel ongoing conversion
 * @param {Object} socket - Socket.IO instance
 */
export function cancelConversion(socket) {
    if (!currentConversionId) {
        showToast('No conversion in progress', 'warning');
        return;
    }

    if (socket && socket.connected) {
        socket.emit('cancel_conversion', {
            conversion_id: currentConversionId
        });
        showToast('Cancelling conversion...', 'info');
    } else {
        showToast('Cannot cancel: WebSocket not connected', 'error');
    }
}

/**
 * Load voice profiles for selection
 * @param {string} containerId - Container for profile selector
 * @returns {Promise<Array>} Array of profiles
 */
export async function loadVoiceProfiles(containerId) {
    try {
        const response = await fetch('/api/v1/voice/profiles');
        if (!response.ok) {
            throw new Error('Failed to load profiles');
        }

        const profiles = await response.json();

        // Populate selector
        const container = document.getElementById(containerId);
        if (container && container.tagName === 'SELECT') {
            container.innerHTML = '<option value="">Select a voice profile...</option>';

            profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.profile_id;
                option.textContent = `Profile ${profile.profile_id.substring(0, 8)}... (${formatDuration(profile.audio_duration)})`;
                container.appendChild(option);
            });

            // Pre-select if stored in localStorage
            const selectedId = localStorage.getItem('selectedProfileId');
            if (selectedId) {
                container.value = selectedId;
            }
        }

        return profiles;

    } catch (error) {
        console.error('Error loading profiles:', error);
        showToast('Failed to load voice profiles', 'error');
        return [];
    }
}

/**
 * Initialize song conversion form
 * @param {string} formId - Form element ID
 * @param {Object} socket - Socket.IO instance
 */
export function initSongConversionForm(formId, socket) {
    // COMMENT 1 FIX: Idempotence guard to prevent duplicate initialization
    if (window.__songConversionInitialized) {
        console.warn('Song conversion form already initialized, skipping duplicate init');
        return;
    }
    window.__songConversionInitialized = true;

    const form = document.getElementById(formId);
    if (!form) {
        console.error(`Form ${formId} not found`);
        return;
    }

    // Initialize progress bar
    progressBar = createProgressBar('conversion-progress');

    // Load profiles on form init
    loadVoiceProfiles('profile-selector');

    // COMMENT 3 FIX: Wire upload area with click listener
    const uploadArea = document.getElementById('upload-area');
    const songInput = document.getElementById('song-file');
    const fileInfo = document.getElementById('file-info');
    const fileInfoText = document.getElementById('file-info-text');

    if (uploadArea && songInput) {
        // Click to browse
        uploadArea.addEventListener('click', (e) => {
            if (e.target !== uploadArea && !uploadArea.contains(e.target)) return;
            songInput.click();
        });

        // Drag and drop support
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', async (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                // Set file to hidden input
                songInput.files = files;
                await updateFileInfo(files[0]);
            }
        });

        // File input change handler
        songInput.addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                await updateFileInfo(e.target.files[0]);
            }
        });

        // Helper to update file info display
        async function updateFileInfo(file) {
            try {
                const validation = await validateAudioFile(file);
                if (!validation.valid) {
                    showToast(validation.error, 'error');
                    songInput.value = '';
                    return;
                }

                const metadata = await getAudioMetadata(file);
                fileInfoText.innerHTML = `
                    <strong>${metadata.filename}</strong><br>
                    ${formatFileSize(metadata.size)} • ${metadata.duration ? formatDuration(metadata.duration) : 'Duration unknown'}
                `;
                fileInfo.classList.add('visible');
            } catch (error) {
                showToast('Error reading file: ' + error.message, 'error');
                songInput.value = '';
            }
        }
    }

    // COMMENT 3: Import needed functions for file handling
    import('./audio_utils.js').then(module => {
        window.getAudioMetadata = module.getAudioMetadata;
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const songInput = form.querySelector('input[name="song"]');
        const profileSelect = form.querySelector('select[name="profile_id"]');
        const vocalVolumeInput = form.querySelector('input[name="vocal_volume"]');
        const instrumentalVolumeInput = form.querySelector('input[name="instrumental_volume"]');
        const returnStemsInput = form.querySelector('input[name="return_stems"]');

        // Validate inputs
        if (!songInput.files || songInput.files.length === 0) {
            showToast('Please select a song file', 'error');
            return;
        }

        if (!profileSelect.value) {
            showToast('Please select a voice profile', 'error');
            return;
        }

        const songFile = songInput.files[0];
        const profileId = profileSelect.value;
        const options = {
            vocalVolume: parseFloat(vocalVolumeInput?.value || 1.0),
            instrumentalVolume: parseFloat(instrumentalVolumeInput?.value || 0.9),
            returnStems: returnStemsInput?.checked || false
        };

        try {
            // Reset progress
            progressBar.reset();
            document.getElementById('conversion-results').innerHTML = '';

            // COMMENT 2 FIX: Show progress UI before emitting
            const progressSection = document.getElementById('conversion-progress');
            const configSection = document.getElementById('config-section');
            const uploadSection = document.getElementById('upload-section');

            if (progressSection) progressSection.classList.remove('hidden');
            if (configSection) configSection.classList.add('hidden');
            if (uploadSection) uploadSection.classList.add('hidden');

            if (!socket || !socket.connected) {
                showToast('WebSocket not connected. Please refresh the page.', 'error');
                throw new Error('WebSocket not connected');
            }

            currentConversionId = generateUUID();
            conversionStartTime = Date.now();

            const songBase64 = await encodeAudioToBase64(songFile);

            // COMMENT 6 FIX: Include MIME type and filename in payload
            const songMimeType = songFile.type || 'audio/wav';
            const songFilename = songFile.name || 'song.wav';

            socket.emit('convert_song_stream', {
                conversion_id: currentConversionId,
                song_data: songBase64,
                song_mime: songMimeType,
                song_filename: songFilename,
                target_profile_id: profileId,
                vocal_volume: options.vocalVolume,
                instrumental_volume: options.instrumentalVolume,
                return_stems: options.returnStems
            });

        } catch (error) {
            console.error('Conversion error:', error);
            showToast('Conversion failed: ' + error.message, 'error', 5000);

            if (progressBar) {
                progressBar.setError('Conversion failed: ' + error.message);
            }
        }
    });

    // Cancel button
    const cancelBtn = document.getElementById('cancel-conversion-btn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', () => {
            cancelConversion(socket);
        });
    }

    // COMMENT 3 FIX: Register socket event listeners here (moved from DOMContentLoaded)
    socket.on('conversion_progress', (data) => {
        if (data.conversion_id === currentConversionId) {
            updateConversionProgress(data);
        }
    });

    socket.on('conversion_complete', (data) => {
        if (data.conversion_id === currentConversionId) {
            displayConversionResults(data, 'conversion-results');
            const progressSection = document.getElementById('conversion-progress');
            if (progressSection) progressSection.classList.add('hidden');
        }
    });

    socket.on('conversion_error', (data) => {
        if (data.conversion_id === currentConversionId) {
            showToast(`Conversion Error: ${data.error}`, 'error');
            if (progressBar) {
                progressBar.setError('Conversion failed: ' + data.error);
            }
        }
    });

    socket.on('conversion_cancelled', (data) => {
        if (data.conversion_id === currentConversionId) {
            showToast('Conversion Cancelled', 'warning');
            if (progressBar) {
                progressBar.setError('Conversion Cancelled');
            }
        }
    });

    // COMMENT 7 FIX: Step advancement logic
    const profileSelector = document.getElementById('profile-selector');
    const uploadSectionEl = document.getElementById('upload-section');
    const configSectionEl = document.getElementById('config-section');
    const stepProfile = document.getElementById('step-profile');
    const stepUpload = document.getElementById('step-upload');
    const stepConfigure = document.getElementById('step-configure');

    // When profile is selected, show upload section
    if (profileSelector) {
        profileSelector.addEventListener('change', (e) => {
            if (e.target.value) {
                // Mark profile step as completed
                if (stepProfile) {
                    stepProfile.classList.remove('active');
                    stepProfile.classList.add('completed');
                }
                // Activate upload step
                if (stepUpload) {
                    stepUpload.classList.add('active');
                }
                // Show upload section
                if (uploadSectionEl) {
                    uploadSectionEl.classList.remove('hidden');
                }
            }
        });
    }

    // When song file is chosen, show config section
    if (songInput) {
        songInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                // Mark upload step as completed
                if (stepUpload) {
                    stepUpload.classList.remove('active');
                    stepUpload.classList.add('completed');
                }
                // Activate configure step
                if (stepConfigure) {
                    stepConfigure.classList.add('active');
                }
                // Show config section
                if (configSectionEl) {
                    configSectionEl.classList.remove('hidden');
                }
            }
        });
    }
}

/**
 * Generate UUID v4
 * @returns {string} UUID
 */
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Global functions for UI buttons
window.downloadConvertedAudio = function() {
    if (window.currentAudioBlob) {
        downloadAudio(window.currentAudioBlob, 'converted_song.wav');
    } else {
        showToast('No audio to download', 'error');
    }
};

window.convertAnotherSong = function() {
    document.getElementById('conversion-results').innerHTML = '';
    document.getElementById('conversion-progress').style.display = 'block';
    progressBar.reset();

    const form = document.querySelector('form[id*="conversion"]');
    if (form) {
        form.querySelector('input[name="song"]').value = '';
    }

    showToast('Ready to convert another song', 'info');
};

// COMMENT 1 FIX: Initialization is centralized in app.js global DOMContentLoaded handler
// Template extra_scripts block is now empty (removed inline initializer to prevent duplicates)
// Socket event listeners are registered once inside initSongConversionForm (with idempotence guard)