/**
 * Audio Utilities Module
 * Provides common audio operations for web interface
 */

/**
 * Validate audio file against requirements
 * @param {File} file - Audio file to validate
 * @param {Object} options - Validation options
 * @returns {Promise<{valid: boolean, error: string, warnings: array}>}
 */
export async function validateAudioFile(file, options = {}) {
    const {
        maxSize = 100 * 1024 * 1024, // 100MB
        allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg'],
        minDuration = null,
        maxDuration = null
    } = options;

    const warnings = [];

    // Check file type
    if (!allowedTypes.some(type => file.type === type || file.name.toLowerCase().endsWith(type.split('/')[1]))) {
        return { valid: false, error: `Invalid file type. Allowed: ${allowedTypes.join(', ')}`, warnings };
    }

    // Check file size
    if (file.size > maxSize) {
        return { valid: false, error: `File too large. Maximum size: ${formatFileSize(maxSize)}`, warnings };
    }

    // Check duration if specified
    if (minDuration || maxDuration) {
        try {
            const duration = await getAudioDuration(file);

            if (minDuration && duration < minDuration) {
                return { valid: false, error: `Audio too short. Minimum: ${formatDuration(minDuration)}`, warnings };
            }

            if (maxDuration && duration > maxDuration) {
                return { valid: false, error: `Audio too long. Maximum: ${formatDuration(maxDuration)}`, warnings };
            }

            // Add duration warning for edge cases
            if (duration < 5) {
                warnings.push('Very short audio may produce suboptimal results');
            }
        } catch (error) {
            warnings.push('Could not determine audio duration');
        }
    }

    return { valid: true, error: null, warnings };
}

/**
 * Get audio duration using Web Audio API
 * @param {File|Blob} file - Audio file
 * @returns {Promise<number>} Duration in seconds
 */
export async function getAudioDuration(file) {
    return new Promise((resolve, reject) => {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const reader = new FileReader();

        reader.onload = async (e) => {
            try {
                const arrayBuffer = e.target.result;
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                resolve(audioBuffer.duration);
            } catch (error) {
                reject(new Error('Failed to decode audio: ' + error.message));
            } finally {
                audioContext.close();
            }
        };

        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsArrayBuffer(file);
    });
}

/**
 * Get comprehensive audio metadata
 * @param {File} file - Audio file
 * @returns {Promise<Object>} Metadata object
 */
export async function getAudioMetadata(file) {
    const metadata = {
        filename: file.name,
        size: file.size,
        type: file.type,
        lastModified: new Date(file.lastModified)
    };

    try {
        metadata.duration = await getAudioDuration(file);
    } catch (error) {
        metadata.duration = null;
        metadata.durationError = error.message;
    }

    return metadata;
}

/**
 * Encode audio blob to base64 string
 * @param {Blob} audioBlob - Audio blob
 * @returns {Promise<string>} Base64 encoded string
 */
export function encodeAudioToBase64(audioBlob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(audioBlob);
    });
}

/**
 * Decode base64 string to audio blob
 * @param {string} base64String - Base64 encoded audio
 * @param {string} mimeType - MIME type (default: audio/wav)
 * @returns {Blob} Audio blob
 */
export function decodeBase64ToAudio(base64String, mimeType = 'audio/wav') {
    const binaryString = atob(base64String);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return new Blob([bytes], { type: mimeType });
}

/**
 * Create object URL from audio blob
 * @param {Blob} audioBlob - Audio blob
 * @returns {string} Object URL
 */
export function createAudioURL(audioBlob) {
    return URL.createObjectURL(audioBlob);
}

/**
 * Create audio player element
 * @param {Blob} audioBlob - Audio blob
 * @param {string} containerId - Container element ID
 * @returns {HTMLAudioElement} Audio element
 */
export function createAudioPlayer(audioBlob, containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
        throw new Error(`Container ${containerId} not found`);
    }

    const audio = document.createElement('audio');
    audio.controls = true;
    audio.className = 'audio-player';
    audio.src = createAudioURL(audioBlob);

    container.innerHTML = '';
    container.appendChild(audio);

    return audio;
}

/**
 * Download audio blob as file
 * @param {Blob} audioBlob - Audio blob
 * @param {string} filename - Download filename
 */
export function downloadAudio(audioBlob, filename = 'autovoice_output.wav') {
    const url = createAudioURL(audioBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

/**
 * Create progress bar component
 * COMMENT 7 FIX: Modified to work with existing elements if present
 * @param {string} containerId - Container element ID
 * @returns {Object} Progress bar controller
 */
export function createProgressBar(containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
        throw new Error(`Container ${containerId} not found`);
    }

    // COMMENT 7 FIX: Check if elements already exist in markup
    let progressFill = document.getElementById('progress-fill');
    let progressStage = document.getElementById('progress-stage');

    // If existing elements found, use them instead of creating new markup
    if (progressFill && progressStage) {
        // Use existing markup from template
        return {
            setProgress(percent) {
                const clampedPercent = Math.max(0, Math.min(100, percent));
                progressFill.style.width = clampedPercent + '%';
                progressFill.textContent = Math.round(clampedPercent) + '%';
            },

            setStage(stageName) {
                progressStage.textContent = stageName;
                progressStage.className = 'progress-stage';
            },

            setError(errorMessage) {
                progressStage.textContent = errorMessage;
                progressStage.className = 'progress-stage error';
                progressFill.className = 'progress-fill error';
            },

            reset() {
                progressFill.style.width = '0%';
                progressFill.className = 'progress-fill';
                progressFill.textContent = '0%';
                progressStage.textContent = 'Initializing...';
                progressStage.className = 'progress-stage';
            }
        };
    } else {
        // Create new markup if elements don't exist
        container.innerHTML = `
            <div class="progress-container">
                <div class="progress-bar" style="width: 0%"></div>
                <div class="progress-text">0%</div>
            </div>
            <div class="stage-indicator">Initializing...</div>
        `;

        const progressBar = container.querySelector('.progress-bar');
        const progressText = container.querySelector('.progress-text');
        const stageIndicator = container.querySelector('.stage-indicator');

        return {
            setProgress(percent) {
                const clampedPercent = Math.max(0, Math.min(100, percent));
                progressBar.style.width = clampedPercent + '%';
                progressText.textContent = Math.round(clampedPercent) + '%';
            },

            setStage(stageName) {
                stageIndicator.textContent = stageName;
                stageIndicator.className = 'stage-indicator';
            },

            setError(errorMessage) {
                stageIndicator.textContent = errorMessage;
                stageIndicator.className = 'stage-indicator error';
                progressBar.className = 'progress-bar error';
            },

            reset() {
                progressBar.style.width = '0%';
                progressBar.className = 'progress-bar';
                progressText.textContent = '0%';
                stageIndicator.textContent = 'Initializing...';
                stageIndicator.className = 'stage-indicator';
            }
        };
    }
}

/**
 * Estimate time remaining based on progress
 * @param {number} progress - Current progress (0-100)
 * @param {number} elapsedTime - Elapsed time in seconds
 * @returns {string} Formatted time remaining
 */
export function estimateTimeRemaining(progress, elapsedTime) {
    if (progress <= 0) return 'Calculating...';
    if (progress >= 100) return 'Complete';

    const remainingSeconds = (elapsedTime / progress) * (100 - progress);
    return `~${formatDuration(remainingSeconds)} remaining`;
}

/**
 * Create drag-and-drop file upload area
 * @param {string} containerId - Container element ID
 * @param {Object} options - Upload options
 * @returns {Object} Upload area controller
 */
export function createFileUploadArea(containerId, options = {}) {
    const {
        accept = 'audio/*',
        multiple = false,
        onFileSelect = null
    } = options;

    const container = document.getElementById(containerId);
    if (!container) {
        throw new Error(`Container ${containerId} not found`);
    }

    container.innerHTML = `
        <div class="file-upload-area" id="${containerId}-dropzone">
            <input type="file" id="${containerId}-input" accept="${accept}" ${multiple ? 'multiple' : ''} style="display: none;">
            <div class="upload-prompt">
                <i class="upload-icon">📁</i>
                <p>Drag and drop audio file here or click to browse</p>
                <small>Supported formats: WAV, MP3, FLAC, OGG (Max 100MB)</small>
            </div>
            <div class="file-preview" style="display: none;"></div>
            <div class="upload-error" style="display: none;"></div>
        </div>
    `;

    const dropzone = container.querySelector('.file-upload-area');
    const fileInput = container.querySelector(`#${containerId}-input`);
    const preview = container.querySelector('.file-preview');
    const errorDiv = container.querySelector('.upload-error');
    let selectedFile = null;

    // Click to browse
    dropzone.addEventListener('click', () => fileInput.click());

    // Drag and drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('drag-over');
    });

    dropzone.addEventListener('drop', async (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            await handleFileSelect(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            await handleFileSelect(e.target.files[0]);
        }
    });

    async function handleFileSelect(file) {
        selectedFile = file;
        errorDiv.style.display = 'none';

        try {
            const metadata = await getAudioMetadata(file);

            preview.innerHTML = `
                <div class="file-info">
                    <div class="file-name">${metadata.filename}</div>
                    <div class="file-details">
                        ${formatFileSize(metadata.size)} •
                        ${metadata.duration ? formatDuration(metadata.duration) : 'Duration unknown'}
                    </div>
                    <button class="clear-file" onclick="this.closest('.file-upload-area').dispatchEvent(new Event('clearFile'))">✕</button>
                </div>
            `;
            preview.style.display = 'block';

            if (onFileSelect) {
                onFileSelect(file, metadata);
            }
        } catch (error) {
            setError('Failed to read file: ' + error.message);
        }
    }

    function setError(message) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        preview.style.display = 'none';
        selectedFile = null;
    }

    dropzone.addEventListener('clearFile', () => {
        selectedFile = null;
        fileInput.value = '';
        preview.style.display = 'none';
        errorDiv.style.display = 'none';
    });

    return {
        getFile() {
            return selectedFile;
        },

        clear() {
            dropzone.dispatchEvent(new Event('clearFile'));
        },

        setError(message) {
            setError(message);
        }
    };
}

/**
 * Upload file with progress tracking
 * @param {File} file - File to upload
 * @param {string} url - Upload URL
 * @param {FormData} formData - Form data
 * @param {Function} onProgress - Progress callback
 * @returns {Promise<Object>} Response data
 */
export function uploadFileWithProgress(file, url, formData, onProgress) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable && onProgress) {
                const percentComplete = (e.loaded / e.total) * 100;
                onProgress(percentComplete);
            }
        };

        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    resolve(response);
                } catch (error) {
                    resolve({ success: true });
                }
            } else {
                reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`));
            }
        };

        xhr.onerror = () => reject(new Error('Network error during upload'));
        xhr.ontimeout = () => reject(new Error('Upload timeout'));

        xhr.open('POST', url);
        xhr.send(formData);
    });
}

/**
 * Format file size to human-readable string
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted size
 */
export function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format duration to MM:SS or HH:MM:SS
 * @param {number} seconds - Duration in seconds
 * @returns {string} Formatted duration
 */
export function formatDuration(seconds) {
    if (!seconds || seconds < 0) return '00:00';

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Show toast notification
 * @param {string} message - Notification message
 * @param {string} type - Notification type (info, success, warning, error)
 * @param {number} duration - Display duration in ms
 */
export function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('show');
    }, 10);

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * Show loading spinner
 * @param {string} containerId - Container element ID
 * @param {string} message - Loading message
 * @returns {Function} Function to hide spinner
 */
export function showLoadingSpinner(containerId, message = 'Loading...') {
    const container = document.getElementById(containerId);
    if (!container) return () => {};

    const spinner = document.createElement('div');
    spinner.className = 'loading-spinner';
    spinner.innerHTML = `
        <div class="spinner"></div>
        <div class="loading-message">${message}</div>
    `;

    container.appendChild(spinner);

    return () => spinner.remove();
}
