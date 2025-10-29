/**
 * Voice Cloning Module
 * Handles voice profile creation workflow
 */

import {
    validateAudioFile,
    getAudioMetadata,
    formatDuration,
    formatFileSize,
    showToast,
    showLoadingSpinner,
    createAudioPlayer
} from './audio_utils.js';

/**
 * Create voice profile from audio sample
 * @param {File} audioFile - Voice sample audio file
 * @param {string} userId - Optional user ID
 * @returns {Promise<Object>} Profile creation result
 */
export async function createVoiceProfile(audioFile, userId = null) {
    // Validate audio file
    const validation = await validateAudioFile(audioFile, {
        minDuration: 5,
        maxDuration: 60
    });

    if (!validation.valid) {
        throw new Error(validation.error);
    }

    // Show validation warnings
    if (validation.warnings.length > 0) {
        validation.warnings.forEach(warning => {
            showToast(warning, 'warning');
        });
    }

    // Create form data
    const formData = new FormData();
    formData.append('reference_audio', audioFile);
    if (userId) {
        formData.append('user_id', userId);
    }

    // Show loading indicator
    const hideSpinner = showLoadingSpinner('voice-cloning-result', 'Creating voice profile...');

    try {
        // POST to voice cloning endpoint
        const response = await fetch('/api/v1/voice/clone', {
            method: 'POST',
            body: formData
        });

        hideSpinner();

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        // Display success
        showToast('Voice profile created successfully!', 'success');

        return result;

    } catch (error) {
        hideSpinner();
        throw error;
    }
}

/**
 * Validate audio file before upload
 * @param {File} file - Audio file to validate
 * @returns {Promise<{valid: boolean, error: string}>}
 */
export async function validateAudioFileForCloning(file) {
    return await validateAudioFile(file, {
        minDuration: 5,
        maxDuration: 60,
        allowedTypes: ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg']
    });
}

/**
 * Preview audio file before cloning
 * @param {File} file - Audio file
 * @param {string} containerId - Container for audio player
 * @returns {Promise<Object>} Audio metadata
 */
export async function previewAudioFile(file, containerId) {
    const metadata = await getAudioMetadata(file);

    // Create audio player for preview
    const audioBlob = new Blob([file], { type: file.type });
    createAudioPlayer(audioBlob, containerId);

    // Show metadata
    const container = document.getElementById(containerId);
    const metadataDiv = document.createElement('div');
    metadataDiv.className = 'audio-metadata';
    metadataDiv.innerHTML = `
        <div class="metadata-item">
            <span class="metadata-label">Duration:</span>
            <span class="metadata-value">${metadata.duration ? formatDuration(metadata.duration) : 'Unknown'}</span>
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Size:</span>
            <span class="metadata-value">${formatFileSize(metadata.size)}</span>
        </div>
    `;
    container.appendChild(metadataDiv);

    // Show duration warning if out of range
    if (metadata.duration && metadata.duration < 5) {
        showToast('Audio is very short. For best results, use 30-60 seconds.', 'warning', 5000);
    } else if (metadata.duration && metadata.duration > 60) {
        showToast('Audio is longer than 60 seconds. Only the first 60 seconds will be used.', 'info', 5000);
    }

    return metadata;
}

/**
 * Display voice profile details
 * @param {Object} profile - Voice profile data
 * @param {string} containerId - Container element ID
 */
export function displayProfileDetails(profile, containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }

    // Format vocal range
    const vocalRange = profile.vocal_range ?
        `${formatFrequencyToNote(profile.vocal_range.min_f0)} - ${formatFrequencyToNote(profile.vocal_range.max_f0)} (${Math.round(profile.vocal_range.min_f0)}-${Math.round(profile.vocal_range.max_f0)} Hz)` :
        'Not available';

    container.innerHTML = `
        <div class="profile-details">
            <h3>Voice Profile Created Successfully</h3>

            <div class="profile-info">
                <div class="info-row">
                    <span class="info-label">Profile ID:</span>
                    <span class="info-value">
                        <code>${profile.profile_id}</code>
                        <button class="copy-btn" onclick="navigator.clipboard.writeText('${profile.profile_id}')">Copy</button>
                    </span>
                </div>

                <div class="info-row">
                    <span class="info-label">Created:</span>
                    <span class="info-value">${new Date(profile.created_at).toLocaleString()}</span>
                </div>

                <div class="info-row">
                    <span class="info-label">Duration:</span>
                    <span class="info-value">${formatDuration(profile.audio_duration)}</span>
                </div>

                <div class="info-row">
                    <span class="info-label">Vocal Range:</span>
                    <span class="info-value">${vocalRange}</span>
                </div>

                ${profile.user_id ? `
                <div class="info-row">
                    <span class="info-label">User ID:</span>
                    <span class="info-value">${profile.user_id}</span>
                </div>
                ` : ''}
            </div>

            <div class="profile-actions">
                <button class="btn btn-primary" onclick="window.useProfileForConversion('${profile.profile_id}')">
                    Use for Song Conversion
                </button>
                <button class="btn btn-secondary" onclick="window.createAnotherProfile()">
                    Create Another Profile
                </button>
            </div>
        </div>
    `;

    // Store profile ID for later use
    localStorage.setItem('lastCreatedProfileId', profile.profile_id);
}

/**
 * Convert frequency to musical note
 * @param {number} frequency - Frequency in Hz
 * @returns {string} Note name (e.g., "C4")
 */
function formatFrequencyToNote(frequency) {
    if (!frequency) return 'Unknown';

    const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const a4 = 440;
    const c0 = a4 * Math.pow(2, -4.75);

    const halfSteps = Math.round(12 * Math.log2(frequency / c0));
    const octave = Math.floor(halfSteps / 12);
    const noteIndex = halfSteps % 12;

    return noteNames[noteIndex] + octave;
}

/**
 * Initialize voice cloning form
 * @param {string} formId - Form element ID
 */
export function initVoiceCloningForm(formId) {
    const form = document.getElementById(formId);
    if (!form) {
        console.error(`Form ${formId} not found`);
        return;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const audioInput = form.querySelector('input[type="file"]');
        const userIdInput = form.querySelector('input[name="user_id"]');

        if (!audioInput.files || audioInput.files.length === 0) {
            showToast('Please select an audio file', 'error');
            return;
        }

        const audioFile = audioInput.files[0];
        const userId = userIdInput ? userIdInput.value : null;

        try {
            // Validate file first
            const validation = await validateAudioFileForCloning(audioFile);
            if (!validation.valid) {
                showToast(validation.error, 'error');
                return;
            }

            // Create profile
            const profile = await createVoiceProfile(audioFile, userId);

            // Display results
            displayProfileDetails(profile, 'voice-cloning-result');

            // Clear form
            form.reset();

        } catch (error) {
            console.error('Error creating voice profile:', error);
            showToast('Failed to create voice profile: ' + error.message, 'error', 5000);
        }
    });

    // File input change handler
    const audioInput = form.querySelector('input[type="file"]');
    if (audioInput) {
        audioInput.addEventListener('change', async (e) => {
            if (e.target.files && e.target.files.length > 0) {
                const file = e.target.files[0];

                // Validate immediately
                const validation = await validateAudioFileForCloning(file);
                if (!validation.valid) {
                    showToast(validation.error, 'error');
                    e.target.value = '';
                    return;
                }

                // Show warnings
                if (validation.warnings.length > 0) {
                    validation.warnings.forEach(warning => {
                        showToast(warning, 'warning');
                    });
                }

                // Preview audio
                try {
                    await previewAudioFile(file, 'audio-preview');
                } catch (error) {
                    console.warn('Could not preview audio:', error);
                }
            }
        });
    }
}

// Global function for "Use for Conversion" button
window.useProfileForConversion = function(profileId) {
    // Store profile ID
    localStorage.setItem('selectedProfileId', profileId);

    // Navigate to song conversion tab
    window.location.href = '/song-conversion';
};

// Global function for "Create Another Profile" button
window.createAnotherProfile = function() {
    const resultContainer = document.getElementById('voice-cloning-result');
    if (resultContainer) {
        resultContainer.innerHTML = '';
    }

    const form = document.querySelector('form[id*="cloning"]');
    if (form) {
        form.reset();
    }

    showToast('Ready to create another voice profile', 'info');
};