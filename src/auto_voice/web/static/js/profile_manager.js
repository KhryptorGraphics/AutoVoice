/**
 * Profile Manager Module
 * Handles voice profile CRUD operations and management
 */

import {
    formatDuration,
    showToast,
    showLoadingSpinner
} from './audio_utils.js';

// Profile cache
let cachedProfiles = null;
let selectedProfileId = null;

/**
 * Load all voice profiles
 * @param {string} userId - Optional user ID filter
 * @returns {Promise<Array>} Array of profiles
 */
export async function loadProfiles(userId = null) {
    try {
        const url = userId ?
            `/api/v1/voice/profiles?user_id=${encodeURIComponent(userId)}` :
            '/api/v1/voice/profiles';

        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const profiles = await response.json();
        cachedProfiles = profiles;

        return profiles;

    } catch (error) {
        console.error('Error loading profiles:', error);
        showToast('Failed to load profiles: ' + error.message, 'error');
        return [];
    }
}

/**
 * Display profile list in UI
 * @param {Array} profiles - Array of profile objects
 * @param {string} containerId - Container element ID (tbody for table rows)
 */
export function displayProfileList(profiles, containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }

    if (!profiles || profiles.length === 0) {
        // COMMENT 5 FIX: Handle empty state properly for tbody
        container.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; padding: 2rem;">
                    <p style="color: #6c757d;">No voice profiles found.</p>
                </td>
            </tr>
        `;
        return;
    }

    // COMMENT 5 FIX: Create table rows only (container is tbody)
    let html = '';

    profiles.forEach(profile => {
        const createdAt = formatTimestamp(profile.created_at);
        const duration = formatDuration(profile.audio_duration);
        const vocalRange = formatVocalRange(
            profile.vocal_range?.min_f0,
            profile.vocal_range?.max_f0
        );

        html += `
            <tr class="profile-row ${selectedProfileId === profile.profile_id ? 'selected' : ''}">
                <td>
                    <code class="profile-id" title="${profile.profile_id}">
                        ${truncateProfileId(profile.profile_id)}
                    </code>
                </td>
                <td>${createdAt}</td>
                <td>${duration}</td>
                <td>${vocalRange}</td>
                <td class="actions">
                    <button class="btn btn-sm" onclick="window.viewProfileDetails('${profile.profile_id}')">
                        View
                    </button>
                    <button class="btn btn-sm btn-primary" onclick="window.selectProfile('${profile.profile_id}')">
                        Select
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="window.deleteProfileConfirm('${profile.profile_id}')">
                        Delete
                    </button>
                </td>
            </tr>
        `;
    });

    // COMMENT 5 FIX: Set innerHTML directly as table rows
    container.innerHTML = html;
}

/**
 * Get profile details by ID
 * @param {string} profileId - Profile ID
 * @returns {Promise<Object>} Profile details
 */
export async function getProfileDetails(profileId) {
    try {
        const response = await fetch(`/api/v1/voice/profiles/${profileId}`);

        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('Profile not found');
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const profile = await response.json();
        return profile;

    } catch (error) {
        console.error('Error getting profile details:', error);
        showToast('Failed to load profile details: ' + error.message, 'error');
        throw error;
    }
}

/**
 * Display profile details in modal or panel
 * @param {string} profileId - Profile ID
 * @param {string} containerId - Container element ID
 */
// COMMENT 4 FIX: Update default container ID to 'details-modal'
export async function showProfileDetailsModal(profileId, containerId = 'details-modal') {
    const hideSpinner = showLoadingSpinner(containerId, 'Loading profile details...');

    try {
        const profile = await getProfileDetails(profileId);
        hideSpinner();

        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        container.innerHTML = `
            <div class="modal-overlay" onclick="this.parentElement.style.display='none'">
                <div class="modal-content" onclick="event.stopPropagation()">
                    <div class="modal-header">
                        <h2>Voice Profile Details</h2>
                        <button class="close-btn" onclick="document.getElementById('${containerId}').style.display='none'">
                            ×
                        </button>
                    </div>

                    <div class="modal-body">
                        <div class="profile-details-grid">
                            <div class="detail-row">
                                <span class="label">Profile ID:</span>
                                <span class="value">
                                    <code>${profile.profile_id}</code>
                                    <button class="copy-btn" onclick="navigator.clipboard.writeText('${profile.profile_id}')">
                                        Copy
                                    </button>
                                </span>
                            </div>

                            ${profile.user_id ? `
                            <div class="detail-row">
                                <span class="label">User ID:</span>
                                <span class="value">${profile.user_id}</span>
                            </div>
                            ` : ''}

                            <div class="detail-row">
                                <span class="label">Created:</span>
                                <span class="value">${new Date(profile.created_at).toLocaleString()}</span>
                            </div>

                            <div class="detail-row">
                                <span class="label">Duration:</span>
                                <span class="value">${formatDuration(profile.audio_duration)}</span>
                            </div>

                            ${profile.vocal_range ? `
                            <div class="detail-row">
                                <span class="label">Vocal Range:</span>
                                <span class="value">
                                    ${formatVocalRange(profile.vocal_range.min_f0, profile.vocal_range.max_f0)}
                                </span>
                            </div>
                            ` : ''}

                            ${profile.embedding_stats ? `
                            <div class="detail-section">
                                <h3>Embedding Statistics</h3>
                                <div class="detail-row">
                                    <span class="label">Mean:</span>
                                    <span class="value">${profile.embedding_stats.mean?.toFixed(4) || 'N/A'}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="label">Std:</span>
                                    <span class="value">${profile.embedding_stats.std?.toFixed(4) || 'N/A'}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="label">Norm:</span>
                                    <span class="value">${profile.embedding_stats.norm?.toFixed(4) || 'N/A'}</span>
                                </div>
                            </div>
                            ` : ''}

                            ${profile.timbre_features ? `
                            <div class="detail-section">
                                <h3>Timbre Features</h3>
                                ${Object.entries(profile.timbre_features).map(([key, value]) => `
                                    <div class="detail-row">
                                        <span class="label">${key}:</span>
                                        <span class="value">${typeof value === 'number' ? value.toFixed(2) : value}</span>
                                    </div>
                                `).join('')}
                            </div>
                            ` : ''}
                        </div>
                    </div>

                    <div class="modal-actions">
                        <button class="btn btn-primary" onclick="window.useProfileForConversion('${profile.profile_id}')">
                            Use for Conversion
                        </button>
                        <button class="btn btn-danger" onclick="window.deleteProfileConfirm('${profile.profile_id}')">
                            Delete Profile
                        </button>
                        <button class="btn btn-secondary" onclick="document.getElementById('${containerId}').style.display='none'">
                            Close
                        </button>
                    </div>
                </div>
            </div>
        `;

        container.style.display = 'block';

    } catch (error) {
        hideSpinner();
        showToast('Failed to load profile details', 'error');
    }
}

/**
 * Delete profile by ID
 * @param {string} profileId - Profile ID
 * @returns {Promise<boolean>} Success status
 */
export async function deleteProfile(profileId) {
    try {
        const response = await fetch(`/api/v1/voice/profiles/${profileId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('Profile not found');
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        // Invalidate cache
        cachedProfiles = null;

        showToast('Profile deleted successfully', 'success');
        return true;

    } catch (error) {
        console.error('Error deleting profile:', error);
        showToast('Failed to delete profile: ' + error.message, 'error');
        return false;
    }
}

/**
 * Show confirmation dialog for profile deletion
 * @param {string} profileId - Profile ID
 */
export function showDeleteConfirmation(profileId) {
    const confirmed = confirm(
        `Delete profile ${truncateProfileId(profileId)}?\n\nThis action cannot be undone.`
    );

    if (confirmed) {
        deleteProfile(profileId).then(success => {
            if (success) {
                // Refresh list
                refreshProfiles();

                // Clear selection if deleted
                if (selectedProfileId === profileId) {
                    selectedProfileId = null;
                    localStorage.removeItem('selectedProfileId');
                }
            }
        });
    }
}

/**
 * Select profile for use
 * @param {string} profileId - Profile ID
 */
export function selectProfile(profileId) {
    selectedProfileId = profileId;
    localStorage.setItem('selectedProfileId', profileId);

    // Update UI
    document.querySelectorAll('.profile-row').forEach(row => {
        row.classList.remove('selected');
    });

    const selectedRow = document.querySelector(`[onclick*="${profileId}"]`)?.closest('.profile-row');
    if (selectedRow) {
        selectedRow.classList.add('selected');
    }

    showToast('Profile selected', 'success');
}

/**
 * Refresh profiles from API
 * @param {string} containerId - Optional container to update
 */
export async function refreshProfiles(containerId = 'profile-list') {
    const profiles = await loadProfiles();
    if (containerId) {
        displayProfileList(profiles, containerId);
    }
    return profiles;
}

/**
 * Search profiles by query
 * @param {string} query - Search query (user_id or profile_id)
 * @param {string} containerId - Container to update
 */
export async function searchProfiles(query, containerId = 'profile-list') {
    if (!query) {
        await refreshProfiles(containerId);
        return;
    }

    const allProfiles = cachedProfiles || await loadProfiles();

    const filtered = allProfiles.filter(profile =>
        profile.profile_id.toLowerCase().includes(query.toLowerCase()) ||
        (profile.user_id && profile.user_id.toLowerCase().includes(query.toLowerCase()))
    );

    displayProfileList(filtered, containerId);

    if (filtered.length === 0) {
        showToast('No profiles found matching query', 'info');
    }
}

/**
 * Format vocal range to readable string
 * @param {number} minF0 - Minimum F0
 * @param {number} maxF0 - Maximum F0
 * @returns {string} Formatted range
 */
function formatVocalRange(minF0, maxF0) {
    if (!minF0 || !maxF0) return 'Not available';

    const minNote = formatFrequencyToNote(minF0);
    const maxNote = formatFrequencyToNote(maxF0);

    return `${minNote} - ${maxNote} (${Math.round(minF0)}-${Math.round(maxF0)} Hz)`;
}

/**
 * Format frequency to musical note
 * @param {number} frequency - Frequency in Hz
 * @returns {string} Note name
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
 * Format timestamp to relative or absolute time
 * @param {string} isoTimestamp - ISO timestamp
 * @returns {string} Formatted time
 */
function formatTimestamp(isoTimestamp) {
    const date = new Date(isoTimestamp);
    const now = new Date();
    const diffSeconds = Math.floor((now - date) / 1000);

    if (diffSeconds < 60) return 'Just now';
    if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)} minutes ago`;
    if (diffSeconds < 86400) return `${Math.floor(diffSeconds / 3600)} hours ago`;
    if (diffSeconds < 604800) return `${Math.floor(diffSeconds / 86400)} days ago`;

    return date.toLocaleDateString();
}

/**
 * Truncate profile ID for display
 * @param {string} profileId - Full profile ID
 * @param {number} length - Truncation length
 * @returns {string} Truncated ID
 */
function truncateProfileId(profileId, length = 8) {
    if (!profileId) return '';
    return profileId.substring(0, length) + '...';
}

// Global functions for UI buttons
// COMMENT 4 FIX: Pass 'details-modal' explicitly when calling from template
window.viewProfileDetails = function(profileId) {
    showProfileDetailsModal(profileId, 'details-modal');
};

window.selectProfile = function(profileId) {
    selectProfile(profileId);
};

window.deleteProfileConfirm = function(profileId) {
    showDeleteConfirmation(profileId);
};

window.refreshProfileList = function() {
    refreshProfiles('profile-list');
};

window.useProfileForConversion = function(profileId) {
    selectProfile(profileId);

    // Navigate to song conversion
    window.location.href = '/song-conversion';
};