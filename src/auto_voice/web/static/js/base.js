// Base JavaScript for AutoVoice Web Interface

// Global variables
let apiBaseUrl = window.location.origin + '/api';
let wsConnection = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeAlerts();
    checkGPUStatus();
    initializeAPILatencyCheck();

    // Initialize API docs link
    const apiDocsLink = document.getElementById('api-docs-link');
    if (apiDocsLink) {
        apiDocsLink.href = '/docs';
    }
});

// Initialize alert dismissal
function initializeAlerts() {
    const alertCloseButtons = document.querySelectorAll('.alert-close');
    alertCloseButtons.forEach(button => {
        button.addEventListener('click', function() {
            this.parentElement.style.display = 'none';
        });
    });
}

// Check GPU status via API
async function checkGPUStatus() {
    const statusElement = document.getElementById('gpu-status-text');

    try {
        const response = await fetch(`${apiBaseUrl}/v1/gpu/status`);
        const data = await response.json();

        if (data.available) {
            statusElement.textContent = `GPU: ${data.device_name}`;
            statusElement.parentElement.style.color = '#48bb78';
        } else {
            statusElement.textContent = 'GPU: Not Available';
            statusElement.parentElement.style.color = '#f56565';
        }
    } catch (error) {
        statusElement.textContent = 'GPU: Error';
        statusElement.parentElement.style.color = '#f6ad55';
        console.error('Error checking GPU status:', error);
    }
}

// API Latency monitoring
function initializeAPILatencyCheck() {
    const latencyElement = document.getElementById('latency-text');

    setInterval(async () => {
        const startTime = performance.now();

        try {
            await fetch(`${apiBaseUrl}/v1/health`);
            const endTime = performance.now();
            const latency = Math.round(endTime - startTime);

            if (latencyElement) {
                latencyElement.textContent = latency.toString();

                // Color code based on latency
                const parent = latencyElement.parentElement;
                if (latency < 100) {
                    parent.style.color = '#48bb78'; // Green
                } else if (latency < 300) {
                    parent.style.color = '#f6ad55'; // Yellow
                } else {
                    parent.style.color = '#f56565'; // Red
                }
            }
        } catch (error) {
            if (latencyElement) {
                latencyElement.textContent = '--';
            }
        }
    }, 5000); // Check every 5 seconds
}

// Utility function to format bytes
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Utility function to format percentages
function formatPercentage(value) {
    return `${Math.round(value)}%`;
}

// Utility function to format time
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// API request wrapper with error handling
async function apiRequest(endpoint, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers,
        },
    };

    try {
        const response = await fetch(`${apiBaseUrl}${endpoint}`, mergedOptions);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`API request failed for ${endpoint}:`, error);
        throw error;
    }
}

// Show notification
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;

    const iconClass = type === 'error' ? 'exclamation-triangle' :
                     type === 'success' ? 'check-circle' : 'info-circle';

    alertDiv.innerHTML = `
        <div>
            <i class="fas fa-${iconClass}"></i>
            ${message}
        </div>
        <button class="alert-close">&times;</button>
    `;

    const mainContent = document.querySelector('.main-content');
    mainContent.insertBefore(alertDiv, mainContent.firstChild);

    // Auto dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.style.display = 'none';
    }, 5000);

    // Add close button functionality
    alertDiv.querySelector('.alert-close').addEventListener('click', function() {
        alertDiv.style.display = 'none';
    });
}

// Export utilities for other scripts
window.AutoVoiceUtils = {
    apiRequest,
    showNotification,
    formatBytes,
    formatPercentage,
    formatTime,
};