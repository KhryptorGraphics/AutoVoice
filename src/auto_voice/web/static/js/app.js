// ES6 Module Imports
import { formatDuration, formatFileSize } from './audio_utils.js';
import { initVoiceCloningForm } from './voice_cloning.js';
import { initSongConversionForm } from './song_conversion.js';
import { refreshProfiles, searchProfiles, displayProfileList } from './profile_manager.js';
import socket from './websocket.js';

// Tab switching functionality
function showTab(tabName, element) {
    const tabs = document.querySelectorAll('.tab-content');
    const buttons = document.querySelectorAll('.tab-button');

    tabs.forEach(tab => {
        tab.classList.remove('active');
    });

    buttons.forEach(button => {
        button.classList.remove('active');
    });

    document.getElementById(tabName).classList.add('active');
    if (element) {
        element.classList.add('active');
    }


    // Handle tab-specific initialization
    if (tabName === 'status') {
        updateStatus();
    } else if (tabName === 'song-conversion') {
        loadProfiles();
    } else if (tabName === 'profiles') {
        refreshProfiles('profile-list');
    }
}

// Global helper to load profiles into all selectors
async function loadProfiles() {
    try {
        const response = await fetch('/api/v1/voice/profiles');
        const profiles = await response.json();

        if (profiles) {
            const songProfileSelector = document.getElementById('profile-selector');
            if (songProfileSelector) {
                songProfileSelector.innerHTML = '<option value="">Select a voice profile...</option>';
                profiles.forEach(profile => {
                    const option = document.createElement('option');
                    option.value = profile.profile_id;
                    option.textContent = `Profile ${profile.profile_id.substring(0, 8)}...`;
                    songProfileSelector.appendChild(option);
                });
            }
        }
    } catch (error) {
        console.error('Error loading profiles:', error);
    }
}

// Update slider values
const speedSlider = document.getElementById('speed');
if (speedSlider) {
    speedSlider.addEventListener('input', function() {
        document.getElementById('speed-value').textContent = this.value;
    });
}

const pitchSlider = document.getElementById('pitch');
if(pitchSlider) {
    pitchSlider.addEventListener('input', function() {
        document.getElementById('pitch-value').textContent = this.value;
    });
}

const pitchShiftSlider = document.getElementById('pitch-shift');
if (pitchShiftSlider) {
    pitchShiftSlider.addEventListener('input', function() {
        document.getElementById('pitch-shift-value').textContent = this.value;
    });
}


// Text to Speech form submission
const synthesizeForm = document.getElementById('synthesize-form');
if (synthesizeForm) {
    synthesizeForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = {
            text: document.getElementById('text').value,
            speaker_id: parseInt(document.getElementById('speaker').value),
            speed: parseFloat(document.getElementById('speed').value),
            pitch: parseFloat(document.getElementById('pitch').value)
        };
        
        try {
            const response = await fetch('/api/v1/synthesize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                throw new Error('Synthesis failed');
            }
            
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            
            document.getElementById('synthesize-audio').src = audioUrl;
            document.getElementById('synthesize-result').style.display = 'block';
            
        } catch (error) {
            alert('Error: ' + error.message);
        }
    });
}

// Voice Conversion form submission
const convertForm = document.getElementById('convert-form');
if (convertForm) {
    convertForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        formData.append('audio', document.getElementById('audio-file').files[0]);
        formData.append('target_speaker_id', document.getElementById('target-speaker').value);
        formData.append('pitch_shift', document.getElementById('pitch-shift').value);
        
        try {
            const response = await fetch('/api/v1/convert', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Conversion failed');
            }
            
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            
            document.getElementById('convert-audio').src = audioUrl;
            document.getElementById('convert-result').style.display = 'block';
            
        } catch (error) {
            alert('Error: ' + error.message);
        }
    });
}


// Update GPU status
async function updateStatus() {
    try {
        const response = await fetch('/api/v1/gpu_status');
        const status = await response.json();
        
        document.getElementById('cuda-available').textContent = status.cuda_available ? 'Yes' : 'No';
        
        if (status.cuda_available) {
            document.getElementById('device-name').textContent = status.device_name || status.device;
            document.getElementById('memory-used').textContent = 
                `${(status.memory_allocated || 0).toFixed(2)} GB / ${(status.memory_reserved || 0).toFixed(2)} GB`;
            document.getElementById('memory-total').textContent = 
                `${(status.memory_total || 0).toFixed(2)} GB`;
        } else {
            document.getElementById('device-name').textContent = 'CPU';
            document.getElementById('memory-used').textContent = 'N/A';
            document.getElementById('memory-total').textContent = 'N/A';
        }
        
    } catch (error) {
        console.error('Error fetching status:', error);
    }
}

// Download audio function
function downloadAudio(audioId) {
    const audio = document.getElementById(audioId);
    const link = document.createElement('a');
    link.href = audio.src;
    link.download = 'autovoice_output.wav';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Load speaker list and initialize modules on page load
window.addEventListener('DOMContentLoaded', async function() {
    try {
        // Load speakers for TTS and voice conversion
        const response = await fetch('/api/v1/speakers');
        const data = await response.json();

        if (data.speakers) {
            const speakerSelects = ['speaker', 'target-speaker'];
            speakerSelects.forEach(selectId => {
                const select = document.getElementById(selectId);
                if (select) {
                    select.innerHTML = '';
                    data.speakers.forEach(speaker => {
                        const option = document.createElement('option');
                        option.value = speaker.id;
                        option.textContent = `${speaker.name} (${speaker.gender})`;
                        select.appendChild(option);
                    });
                }
            });
        }

        // Initialize voice cloning form with profile creation
        if (document.getElementById('voice-cloning-form')) {
            initVoiceCloningForm('voice-cloning-form');
        }

        // Initialize song conversion form with WebSocket support
        if (document.getElementById('song-conversion-form')) {
            initSongConversionForm('song-conversion-form', socket);
        }

        // Initialize profile search functionality
        const profileSearch = document.getElementById('profile-search');
        if (profileSearch) {
            profileSearch.addEventListener('input', function(e) {
                searchProfiles(e.target.value, 'profile-list');
            });
        }

        // Load profiles initially
        await loadProfiles();

    } catch (error) {
        console.error('Error during initialization:', error);
    }
});

// Make showTab globally accessible
window.showTab = showTab;
window.loadProfiles = loadProfiles;
window.downloadAudio = downloadAudio;