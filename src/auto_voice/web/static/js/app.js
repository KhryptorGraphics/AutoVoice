// Tab switching functionality
function showTab(tabName) {
    const tabs = document.querySelectorAll('.tab-content');
    const buttons = document.querySelectorAll('.tab-button');
    
    tabs.forEach(tab => {
        tab.classList.remove('active');
    });
    
    buttons.forEach(button => {
        button.classList.remove('active');
    });
    
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
    
    if (tabName === 'status') {
        updateStatus();
    }
}

// Update slider values
document.getElementById('speed').addEventListener('input', function() {
    document.getElementById('speed-value').textContent = this.value;
});

document.getElementById('pitch').addEventListener('input', function() {
    document.getElementById('pitch-value').textContent = this.value;
});

document.getElementById('pitch-shift').addEventListener('input', function() {
    document.getElementById('pitch-shift-value').textContent = this.value;
});

// Text to Speech form submission
document.getElementById('synthesize-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = {
        text: document.getElementById('text').value,
        speaker_id: parseInt(document.getElementById('speaker').value),
        speed: parseFloat(document.getElementById('speed').value),
        pitch: parseFloat(document.getElementById('pitch').value)
    };
    
    try {
        const response = await fetch('/api/synthesize', {
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

// Voice Conversion form submission
document.getElementById('convert-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    formData.append('audio', document.getElementById('audio-file').files[0]);
    formData.append('target_speaker_id', document.getElementById('target-speaker').value);
    formData.append('pitch_shift', document.getElementById('pitch-shift').value);
    
    try {
        const response = await fetch('/api/convert', {
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

// Voice Cloning form submission
document.getElementById('clone-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    formData.append('reference', document.getElementById('reference-audio').files[0]);
    formData.append('text', document.getElementById('clone-text').value);
    
    try {
        const response = await fetch('/api/clone', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Cloning failed');
        }
        
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        
        document.getElementById('clone-audio').src = audioUrl;
        document.getElementById('clone-result').style.display = 'block';
        
    } catch (error) {
        alert('Error: ' + error.message);
    }
});

// Update GPU status
async function updateStatus() {
    try {
        const response = await fetch('/api/gpu_status');
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

// Load speaker list on page load
window.addEventListener('DOMContentLoaded', async function() {
    try {
        const response = await fetch('/api/speakers');
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
    } catch (error) {
        console.error('Error loading speakers:', error);
    }
});
