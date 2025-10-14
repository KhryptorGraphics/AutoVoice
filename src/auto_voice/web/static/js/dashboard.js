// Dashboard JavaScript - GPU monitoring and visualization

let gpuChart = null;
let memoryChart = null;
let gpuDataHistory = [];
let memoryDataHistory = [];
const maxDataPoints = 50;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    loadSystemInfo();
    loadKernelMetrics();

    // Register WebSocket handlers
    if (window.wsManager) {
        window.wsManager.on('gpu_stats', updateDashboard);
        window.wsManager.on('kernel_metrics', updateKernelTable);
    }

    // Periodic updates
    setInterval(() => {
        fetchGPUStats();
    }, 2000); // Update every 2 seconds
});

// Initialize Chart.js charts
function initializeCharts() {
    const gpuCtx = document.getElementById('gpu-chart').getContext('2d');
    const memoryCtx = document.getElementById('memory-chart').getContext('2d');

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0
        },
        scales: {
            x: {
                display: false
            },
            y: {
                beginAtZero: true,
                max: 100,
                ticks: {
                    color: '#a0aec0',
                    callback: function(value) {
                        return value + '%';
                    }
                }
            }
        },
        plugins: {
            legend: {
                display: false
            }
        }
    };

    gpuChart = new Chart(gpuCtx, {
        type: 'line',
        data: {
            labels: Array(maxDataPoints).fill(''),
            datasets: [{
                data: [],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });

    // Memory chart with custom scale
    const memoryOptions = {
        ...chartOptions,
        scales: {
            ...chartOptions.scales,
            y: {
                beginAtZero: true,
                ticks: {
                    color: '#a0aec0',
                    callback: function(value) {
                        return (value / 1024).toFixed(1) + ' GB';
                    }
                }
            }
        }
    };

    memoryChart = new Chart(memoryCtx, {
        type: 'line',
        data: {
            labels: Array(maxDataPoints).fill(''),
            datasets: [{
                data: [],
                borderColor: '#764ba2',
                backgroundColor: 'rgba(118, 75, 162, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: memoryOptions
    });
}

// Update charts with new data
function updateGPUCharts(stats) {
    // Update GPU utilization chart
    gpuDataHistory.push(stats.utilization);
    if (gpuDataHistory.length > maxDataPoints) {
        gpuDataHistory.shift();
    }
    gpuChart.data.datasets[0].data = gpuDataHistory;
    gpuChart.update('none');

    // Update memory chart
    memoryDataHistory.push(stats.memory_used);
    if (memoryDataHistory.length > maxDataPoints) {
        memoryDataHistory.shift();
    }
    memoryChart.data.datasets[0].data = memoryDataHistory;
    memoryChart.update('none');
}

// Fetch GPU stats from API
async function fetchGPUStats() {
    try {
        const stats = await window.AutoVoiceUtils.apiRequest('/v1/gpu/metrics');
        updateDashboard(stats);
        updateGPUCharts(stats);
    } catch (error) {
        console.error('Failed to fetch GPU stats:', error);
    }
}

// Update dashboard UI elements
function updateDashboard(stats) {
    // Update stat cards
    document.getElementById('gpu-utilization').textContent = `${stats.utilization}%`;

    const memoryUsedGB = (stats.memory_used / 1024).toFixed(2);
    const memoryTotalGB = (stats.memory_total / 1024).toFixed(2);
    document.getElementById('gpu-memory').textContent = `${memoryUsedGB} / ${memoryTotalGB} GB`;

    document.getElementById('gpu-temperature').textContent = `${stats.temperature}°C`;

    if (stats.audio_throughput) {
        document.getElementById('audio-throughput').textContent = `${stats.audio_throughput.toLocaleString()}`;
    }

    // Color code temperature
    const tempElement = document.getElementById('gpu-temperature').parentElement.parentElement;
    const temp = stats.temperature;
    if (temp < 70) {
        tempElement.style.borderLeft = '4px solid #48bb78';
    } else if (temp < 80) {
        tempElement.style.borderLeft = '4px solid #f6ad55';
    } else {
        tempElement.style.borderLeft = '4px solid #f56565';
    }

    // Color code utilization
    const utilElement = document.getElementById('gpu-utilization').parentElement.parentElement;
    const util = stats.utilization;
    if (util < 70) {
        utilElement.style.borderLeft = '4px solid #48bb78';
    } else if (util < 90) {
        utilElement.style.borderLeft = '4px solid #f6ad55';
    } else {
        utilElement.style.borderLeft = '4px solid #f56565';
    }
}

// Load system information
async function loadSystemInfo() {
    try {
        const info = await window.AutoVoiceUtils.apiRequest('/v1/gpu/info');

        document.getElementById('gpu-model').textContent = info.device_name || '--';
        document.getElementById('driver-version').textContent = info.driver_version || '--';
        document.getElementById('cuda-version').textContent = info.cuda_version || '--';
        document.getElementById('pytorch-version').textContent = info.pytorch_version || '--';
        document.getElementById('compute-capability').textContent = info.compute_capability || '--';

        const totalMemoryGB = (info.total_memory / 1024 / 1024 / 1024).toFixed(2);
        document.getElementById('total-memory').textContent = `${totalMemoryGB} GB`;
    } catch (error) {
        console.error('Failed to load system info:', error);
    }
}

// Load kernel metrics
async function loadKernelMetrics() {
    try {
        const metrics = await window.AutoVoiceUtils.apiRequest('/v1/kernels/metrics');
        updateKernelTable(metrics);
    } catch (error) {
        console.error('Failed to load kernel metrics:', error);
        document.getElementById('kernel-metrics').innerHTML =
            '<tr><td colspan="6" class="loading">Failed to load metrics</td></tr>';
    }
}

// Update kernel metrics table
function updateKernelTable(metrics) {
    const tbody = document.getElementById('kernel-metrics');

    if (!metrics || metrics.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="loading">No kernel metrics available</td></tr>';
        return;
    }

    let html = '';
    metrics.forEach(metric => {
        const statusClass = metric.avg_time < 10 ? 'status-optimal' :
                           metric.avg_time < 50 ? 'status-warning' : 'status-critical';

        const statusText = metric.avg_time < 10 ? 'Optimal' :
                          metric.avg_time < 50 ? 'Normal' : 'Slow';

        html += `
            <tr>
                <td>${metric.name}</td>
                <td>${metric.executions.toLocaleString()}</td>
                <td>${metric.avg_time.toFixed(3)}</td>
                <td>${metric.min_time.toFixed(3)}</td>
                <td>${metric.max_time.toFixed(3)}</td>
                <td><span class="${statusClass}">${statusText}</span></td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
}

// Add styles for status indicators
const style = document.createElement('style');
style.textContent = `
    .status-optimal { color: #48bb78; }
    .status-warning { color: #f6ad55; }
    .status-critical { color: #f56565; }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-dot.green { background: #48bb78; }
    .status-dot.yellow { background: #f6ad55; }
    .status-dot.red { background: #f56565; }
    .queue-item {
        background: var(--darker-bg);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .queue-item-info {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .progress-bar {
        height: 4px;
        background: var(--border-color);
        border-radius: 2px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: var(--gradient);
        transition: width 0.3s;
    }
    .queue-empty {
        text-align: center;
        color: var(--text-secondary);
        padding: 2rem;
    }
    .queue-empty i {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
`;
document.head.appendChild(style);

// Export update function for WebSocket
window.updateGPUCharts = updateGPUCharts;