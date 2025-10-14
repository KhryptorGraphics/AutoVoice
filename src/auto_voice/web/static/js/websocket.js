// WebSocket connection handler for real-time updates

class WebSocketManager {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 5000;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.eventHandlers = {};
        this.isConnected = false;
        this.connectWebSocket();
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
                this.sendMessage('subscribe', { channels: ['gpu_stats', 'processing_status'] });
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.attemptReconnect();
            };
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.attemptReconnect();
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

            setTimeout(() => {
                this.connectWebSocket();
            }, this.reconnectInterval);
        } else {
            console.error('Max reconnection attempts reached');
            this.updateConnectionStatus(false, 'Failed to connect');
        }
    }

    sendMessage(type, data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: type,
                data: data,
                timestamp: Date.now()
            }));
        } else {
            console.warn('WebSocket is not connected');
        }
    }

    handleMessage(message) {
        const { type, data } = message;

        // Trigger registered event handlers
        if (this.eventHandlers[type]) {
            this.eventHandlers[type].forEach(handler => handler(data));
        }

        // Handle specific message types
        switch (type) {
            case 'gpu_stats':
                this.updateGPUStats(data);
                break;
            case 'processing_status':
                this.updateProcessingStatus(data);
                break;
            case 'kernel_metrics':
                this.updateKernelMetrics(data);
                break;
            case 'audio_progress':
                this.updateAudioProgress(data);
                break;
            case 'error':
                this.handleError(data);
                break;
        }
    }

    updateGPUStats(stats) {
        // Update GPU utilization
        const gpuUtilElement = document.getElementById('gpu-utilization');
        if (gpuUtilElement) {
            gpuUtilElement.textContent = `${stats.utilization}%`;
        }

        // Update memory usage
        const gpuMemoryElement = document.getElementById('gpu-memory');
        if (gpuMemoryElement) {
            const memoryUsedGB = (stats.memory_used / 1024).toFixed(2);
            const memoryTotalGB = (stats.memory_total / 1024).toFixed(2);
            gpuMemoryElement.textContent = `${memoryUsedGB} / ${memoryTotalGB} GB`;
        }

        // Update temperature
        const gpuTempElement = document.getElementById('gpu-temperature');
        if (gpuTempElement) {
            gpuTempElement.textContent = `${stats.temperature}°C`;
        }

        // Update throughput
        const throughputElement = document.getElementById('audio-throughput');
        if (throughputElement && stats.audio_throughput) {
            throughputElement.textContent = `${stats.audio_throughput.toLocaleString()}`;
        }

        // Trigger chart updates if on dashboard
        if (window.updateGPUCharts) {
            window.updateGPUCharts(stats);
        }
    }

    updateProcessingStatus(status) {
        const queueElement = document.getElementById('processing-queue');
        if (!queueElement) return;

        if (status.tasks && status.tasks.length > 0) {
            let queueHTML = '';
            status.tasks.forEach(task => {
                const progressWidth = task.progress || 0;
                queueHTML += `
                    <div class="queue-item">
                        <div class="queue-item-info">
                            <span class="queue-item-name">${task.name}</span>
                            <span class="queue-item-status">${task.status}</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${progressWidth}%"></div>
                        </div>
                    </div>
                `;
            });
            queueElement.innerHTML = queueHTML;
        } else {
            queueElement.innerHTML = `
                <div class="queue-empty">
                    <i class="fas fa-inbox"></i>
                    <p>No active processing tasks</p>
                </div>
            `;
        }
    }

    updateKernelMetrics(metrics) {
        const metricsTable = document.getElementById('kernel-metrics');
        if (!metricsTable) return;

        let tableHTML = '';
        metrics.forEach(metric => {
            const statusColor = metric.status === 'optimal' ? 'green' :
                               metric.status === 'warning' ? 'yellow' : 'red';

            tableHTML += `
                <tr>
                    <td>${metric.name}</td>
                    <td>${metric.executions}</td>
                    <td>${metric.avg_time.toFixed(3)}</td>
                    <td>${metric.min_time.toFixed(3)}</td>
                    <td>${metric.max_time.toFixed(3)}</td>
                    <td><span class="status-dot ${statusColor}"></span> ${metric.status}</td>
                </tr>
            `;
        });

        metricsTable.innerHTML = tableHTML || '<tr><td colspan="6" class="loading">No metrics available</td></tr>';
    }

    updateAudioProgress(progress) {
        // Update processing progress for voice synthesis
        const statusText = document.getElementById('status-text');
        if (statusText) {
            statusText.textContent = `${progress.stage} - ${progress.percent}%`;
        }

        // Update any progress bars
        const progressBars = document.querySelectorAll('.synthesis-progress');
        progressBars.forEach(bar => {
            bar.style.width = `${progress.percent}%`;
        });
    }

    handleError(error) {
        console.error('WebSocket error message:', error);
        if (window.AutoVoiceUtils) {
            window.AutoVoiceUtils.showNotification(error.message, 'error');
        }
    }

    updateConnectionStatus(connected, message = null) {
        const statusElement = document.getElementById('ws-status-text');
        if (statusElement) {
            if (connected) {
                statusElement.textContent = 'Connected';
                statusElement.parentElement.style.color = '#48bb78';
            } else {
                statusElement.textContent = message || 'Disconnected';
                statusElement.parentElement.style.color = '#f56565';
            }
        }
    }

    // Register event handler
    on(eventType, handler) {
        if (!this.eventHandlers[eventType]) {
            this.eventHandlers[eventType] = [];
        }
        this.eventHandlers[eventType].push(handler);
    }

    // Remove event handler
    off(eventType, handler) {
        if (this.eventHandlers[eventType]) {
            this.eventHandlers[eventType] = this.eventHandlers[eventType].filter(h => h !== handler);
        }
    }

    // Close connection
    close() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

// Initialize WebSocket manager
let wsManager = new WebSocketManager();

// Export for global use
window.wsManager = wsManager;