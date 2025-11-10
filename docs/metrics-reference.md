# AutoVoice Metrics Reference

**Version**: 1.0  
**Last Updated**: 2025-11-07  
**Purpose**: Complete mapping of Prometheus metrics to Grafana dashboard panels

---

## Metric Naming Convention

All AutoVoice application metrics use the `autovoice_*` prefix for clear ownership and namespace isolation.

---

## Application Metrics

### HTTP Request Metrics

| Metric Name | Type | Labels | Description | Dashboard Panel |
|------------|------|--------|-------------|-----------------|
| `autovoice_http_requests_total` | Counter | `method`, `endpoint`, `status` | Total HTTP requests | Panel 1: HTTP Requests per Second |
| `autovoice_http_request_duration_seconds` | Histogram | `method`, `endpoint` | HTTP request duration | Panel 4: Synthesis Duration (p95) |

**PromQL Examples:**
```promql
# Request rate
rate(autovoice_http_requests_total[5m])

# Error rate (5xx errors)
rate(autovoice_http_requests_total{status=~"5.."}[5m])
```

---

### WebSocket Metrics

| Metric Name | Type | Labels | Description | Dashboard Panel |
|------------|------|--------|-------------|-----------------|
| `autovoice_websocket_connections_total` | Counter | - | Total WebSocket connections | - |
| `autovoice_active_websocket_connections` | Gauge | - | Current active WebSocket connections | Panel 2: Active WebSocket Connections |
| `autovoice_websocket_events_total` | Counter | `event_type` | Total WebSocket events | - |

**PromQL Examples:**
```promql
# Active connections
autovoice_active_websocket_connections

# Connection rate
rate(autovoice_websocket_connections_total[5m])
```

---

### Synthesis Metrics

| Metric Name | Type | Labels | Description | Dashboard Panel |
|------------|------|--------|-------------|-----------------|
| `autovoice_synthesis_requests_total` | Counter | `speaker_id`, `success` | Total synthesis requests | - |
| `autovoice_synthesis_duration_seconds` | Histogram | - | Synthesis operation duration | Panel 4: Synthesis Duration (p95) |

**PromQL Examples:**
```promql
# Synthesis request rate
rate(autovoice_synthesis_requests_total[5m])

# p95 latency
histogram_quantile(0.95, rate(autovoice_synthesis_duration_seconds_bucket[5m]))

# Error rate
rate(autovoice_synthesis_requests_total{success="false"}[5m]) / rate(autovoice_synthesis_requests_total[5m]) * 100
```

---

### Audio Processing Metrics

| Metric Name | Type | Labels | Description | Dashboard Panel |
|------------|------|--------|-------------|-----------------|
| `autovoice_audio_processing_total` | Counter | `operation`, `success` | Total audio processing operations | - |
| `autovoice_audio_processing_duration_seconds` | Histogram | `operation` | Audio processing duration | - |

**PromQL Examples:**
```promql
# Processing rate by operation
rate(autovoice_audio_processing_total[5m])

# Processing duration by operation
histogram_quantile(0.95, rate(autovoice_audio_processing_duration_seconds_bucket[5m]))
```

---

### Model Metrics

| Metric Name | Type | Labels | Description | Dashboard Panel |
|------------|------|--------|-------------|-----------------|
| `autovoice_model_inference_duration_seconds` | Histogram | - | Model inference time | - |
| `autovoice_model_loaded` | Gauge | - | Whether model is loaded (1=loaded, 0=not loaded) | - |

**PromQL Examples:**
```promql
# Model loaded status
autovoice_model_loaded

# Inference duration p95
histogram_quantile(0.95, rate(autovoice_model_inference_duration_seconds_bucket[5m]))
```

---

### GPU Metrics

| Metric Name | Type | Labels | Description | Dashboard Panel |
|------------|------|--------|-------------|-----------------|
| `autovoice_gpu_memory_used_bytes` | Gauge | `device_id` | GPU memory usage in bytes | Panel 5: GPU Memory Usage |
| `autovoice_gpu_utilization_percent` | Gauge | `device_id` | GPU utilization percentage | Panel 3: GPU Utilization |
| `autovoice_gpu_temperature_celsius` | Gauge | `device_id` | GPU temperature in Celsius | - |

**PromQL Examples:**
```promql
# GPU memory in GB
autovoice_gpu_memory_used_bytes{device_id="0"} / (1024*1024*1024)

# GPU utilization
autovoice_gpu_utilization_percent{device_id="0"}

# GPU temperature
autovoice_gpu_temperature_celsius{device_id="0"}
```

---

## Dashboard Panel Mapping

### Panel 1: HTTP Requests per Second
- **Type**: Timeseries
- **Metric**: `rate(autovoice_http_requests_total[5m])`
- **Unit**: requests per second (reqps)
- **Legend**: `{{method}} {{endpoint}} ({{status}})`

### Panel 2: Active WebSocket Connections
- **Type**: Stat
- **Metric**: `autovoice_active_websocket_connections`
- **Unit**: short
- **Thresholds**: Green (0-5), Yellow (5-8), Red (8+)

### Panel 3: GPU Utilization
- **Type**: Timeseries
- **Metric**: `autovoice_gpu_utilization_percent`
- **Unit**: percent
- **Range**: 0-100%
- **Thresholds**: Green (0-80%), Yellow (80-90%), Red (90-100%)

### Panel 4: Synthesis Duration (p95)
- **Type**: Stat
- **Metric**: `histogram_quantile(0.95, rate(autovoice_synthesis_duration_seconds_bucket[5m]))`
- **Unit**: seconds
- **Thresholds**: Green (<0.1s), Yellow (0.1-0.5s), Red (>0.5s)

### Panel 5: GPU Memory Usage
- **Type**: Timeseries
- **Metric**: `autovoice_gpu_memory_used_bytes / (1024*1024*1024)`
- **Unit**: decgbytes (GB)
- **Thresholds**: Green (0-6GB), Yellow (6-7GB), Red (>7GB)

### Panel 6: Error Rate
- **Type**: Timeseries
- **Metric**: `rate(autovoice_http_requests_total{status=~"5.."}[5m])`
- **Unit**: requests per second (reqps)
- **Thresholds**: Green (<0.1), Yellow (0.1-1), Red (>1)

---

## NVIDIA DCGM Exporter Metrics (Optional)

When using the `nvidia-exporter` service, additional detailed GPU metrics are available:

| Metric Name | Description |
|------------|-------------|
| `DCGM_FI_DEV_GPU_UTIL` | GPU utilization percentage |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory copy utilization |
| `DCGM_FI_DEV_FB_USED` | Frame buffer memory used (bytes) |
| `DCGM_FI_DEV_GPU_TEMP` | GPU temperature |
| `DCGM_FI_DEV_POWER_USAGE` | Power usage (watts) |
| `DCGM_FI_DEV_SM_CLOCK` | SM clock frequency |
| `DCGM_FI_DEV_MEM_CLOCK` | Memory clock frequency |

**Note**: To use DCGM metrics instead of application metrics, update dashboard queries accordingly.

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-07  
**Maintained By**: AutoVoice Operations Team

