# AutoVoice Monitoring Guide

**Version**: 1.0
**Last Updated**: 2025-11-01
**Purpose**: Comprehensive monitoring setup and operational procedures

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Prometheus Setup](#prometheus-setup)
4. [Grafana Setup](#grafana-setup)
5. [Key Metrics Reference](#key-metrics-reference)
6. [Grafana Dashboard Configuration](#grafana-dashboard-configuration)
7. [PromQL Query Examples](#promql-query-examples)
8. [Alert Configuration](#alert-configuration)
9. [Health Check Monitoring](#health-check-monitoring)
10. [Log Aggregation Integration](#log-aggregation-integration)
11. [Troubleshooting Monitoring Issues](#troubleshooting-monitoring-issues)
12. [Monitoring Best Practices](#monitoring-best-practices)
13. [Maintenance Tasks](#maintenance-tasks)
14. [Integration with External Systems](#integration-with-external-systems)
15. [Appendix: Metric Definitions](#appendix-metric-definitions)

---

## Introduction

### Purpose of Monitoring AutoVoice

Monitoring is essential for:
- **Performance tracking**: Measure request rates, latency, and throughput
- **Resource management**: Track GPU, CPU, and memory usage
- **Error detection**: Identify and alert on failures
- **Capacity planning**: Understand usage patterns and scale accordingly
- **SLA compliance**: Ensure service meets performance targets

### Monitoring Stack

AutoVoice uses a modern observability stack:
- **Prometheus**: Time-series metrics collection and storage
- **Grafana**: Visualization and dashboarding
- **JSON Logs**: Structured logging for analysis
- **Docker Health Checks**: Container-level health monitoring

### Key Metrics to Track

**Application Metrics:**
- Request rate (TTS and voice conversion)
- Processing duration (p50, p95, p99)
- Error rate
- Active conversions
- Cache hit rate

**System Metrics:**
- GPU utilization and memory
- CPU and RAM usage
- Disk I/O and space
- Network traffic

**Quality Metrics:**
- Pitch accuracy
- Speaker similarity
- Audio quality scores

### Alert Thresholds and SLOs

**Service Level Objectives (SLOs):**
- Availability: 99.9% uptime
- Latency: p95 <100ms for TTS, <60s for voice conversion
- Error rate: <1% for all requests
- GPU utilization: 60-80% (optimal range)

---

## Quick Start

### Start Monitoring Stack

```bash
# Start AutoVoice with monitoring
docker-compose --profile monitoring up -d

# Verify services are running
docker-compose ps

# Expected output:
# auto_voice_app       running (healthy)
# auto_voice_redis     running (healthy)
# auto_voice_prometheus running
# auto_voice_grafana   running
```

### Access Monitoring Interfaces

**Prometheus:**
- URL: http://localhost:9090
- No authentication required (internal use)
- Check targets: http://localhost:9090/targets

**Grafana:**
- URL: http://localhost:3000
- Default credentials: admin/admin (change on first login)
- Dashboards: Navigate to Dashboards → AutoVoice

### Verify Metrics Collection

```bash
# Check metrics endpoint
curl http://localhost:5000/metrics

# Expected output:
# autovoice_synthesis_requests_total{speaker_id="default",success="true"} 42
# autovoice_http_requests_total{method="POST",endpoint="/synthesize",status="200"} 38
# autovoice_gpu_memory_used_bytes{device_id="0"} 4294967296
# autovoice_gpu_utilization_percent{device_id="0"} 75.5
# ...

# Check Prometheus is scraping
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Expected output:
# {"job": "autovoice", "health": "up"}
# {"job": "nvidia-gpu", "health": "up"}
```

### Optional: Enable Node Exporter and cAdvisor

For additional system and container metrics, you can enable node-exporter and cAdvisor:

**1. Add services to docker-compose.yml:**

```yaml
  node-exporter:
    image: prom/node-exporter:latest
    container_name: auto_voice_node_exporter
    restart: unless-stopped
    profiles: ["monitoring"]
    ports:
      - "9100:9100"
    networks:
      - auto-voice-net
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: auto_voice_cadvisor
    restart: unless-stopped
    profiles: ["monitoring"]
    ports:
      - "8080:8080"
    networks:
      - auto-voice-net
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    privileged: true
```

**2. Uncomment scrape configs in config/prometheus.yml:**

```yaml
  # Node exporter for system metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  # cAdvisor for container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

---

## Prometheus Setup

### Configuration File

**Location**: `config/prometheus.yml`

**Scrape Configuration:**
```yaml
global:
  scrape_interval: 10s
  evaluation_interval: 10s

scrape_configs:
  - job_name: 'autovoice'
    static_configs:
      - targets: ['auto-voice-app:5000']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### Scrape Targets

**AutoVoice Application:**
- Target: `auto-voice-app:5000/metrics`
- Interval: 10 seconds
- Timeout: 5 seconds
- Metrics: Application-specific metrics (TTS, VC, GPU)

**Prometheus Self-Monitoring:**
- Target: `localhost:9090/metrics`
- Interval: 10 seconds
- Metrics: Prometheus internal metrics

**Optional Targets:**
- Node Exporter: `node-exporter:9100` (system metrics) - See "Optional: Enable Node Exporter and cAdvisor" section
- cAdvisor: `cadvisor:8080` (container metrics) - See "Optional: Enable Node Exporter and cAdvisor" section
- NVIDIA GPU Exporter: `nvidia-exporter:9400` (detailed GPU metrics) - Enabled by default with monitoring profile

### GPU Metrics Sources

AutoVoice provides GPU metrics from two sources:

**1. Application Metrics (Default):**
- Metrics: `autovoice_gpu_utilization_percent`, `autovoice_gpu_memory_used_bytes`
- Source: Application code using pynvml/nvitop
- Scrape target: `auto-voice-app:5000/metrics`
- Use case: Basic GPU monitoring integrated with application metrics

**2. NVIDIA DCGM Exporter (Detailed):**
- Metrics: `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_MEM_COPY_UTIL`, `DCGM_FI_DEV_GPU_TEMP`, etc.
- Source: NVIDIA Data Center GPU Manager (DCGM)
- Scrape target: `nvidia-exporter:9400/metrics`
- Use case: Detailed GPU telemetry, temperature, power, clock speeds
- Enabled: Automatically with `--profile monitoring`

**Grafana Dashboard Configuration:**
The AutoVoice dashboard uses application metrics by default. To use NVIDIA DCGM metrics:
1. Update panel queries in `config/grafana/dashboards/autovoice-overview.json`
2. Replace `autovoice_gpu_utilization_percent` with `DCGM_FI_DEV_GPU_UTIL`
3. Replace `autovoice_gpu_memory_used_bytes` with `DCGM_FI_DEV_FB_USED` (in bytes)
4. Or create a dashboard variable to switch between sources

### Retention Policy

**Default Configuration:**
- Retention time: 30 days
- Max storage: 10GB
- Storage location: `prometheus-data` Docker volume

**Adjust Retention:**
Edit `docker-compose.yml` Prometheus command:
```yaml
command:
  - '--config.file=/etc/prometheus/prometheus.yml'
  - '--storage.tsdb.path=/prometheus'
  - '--storage.tsdb.retention.time=30d'
  - '--storage.tsdb.retention.size=10GB'
```

### Storage Location

**Docker Volume:**
```bash
# Inspect volume
docker volume inspect autovoice_prometheus-data

# Backup Prometheus data
docker run --rm -v autovoice_prometheus-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/prometheus-backup-$(date +%Y%m%d).tar.gz /data
```



---

## Grafana Setup

### Default Credentials

**Initial Login:**
- Username: `admin`
- Password: `admin` (or value from `GRAFANA_ADMIN_PASSWORD` environment variable)
- **Action Required**: Change password on first login

### Data Source Configuration

**Automatic Configuration:**
Prometheus datasource is auto-configured via provisioning:
- File: `config/grafana/datasources/prometheus.yml`
- Name: `Prometheus`
- URL: `http://prometheus:9090`
- Access: `proxy`

**Manual Configuration (if needed):**
1. Navigate to Configuration → Data Sources
2. Click "Add data source"
3. Select "Prometheus"
4. Set URL: `http://prometheus:9090`
5. Click "Save & Test"

### Dashboard Provisioning

**Automatic Loading:**
Place dashboard JSON files in `config/grafana/dashboards/`:
- `autovoice-overview.json` - Application overview
- `autovoice-performance.json` - Performance metrics
- `autovoice-gpu.json` - GPU monitoring

**Note**: GPU utilization metrics are provided by the nvidia-exporter service when enabled via the `monitoring` profile in `docker-compose.yml`. Enable the GPU exporter by uncommenting and configuring the nvidia-exporter service for detailed GPU metrics.

**Manual Import:**
1. Navigate to Dashboards → Import
2. Upload JSON file or paste JSON
3. Select Prometheus datasource
4. Click "Import"

---

## Key Metrics Reference

### Metric Naming Convention

All AutoVoice application metrics use the `autovoice_*` prefix for clear ownership and namespace isolation. This convention:
- Prevents naming conflicts with other services
- Makes it easy to identify AutoVoice metrics in Prometheus
- Follows Prometheus best practices for application-specific metrics

### Application Metrics

**Request Metrics:**
- `autovoice_synthesis_requests_total{speaker_id, success}` - Total synthesis requests (counter)
  - Labels: `speaker_id` (speaker identifier), `success` (true, false)
- `autovoice_http_requests_total{method, endpoint, status}` - Total HTTP requests (counter)
  - Labels: `method` (GET, POST), `endpoint` (path), `status` (HTTP status code)
- `autovoice_websocket_connections_total` - Total WebSocket connections (counter)
- `autovoice_active_websocket_connections` - Current active WebSocket connections (gauge)

**Duration Metrics:**
- `autovoice_synthesis_duration_seconds` - Synthesis processing duration (histogram)
  - Buckets: 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0
- `autovoice_http_request_duration_seconds{method, endpoint}` - HTTP request duration (histogram)
- `autovoice_audio_processing_duration_seconds{operation}` - Audio processing duration (histogram)
- `autovoice_model_inference_duration_seconds` - Model inference time (histogram)

**GPU Metrics:**
- `autovoice_gpu_memory_used_bytes{device_id}` - GPU memory usage in bytes (gauge)
- `autovoice_gpu_utilization_percent{device_id}` - GPU utilization percentage (gauge)
- `autovoice_gpu_temperature_celsius{device_id}` - GPU temperature (gauge)

**Model Metrics:**
- `autovoice_model_loaded` - Whether model is loaded (gauge, 1=loaded, 0=not loaded)

**Audio Processing Metrics:**
- `autovoice_audio_processing_total{operation, success}` - Total audio processing operations (counter)

### System Metrics (from Node Exporter)

**CPU:**
- `node_cpu_seconds_total{mode}` - CPU time by mode
- `node_load1`, `node_load5`, `node_load15` - Load averages

**Memory:**
- `node_memory_MemTotal_bytes` - Total memory
- `node_memory_MemAvailable_bytes` - Available memory
- `node_memory_MemFree_bytes` - Free memory

**Disk:**
- `node_disk_io_time_seconds_total` - Disk I/O time
- `node_filesystem_avail_bytes` - Available disk space
- `node_filesystem_size_bytes` - Total disk space

**Network:**
- `node_network_receive_bytes_total` - Network bytes received
- `node_network_transmit_bytes_total` - Network bytes transmitted

---

## Grafana Dashboard Configuration

### Dashboard 1: AutoVoice Overview

**Purpose**: High-level application health and performance

**Panels:**

**Row 1: Request Metrics**
1. **Request Rate** (Time Series)
   - Query: `rate(autovoice_synthesis_requests_total[5m])`
   - Y-axis: Requests/second
   - Legend: By speaker_id

2. **Error Rate** (Time Series)
   - Query: `rate(autovoice_synthesis_requests_total{success="false"}[5m]) / rate(autovoice_synthesis_requests_total[5m]) * 100`
   - Y-axis: Percentage
   - Threshold: 5% (red line)

3. **Total Requests** (Stat)
   - Query: `sum(autovoice_synthesis_requests_total)`
   - Format: Number with commas

**Row 2: Performance Metrics**
1. **Synthesis Latency** (Time Series)
   - Query p50: `histogram_quantile(0.50, rate(autovoice_synthesis_duration_seconds_bucket[5m]))`
   - Query p95: `histogram_quantile(0.95, rate(autovoice_synthesis_duration_seconds_bucket[5m]))`
   - Query p99: `histogram_quantile(0.99, rate(autovoice_synthesis_duration_seconds_bucket[5m]))`
   - Y-axis: Seconds
   - Threshold: 0.1s

2. **Active WebSocket Connections** (Gauge)
   - Query: `autovoice_active_websocket_connections`
   - Min: 0, Max: 10
   - Thresholds: Green 0-5, Yellow 5-8, Red 8-10

**Row 3: GPU Metrics**
1. **GPU Memory Usage** (Area Graph)
   - Query: `autovoice_gpu_memory_used_bytes / 1024 / 1024 / 1024`
   - Y-axis: Gigabytes
   - Threshold: 6GB (warning), 7GB (critical)

2. **GPU Utilization** (Gauge)
   - Query: `autovoice_gpu_utilization_percent`
   - Min: 0, Max: 100
   - Thresholds: Green 60-80%, Yellow 80-90%, Red >90%

---

## PromQL Query Examples

### Request Rate

**Synthesis Requests per Second:**
```promql
rate(autovoice_synthesis_requests_total[5m])
```

**HTTP Requests per Second:**
```promql
rate(autovoice_http_requests_total[5m])
```

**Total Requests per Minute:**
```promql
rate(autovoice_synthesis_requests_total[1m]) * 60
```

### Error Rate

**Error Percentage:**
```promql
(rate(autovoice_synthesis_requests_total{success="false"}[5m]) / rate(autovoice_synthesis_requests_total[5m])) * 100
```

**Errors per Minute:**
```promql
rate(autovoice_synthesis_requests_total{success="false"}[1m]) * 60
```

### Latency

**p50 Latency:**
```promql
histogram_quantile(0.50, rate(autovoice_synthesis_duration_seconds_bucket[5m]))
```

**p95 Latency:**
```promql
histogram_quantile(0.95, rate(autovoice_synthesis_duration_seconds_bucket[5m]))
```

**p99 Latency:**
```promql
histogram_quantile(0.99, rate(autovoice_synthesis_duration_seconds_bucket[5m]))
```

### GPU Metrics

**GPU Memory Usage (GB):**
```promql
autovoice_gpu_memory_used_bytes{device_id="0"} / 1024 / 1024 / 1024
```

**GPU Utilization:**
```promql
autovoice_gpu_utilization_percent{device_id="0"}
```

**GPU Temperature:**
```promql
autovoice_gpu_temperature_celsius{device_id="0"}
```


---

## Alert Configuration

### Critical Alerts

**Service Down:**
```yaml
- alert: AutoVoiceDown
  expr: up{job="autovoice"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "AutoVoice service is down"
    description: "AutoVoice has been down for more than 1 minute"
```

**High Error Rate:**
```yaml
- alert: HighErrorRate
  expr: (rate(autovoice_synthesis_requests_total{success="false"}[5m]) / rate(autovoice_synthesis_requests_total[5m])) * 100 > 5
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"
    description: "Error rate is {{ $value }}% (threshold: 5%)"
```

**GPU High Utilization:**
```yaml
- alert: GPUHighUtilization
  expr: autovoice_gpu_utilization_percent{device_id="0"} > 95
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "GPU utilization critically high"
    description: "GPU utilization is {{ $value }}%"
```

**High Latency:**
```yaml
- alert: HighLatency
  expr: histogram_quantile(0.95, rate(autovoice_synthesis_duration_seconds_bucket[5m])) > 0.5
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High synthesis latency detected"
    description: "p95 latency is {{ $value }}s (threshold: 0.5s)"
```

### Warning Alerts

**Elevated Error Rate:**
```yaml
- alert: ElevatedErrorRate
  expr: (rate(autovoice_synthesis_requests_total{success="false"}[5m]) / rate(autovoice_synthesis_requests_total[5m])) * 100 > 2
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Elevated error rate"
    description: "Error rate is {{ $value }}% (threshold: 2%)"
```

**High GPU Utilization:**
```yaml
- alert: HighGPUUtilization
  expr: autovoice_gpu_utilization_percent > 85
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "High GPU utilization"
    description: "GPU utilization is {{ $value }}%"
```

**Model Not Loaded:**
```yaml
- alert: ModelNotLoaded
  expr: autovoice_model_loaded == 0
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Model not loaded"
    description: "AutoVoice model is not loaded"
```

---

## Troubleshooting Monitoring Issues

### Prometheus Not Scraping

**Symptoms:**
- Targets show as "DOWN" in Prometheus UI
- No metrics data in Grafana

**Diagnosis:**
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health, lastError: .lastError}'

# Check network connectivity
docker exec auto_voice_prometheus ping auto-voice-app

# Check metrics endpoint directly
curl http://localhost:5000/metrics
```

**Solutions:**
1. Verify AutoVoice service is running: `docker ps | grep auto-voice-app`
2. Check Prometheus configuration: `cat config/prometheus.yml`
3. Restart Prometheus: `docker-compose restart prometheus`
4. Check Prometheus logs: `docker logs auto_voice_prometheus`

### Grafana Not Showing Data

**Symptoms:**
- Dashboards show "No data"
- Queries return empty results

**Diagnosis:**
```bash
# Test Prometheus data source in Grafana
# Navigate to: Configuration → Data Sources → Prometheus → Test

# Test query directly in Prometheus
curl 'http://localhost:9090/api/v1/query?query=up{job="autovoice"}'

# Check Grafana logs
docker logs auto_voice_grafana
```

**Solutions:**
1. Verify Prometheus data source URL: `http://prometheus:9090`
2. Check time range in Grafana (ensure it covers recent data)
3. Test PromQL query in Prometheus UI first
4. Verify metrics are being collected: `curl http://localhost:5000/metrics`

### Missing Metrics

**Symptoms:**
- Specific metrics not appearing
- Incomplete metric data

**Diagnosis:**
```bash
# List all available metrics
curl http://localhost:5000/metrics | grep autovoice

# Check if metrics are being scraped
curl 'http://localhost:9090/api/v1/label/__name__/values' | jq '.data[] | select(startswith("autovoice"))'
```

**Solutions:**
1. Verify metrics are exposed: `curl http://localhost:5000/metrics | grep <metric_name>`
2. Check Prometheus scrape interval (default: 10s)
3. Restart AutoVoice service: `docker-compose restart auto-voice-app`
4. Check for metric naming conflicts

---

## Maintenance Tasks

### Daily Tasks

- Review Grafana dashboards for anomalies
- Check alert status in Prometheus
- Verify all targets are up: http://localhost:9090/targets
- Review error logs: `docker logs auto_voice_app | grep ERROR`

### Weekly Tasks

- Review metric retention and storage usage
- Analyze slow queries and optimize
- Update dashboard thresholds based on trends
- Clean old Prometheus data if needed: `docker volume ls | grep prometheus`

### Monthly Tasks

- Review and update alert rules
- Archive old metrics if needed
- Update Grafana dashboards with new metrics
- Review monitoring documentation

---

## Best Practices

### Metric Collection

- **Use histograms for latency metrics** (not averages)
- **Include labels for filtering** (status, endpoint, user_id)
- **Avoid high-cardinality labels** (e.g., request_id, timestamp)
- **Set appropriate bucket boundaries** for histograms

### Dashboard Design

- **Use consistent time ranges** across panels
- **Include threshold lines** for SLOs
- **Use appropriate visualization types** (line, gauge, heatmap)
- **Add annotations** for deployments and incidents

### Alert Configuration

- **Set appropriate thresholds** based on SLOs
- **Use "for" duration** to avoid flapping alerts
- **Include runbook links** in alert annotations
- **Test alerts** before deploying to production

---

## Summary

This monitoring guide provides comprehensive setup and operational procedures for AutoVoice production deployment. Key components:

- ✅ Prometheus for metrics collection
- ✅ Grafana for visualization
- ✅ Pre-configured dashboards with timeseries panels (Grafana 8+)
- ✅ Alert rules for critical issues
- ✅ Troubleshooting procedures
- ✅ Maintenance schedules
- ✅ Comprehensive metrics reference with dashboard mapping

For additional information, refer to:
- `docs/metrics-reference.md` - Complete metrics-to-dashboard mapping
- `docs/runbook.md` - Operational procedures
- `docs/production_readiness_checklist.md` - Deployment checklist
- `config/prometheus.yml` - Prometheus configuration
- `config/grafana/` - Grafana provisioning

---

**Document Version**: 1.1
**Last Updated**: 2025-11-07
**Maintained By**: AutoVoice Operations Team
