# AutoVoice Log Rotation Policy

**Version**: 1.0  
**Last Updated**: 2025-11-01  
**Purpose**: Document log rotation configuration and operational procedures

---

## Overview

### Purpose of Log Rotation

Log rotation prevents disk space exhaustion by:
- Automatically archiving old log files
- Limiting total disk usage
- Maintaining performance
- Complying with retention policies

### Implementation

AutoVoice uses Python's `logging.handlers.RotatingFileHandler` for automatic log rotation.

---

## Current Configuration

### Log Files

**Application Log: `logs/autovoice.log`**
- Handler: `file` (RotatingFileHandler)
- Level: DEBUG
- Format: JSON (structured logging)
- Max size: 10MB (10,485,760 bytes)
- Backup count: 5 files
- Total retention: ~50MB (10MB × 5 backups)
- Encoding: UTF-8

**Error Log: `logs/error.log`**
- Handler: `error_file` (RotatingFileHandler)
- Level: ERROR
- Format: JSON (structured logging)
- Max size: 10MB (10,485,760 bytes)
- Backup count: 5 files
- Total retention: ~50MB (10MB × 5 backups)
- Encoding: UTF-8

### Rotation Behavior

- **Automatic rotation** when file reaches 10MB
- Rotated files named: `autovoice.log.1`, `autovoice.log.2`, ..., `autovoice.log.5`
- Oldest file (`.5`) is deleted when new rotation occurs
- Rotation happens synchronously during log write

### Configuration Location

**File**: `config/logging_config.yaml`

```yaml
handlers:
  file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/autovoice.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: json
    encoding: utf-8
```

---

## Log File Naming Convention

### Active Files
- `logs/autovoice.log` - Current application log
- `logs/error.log` - Current error log

### Rotated Files
- `logs/autovoice.log.1` - Most recent backup (1 rotation ago)
- `logs/autovoice.log.2` - 2 rotations ago
- `logs/autovoice.log.3` - 3 rotations ago
- `logs/autovoice.log.4` - 4 rotations ago
- `logs/autovoice.log.5` - Oldest backup (5 rotations ago)

---

## Retention Policy

### Space-Based Retention
- Maximum disk usage per log type: 50MB
- Total for both logs: ~100MB
- Automatic cleanup when limit reached

### Time-Based Retention
- Retention period: Depends on log volume
- Typical retention: 1-7 days for high-volume applications
- Calculation: (10MB × 5 backups) / (average MB per day)

---

## Log Format

### JSON Format (Production)
```json
{
  "asctime": "2025-11-01 14:30:45",
  "name": "auto_voice.inference",
  "levelname": "INFO",
  "filename": "singing_conversion_pipeline.py",
  "lineno": 123,
  "message": "Conversion completed successfully",
  "processing_time": 45.2,
  "user_id": "user123",
  "conversion_id": "conv-550e8400"
}
```

---

## Monitoring Log Rotation

### Check Current Log Sizes
```bash
ls -lh logs/
```

### Check Total Disk Usage
```bash
du -sh logs/
```

### Monitor Rotation Events
```bash
watch -n 5 'ls -lh logs/autovoice.log*'
```

---

## Operational Procedures

### Manual Log Rotation (if needed)
```bash
# Stop application
sudo systemctl stop autovoice

# Manually rotate logs
mv logs/autovoice.log.4 logs/autovoice.log.5
mv logs/autovoice.log.3 logs/autovoice.log.4
mv logs/autovoice.log.2 logs/autovoice.log.3
mv logs/autovoice.log.1 logs/autovoice.log.2
mv logs/autovoice.log logs/autovoice.log.1
touch logs/autovoice.log

# Start application
sudo systemctl start autovoice
```

### Log Archival
```bash
# Create archive directory
mkdir -p logs/archive/$(date +%Y%m)

# Move rotated logs to archive
mv logs/autovoice.log.* logs/archive/$(date +%Y%m)/
mv logs/error.log.* logs/archive/$(date +%Y%m)/

# Compress archives
tar -czf logs/archive/logs_$(date +%Y%m%d).tar.gz logs/archive/$(date +%Y%m)/
rm -rf logs/archive/$(date +%Y%m)/
```

---

## Best Practices

### Configuration
- Set max file size based on disk space and retention needs
- Use JSON format for production (easier parsing)
- Separate error logs for quick troubleshooting
- Use appropriate log levels (DEBUG for dev, INFO for prod)

### Monitoring
- Monitor disk usage: Alert when >80% full
- Track log volume: Unusual spikes may indicate issues
- Review error logs daily
- Archive old logs regularly

### Security
- Restrict log file permissions: `chmod 640 logs/*.log`
- Avoid logging sensitive data (passwords, tokens, PII)
- Sanitize user inputs in log messages
- Encrypt archived logs if containing sensitive data

---

## Docker Considerations

### Volume Mounts
Ensure logs directory is mounted in `docker-compose.yml`:
```yaml
volumes:
  - ./logs:/app/logs
```

### Docker Logging Driver
Docker Compose configures JSON file driver:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### Combined Retention
- Application logs: 50MB (5 × 10MB)
- Docker logs: 30MB (3 × 10MB)
- Total per container: ~80MB

---

## Summary

**Current Policy:**
- ✅ Automatic rotation at 10MB
- ✅ 5 backup files per log type
- ✅ ~50MB retention per log type
- ✅ JSON format for structured logging
- ✅ Separate error logs
- ✅ UTF-8 encoding

**Recommendations:**
- ✅ Policy is appropriate for production
- ⚠️ Consider increasing retention if disk space allows
- ⚠️ Implement automated archival for long-term storage
- ⚠️ Set up log aggregation for multi-instance deployments

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-01  
**Maintained By**: AutoVoice Operations Team

