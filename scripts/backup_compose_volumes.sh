#!/usr/bin/env bash
# Archive AutoVoice docker-compose named volumes for operator-controlled backups.
set -euo pipefail

OUTPUT_DIR="${1:-reports/backups/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUTPUT_DIR"

volumes=(
  autovoice-app-state
  autovoice-profiles
  autovoice-samples
  autovoice-trained-models
  autovoice-checkpoints
  autovoice-training-vocals
  autovoice-youtube-audio
  autovoice-separated-youtube
  autovoice-diarized-youtube
  autovoice-uploads
  autovoice-outputs
  autovoice-swarm-runs
  autovoice-swarm-memory
  autovoice-logs
)

for volume in "${volumes[@]}"; do
  docker run --rm \
    -v "${volume}:/source:ro" \
    -v "$(pwd)/${OUTPUT_DIR}:/backup" \
    alpine:3.20 \
    sh -c "cd /source && tar -czf /backup/${volume}.tgz ."
done

printf 'Backed up %s volumes to %s\n' "${#volumes[@]}" "$OUTPUT_DIR"
