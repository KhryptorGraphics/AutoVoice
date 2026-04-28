#!/usr/bin/env bash
# Restore AutoVoice docker-compose named volumes from backup archives.
set -euo pipefail

BACKUP_DIR="${1:?usage: scripts/restore_compose_volumes.sh <backup-dir>}"

for archive in "$BACKUP_DIR"/autovoice-*.tgz; do
  [ -e "$archive" ] || continue
  volume="$(basename "$archive" .tgz)"
  docker volume create "$volume" >/dev/null
  docker run --rm \
    -v "${volume}:/target" \
    -v "$(cd "$BACKUP_DIR" && pwd):/backup:ro" \
    alpine:3.20 \
    sh -c "rm -rf /target/* && tar -xzf /backup/$(basename "$archive") -C /target"
done

printf 'Restored AutoVoice volumes from %s\n' "$BACKUP_DIR"
