#!/usr/bin/env bash
# Backup Postgres + list MinIO volume health for a compose install. Operator-owned off-box copy.
# Usage: ./bin/backup.sh [outdir]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
OUT="${1:-$HERE/../../../.local/backups}"
mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PROJECT="${COMPOSE_PROJECT:-vexa-v012}"
PG=$(docker compose -p "$PROJECT" -f docker-compose.yml ps -q postgres)
if [ -z "$PG" ]; then
  echo "✗ postgres container not running (project=$PROJECT)" >&2
  exit 1
fi
FILE="$OUT/vexa-pg-${STAMP}.sql.gz"
echo "→ pg_dump → $FILE"
docker exec -t "$PG" pg_dump -U postgres -d vexa | gzip > "$FILE"
echo "✓ postgres dump $(du -h "$FILE" | awk '{print $1}')"
# MinIO: document volume name; optional mc if available
echo "→ MinIO volume (copy off-box separately):"
docker volume ls --format '{{.Name}}' | grep -E "${PROJECT}.*minio|minio-data" || true
echo "  docker run --rm -v ${PROJECT}_minio-data:/data -v \"$OUT\":/out alpine tar czf /out/minio-${STAMP}.tgz -C /data ."
echo "✓ backup helper finished — encrypt and copy $OUT off this host"
