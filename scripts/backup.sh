#!/bin/bash
set -e

# Configuration
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"

echo "[INFO] Starting backup for $DATE..."

# 1. Postgres Backup (Compressed)
if [ -f "$POSTGRES_PASSWORD_FILE" ]; then
    export PGPASSWORD=$(cat "$POSTGRES_PASSWORD_FILE")
else
    export PGPASSWORD="$POSTGRES_PASSWORD"
fi

# Default values if env vars are not set
PG_USER=${POSTGRES_USER:-rag_user}
PG_DB=${POSTGRES_DB:-rag_db}

if pg_isready -h postgres -U "$PG_USER" -d "$PG_DB" > /dev/null 2>&1; then
    echo "[INFO] Backing up Postgres..."
    pg_dump -h postgres -U "$PG_USER" -d "$PG_DB" | gzip > "$BACKUP_DIR/db_$DATE.sql.gz"
else
    echo "[WARN] Postgres not available, skipping DB backup."
fi

# 2. Redis Backup
# Redis saves dump.rdb to /data/dump.rdb (mapped to volume)
# We just copy it to the backup folder
echo "[INFO] Backing up Redis..."
if [ -f "/var/lib/redis/dump.rdb" ]; then
    cp "/var/lib/redis/dump.rdb" "$BACKUP_DIR/redis_$DATE.rdb"
else
    echo "[WARN] Redis dump.rdb not found."
fi

# 3. ChromaDB Backup
echo "[INFO] Backing up ChromaDB..."
if [ -d "/chroma_data" ]; then
    tar -czf "$BACKUP_DIR/chroma_$DATE.tar.gz" -C /chroma_data .
else
    echo "[WARN] ChromaDB data directory not found."
fi

# 4. Cleanup Old Backups
echo "[INFO] Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -type f -mtime +$RETENTION_DAYS -delete

echo "[SUCCESS] Backup completed: $DATE"
ls -lh "$BACKUP_DIR"
