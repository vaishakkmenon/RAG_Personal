#!/bin/bash
set -e

# =============================================================================
# Disaster Recovery - Restore Script
# Restores PostgreSQL, Redis, and ChromaDB from backups
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="${BACKUP_DIR:-./backups}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
CHROMA_DATA_DIR="${CHROMA_DATA_DIR:-./data/chroma}"

# Helper functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --date DATE     Restore from specific date (format: YYYYMMDD_HHMMSS)"
    echo "  -l, --list          List available backups"
    echo "  -f, --force         Skip confirmation prompt"
    echo "  --dry-run           Show what would be restored without making changes"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --list                    # List available backups"
    echo "  $0                           # Restore from latest backup (interactive)"
    echo "  $0 -d 20251223_030000        # Restore from specific backup"
    echo "  $0 -d 20251223_030000 -f     # Restore without confirmation"
    exit 0
}

list_backups() {
    log_info "Available backups in $BACKUP_DIR:"
    echo ""
    echo "PostgreSQL backups:"
    ls -lh "$BACKUP_DIR"/db_*.sql.gz 2>/dev/null || echo "  (none found)"
    echo ""
    echo "Redis backups:"
    ls -lh "$BACKUP_DIR"/redis_*.rdb 2>/dev/null || echo "  (none found)"
    echo ""
    echo "ChromaDB backups:"
    ls -lh "$BACKUP_DIR"/chroma_*.tar.gz 2>/dev/null || echo "  (none found)"
    exit 0
}

# Parse arguments
DATE=""
FORCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--date) DATE="$2"; shift 2 ;;
        -l|--list) list_backups ;;
        -f|--force) FORCE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) log_error "Unknown option: $1"; usage ;;
    esac
done

# Verify backup directory exists
if [ ! -d "$BACKUP_DIR" ]; then
    log_error "Backup directory not found: $BACKUP_DIR"
    exit 1
fi

# Find latest backup date if not specified
if [ -z "$DATE" ]; then
    LATEST_DB=$(ls -t "$BACKUP_DIR"/db_*.sql.gz 2>/dev/null | head -1)
    if [ -z "$LATEST_DB" ]; then
        log_error "No database backups found in $BACKUP_DIR"
        exit 1
    fi
    DATE=$(basename "$LATEST_DB" | sed 's/db_\(.*\)\.sql\.gz/\1/')
    log_info "Using latest backup: $DATE"
fi

# Verify backup files exist
DB_BACKUP="$BACKUP_DIR/db_${DATE}.sql.gz"
REDIS_BACKUP="$BACKUP_DIR/redis_${DATE}.rdb"
CHROMA_BACKUP="$BACKUP_DIR/chroma_${DATE}.tar.gz"

log_info "Checking backup files for date: $DATE"

MISSING_FILES=false
if [ -f "$DB_BACKUP" ]; then
    log_info "  PostgreSQL: $(ls -lh "$DB_BACKUP" | awk '{print $5}')"
else
    log_warn "  PostgreSQL: NOT FOUND - $DB_BACKUP"
    MISSING_FILES=true
fi

if [ -f "$REDIS_BACKUP" ]; then
    log_info "  Redis: $(ls -lh "$REDIS_BACKUP" | awk '{print $5}')"
else
    log_warn "  Redis: NOT FOUND - $REDIS_BACKUP"
    MISSING_FILES=true
fi

if [ -f "$CHROMA_BACKUP" ]; then
    log_info "  ChromaDB: $(ls -lh "$CHROMA_BACKUP" | awk '{print $5}')"
else
    log_warn "  ChromaDB: NOT FOUND - $CHROMA_BACKUP"
    MISSING_FILES=true
fi

if [ "$MISSING_FILES" = true ]; then
    log_warn "Some backup files are missing. Partial restore will be attempted."
fi

# Dry run - just show what would happen
if [ "$DRY_RUN" = true ]; then
    echo ""
    log_info "=== DRY RUN - No changes will be made ==="
    echo "Would restore from backup dated: $DATE"
    echo "Steps that would be performed:"
    echo "  1. Stop api and caddy services"
    [ -f "$DB_BACKUP" ] && echo "  2. Restore PostgreSQL from $DB_BACKUP"
    [ -f "$REDIS_BACKUP" ] && echo "  3. Stop redis, restore from $REDIS_BACKUP, start redis"
    [ -f "$CHROMA_BACKUP" ] && echo "  4. Restore ChromaDB from $CHROMA_BACKUP"
    echo "  5. Restart all services"
    echo "  6. Run health check"
    exit 0
fi

# Confirmation prompt
if [ "$FORCE" = false ]; then
    echo ""
    echo "=========================================="
    echo "  WARNING: DESTRUCTIVE OPERATION"
    echo "=========================================="
    echo ""
    echo "This will restore from backup: $DATE"
    echo "Current data will be OVERWRITTEN."
    echo ""
    read -p "Type 'RESTORE' to confirm: " CONFIRM
    if [ "$CONFIRM" != "RESTORE" ]; then
        log_info "Restore cancelled."
        exit 0
    fi
fi

echo ""
log_info "Starting restoration process..."
RESTORE_START=$(date +%s)

# Step 1: Stop application services (keep databases running for restore)
log_info "Step 1/6: Stopping application services..."
docker compose -f "$COMPOSE_FILE" stop api caddy 2>/dev/null || true

# Step 2: Restore PostgreSQL
if [ -f "$DB_BACKUP" ]; then
    log_info "Step 2/6: Restoring PostgreSQL..."

    # Verify backup integrity
    if ! gzip -t "$DB_BACKUP" 2>/dev/null; then
        log_error "PostgreSQL backup is corrupted!"
        exit 1
    fi

    # Drop and recreate database, then restore
    gunzip -c "$DB_BACKUP" | docker compose -f "$COMPOSE_FILE" exec -T postgres \
        psql -U rag_user -d postgres -c "DROP DATABASE IF EXISTS rag_db;" 2>/dev/null || true

    docker compose -f "$COMPOSE_FILE" exec -T postgres \
        psql -U rag_user -d postgres -c "CREATE DATABASE rag_db;" 2>/dev/null || true

    gunzip -c "$DB_BACKUP" | docker compose -f "$COMPOSE_FILE" exec -T postgres \
        psql -U rag_user -d rag_db

    log_info "  PostgreSQL restored successfully."
else
    log_warn "Step 2/6: Skipping PostgreSQL (backup not found)"
fi

# Step 3: Restore Redis
if [ -f "$REDIS_BACKUP" ]; then
    log_info "Step 3/6: Restoring Redis..."

    # Stop Redis to replace the dump file
    docker compose -f "$COMPOSE_FILE" stop redis

    # Get the Redis data volume mount point
    REDIS_CONTAINER=$(docker compose -f "$COMPOSE_FILE" ps -q redis 2>/dev/null)
    if [ -n "$REDIS_CONTAINER" ]; then
        # Copy backup to volume via temporary container
        docker run --rm \
            -v "$(docker volume ls -q | grep redis_data | head -1)":/data \
            -v "$(realpath "$REDIS_BACKUP")":/backup.rdb:ro \
            alpine sh -c "cp /backup.rdb /data/dump.rdb && chown 999:999 /data/dump.rdb"
    else
        # Fallback: copy directly if volume is bind-mounted
        cp "$REDIS_BACKUP" ./data/redis/dump.rdb 2>/dev/null || \
            log_warn "Could not copy Redis backup directly"
    fi

    # Restart Redis
    docker compose -f "$COMPOSE_FILE" start redis
    sleep 2  # Give Redis time to load the dump

    log_info "  Redis restored successfully."
else
    log_warn "Step 3/6: Skipping Redis (backup not found)"
fi

# Step 4: Restore ChromaDB
if [ -f "$CHROMA_BACKUP" ]; then
    log_info "Step 4/6: Restoring ChromaDB..."

    # Verify backup integrity
    if ! tar -tzf "$CHROMA_BACKUP" >/dev/null 2>&1; then
        log_error "ChromaDB backup is corrupted!"
        exit 1
    fi

    # Clear existing data and restore
    if [ -d "$CHROMA_DATA_DIR" ]; then
        rm -rf "${CHROMA_DATA_DIR:?}"/*
        tar -xzf "$CHROMA_BACKUP" -C "$CHROMA_DATA_DIR"
        log_info "  ChromaDB restored successfully."
    else
        log_error "ChromaDB data directory not found: $CHROMA_DATA_DIR"
    fi
else
    log_warn "Step 4/6: Skipping ChromaDB (backup not found)"
fi

# Step 5: Restart all services
log_info "Step 5/6: Restarting all services..."
docker compose -f "$COMPOSE_FILE" up -d

# Step 6: Health check (using Docker's health status)
log_info "Step 6/6: Running health checks..."
sleep 5  # Give services time to start

MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    STATUS=$(docker inspect --format="{{.State.Health.Status}}" rag_api 2>/dev/null)
    if [ "$STATUS" = "healthy" ]; then
        log_info "  API health check: PASSED (Docker reports healthy)"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Waiting for API... ($RETRY_COUNT/$MAX_RETRIES) - Status: $STATUS"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log_error "  API health check: FAILED after ${MAX_RETRIES} attempts"
    log_error "  Check logs with: docker compose logs api"
    exit 1
fi

# Show container status
log_info "Container status:"
docker compose -f "$COMPOSE_FILE" ps api

# Summary
RESTORE_END=$(date +%s)
DURATION=$((RESTORE_END - RESTORE_START))

echo ""
echo "=========================================="
log_info "RESTORATION COMPLETE"
echo "=========================================="
echo "  Backup date: $DATE"
echo "  Duration: ${DURATION} seconds"
echo "  Time: $(date)"
echo ""
echo "Verify your data and test the application."
echo "If issues occur, check logs: docker compose logs -f"
