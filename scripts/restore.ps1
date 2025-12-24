# =============================================================================
# Disaster Recovery - Restore Script (Windows PowerShell)
# Restores PostgreSQL, Redis, and ChromaDB from backups
# =============================================================================

param(
    [string]$Date = "",
    [switch]$List,
    [switch]$Force,
    [switch]$DryRun,
    [switch]$Help
)

# Configuration
$BackupDir = if ($env:BACKUP_DIR) { $env:BACKUP_DIR } else { ".\backups" }
$ComposeFile = if ($env:COMPOSE_FILE) { $env:COMPOSE_FILE } else { "docker-compose.yml" }
$ChromaDataDir = if ($env:CHROMA_DATA_DIR) { $env:CHROMA_DATA_DIR } else { ".\data\chroma" }

# Helper functions
function Write-Info { param($Message) Write-Host "[INFO] $Message" -ForegroundColor Green }
function Write-Warn { param($Message) Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Err { param($Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

function Show-Usage {
    Write-Host @"
Usage: .\restore.ps1 [OPTIONS]

Options:
  -Date DATE      Restore from specific date (format: YYYYMMDD_HHMMSS)
  -List           List available backups
  -Force          Skip confirmation prompt
  -DryRun         Show what would be restored without making changes
  -Help           Show this help message

Examples:
  .\restore.ps1 -List                    # List available backups
  .\restore.ps1                          # Restore from latest backup (interactive)
  .\restore.ps1 -Date 20251223_030000    # Restore from specific backup
  .\restore.ps1 -Date 20251223_030000 -Force  # Restore without confirmation
"@
    exit 0
}

function Show-Backups {
    Write-Info "Available backups in ${BackupDir}:"
    Write-Host ""

    Write-Host "PostgreSQL backups:"
    $dbBackups = Get-ChildItem -Path $BackupDir -Filter "db_*.sql.gz" -ErrorAction SilentlyContinue
    if ($dbBackups) {
        $dbBackups | ForEach-Object { Write-Host "  $($_.Name) - $([math]::Round($_.Length/1KB, 2)) KB" }
    } else {
        Write-Host "  (none found)"
    }

    Write-Host ""
    Write-Host "Redis backups:"
    $redisBackups = Get-ChildItem -Path $BackupDir -Filter "redis_*.rdb" -ErrorAction SilentlyContinue
    if ($redisBackups) {
        $redisBackups | ForEach-Object { Write-Host "  $($_.Name) - $([math]::Round($_.Length/1KB, 2)) KB" }
    } else {
        Write-Host "  (none found)"
    }

    Write-Host ""
    Write-Host "ChromaDB backups:"
    $chromaBackups = Get-ChildItem -Path $BackupDir -Filter "chroma_*.tar.gz" -ErrorAction SilentlyContinue
    if ($chromaBackups) {
        $chromaBackups | ForEach-Object { Write-Host "  $($_.Name) - $([math]::Round($_.Length/1MB, 2)) MB" }
    } else {
        Write-Host "  (none found)"
    }

    exit 0
}

# Handle help and list flags
if ($Help) { Show-Usage }
if ($List) { Show-Backups }

# Verify backup directory exists
if (-not (Test-Path $BackupDir)) {
    Write-Err "Backup directory not found: $BackupDir"
    exit 1
}

# Find latest backup date if not specified
if (-not $Date) {
    $latestDb = Get-ChildItem -Path $BackupDir -Filter "db_*.sql.gz" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1

    if (-not $latestDb) {
        Write-Err "No database backups found in $BackupDir"
        exit 1
    }

    $Date = $latestDb.Name -replace "db_(.*)\.sql\.gz", '$1'
    Write-Info "Using latest backup: $Date"
}

# Define backup file paths
$DbBackup = Join-Path $BackupDir "db_$Date.sql.gz"
$RedisBackup = Join-Path $BackupDir "redis_$Date.rdb"
$ChromaBackup = Join-Path $BackupDir "chroma_$Date.tar.gz"

Write-Info "Checking backup files for date: $Date"

$missingFiles = $false

if (Test-Path $DbBackup) {
    $size = [math]::Round((Get-Item $DbBackup).Length / 1KB, 2)
    Write-Info "  PostgreSQL: $size KB"
} else {
    Write-Warn "  PostgreSQL: NOT FOUND - $DbBackup"
    $missingFiles = $true
}

if (Test-Path $RedisBackup) {
    $size = [math]::Round((Get-Item $RedisBackup).Length / 1KB, 2)
    Write-Info "  Redis: $size KB"
} else {
    Write-Warn "  Redis: NOT FOUND - $RedisBackup"
    $missingFiles = $true
}

if (Test-Path $ChromaBackup) {
    $size = [math]::Round((Get-Item $ChromaBackup).Length / 1MB, 2)
    Write-Info "  ChromaDB: $size MB"
} else {
    Write-Warn "  ChromaDB: NOT FOUND - $ChromaBackup"
    $missingFiles = $true
}

if ($missingFiles) {
    Write-Warn "Some backup files are missing. Partial restore will be attempted."
}

# Dry run - just show what would happen
if ($DryRun) {
    Write-Host ""
    Write-Info "=== DRY RUN - No changes will be made ==="
    Write-Host "Would restore from backup dated: $Date"
    Write-Host "Steps that would be performed:"
    Write-Host "  1. Stop api and caddy services"
    if (Test-Path $DbBackup) { Write-Host "  2. Restore PostgreSQL from $DbBackup" }
    if (Test-Path $RedisBackup) { Write-Host "  3. Stop redis, restore from $RedisBackup, start redis" }
    if (Test-Path $ChromaBackup) { Write-Host "  4. Restore ChromaDB from $ChromaBackup" }
    Write-Host "  5. Restart all services"
    Write-Host "  6. Run health check"
    exit 0
}

# Confirmation prompt
if (-not $Force) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "  WARNING: DESTRUCTIVE OPERATION" -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "This will restore from backup: $Date"
    Write-Host "Current data will be OVERWRITTEN."
    Write-Host ""
    $confirm = Read-Host "Type 'RESTORE' to confirm"
    if ($confirm -ne "RESTORE") {
        Write-Info "Restore cancelled."
        exit 0
    }
}

Write-Host ""
Write-Info "Starting restoration process..."
$restoreStart = Get-Date

# Step 1: Stop application services
Write-Info "Step 1/6: Stopping application services..."
docker compose -f $ComposeFile stop api caddy 2>$null

# Step 2: Restore PostgreSQL
if (Test-Path $DbBackup) {
    Write-Info "Step 2/6: Restoring PostgreSQL..."

    # Terminate all connections to the database first
    Write-Info "  Terminating existing database connections..."
    docker compose -f $ComposeFile exec -T postgres psql -U rag_user -d postgres -c @"
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = 'rag_db' AND pid <> pg_backend_pid();
"@ 2>$null

    # Drop and recreate database
    docker compose -f $ComposeFile exec -T postgres psql -U rag_user -d postgres -c "DROP DATABASE IF EXISTS rag_db;" 2>$null
    docker compose -f $ComposeFile exec -T postgres psql -U rag_user -d postgres -c "CREATE DATABASE rag_db;" 2>$null

    # Restore from backup (decompress and pipe to psql)
    $backupFullPath = (Resolve-Path $DbBackup).Path
    docker run --rm -v "${backupFullPath}:/backup.sql.gz:ro" alpine sh -c "gunzip -c /backup.sql.gz" |
        docker compose -f $ComposeFile exec -T postgres psql -U rag_user -d rag_db

    Write-Info "  PostgreSQL restored successfully."
} else {
    Write-Warn "Step 2/6: Skipping PostgreSQL (backup not found)"
}

# Step 3: Restore Redis
if (Test-Path $RedisBackup) {
    Write-Info "Step 3/6: Restoring Redis..."

    # Stop Redis
    docker compose -f $ComposeFile stop redis

    # Find the Redis volume and copy backup
    $redisVolume = docker volume ls -q | Where-Object { $_ -match "redis_data" } | Select-Object -First 1

    if ($redisVolume) {
        $backupFullPath = (Resolve-Path $RedisBackup).Path
        # Convert Windows path to Docker-compatible path
        $dockerPath = $backupFullPath -replace "\\", "/" -replace "^([A-Za-z]):", '/$1'

        docker run --rm `
            -v "${redisVolume}:/data" `
            -v "${dockerPath}:/backup.rdb:ro" `
            alpine sh -c "cp /backup.rdb /data/dump.rdb && chown 999:999 /data/dump.rdb"
    } else {
        Write-Warn "Could not find Redis volume, attempting direct copy..."
        Copy-Item $RedisBackup -Destination ".\data\redis\dump.rdb" -Force -ErrorAction SilentlyContinue
    }

    # Restart Redis
    docker compose -f $ComposeFile start redis
    Start-Sleep -Seconds 2

    Write-Info "  Redis restored successfully."
} else {
    Write-Warn "Step 3/6: Skipping Redis (backup not found)"
}

# Step 4: Restore ChromaDB
if (Test-Path $ChromaBackup) {
    Write-Info "Step 4/6: Restoring ChromaDB..."

    if (Test-Path $ChromaDataDir) {
        # Clear existing data
        Remove-Item -Path "$ChromaDataDir\*" -Recurse -Force -ErrorAction SilentlyContinue

        # Extract backup using tar (via docker if tar not available)
        $tarAvailable = Get-Command tar -ErrorAction SilentlyContinue

        if ($tarAvailable) {
            tar -xzf $ChromaBackup -C $ChromaDataDir
        } else {
            $backupFullPath = (Resolve-Path $ChromaBackup).Path
            $chromaFullPath = (Resolve-Path $ChromaDataDir).Path
            $dockerBackupPath = $backupFullPath -replace "\\", "/" -replace "^([A-Za-z]):", '/$1'
            $dockerChromaPath = $chromaFullPath -replace "\\", "/" -replace "^([A-Za-z]):", '/$1'

            docker run --rm `
                -v "${dockerBackupPath}:/backup.tar.gz:ro" `
                -v "${dockerChromaPath}:/restore" `
                alpine sh -c "tar -xzf /backup.tar.gz -C /restore"
        }

        Write-Info "  ChromaDB restored successfully."
    } else {
        Write-Err "ChromaDB data directory not found: $ChromaDataDir"
    }
} else {
    Write-Warn "Step 4/6: Skipping ChromaDB (backup not found)"
}

# Step 5: Restart all services
Write-Info "Step 5/6: Restarting all services..."
docker compose -f $ComposeFile up -d

# Step 6: Health check (using Docker's health status)
Write-Info "Step 6/6: Running health checks..."
Start-Sleep -Seconds 5

$maxRetries = 30
$retryCount = 0
$healthy = $false

while ($retryCount -lt $maxRetries) {
    $status = docker inspect --format="{{.State.Health.Status}}" rag_api 2>$null
    if ($status -eq "healthy") {
        Write-Info "  API health check: PASSED (Docker reports healthy)"
        $healthy = $true
        break
    }
    $retryCount++
    Write-Host "  Waiting for API... ($retryCount/$maxRetries) - Status: $status"
    Start-Sleep -Seconds 2
}

if (-not $healthy) {
    Write-Err "  API health check: FAILED after $maxRetries attempts"
    Write-Err "  Check logs with: docker compose logs api"
    exit 1
}

# Show container status
Write-Info "Container status:"
docker compose -f $ComposeFile ps api

# Summary
$restoreEnd = Get-Date
$duration = ($restoreEnd - $restoreStart).TotalSeconds

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Info "RESTORATION COMPLETE"
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Backup date: $Date"
Write-Host "  Duration: $([math]::Round($duration, 1)) seconds"
Write-Host "  Time: $(Get-Date)"
Write-Host ""
Write-Host "Verify your data and test the application."
Write-Host "If issues occur, check logs: docker compose logs -f"
