# PowerShell Script to Sync Backups from VPS
# Usage: .\scripts\sync_backups.ps1
# Prerequisites: Ensure 'scp' is in your path

# 1. Helper function to parse .env file
function Get-EnvVar {
    param($Name, $Default)
    $envFile = "$PSScriptRoot\..\.env"
    if (Test-Path $envFile) {
        $lines = Get-Content $envFile
        foreach ($line in $lines) {
            if ($line -match "^$Name=(.*)$") {
                return $matches[1].Trim()
            }
        }
    }
    return $Default
}

# 2. Load Configuration from .env or default
$VPS_USER = Get-EnvVar -Name "VPS_USER" -Default "root"
$VPS_HOST = Get-EnvVar -Name "VPS_HOST" -Default "api.vaishakmenon.com"
$REMOTE_DIR = "/opt/rag-personal/backups/"
$LOCAL_DIR = "$PSScriptRoot\..\backups"

# Ensure local backup directory exists
if (-not (Test-Path -Path $LOCAL_DIR)) {
    New-Item -ItemType Directory -Path $LOCAL_DIR | Out-Null
    Write-Host "Created local backup directory: $LOCAL_DIR" -ForegroundColor Green
}

Write-Host "🚀 Starting Backup Sync from $VPS_USER@$VPS_HOST..." -ForegroundColor Cyan
Write-Host "   Remote: $REMOTE_DIR"
Write-Host "   Local:  $LOCAL_DIR"

if ($VPS_HOST -eq "your-vps-ip-here") {
    Write-Host "❌ Error: VPS_HOST is not set in .env file." -ForegroundColor Red
    exit 1
}

# Check if SSH key is defined in environment variable, otherwise assume standard ssh agent or default key
$SCP_CMD = "scp -r $VPS_USER@${VPS_HOST}:${REMOTE_DIR}* '$LOCAL_DIR'"

Write-Host "   Executing: $SCP_CMD" -ForegroundColor DarkGray

try {
    # Run SCP
    # Note: This will prompt for password if SSH key is not set up
    scp -r "$VPS_USER@${VPS_HOST}:${REMOTE_DIR}*" "$LOCAL_DIR"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Backups synced successfully!" -ForegroundColor Green
        Get-ChildItem $LOCAL_DIR | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
    }
    else {
        Write-Host "❌ SCP failed with exit code $LASTEXITCODE" -ForegroundColor Red
        Write-Host "   Tip: Ensure your VPN/SSH connection is active and hostname is correct." -ForegroundColor Yellow
    }
}
catch {
    Write-Host "❌ Error executing SCP: $_" -ForegroundColor Red
}

Write-Host "Press any key to exit..."
$host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
