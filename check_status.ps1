$logFile = "C:\Temp\oci_retry_log.txt"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "OCI Instance Retry Status" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$pythonRunning = Get-Process python -ErrorAction SilentlyContinue
Write-Host "Python running: $(if($pythonRunning){'Yes ✓ (PID: ' + $pythonRunning.Id + ')'}else{'No ✗'})" -ForegroundColor $(if($pythonRunning){'Green'}else{'Red'})

if (Test-Path $logFile) {
    $attempts = (Get-Content $logFile | Select-String "Attempt").Count
    $lastUpdate = (Get-Item $logFile).LastWriteTime
    Write-Host "Total attempts: $attempts" -ForegroundColor Yellow
    Write-Host "Last update: $lastUpdate" -ForegroundColor Gray
    Write-Host "`nLast 15 lines:" -ForegroundColor Cyan
    Get-Content $logFile -Tail 15
} else {
    Write-Host "Log file not found!" -ForegroundColor Red
}

Write-Host "`n========================================`n" -ForegroundColor Cyan