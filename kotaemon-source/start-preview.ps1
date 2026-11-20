$ErrorActionPreference = "Stop"
$python = Join-Path $env:LOCALAPPDATA "RFQKotaemon\runtime\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Dedicated Python environment was not found."
}
Push-Location -LiteralPath $PSScriptRoot
try {
    Write-Host "Read-only preview. No uploads, QA requests or business documents."
    Write-Host "After the server starts, open http://127.0.0.1:7860"
    Write-Host "Keep this window open. Press Ctrl+C to stop."
    & $python -B (Join-Path $PSScriptRoot "preview_kotaemon.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Preview failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
