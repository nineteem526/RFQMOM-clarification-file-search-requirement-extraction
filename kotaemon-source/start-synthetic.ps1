$ErrorActionPreference = "Stop"
$python = Join-Path $env:LOCALAPPDATA "RFQKotaemon\runtime\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Dedicated Python environment was not found."
}
Push-Location -LiteralPath $PSScriptRoot
try {
    Write-Host "Synthetic-only QA. Do not enter real project information."
    Write-Host "Questions and synthetic evidence will go to the company API."
    Write-Host "Open http://127.0.0.1:7861 after startup. Ctrl+C stops the server."
    & $python -B (Join-Path $PSScriptRoot "preview_kotaemon.py") --synthetic --port 7861
    if ($LASTEXITCODE -ne 0) {
        throw "Synthetic QA failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
