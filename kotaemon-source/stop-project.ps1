param(
    [ValidateRange(1024,65535)][int]$Port = 7863,
    [ValidatePattern('^[a-z0-9][a-z0-9-]{0,47}$')][string]$Project = "project-demo"
)
$ErrorActionPreference = "Stop"
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if (-not $listeners) {
    Write-Host "Project service is not running."
    exit 0
}
$healthHost = $listeners |
    Where-Object LocalAddress -notin @("0.0.0.0", "::", "::1") |
    Select-Object -First 1 -ExpandProperty LocalAddress
if (-not $healthHost) {
    $healthHost = "127.0.0.1"
}
$health = Invoke-RestMethod "http://${healthHost}:$Port/project-health" -TimeoutSec 5
if ($health.app -ne "rfq-kotaemon-project" -or $health.project -ne $Project) {
    throw "This port does not belong to the requested project. Nothing stopped."
}
$processIds = @($listeners.OwningProcess | Select-Object -Unique)
if ($processIds.Count -ne 1) {
    throw "Ambiguous listener. Nothing stopped."
}
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($processIds[0])"
$runtime = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "RFQKotaemon\runtime")) + "\"
if (-not $process.ExecutablePath -or
    -not $process.ExecutablePath.StartsWith($runtime, [StringComparison]::OrdinalIgnoreCase) -or
    $process.CommandLine -notmatch 'project_app\.py' -or
    $process.CommandLine -notmatch ("--project\s+" + [regex]::Escape($Project) + '(?:\s|$|")')) {
    throw "Listener executable could not be verified. Nothing stopped."
}
Stop-Process -Id $processIds[0]
Write-Host "Stopped project $Project. Saved files remain in the local project directory."
