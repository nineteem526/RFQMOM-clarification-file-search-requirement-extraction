param(
    [ValidateRange(1024,65535)][int]$Port = 7863,
    [ValidatePattern('^[a-z0-9][a-z0-9-]{0,47}$')][string]$Project = "project-demo",
    [switch]$Lan
)
$ErrorActionPreference = "Stop"
$python = Join-Path $env:LOCALAPPDATA "RFQKotaemon\runtime\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Dedicated Python environment was not found."
}
$lanIp = $null
$remoteAddress = $null
if ($Lan) {
    $profiles = @(Get-NetConnectionProfile | Where-Object NetworkCategory -eq "DomainAuthenticated")
    if ($profiles.Count -ne 1) {
        throw "LAN sharing requires exactly one active DomainAuthenticated network profile."
    }
    $addresses = @(Get-NetIPAddress -InterfaceIndex $profiles[0].InterfaceIndex `
        -AddressFamily IPv4 | Where-Object {
            $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*"
        })
    if ($addresses.Count -ne 1) {
        throw "Could not safely determine one company-network IPv4 address."
    }
    $lanIp = $addresses[0].IPAddress
    $prefix = $addresses[0].PrefixLength
    $bytes = [Net.IPAddress]::Parse($lanIp).GetAddressBytes()
    $networkBytes = for ($index = 0; $index -lt 4; $index++) {
        $remainingBits = $prefix - (8 * $index)
        $mask = if ($remainingBits -ge 8) { 255 }
            elseif ($remainingBits -le 0) { 0 }
            else { (255 -shl (8 - $remainingBits)) -band 255 }
        $bytes[$index] -band $mask
    }
    $network = [Net.IPAddress]::new([byte[]]$networkBytes).ToString()
}
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    $health = $null
    $healthHosts = @("127.0.0.1")
    if ($Lan) {
        $healthHosts += $lanIp
    }
    foreach ($healthHost in $healthHosts) {
        try {
            $health = Invoke-RestMethod -Uri "http://${healthHost}:$Port/project-health" -TimeoutSec 2
            break
        }
        catch { }
    }
    if ($health -and $health.app -eq "rfq-kotaemon-project" -and $health.project -eq $Project) {
        $expectedMode = if ($Lan) { "company-lan" } else { "loopback" }
        if ($health.access_mode -eq $expectedMode) {
            Start-Process $health.access_url
            Write-Host "Opened existing project assistant: $($health.access_url)"
            exit 0
        }
        if (-not $health) {
            Write-Host "The existing listener could not be verified as this project assistant."
        }
        Write-Host "The running service uses $($health.access_mode) mode."
        Write-Host "Stop it with 停止项目助手.cmd, then start the requested access mode."
        exit 1
    }
    Write-Host "Port $Port is already in use. No process was stopped."
    Write-Host "Open http://127.0.0.1:$Port if this is your service."
    Write-Host "For a different project use a distinct ID and port."
    exit 1
}
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Lan) {
        Write-Host "LAN URL: http://${lanIp}:$Port"
        Write-Host "Any reachable user on this company subnet can query and view indexed evidence."
        Write-Host "No administrator rights or firewall settings are changed."
        Write-Host "If peers cannot connect, ask IT to allow TCP/$Port from $network/$prefix."
        Write-Host "This mode has no login; keep the server running only while sharing."
    }
    else {
        Write-Host "Project: $Project. Local URL: http://127.0.0.1:$Port"
    }
    Write-Host "Scan, select files and click Ingest. No approval or review steps."
    Write-Host "Questions and retrieved evidence go only to the company API."
    Write-Host "Keep this window open. Ctrl+C stops the service."
    if ($Lan) {
        & $python -B (Join-Path $PSScriptRoot "project_app.py") `
            --project $Project --port $Port --lan-ip $lanIp --open-browser
    }
    else {
        & $python -B (Join-Path $PSScriptRoot "project_app.py") `
            --project $Project --port $Port --open-browser
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Project service failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
