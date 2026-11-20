$ErrorActionPreference = "Stop"
$runtime = Join-Path $env:LOCALAPPDATA "RFQKotaemon\runtime"
$python = Join-Path $runtime ".venv\Scripts\python.exe"
$source = Join-Path $runtime "kotaemon"
$expected = "9ad3e4e49aa35b8acddd235918a5d9753c1cfdf9"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Dedicated Python environment missing."
}
$actual = & git -C $source rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actual.Trim() -ne $expected) {
    throw "Kotaemon source commit does not match the approved fixed version."
}
$changes = & git -C $source status --porcelain
if ($LASTEXITCODE -ne 0 -or $changes) {
    throw "Upstream source has local changes; review before installation."
}
& $python -B (Join-Path $PSScriptRoot "lock_constraints.py") (Join-Path $source "uv.lock") (Join-Path $PSScriptRoot "upstream-constraints.txt")
if ($LASTEXITCODE -ne 0) { throw "Upstream constraint generation failed." }
& $python -m pip install --index-url https://pypi.org/simple --disable-pip-version-check -r (Join-Path $PSScriptRoot "required-dependencies.in") -c (Join-Path $PSScriptRoot "upstream-constraints.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $python -m pip install --index-url https://pypi.org/simple --disable-pip-version-check -c (Join-Path $PSScriptRoot "upstream-constraints.txt") (Join-Path $source "libs\kotaemon") (Join-Path $source "libs\ktem")
if ($LASTEXITCODE -ne 0) { throw "Fixed-source package installation failed." }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency integrity check failed." }
& $python -m pip freeze --all | Out-File -LiteralPath (Join-Path $PSScriptRoot "installed-freeze.txt") -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw "Could not record installed versions." }
Write-Output "Installation complete. This does not start or validate the Kotaemon application."
