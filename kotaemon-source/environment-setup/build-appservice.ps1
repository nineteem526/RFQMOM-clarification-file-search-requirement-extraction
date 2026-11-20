param(
    [ValidatePattern('^[a-z0-9][a-z0-9._:/-]*$')]
    [string]$ImageTag = "rfq-kotaemon-appservice:local"
)

$ErrorActionPreference = "Stop"
$docker = Get-Command docker -ErrorAction Stop
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot "embedding-download-manifest.json") -Raw |
    ConvertFrom-Json
$runtimeHome = if ($env:RFQ_RUNTIME_HOME) {
    $env:RFQ_RUNTIME_HOME
} else {
    Join-Path $env:LOCALAPPDATA "RFQKotaemon"
}
$runtimeVenv = if ($env:RFQ_RUNTIME_VENV) {
    $env:RFQ_RUNTIME_VENV
} else {
    Join-Path $runtimeHome "runtime\.venv"
}
$python = Join-Path $runtimeVenv "Scripts\python.exe"
$modelSource = if ($env:RFQ_EMBEDDING_DIR) {
    $env:RFQ_EMBEDDING_DIR
} elseif ($manifest.local_directory -match '^%LOCALAPPDATA%[\\/]RFQKotaemon[\\/](.+)$') {
    Join-Path $runtimeHome $Matches[1]
} else {
    [Environment]::ExpandEnvironmentVariables($manifest.local_directory)
}
$tiktokenSource = if ($env:RFQ_TIKTOKEN_DATA) {
    $env:RFQ_TIKTOKEN_DATA
} else {
    Join-Path $runtimeHome "resources\tiktoken"
}
$nltkArchives = Join-Path $runtimeHome "runtime\downloads\nltk"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Existing RFQ Python environment is missing: $python"
}

$context = [System.IO.Path]::GetFullPath((Join-Path ([System.IO.Path]::GetTempPath()) (
    "rfq-kotaemon-appservice-" + [guid]::NewGuid().ToString("N")
)))
try {
    & $python -B (Join-Path $PSScriptRoot "stage_appservice.py") `
        --context $context --model-dir $modelSource `
        --nltk-archives $nltkArchives --tiktoken-dir $tiktokenSource
    if ($LASTEXITCODE -ne 0) {
        throw "Build context preparation failed with exit code $LASTEXITCODE."
    }

    & $docker.Source build --pull --tag $ImageTag --file (Join-Path $context "Dockerfile") $context
    if ($LASTEXITCODE -ne 0) {
        throw "Docker image build failed with exit code $LASTEXITCODE."
    }
    Write-Output "Built image: $ImageTag"
    Write-Warning "Confirm company authorization for the included BGE, NLTK and tiktoken resources before publishing the image."
}
finally {
    if (Test-Path -LiteralPath $context -PathType Container) {
        Remove-Item -LiteralPath $context -Recurse -Force
    }
}
