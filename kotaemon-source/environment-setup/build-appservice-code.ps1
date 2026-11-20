param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [string]$OpenCvHeadlessWheel
)

$ErrorActionPreference = "Stop"
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
$python = @(
    (Join-Path $runtimeVenv "Scripts\python.exe"),
    (Join-Path $runtimeVenv "python.exe")
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $python) {
    throw "Python executable is missing from the configured RFQ runtime: $runtimeVenv"
}
$upstreamRoot = Join-Path (Split-Path $runtimeVenv -Parent) "kotaemon"
$modelManifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot "embedding-download-manifest.json") -Raw |
    ConvertFrom-Json
$modelSource = if ($env:RFQ_EMBEDDING_DIR) {
    $env:RFQ_EMBEDDING_DIR
} elseif ($modelManifest.local_directory -match '^%LOCALAPPDATA%[\\/]RFQKotaemon[\\/](.+)$') {
    Join-Path $runtimeHome $Matches[1]
} else {
    [Environment]::ExpandEnvironmentVariables($modelManifest.local_directory)
}
$tokenizerSource = if ($env:RFQ_TIKTOKEN_DATA) {
    $env:RFQ_TIKTOKEN_DATA
} else {
    Join-Path $runtimeHome "resources\tiktoken"
}
$stopwordsSource = Join-Path $runtimeVenv `
    "Lib\site-packages\llama_index\core\_static\nltk_cache\corpora\stopwords"
$iconsSource = Join-Path $upstreamRoot "libs\ktem\ktem\assets\icons"
$nltkArchives = Join-Path $runtimeHome "runtime\downloads\nltk"

foreach ($required in @(
    $python,
    $modelSource,
    $tokenizerSource,
    $nltkArchives,
    $stopwordsSource,
    $iconsSource
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required approved local runtime asset is missing: $required"
    }
}

$output = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $output) {
    throw "Refusing to overwrite an existing deployment package: $output"
}

$optionalArguments = @()
if ($OpenCvHeadlessWheel) {
    if (-not (Test-Path -LiteralPath $OpenCvHeadlessWheel -PathType Leaf)) {
        throw "The requested OpenCV headless wheel does not exist."
    }
    $optionalArguments += @("--opencv-headless-wheel", [System.IO.Path]::GetFullPath($OpenCvHeadlessWheel))
}

& $python -B (Join-Path $PSScriptRoot "build_appservice_code.py") `
    --output $output `
    --model-dir $modelSource `
    --nltk-archives $nltkArchives `
    --tiktoken-dir $tokenizerSource `
    --icons-dir $iconsSource `
    --stopwords-dir $stopwordsSource `
    @optionalArguments
if ($LASTEXITCODE -ne 0) {
    throw "App Service code package preparation failed with exit code $LASTEXITCODE."
}
Write-Warning "The package includes the BGE, NLTK and tiktoken runtime assets. Confirm their approval before deploying."
