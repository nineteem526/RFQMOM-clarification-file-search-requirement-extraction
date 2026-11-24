$ErrorActionPreference = "Stop"

$ocrRoot = if ($env:RFQ_OCR_ROOT) { $env:RFQ_OCR_ROOT } else { Join-Path $env:LOCALAPPDATA "RFQKotaemon\ocr" }
$venvDir = Join-Path $ocrRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

$basePythonCandidates = @(
    (Join-Path $env:APPDATA "uv\python\cpython-3.11.15-windows-x86_64-none\python.exe"),
    (Join-Path $env:LOCALAPPDATA "RFQKotaemon\runtime\.venv\Scripts\python.exe")
)
$basePython = $basePythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $basePython) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $basePython = $cmd.Source }
}
if (-not $basePython) {
    throw "No suitable Python 3.11 interpreter found for creating isolated OCR environment."
}

New-Item -ItemType Directory -Force -Path $ocrRoot | Out-Null
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Output "Creating isolated OCR virtual environment at $venvDir ..."
    & $basePython -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create OCR venv." }
}

$requirements = Join-Path $PSScriptRoot "ocr-resolved-requirements.txt"
$manifestSrc = Join-Path $PSScriptRoot "ocr-model-manifest.json"
$manifestDst = Join-Path $ocrRoot "model-manifest.json"

& $venvPython -m pip install --upgrade pip --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip in OCR venv." }

& $venvPython -m pip install --index-url https://pypi.org/simple --disable-pip-version-check -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install OCR requirements." }

& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "OCR dependency integrity check failed." }

Copy-Item -LiteralPath $manifestSrc -Destination $manifestDst -Force

$verifyCode = @'
import hashlib, json, os
from pathlib import Path
import rapidocr_onnxruntime

ocr_root = Path(os.environ["OCR_ROOT"])
manifest = json.loads((ocr_root / "model-manifest.json").read_text(encoding="utf-8"))
pkg = Path(rapidocr_onnxruntime.__file__).parent
for name, expected in manifest["models"].items():
    p = pkg / "models" / name
    assert p.is_file(), f"Missing OCR model: {p}"
    actual = hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
    assert actual == expected, f"Hash mismatch for {name}: {actual} != {expected}"
print("OCR models verified:", len(manifest["models"]))
'@

$env:OCR_ROOT = $ocrRoot
$verifyCode | & $venvPython -
if ($LASTEXITCODE -ne 0) { throw "OCR model SHA-256 verification failed." }

Write-Output "Offline RapidOCR runtime installed and verified at $ocrRoot"
