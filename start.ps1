$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment missing. Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}
& $python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
exit $LASTEXITCODE
