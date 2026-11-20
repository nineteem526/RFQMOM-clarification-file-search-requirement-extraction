#!/usr/bin/env bash
set -euo pipefail

app_path="${APP_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
app_path="$(cd "$app_path" && pwd)"
export APP_PATH="$app_path"
cd "$app_path"

export RFQ_APP_SERVICE=true
export RFQ_SINGLE_PROJECT_MODE=true
export RFQ_PROJECT_ID="${RFQ_PROJECT_ID:-rfq-poc}"
export RFQ_RUNTIME_HOME="${RFQ_RUNTIME_HOME:-/home/rfq-data}"
export RFQ_AUTHORIZED_SOURCE="${RFQ_AUTHORIZED_SOURCE:-/home/rfq-source}"
export RFQ_PUBLIC_ORIGIN="${RFQ_PUBLIC_ORIGIN:-https://${WEBSITE_HOSTNAME:?WEBSITE_HOSTNAME is required}}"
export RFQ_OPENCV_HEADLESS_DIR="${RFQ_OPENCV_HEADLESS_DIR:-$app_path/_appservice-assets/ocr/opencv-headless}"

if [ "$RFQ_AUTHORIZED_SOURCE" = "/home/rfq-source" ]; then
    mkdir -p "$RFQ_AUTHORIZED_SOURCE"
fi

exec python -B "$app_path/project_app.py"
