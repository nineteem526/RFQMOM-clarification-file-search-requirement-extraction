#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
context="$(mktemp -d "${RUNNER_TEMP:?}/rfq-context.XXXXXX")"
home="$(mktemp -d "${RUNNER_TEMP:?}/rfq-smoke.XXXXXX")"
image="rfq-kotaemon-ci:${GITHUB_SHA:?}-${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}"
container=""
missing_container=""
cleanup() {
    status=$?
    if [ -n "$container" ]; then
        if [ "$status" -ne 0 ]; then
            docker logs "$container" >&2 || true
        fi
        docker rm --force "$container" >/dev/null 2>&1 || true
    fi
    if [ -n "$missing_container" ]; then
        docker rm --force "$missing_container" >/dev/null 2>&1 || true
    fi
    if ! rm -rf -- "$context" "$home"; then
        echo "Failed to clean up synthetic smoke-test files." >&2
        status=1
    fi
    exit "$status"
}
trap cleanup EXIT
mkdir -p "$home/rfq-source"

python3 -B "$root/environment-setup/stage_appservice.py" --context "$context"
docker build --pull --tag "$image" --file "$context/Dockerfile" "$context"

missing_container="rfq-missing-${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}"
set +e
timeout 30s docker run --rm --name "$missing_container" \
    -e RFQ_PUBLIC_ORIGIN=https://rfq-ci.example.invalid \
    -e RFQ_AUTHORIZED_SOURCE=/missing-rfq-source \
    "$image" > "$home/missing-source.log" 2>&1
missing_status=$?
set -e
docker rm --force "$missing_container" >/dev/null 2>&1 || true
if [ "$missing_status" -eq 0 ] || [ "$missing_status" -eq 124 ]; then
    echo "Container started without its authorized source directory." >&2
    exit 1
fi
if ! grep -q "must be an existing absolute directory" "$home/missing-source.log"; then
    cat "$home/missing-source.log" >&2
    echo "Missing-source check failed for an unexpected reason." >&2
    exit 1
fi

container="$(docker create \
    --publish 127.0.0.1:8000:8000 \
    -e RFQ_PUBLIC_ORIGIN=https://rfq-ci.example.invalid \
    -e RFQ_PROJECT_ID=rfq-ci \
    "$image")"
docker cp "$home/rfq-source" "$container:/home/rfq-source"
docker start "$container" >/dev/null

status=""
for _ in $(seq 1 120); do
    status="$(curl --silent --output /dev/null --write-out '%{http_code}' \
        --max-time 2 -H 'Host: rfq-ci.example.invalid' \
        http://127.0.0.1:8000/project-health || true)"
    [ "$status" = "403" ] && break
    if [ "$(docker inspect --format '{{.State.Running}}' "$container")" != "true" ]; then
        echo "Container exited before becoming ready." >&2
        exit 1
    fi
    sleep 2
done
if [ "$status" != "403" ]; then
    echo "Container did not reject an unauthenticated request before timeout." >&2
    exit 1
fi

principal="$(printf '%s' '{"auth_typ":"aad","claims":[{"typ":"oid","val":"fixture"}]}' | base64 | tr -d '\n')"
curl --fail --silent --show-error --max-time 10 \
    -H 'Host: rfq-ci.example.invalid' \
    -H 'Origin: https://rfq-ci.example.invalid' \
    -H "X-MS-CLIENT-PRINCIPAL: $principal" \
    http://127.0.0.1:8000/project-health |
    python3 -c 'import json,sys; p=json.load(sys.stdin); assert p["app"] == "rfq-kotaemon-project" and p["project"] == "rfq-ci"'
curl --fail --silent --show-error --output /dev/null --max-time 10 \
    -H 'Host: rfq-ci.example.invalid' \
    -H "X-MS-CLIENT-PRINCIPAL: $principal" \
    http://127.0.0.1:8000/

docker exec -i "$container" python -B - <<'PY'
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from pilot_runtime import prepare_runtime
from preview_kotaemon import local_only

prepare_runtime()
sys.addaudithook(local_only)

import tiktoken
from local_embedding import LocalBGEEmbeddings
from project_workspace import ProjectWorkspace
from query_cache import pipeline_fingerprint

assert len(pipeline_fingerprint()) == 64
for name in ("cl100k_base", "o200k_base"):
    assert tiktoken.get_encoding(name).encode("RFQ fixture")
vector = LocalBGEEmbeddings().invoke("RFQ fixture")[0].embedding
assert len(vector) == 512 and all(math.isfinite(value) for value in vector)

source = Path("/home/rfq-source")
(source / "fixture.txt").write_text(
    "The platform design load shall be 3.0 kN/m2.", encoding="utf-8"
)
workspace = ProjectWorkspace(Path("/home/rfq-ci-data"), "rfq-ci-fixture", source)
scan = workspace.scan()
report = workspace.ingest(scan["scan_id"], ["fixture.txt"], True)
if len(report) != 1 or "error" in report[0]:
    raise RuntimeError(f"Synthetic fixture ingestion failed: {report}")

class FixtureChat:
    def invoke(self, messages):
        payload = json.loads(messages[1].content)
        return SimpleNamespace(content=json.dumps({
            "ids": [item["id"] for item in payload["evidence"]]
        }))

_, evidence, _, _ = workspace.ask("platform design load", chat=FixtureChat())
assert any("3.0 kN/m2" in document.content for document in evidence)
PY
docker exec "$container" /opt/rfq-ocr/.venv/bin/python -B -c \
    'import fitz; doc=fitz.open(); page=doc.new_page(); page.insert_text((72,72), "RFQ 123", fontsize=24); doc.save("/home/rfq-ci-ocr.pdf"); doc.close()'
docker exec "$container" /opt/rfq-ocr/.venv/bin/python -B \
    /app/local_ocr.py --worker /home/rfq-ci-ocr.pdf 1 |
    python3 -c 'import json,sys; p=json.load(sys.stdin); text=" ".join(line["text"] for line in p["lines"]).upper(); assert p["page"] == 1 and "RFQ" in text and "123" in text'

echo "Container smoke checks passed with synthetic fixtures only."
