"""Exact-request fingerprints; no credentials or semantic answer matching."""

from datetime import date
import hashlib
import json
import os
from pathlib import Path


PIPELINE_FILES = (
    "query_cache.py", "query_jobs.py", "project_workspace.py", "project_retrieval.py",
    "project_answer.py", "project_intent.py", "company_chat.py", "check_company_api.py",
    "local_embedding.py", "synthetic_qa.py", "citation_location.py",
    "environment-setup/embedding-download-manifest.json",
)


def pipeline_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in PIPELINE_FILES:
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def request_key(workspace, question, mode, context, pipeline):
    with workspace.connection() as conn:
        state = dict(conn.execute("SELECT key,value FROM state"))
    scans = sorted((workspace.root / "scans").glob("*.json"))
    payload = {
        "project_root": str(workspace.root),
        "source": str(workspace.source_root),
        "question": question.strip(),
        "mode": mode,
        "previous_user_question": context.strip(),
        "revision": state["revision"],
        "embedding": state["embedding_model"],
        "scan": hashlib.sha256(scans[-1].read_bytes()).hexdigest() if scans else "",
        "pipeline": pipeline,
        "endpoint": os.environ.get("DUERR_API_ENDPOINT", ""),
        "deployment": os.environ.get("DUERR_MODEL", "gpt-5.2"),
        "api_version": os.environ.get("DUERR_API_VERSION", "2025-04-01-preview"),
        # Coverage includes warnings for dates later than today.
        "local_date": date.today().isoformat(),
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()


def reusable(result):
    from project_workspace import coverage_incomplete

    return bool(
        result["answer"] and result["evidence"]
        and not coverage_incomplete(result["coverage"])
        and not any(entry.get("errors") or entry.get("intent_notice")
                    or entry.get("meeting_source_missing") for entry in result["coverage"])
    )
