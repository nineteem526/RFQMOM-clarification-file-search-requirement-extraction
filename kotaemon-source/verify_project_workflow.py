"""Exercise the project workflow using generated fixtures, never business documents."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from pilot_runtime import prepare_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    runtime = prepare_runtime()
    os.environ["THEFLOW_SETTINGS_MODULE"] = "preview_settings"
    from theflow.settings import settings
    from preview_kotaemon import company_network_boundary, local_only
    import sys

    sys.addaudithook(company_network_boundary() if args.live else local_only)
    from project_workspace import ProjectWorkspace
    from synthetic_qa import fixture_documents
    from types import SimpleNamespace

    class SelectAll:
        def invoke(self, messages):
            payload = json.loads(messages[1].content)
            if '"sections"' in messages[0].content:
                first = payload["evidence"][0]
                return SimpleNamespace(content=json.dumps({"sections": [{
                    "title": "虚构荷载", "points": [{
                        "text": "荷载及条件见原文。",
                        "citations": [{"id": first["id"], "quote": first["text"]}],
                    }],
                }]}))
            if '"ids"' in messages[0].content:
                return SimpleNamespace(content=json.dumps({
                    "ids": [entry["id"] for entry in payload["evidence"]]
                }))
            return SimpleNamespace(content=json.dumps({"excerpts": [
                {"id": entry["id"], "quote": entry["text"]}
                for entry in payload["evidence"]
            ]}))

    with tempfile.TemporaryDirectory(prefix="workflow-check-", dir=runtime) as directory:
        base = Path(directory)
        source = base / "sources"
        fixture_documents(source)
        workspace = ProjectWorkspace(base / "data", "fixture-project", source)
        scan = workspace.scan()
        names = [row["path"] for row in scan["files"]
                 if Path(row["path"]).suffix in {".txt", ".docx", ".xlsx"}]
        report = workspace.ingest(scan["scan_id"], names, confirmed=True)
        if any("error" in row for row in report):
            raise RuntimeError(json.dumps(report, ensure_ascii=False))
        identity, evidence, _, coverage = workspace.ask(
            "钢结构平台荷载有哪些新旧要求？",
            chat=None if args.live else SelectAll(),
            generate_answer=True,
        )
        text = "\n".join(doc.content for doc in evidence)
        if not all(term in text for term in ["3.0 kN/m²", "5.0 kN/m²", "待业主批准"]):
            raise RuntimeError("Old/new requirements or approval conditions missing.")
        workspace.save_result(identity)
        exported = workspace.export(identity)
        if not exported.is_file():
            raise RuntimeError("Export not created.")
        if workspace.result(identity)["stale"]:
            raise RuntimeError("Unchanged fixture result incorrectly marked stale.")
        answer = json.loads(workspace.result(identity)["answer"])
        if not answer["sections"]:
            raise RuntimeError("Cited answer missing.")
        if args.live and not all(
            term in json.dumps(answer, ensure_ascii=False)
            for term in ["3.0", "5.0", "批准"]
        ):
            raise RuntimeError("Synthesis lost old/new values or approval conditions.")
        print(json.dumps({
            "project_workflow": "passed", "parsed_files": len(names),
            "save_export": "passed", "company_api_called": args.live,
            "cited_synthesis": "passed",
            "business_documents_read": False,
        }), flush=True)


if __name__ == "__main__":
    main()
