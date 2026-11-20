"""Verify actual fixture parsing/indexing/retrieval, optionally company chat."""

import argparse
import json
import os

from pilot_runtime import prepare_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    root = prepare_runtime()
    os.environ["THEFLOW_SETTINGS_MODULE"] = "preview_settings"
    from theflow.settings import settings
    from preview_kotaemon import company_network_boundary, local_only
    import sys

    sys.addaudithook(company_network_boundary() if args.live else local_only)
    from synthetic_qa import SyntheticKnowledge, validate_excerpts

    knowledge = SyntheticKnowledge(root / "synthetic-only-v1" / "sources")
    if len(knowledge.documents) != 7:
        raise RuntimeError("Expected 7 parsed fixture records.")
    queries = [
        ("钢结构平台荷载有哪些新旧要求？", ["3.0 kN/m²", "5.0 kN/m²"]),
        ("PLC-S7-1500电压是多少？", ["24", "V DC", "PLC-S7-1500"]),
        ("IF-002备用接口供货边界是什么？", ["仅预留端子，不含现场接线"]),
    ]
    for number, (query, required) in enumerate(queries, 1):
        retrieved = knowledge.retrieve(query)
        text = "\n".join(doc.content for doc in retrieved)
        if not all(value in text for value in required):
            raise RuntimeError(f"Retrieval case {number} missed required evidence.")
        print(json.dumps({"retrieval_case": number, "passed": True}), flush=True)
    if args.live:
        selected, retrieved = knowledge.answer(queries[0][0])
        text = "\n".join(doc.content for doc in selected)
        if not all(term in text for term in ["3.0 kN/m²", "5.0 kN/m²", "待业主批准"]):
            raise RuntimeError("Live answer missed old/new evidence or approval condition.")
        print(json.dumps({"live_retrieval_and_chat": "passed",
                          "business_text_sent": False}), flush=True)
    else:
        documents = knowledge.retrieve(queries[0][0])
        validate_excerpts(json.dumps({"excerpts": [
            {"id": doc.id_, "quote": doc.content} for doc in documents
        ]}), documents)
    print(json.dumps({
        "parsed_files": 3, "parsed_records": 7,
        "formats": ["TXT", "DOCX", "XLSX"],
        "kotaemon_indexing_and_retrieval": True,
        "company_api_called": args.live, "business_documents_read": False,
    }), flush=True)


if __name__ == "__main__":
    main()
