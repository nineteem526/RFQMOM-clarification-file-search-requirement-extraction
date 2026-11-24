"""Reproducible End-to-End RAG Evaluation Script (Mock & Live API modes separated).

Evaluates per-question pipeline stages without logging secrets or private source text:
1. Retrieval candidates & evidence coverage
2. LLM invocation status (success / call_failed / verification_failed / no_evidence)
3. Verification pass/reject breakdown, self-correction recovery rate, API calls, tokens, latency

Usage:
  python eval_e2e_rag.py --mode mock --dataset both
  python eval_e2e_rag.py --mode live --dataset cn --limit 8
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import rag_engine
from web_app import _extract_text_from_file


def _mock_llm_for_case(question: str, citations: list[dict], _project_name: str, **_kwargs):
    """Deterministic mock simulating typical DeepSeek JSON output (including format variants)."""
    if not citations:
        return None
    top = citations[0]
    line = next((s.strip() for s in top["content"].splitlines() if s.strip()), top["content"][:100])
    return {
        "conclusion": f"依据原文，{line[:120]} [1]。请核对原文。",
        "basis": [{"index": 1, "source": top["file_name"], "detail": f"{line[:100]}"}],
        "table": [{"item": "主要条款", "value": f"{line[:80]} [1]"}],
        "llm_provider": "模拟在线模型（mock）",
        "_usage": {"prompt_tokens": 180, "completion_tokens": 65, "total_tokens": 245},
        "_latency_ms": 12.0,
    }


def eval_cn_e2e(fixture_path: Path, mode: str) -> dict:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    candidates = [
        {
            "id": ev["id"],
            "doc_id": ev["source"].split()[0],
            "doc_name": ev["source"].split()[0],
            "page": 1,
            "section": ev["source"],
            "content": ev["text"],
            "embedding_vec": rag_engine.embed_text(ev["text"]),
        }
        for ev in data["evidence"]
    ]
    stats = {
        "dataset": "CN_RFQ_P0_8Q",
        "mode": mode,
        "total_cases": len(data["cases"]),
        "grounded_cases": 0,
        "refusal_cases": 0,
        "evidence_hit": 0,
        "refusal_ok": 0,
        "llm_success": 0,
        "llm_call_failed": 0,
        "verification_failed": 0,
        "retried_recovered": 0,
        "total_api_calls": 0,
        "total_tokens": 0,
        "total_llm_latency_ms": 0.0,
        "cases": [],
    }
    for case in data["cases"]:
        expected = set(case["expected_evidence"])
        res = rag_engine.hybrid_search(case["question"], candidates, similarity_threshold=0.62, top_k=5)
        hit_ids = [h["id"] for h in res["hits"]]
        is_refusal = len(expected) == 0
        if is_refusal:
            stats["refusal_cases"] += 1
            if not hit_ids:
                stats["refusal_ok"] += 1
        else:
            stats["grounded_cases"] += 1
            if expected.intersection(hit_ids):
                stats["evidence_hit"] += 1

        if mode == "mock":
            with patch.object(rag_engine, "_call_external_llm", side_effect=_mock_llm_for_case):
                ans = rag_engine.generate_structured_answer(
                    case["question"], res["hits"], "cn-rfq-eval", res, allow_external=True
                )
        else:
            ans = rag_engine.generate_structured_answer(
                case["question"], res["hits"], "cn-rfq-eval", res, allow_external=True
            )

        st = ans.get("answer_status", "unknown")
        ls = ans.get("llm_stats") or {}
        if st == "llm_success":
            stats["llm_success"] += 1
            if ls.get("retried"):
                stats["retried_recovered"] += 1
        elif st == "llm_call_failed":
            stats["llm_call_failed"] += 1
        elif st == "verification_failed":
            stats["verification_failed"] += 1

        stats["total_api_calls"] += int(ls.get("api_calls", 0))
        stats["total_tokens"] += int(ls.get("total_tokens", 0))
        stats["total_llm_latency_ms"] = round(stats["total_llm_latency_ms"] + float(ls.get("latency_ms", 0.0)), 1)
        stats["cases"].append({
            "id": case["id"],
            "expected_count": len(expected),
            "hit_count": len(hit_ids),
            "answer_status": st,
            "diag_code": (ans.get("diagnostic") or {}).get("code"),
            "api_calls": ls.get("api_calls", 0),
            "tokens": ls.get("total_tokens", 0),
            "latency_ms": ls.get("latency_ms", 0.0),
        })
    return stats


def eval_makita_e2e(pdf_path: Path, jsonl_path: Path, mode: str, limit: int = 10) -> dict:
    parsed = _extract_text_from_file(pdf_path, chunk_size=480, chunk_overlap=64, ocr_enabled=False)
    chunks = parsed["chunks"] if isinstance(parsed, dict) else parsed
    candidates = [
        {
            "id": f"m-{idx}",
            "doc_id": "makita_drill.pdf",
            "doc_name": pdf_path.name,
            "page": int(ch["page"]),
            "section": ch["section"],
            "content": ch["content"],
            "embedding_vec": rag_engine.embed_text(ch["content"]),
        }
        for idx, ch in enumerate(chunks, 1)
    ]
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit > 0:
        rows = rows[:limit]

    stats = {
        "dataset": f"Makita_XFD12_{len(rows)}Q",
        "mode": mode,
        "total_cases": len(rows),
        "page_hit_at_5": 0,
        "answerable_cases": 0,
        "llm_success": 0,
        "llm_call_failed": 0,
        "verification_failed": 0,
        "no_evidence": 0,
        "total_api_calls": 0,
        "total_tokens": 0,
        "total_llm_latency_ms": 0.0,
    }
    for row in rows:
        raw_pg = row.get("gt_page_number")
        gt_page = int(str(raw_pg).strip()) if raw_pg is not None and str(raw_pg).strip().isdigit() else None
        res = rag_engine.hybrid_search(row["question_text"], candidates, similarity_threshold=0.55, top_k=5)
        if gt_page is not None:
            stats["answerable_cases"] += 1
            if gt_page in [h["page"] for h in res["hits"][:5]]:
                stats["page_hit_at_5"] += 1
        if mode == "mock":
            with patch.object(rag_engine, "_call_external_llm", side_effect=_mock_llm_for_case):
                ans = rag_engine.generate_structured_answer(
                    row["question_text"], res["hits"], "makita-eval", res, allow_external=True
                )
        else:
            ans = rag_engine.generate_structured_answer(
                row["question_text"], res["hits"], "makita-eval", res, allow_external=True
            )
        st = ans.get("answer_status", "unknown")
        ls = ans.get("llm_stats") or {}
        if st in stats:
            stats[st] += 1
        stats["total_api_calls"] += int(ls.get("api_calls", 0))
        stats["total_tokens"] += int(ls.get("total_tokens", 0))
        stats["total_llm_latency_ms"] = round(stats["total_llm_latency_ms"] + float(ls.get("latency_ms", 0.0)), 1)
    return stats


def main():
    parser = argparse.ArgumentParser(description="End-to-end RAG evaluation (mock vs live)")
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    parser.add_argument("--dataset", choices=("cn", "makita", "both"), default="both")
    parser.add_argument("--limit", type=int, default=35)
    args = parser.parse_args()

    if args.mode == "live" and not rag_engine.llm_config_status().get("configured"):
        raise SystemExit("Live mode requires configured LLM environment variables (e.g. RFQ_LLM_PROVIDER=siliconflow, SILICONFLOW_MODEL, SILICONFLOW_API_KEY).")

    ws_root = Path(__file__).resolve().parents[2]
    report = {}
    if args.dataset in ("cn", "both"):
        report["cn_rfq"] = eval_cn_e2e(ws_root / "p0-fixture.json", args.mode)
    if args.dataset in ("makita", "both"):
        report["makita"] = eval_makita_e2e(
            ws_root / "公开测试数据" / "Makita_XFD12_manual.pdf",
            ws_root / "公开测试数据" / "Makita_XFD12_35题.jsonl",
            args.mode,
            limit=args.limit,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
