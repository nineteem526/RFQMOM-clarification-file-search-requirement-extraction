"""Comprehensive RAG Benchmark Suite (Real PDF TechManualQA 35Q + Chinese RFQ 8Q).

Measures:
- Real 20-page PDF (Makita_XFD12_manual.pdf, 35 questions):
  Page Recall@1/@3/@5, Answer Snippet Hit@1/@3/@5, Page MRR, Latency (ms), Bad Cases
- Chinese RFQ Fixture (p0-fixture.json, 7 grounded + 1 refusal):
  Evidence Recall@1/@3/@5, MRR, Refusal Accuracy, Latency (ms)
"""

import argparse
import json
from pathlib import Path
import re

from rag_engine import embed_text, hybrid_search
from web_app import _extract_text_from_file


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\\n", " ").replace("\u00ad", "")).strip().casefold()


def _snippet_in_chunks(snippet: str, chunks: list[dict]) -> bool:
    needle = _norm(snippet)
    if not needle:
        return False
    # Whole reference snippet in a single chunk. Word boundaries avoid counting
    # partial IDs and numbers such as 18 in 118 or IP54 in IP540.
    left = r"(?<!\w)" if needle[0].isalnum() else ""
    right = r"(?!\w)" if needle[-1].isalnum() else ""
    pattern = re.compile(left + re.escape(needle) + right)
    return any(pattern.search(_norm(chunk.get("content", ""))) for chunk in chunks)


def eval_makita_pdf(
    pdf_path: Path,
    jsonl_path: Path,
    retrieval_mode: str = "混合检索（语义 + 关键词）",
    threshold: float = 0.55,
    chunk_size: int = 480,
    chunk_overlap: int = 64,
    show_bad_cases: bool = True,
    include_cases: bool = False,
) -> dict:
    raw_chunks = _extract_text_from_file(
        pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, ocr_enabled=False
    )
    candidates = []
    for idx, ch in enumerate(raw_chunks, 1):
        vec = embed_text(ch["content"])
        candidates.append({
            "id": f"m-{idx}",
            "doc_id": "makita_drill.pdf",
            "doc_name": pdf_path.name,
            "page": int(ch["page"]),
            "section": ch["section"],
            "content": ch["content"],
            "embedding_vec": vec,
        })

    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    p1 = p3 = p5 = s1 = s3 = s5 = 0
    page_total = 0
    snip_total = 0
    unanswerable_total = 0
    unanswerable_refused = 0
    mrr_sum = 0.0
    latencies = []
    bad_cases = []
    case_results = []

    for row in rows:
        is_unanswerable = str(row.get("category", "")).strip().lower() == "unanswerable" or _norm(row.get("gt_answer_snippet", "")) == "not answered"
        raw_pg = row.get("gt_page_number")
        gt_page = int(str(raw_pg).strip()) if raw_pg is not None and str(raw_pg).strip().isdigit() else None
        gt_snip = row.get("gt_answer_snippet") or ""
        q = row["question_text"]

        res = hybrid_search(
            query=q,
            candidates=candidates,
            retrieval_mode=retrieval_mode,
            similarity_threshold=threshold,
            top_k=5,
        )
        latencies.append(res["latency_ms"])
        hits = res["hits"]
        pages = [h["page"] for h in hits]

        if is_unanswerable:
            unanswerable_total += 1
            if len(hits) == 0:
                unanswerable_refused += 1
            else:
                bad_cases.append({
                    "qid": row["question_id"], "cat": "Unanswerable",
                    "gt_page": None, "got_pages": pages, "q": q[:80], "snip": "should refuse",
                })
            case_results.append({"qid": row["question_id"], "page_ranks": pages, "empty_result": not hits,
                                 "answerable": False})
            continue

        page_ok3 = True
        if gt_page is not None:
            page_total += 1
            if gt_page in pages[:1]:
                p1 += 1
            if gt_page in pages[:3]:
                p3 += 1
            if gt_page in pages[:5]:
                p5 += 1
            page_ok3 = gt_page in pages[:3]
            rr = 0.0
            for rank, pg in enumerate(pages, 1):
                if pg == gt_page:
                    rr = 1.0 / rank
                    break
            mrr_sum += rr

        snip_ok3 = True
        if gt_snip.strip():
            snip_total += 1
            if _snippet_in_chunks(gt_snip, hits[:1]):
                s1 += 1
            if _snippet_in_chunks(gt_snip, hits[:3]):
                s3 += 1
            if _snippet_in_chunks(gt_snip, hits[:5]):
                s5 += 1
            snip_ok3 = _snippet_in_chunks(gt_snip, hits[:3])
        case_results.append({
            "qid": row["question_id"], "answerable": True, "reference_page": gt_page,
            "page_ranks": pages, "empty_result": not hits,
            "exact_snippet_hit_at": [k for k in (1, 3, 5) if _snippet_in_chunks(gt_snip, hits[:k])],
        })

        if not page_ok3 or not snip_ok3:
            bad_cases.append({
                "qid": row["question_id"],
                "cat": row["category"],
                "gt_page": gt_page,
                "got_pages": pages,
                "q": q[:80],
                "snip": _norm(gt_snip)[:60],
            })

    report = {
        "dataset": "Makita_XFD12_35Q",
        "mode": retrieval_mode,
        "chunks": len(candidates),
        "answerable_count": page_total,
        "snippet_count": snip_total,
        "snippet_match": "exact_within_one_chunk",
        "page_recall_at_1": round(p1 / max(1, page_total), 4),
        "page_recall_at_3": round(p3 / max(1, page_total), 4),
        "page_recall_at_5": round(p5 / max(1, page_total), 4),
        "snippet_hit_at_1": round(s1 / max(1, snip_total), 4),
        "snippet_hit_at_3": round(s3 / max(1, snip_total), 4),
        "snippet_hit_at_5": round(s5 / max(1, snip_total), 4),
        "page_mrr": round(mrr_sum / max(1, page_total), 4),
        "unanswerable_refusal": f"{unanswerable_refused}/{unanswerable_total}",
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "bad_case_count": len(bad_cases),
    }
    if show_bad_cases:
        report["sample_bad_cases"] = bad_cases[:8]
    if include_cases:
        report["cases"] = case_results
    return report


def eval_cn_rfq(fixture_path: Path, retrieval_mode: str = "混合检索（语义 + 关键词）", threshold: float = 0.62) -> dict:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    candidates = []
    for ev in data["evidence"]:
        candidates.append({
            "id": ev["id"],
            "doc_id": ev["source"].split()[0],
            "doc_name": ev["source"].split()[0],
            "page": 1,
            "section": ev["source"],
            "content": ev["text"],
            "embedding_vec": embed_text(ev["text"]),
        })
    r1_l, r3_l, r5_l, mrr_l, lats = [], [], [], [], []
    ref_ok = ref_tot = 0
    for case in data["cases"]:
        exp = set(case["expected_evidence"])
        res = hybrid_search(case["question"], candidates, retrieval_mode=retrieval_mode, similarity_threshold=threshold, top_k=5)
        lats.append(res["latency_ms"])
        ids = [h["id"] for h in res["hits"]]
        if not exp:
            ref_tot += 1
            if not ids:
                ref_ok += 1
            continue
        r1_l.append(len(exp & set(ids[:1])) / len(exp))
        r3_l.append(len(exp & set(ids[:3])) / len(exp))
        r5_l.append(len(exp & set(ids[:5])) / len(exp))
        mrr_l.append(next((1.0 / i for i, hid in enumerate(ids, 1) if hid in exp), 0.0))
    return {
        "dataset": "CN_RFQ_P0_8Q",
        "mode": retrieval_mode,
        "recall_at_1": round(sum(r1_l) / len(r1_l), 4),
        "recall_at_3": round(sum(r3_l) / len(r3_l), 4),
        "recall_at_5": round(sum(r5_l) / len(r5_l), 4),
        "mrr": round(sum(mrr_l) / len(mrr_l), 4),
        "refusal": f"{ref_ok}/{ref_tot}",
        "avg_latency_ms": round(sum(lats) / len(lats), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--include-cases", action="store_true")
    parser.add_argument("--hybrid-threshold", type=float, default=0.62)
    args = parser.parse_args()
    workspace_root = args.data_root
    pdf = workspace_root / "公开测试数据" / "Makita_XFD12_manual.pdf"
    jsonl = workspace_root / "公开测试数据" / "Makita_XFD12_35题.jsonl"
    cn_fixture = workspace_root / "p0-fixture.json"

    out = {
        "makita_hybrid": eval_makita_pdf(pdf, jsonl, "混合检索（语义 + 关键词）", threshold=args.hybrid_threshold,
                                        include_cases=args.include_cases),
        "makita_bm25": eval_makita_pdf(pdf, jsonl, "精确关键词检索", threshold=0.55, show_bad_cases=False,
                                      include_cases=args.include_cases),
        "makita_bge": eval_makita_pdf(pdf, jsonl, "语义向量检索", threshold=0.50, show_bad_cases=False,
                                     include_cases=args.include_cases),
        "cn_rfq_hybrid": eval_cn_rfq(cn_fixture, "混合检索（语义 + 关键词）", threshold=0.62),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
