"""P0 Retrieval & Refusal Benchmark for the Multi-Project RFQ RAG Engine.

Evaluates:
- Recall@1, Recall@3, Recall@5, MRR on grounded engineering cases
- Empty-evidence refusal accuracy (evaluated separately, never in Recall denominator)
- Average per-query latency (ms)
Across 3 modes: Keyword-only (BM25), Semantic-only (Local BGE 512d), and Hybrid (RRF).
"""

import json
from pathlib import Path
import sys

from rag_engine import embed_text, hybrid_search


def run_benchmark(fixture_path: Path):
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    evidence_rows = data["evidence"]
    cases = data["cases"]

    # Pre-compute real local BGE 512-dim embeddings once (same as indexing time)
    candidates = []
    for ev in evidence_rows:
        vec = embed_text(ev["text"])
        candidates.append({
            "id": ev["id"],
            "doc_id": ev["source"].split()[0],
            "doc_name": ev["source"].split()[0],
            "page": 1,
            "section": ev["source"],
            "content": ev["text"],
            "embedding_vec": vec,
        })

    modes = [
        ("精确关键词检索", 0.62),
        ("语义向量检索", 0.58),
        ("混合检索（语义 + 关键词）", 0.62),
    ]

    summary = []
    for mode, threshold in modes:
        r1_list, r3_list, r5_list, mrr_list, latencies = [], [], [], [], []
        refusal_total = 0
        refusal_ok = 0

        for case in cases:
            expected = set(case["expected_evidence"])
            res = hybrid_search(
                query=case["question"],
                candidates=candidates,
                retrieval_mode=mode,
                similarity_threshold=threshold,
                top_k=5,
            )
            latencies.append(res["latency_ms"])
            hit_ids = [h["id"] for h in res["hits"]]

            if not expected:
                refusal_total += 1
                if len(hit_ids) == 0:
                    refusal_ok += 1
                continue

            r1 = len(expected.intersection(hit_ids[:1])) / len(expected)
            r3 = len(expected.intersection(hit_ids[:3])) / len(expected)
            r5 = len(expected.intersection(hit_ids[:5])) / len(expected)
            rr = 0.0
            for rank, hid in enumerate(hit_ids, 1):
                if hid in expected:
                    rr = 1.0 / rank
                    break
            r1_list.append(r1)
            r3_list.append(r3)
            r5_list.append(r5)
            mrr_list.append(rr)

        summary.append({
            "mode": mode,
            "threshold": threshold,
            "recall_at_1": round(sum(r1_list) / max(1, len(r1_list)), 4),
            "recall_at_3": round(sum(r3_list) / max(1, len(r3_list)), 4),
            "recall_at_5": round(sum(r5_list) / max(1, len(r5_list)), 4),
            "mrr": round(sum(mrr_list) / max(1, len(mrr_list)), 4),
            "refusal_accuracy": f"{refusal_ok}/{refusal_total}",
            "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        })

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    default_fixture = Path(__file__).resolve().parents[2] / "p0-fixture.json"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default_fixture
    run_benchmark(target)
