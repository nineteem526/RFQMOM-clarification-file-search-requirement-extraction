"""Core RAG Engine for Multi-Project RFQ Workbench.

Provides:
- Local offline BGE-small-zh-v1.5 (512-dim) embeddings via local_embedding.py
- Overlap-aware engineering text chunking (preserves specs, units, and clause boundaries)
- Hybrid retrieval: BM25 lexical scoring (CN + EN model/spec tokens) + BGE cosine similarity + RRF fusion
- Pluggable OpenAI-compatible / Company Azure LLM answer synthesis with deterministic local fallback
"""

import json
import math
import os
import re
import time
from typing import Any

import numpy as np

from local_embedding import LocalEmbeddingError, local_engine

_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._/\-]*|[\u4e00-\u9fff]")
_CN_STOPWORDS = frozenset({
    "的", "了", "是", "在", "和", "与", "及", "或", "有", "吗", "呢", "啊",
    "什", "么", "哪", "些", "多", "少", "请", "问", "这", "那", "对", "于",
    "中", "为", "应", "可", "以", "按", "将", "被", "把", "就", "都", "也",
    "还", "要", "求", "是", "否", "已", "经", "如", "何", "该", "本", "项",
})


def tokenize_engineering(text: str) -> list[str]:
    """Tokenize Chinese + English engineering text preserving model IDs & numeric specs."""
    if not text:
        return []
    raw = _TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    cn_run: list[str] = []

    def flush_cn():
        if not cn_run:
            return
        # Add individual meaningful characters and bigrams for Chinese recall
        for ch in cn_run:
            if ch not in _CN_STOPWORDS:
                tokens.append(ch)
        for i in range(len(cn_run) - 1):
            bg = cn_run[i] + cn_run[i + 1]
            if not (cn_run[i] in _CN_STOPWORDS and cn_run[i + 1] in _CN_STOPWORDS):
                tokens.append(bg)
        cn_run.clear()

    for tok in raw:
        if len(tok) == 1 and "\u4e00" <= tok <= "\u9fff":
            cn_run.append(tok)
        else:
            flush_cn()
            tokens.append(tok)
    flush_cn()
    return tokens


def split_text_with_overlap(text: str, chunk_size: int = 480, chunk_overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks without breaking short clauses."""
    clean = (text or "").strip()
    if not clean:
        return []
    size = max(120, min(int(chunk_size or 480), 1200))
    overlap = max(0, min(int(chunk_overlap or 64), size // 2))
    if len(clean) <= size:
        return [clean]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 <= size:
            buf = (buf + "\n\n" + para).strip() if buf else para
            continue
        if buf:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = (tail + "\n" + para).strip() if tail else para
        else:
            # Single paragraph longer than chunk_size: sliding window
            step = max(1, size - overlap)
            for start in range(0, len(para), step):
                part = para[start:start + size].strip()
                if part:
                    chunks.append(part)
                if start + size >= len(para):
                    break
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def _safe_bge_text(text: str, engine) -> str:
    """Ensure text fits within BGE's strict 512-token limit."""
    s = (text or "").strip()
    if not s:
        return "empty"
    encoded = engine.tokenizer.encode(s)
    if len(encoded.ids) <= 512:
        return s
    # Binary-shrink character length cleanly until <= 510 tokens
    lo, hi = 1, len(s)
    best = s[:256]
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = s[:mid].strip()
        if not cand:
            lo = mid + 1
            continue
        if len(engine.tokenizer.encode(cand).ids) <= 510:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def embed_text(text: str) -> np.ndarray:
    """Compute a normalized 512-dim float32 embedding using the local offline BGE model."""
    engine = local_engine()
    safe = _safe_bge_text(text, engine)
    encoded = engine.encode_input(safe)
    vec = engine.embed_encoded(encoded)
    return np.asarray(vec, dtype=np.float32)


def pack_vector(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack_vector(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr if arr.size == 512 else None


def embedding_health() -> dict[str, Any]:
    try:
        engine = local_engine()
        return {"ok": True, "model": "BAAI/bge-small-zh-v1.5 (ONNX CPU)", "dim": 512, "ready": engine is not None}
    except Exception as exc:
        return {"ok": False, "model": "BAAI/bge-small-zh-v1.5", "error": str(exc)}


def llm_config_status() -> dict[str, Any]:
    """Detect whether an OpenAI-compatible or Company Azure LLM key is present in env."""
    api_key = os.environ.get("RFQ_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    base_url = os.environ.get("RFQ_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
    model = os.environ.get("RFQ_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    if api_key.strip():
        return {
            "configured": True,
            "provider": "openai-compatible",
            "model": model,
            "base_url": base_url or "https://api.openai.com/v1",
            "status_text": f"已连接 ({model})",
        }
    if os.environ.get("DUERR_API_KEY", "").strip():
        return {
            "configured": True,
            "provider": "company-azure",
            "model": os.environ.get("DUERR_MODEL", "gpt-5.2"),
            "status_text": "已连接 (企业内网 Azure)",
        }
    return {
        "configured": False,
        "provider": "local-fallback",
        "model": "本地检索汇总（未配置在线 API）",
        "status_text": "本地检索模式（未配置 API Key）",
    }


def _bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    n_docs = len(docs_tokens)
    if n_docs == 0 or not query_tokens:
        return [0.0] * n_docs
    avgdl = sum(len(t) for t in docs_tokens) / max(1, n_docs)
    df: dict[str, int] = {}
    q_unique = set(query_tokens)
    for toks in docs_tokens:
        present = set(toks)
        for qt in q_unique:
            if qt in present:
                df[qt] = df.get(qt, 0) + 1
    scores = []
    for toks in docs_tokens:
        dl = len(toks)
        if dl == 0:
            scores.append(0.0)
            continue
        tf: dict[str, int] = {}
        for t in toks:
            if t in q_unique:
                tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for qt in query_tokens:
            f = tf.get(qt, 0)
            if f <= 0:
                continue
            idf = math.log((n_docs - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1.0)
            # Boost exact alphanumeric model/spec tokens (e.g., IP54, CAB-001, 24V)
            weight = 2.4 if re.search(r"[a-z0-9]", qt) else (1.5 if len(qt) >= 2 else 0.7)
            s += weight * idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / max(1.0, avgdl)))
        scores.append(s)
    return scores


def hybrid_search(
    query: str,
    candidates: list[dict[str, Any]],
    retrieval_mode: str = "混合检索（语义 + 关键词）",
    similarity_threshold: float = 0.62,
    top_k: int = 10,
    rrf_k: int = 60,
) -> dict[str, Any]:
    """Run semantic, keyword, or RRF hybrid retrieval over a project's chunks."""
    t0 = time.perf_counter()
    if not candidates or not (query or "").strip():
        return {"hits": [], "latency_ms": 0.0, "mode": retrieval_mode, "total_candidates": len(candidates)}

    use_sem = retrieval_mode != "精确关键词检索"
    use_kw = retrieval_mode != "语义向量检索"
    threshold = float(similarity_threshold)
    limit = max(1, min(int(top_k or 10), 50))

    # 1. Keyword BM25
    q_tokens = tokenize_engineering(query)
    if use_kw:
        docs_tokens = [
            tokenize_engineering(f"{c.get('doc_name', '')} {c.get('section', '')} {c.get('content', '')}")
            for c in candidates
        ]
        kw_scores = _bm25_scores(q_tokens, docs_tokens)
    else:
        kw_scores = [0.0] * len(candidates)

    # 2. Semantic BGE cosine similarity
    sem_scores = [0.0] * len(candidates)
    bge_used = False
    if use_sem:
        try:
            q_vec = embed_text(query)
            bge_used = True
            for i, c in enumerate(candidates):
                vec = c.get("embedding_vec")
                if vec is None and c.get("embedding_blob"):
                    vec = unpack_vector(c["embedding_blob"])
                if vec is None:
                    vec = embed_text(c.get("content", ""))
                sem_scores[i] = float(np.dot(q_vec, vec))
        except LocalEmbeddingError:
            bge_used = False

    # 3. Rank & Fuse
    sem_order = sorted(range(len(candidates)), key=lambda i: sem_scores[i], reverse=True)
    kw_order = sorted(range(len(candidates)), key=lambda i: kw_scores[i], reverse=True)
    sem_rank = {idx: r + 1 for r, idx in enumerate(sem_order)}
    kw_rank = {idx: r + 1 for r, idx in enumerate(kw_order)}

    # Specific technical tokens in query (alphanumeric codes or 2-char Chinese terms)
    alnum_terms = [t for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9._/\-]{1,}", query.lower())]
    cn_bigrams = [t for t in q_tokens if len(t) >= 2 and not re.search(r"[a-z0-9]", t)]

    results = []
    for i, c in enumerate(candidates):
        s_score = sem_scores[i]
        k_score = kw_scores[i]
        hay_lower = (c.get("content", "") + " " + c.get("section", "") + " " + c.get("doc_name", "")).lower()
        exact_hits = [t for t in alnum_terms if t in hay_lower] + [t for t in cn_bigrams if t in hay_lower]

        if retrieval_mode == "语义向量检索":
            if s_score < threshold:
                continue
            final_score = s_score
        elif retrieval_mode == "精确关键词检索":
            if k_score <= 0.0:
                continue
            final_score = k_score
        else:
            # Hybrid RRF with strict out-of-scope / empty-evidence gate:
            # Accept if semantic >= threshold OR strong lexical match with solid semantic alignment
            sem_ok = bge_used and s_score >= threshold
            kw_ok = k_score >= 2.2 and (not bge_used or s_score >= max(0.48, threshold - 0.10))
            if not (sem_ok or kw_ok):
                continue
            final_score = 0.0
            if bge_used and s_score >= max(0.30, threshold - 0.25):
                final_score += 1.0 / (rrf_k + sem_rank[i])
            if k_score > 0.0:
                final_score += 1.0 / (rrf_k + kw_rank[i])
            # Slight tie-break boost for exact spec/model matches
            if exact_hits:
                final_score += 0.004 * min(3, len(exact_hits))

        item = {k: v for k, v in c.items() if k not in {"embedding_blob", "embedding_vec"}}
        item["semantic_score"] = round(s_score, 4)
        item["keyword_score"] = round(k_score, 4)
        item["final_score"] = round(final_score, 6)
        item["matched_terms"] = list(dict.fromkeys(exact_hits))[:6]
        results.append(item)

    results.sort(key=lambda x: (x["final_score"], x["semantic_score"], x["keyword_score"]), reverse=True)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "hits": results[:limit],
        "latency_ms": latency_ms,
        "mode": retrieval_mode,
        "bge_used": bge_used,
        "total_candidates": len(candidates),
    }


def _call_external_llm(question: str, citations: list[dict[str, Any]], project_name: str) -> dict[str, Any] | None:
    """Optionally call an OpenAI-compatible or Company Azure LLM if configured via env."""
    cfg = llm_config_status()
    if not cfg["configured"]:
        return None
    evidence_block = "\n\n".join(
        f"[{c['index']}] {c['file_name']} (第 {c['page']} 页 · {c['section']})\n{c['content']}"
        for c in citations
    )
    system_prompt = (
        "你是严谨的工程询价文档(RFQ)智能处理助手。请仅依据提供的原文证据回答问题，严禁编造任何未在证据中出现的数值、型号或结论。"
        "必须严格区分“已确认/已批准”与“待批准/建议/讨论稿”，不得把待批准建议写成已生效要求。"
        "请只输出合法 JSON，包含字段："
        '{"conclusion": "带[1][2]引用标号的中文总结，关键参数可用**加粗**", '
        '"basis": [{"index": 1, "source": "文件名（第X页，章节）", "detail": "该依据的关键内容说明 [1]"}], '
        '"table": [{"item": "参数/属性名", "value": "对应要求及来源 [1]"}]}'
    )
    user_prompt = f"项目：{project_name}\n问题：{question}\n\n检索到的原文证据：\n{evidence_block}"
    try:
        if cfg["provider"] == "openai-compatible":
            from openai import OpenAI

            client = OpenAI(
                api_key=os.environ.get("RFQ_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("RFQ_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or None,
                timeout=45.0,
            )
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
        else:
            from company_chat import CompanyAzureChat

            resp = CompanyAzureChat(completion_budget=2048).invoke(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            raw = resp.text or ""
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("conclusion"):
            parsed["llm_provider"] = cfg["status_text"]
            return parsed
    except Exception:
        return None
    return None


def generate_structured_answer(
    question: str,
    hits: list[dict[str, Any]],
    project_name: str,
    retrieval_meta: dict[str, Any],
    mode: str = "快速问答",
) -> dict[str, Any]:
    """Synthesize a structured, citation-grounded answer (LLM if configured, else deterministic local)."""
    citations = []
    for idx, hit in enumerate(hits[:5], 1):
        snippet = hit["content"].strip()
        hl = hit.get("highlight") or (hit.get("matched_terms") or [""])[0]
        citations.append({
            "index": idx,
            "chunk_id": hit["id"],
            "doc_id": hit["doc_id"],
            "file_name": hit["doc_name"],
            "version": hit.get("version", "Rev.0"),
            "page": hit["page"],
            "total_pages": hit.get("doc_pages", 1),
            "section": hit.get("section", ""),
            "content": snippet,
            "highlight": hl,
            "semantic_score": hit.get("semantic_score", 0.0),
            "keyword_score": hit.get("keyword_score", 0.0),
        })

    if not citations:
        return {
            "mode": mode,
            "retrieval": retrieval_meta,
            "llm_provider": llm_config_status()["status_text"],
            "conclusion": (
                "当前项目的独立数据库中未检索到满足阈值的原文依据，**证据不足，不作推测**。"
                "请确认相关工程文件已上传并索引，或在“设置”中调整检索方式/相似度阈值后重试。"
            ),
            "basis": [],
            "table": [
                {"item": "查询项目", "value": project_name},
                {"item": "检索状态", "value": f"未命中（已扫描 {retrieval_meta.get('total_candidates', 0)} 个文本块，耗时 {retrieval_meta.get('latency_ms', 0)} ms）"},
            ],
            "citations": [],
        }

    # Try external LLM first if configured
    llm_out = _call_external_llm(question, citations, project_name)
    if llm_out:
        return {
            "mode": mode,
            "retrieval": retrieval_meta,
            "llm_provider": llm_out.get("llm_provider", "在线模型"),
            "conclusion": llm_out["conclusion"],
            "basis": llm_out.get("basis", []),
            "table": llm_out.get("table", []),
            "citations": citations,
        }

    # Deterministic local grounded synthesis (zero hallucination)
    basis_items = []
    pending_flag = False
    for c in citations:
        txt = c["content"].replace("\n", " ").strip()
        if any(w in txt for w in ("待批准", "待业主批准", "建议", "讨论稿", "待确认")):
            pending_flag = True
        short = txt[:130] + ("…" if len(txt) > 130 else "")
        basis_items.append({
            "index": c["index"],
            "source": f"{c['file_name']}（第 {c['page']} 页，{c['section']}）",
            "detail": f"{short} [{c['index']}]",
        })

    # Extract concise key sentence from top citation
    top = citations[0]
    top_sentences = [s.strip() for s in re.split(r"[。\n]+", top["content"]) if s.strip()]
    lead_sentence = top_sentences[0][:140] if top_sentences else top["content"][:140]
    for s in top_sentences:
        if any(t in s.lower() for t in (hits[0].get("matched_terms") or [])):
            lead_sentence = s[:150]
            break

    approval_note = " 注意：检索证据中包含**待批准/建议类条款**，尚未自动替代原已确认要求，请结合右侧原文核实审批状态。" if pending_flag else ""
    ref_tags = "".join(f" [{c['index']}]" for c in citations[:2])
    conclusion = (
        f"根据当前项目数据库的混合检索结果（耗时 {retrieval_meta.get('latency_ms', 0)} ms），"
        f"最相关原文指出：**{lead_sentence}**{ref_tags}。{approval_note}"
    )

    summary_rows = [
        {"item": "检索方式", "value": f"{retrieval_meta.get('mode', mode)}（本地 BGE 512维 + BM25）"},
        {"item": "最高语义分", "value": f"{top['semantic_score']:.3f}（共命中 {len(citations)} 条高相关依据）"},
        {"item": "审批/状态提示", "value": "含待批准或建议条款，需核对原件" if pending_flag else "未检出待批准字样，仍以原件审批章为准"},
        {"item": "主要来源", "value": "；".join(f"{c['file_name']} 第 {c['page']} 页 [{c['index']}]" for c in citations[:2])},
    ]

    return {
        "mode": mode,
        "retrieval": retrieval_meta,
        "llm_provider": llm_config_status()["status_text"],
        "conclusion": conclusion,
        "basis": basis_items,
        "table": summary_rows,
        "citations": citations,
    }
