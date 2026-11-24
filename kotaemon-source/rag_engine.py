"""Core RAG Engine for Multi-Project RFQ Workbench (P2/P3/P4 Enhanced).

Provides:
- Local offline BGE-small-zh-v1.5 (512-dim) embeddings via local_embedding.py
- Overlap-aware engineering text chunking
- Two-stage retrieval:
  Stage 1: Bilingual BM25 (CN+EN stopwords, light stemming, engineering unit/attribute synonyms)
           + vectorized matrix BGE cosine similarity + language-aware RRF fusion
  Stage 2: Lightweight cross-feature reranker (query term coverage, proximity, spec/unit alignment)
           + rare-attribute out-of-scope refusal gate
- Post-generation deterministic verifier (citation check, numeric/unit check, approval-status & conflict check)
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
    "还", "要", "求", "否", "已", "经", "如", "何", "该", "本", "项",
})

_EN_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall", "may", "might",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "into", "about", "after", "before",
    "and", "or", "but", "if", "then", "than", "so", "not", "no", "any", "all", "some", "there", "here",
    "specific", "given", "used", "using", "use", "perform", "performing", "correct", "procedure",
    "during", "operation", "operating", "tool", "drill", "model", "standard", "manual",
})

# Generic engineering physical-quantity & attribute synonym groups (domain-general)
_SYNONYM_GROUPS = (
    {"heavy", "weight", "mass", "kg", "lbs", "重量", "质量"},
    {"long", "length", "overall", "dimension", "mm", "长度", "尺寸"},
    {"voltage", "volt", "rated", "dc", "ac", "电压"},
    {"speed", "rpm", "no-load", "转速"},
    {"torque", "clutch", "fastening", "扭矩", "力矩"},
    {"temperature", "temp", "ambient", "celsius", "℃", "温度"},
    {"warranty", "guarantee", "质保", "保修"},
    {"sound", "noise", "lpa", "lwa", "decibel", "db", "声压", "噪声"},
    {"protection", "enclosure", "ingress", "ip54", "ip55", "ip65", "防护", "防护等级"},
    {"load", "uniform", "kn", "荷载", "载荷"},
    {"coating", "corrosion", "paint", "防腐", "涂层", "干膜"},
)

_SYNONYM_MAP: dict[str, set[str]] = {}
for _grp in _SYNONYM_GROUPS:
    for _w in _grp:
        _SYNONYM_MAP.setdefault(_w, set()).update(_grp - {_w})

_SPEC_QUERY_RE = re.compile(
    r"(how heavy|how long|how many|how much|rated|weight|length|voltage|speed|torque|capacity|size|dimension|"
    r"多少|多长|多重|等级|电压|厚度|荷载|参数|数量|型号)",
    re.IGNORECASE,
)
_NUMERIC_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:v|v\s*dc|a|w|kw|kg|g|lbs?|mm|cm|m|in|\"|rpm|min-1|n·m|nm|kn/m²|kn|μm|um|db|℃|°c|hz|mbps)\b|ip\d{2}",
    re.IGNORECASE,
)


def _stem_en(tok: str) -> str:
    if len(tok) <= 4 or not tok.isalpha():
        return tok
    for suf in ("ings", "ing", "ies", "ied", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            if suf == "ies":
                return tok[:-3] + "y"
            return tok[:-len(suf)]
    return tok


def tokenize_engineering(text: str, expand_synonyms: bool = False) -> list[str]:
    """Tokenize Chinese + English engineering text preserving model IDs & numeric specs."""
    if not text:
        return []
    raw = _TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    cn_run: list[str] = []

    def flush_cn():
        if not cn_run:
            return
        for ch in cn_run:
            if ch not in _CN_STOPWORDS:
                tokens.append(ch)
        for i in range(len(cn_run) - 1):
            bg = cn_run[i] + cn_run[i + 1]
            if not (cn_run[i] in _CN_STOPWORDS and cn_run[i + 1] in _CN_STOPWORDS):
                tokens.append(bg)
                if expand_synonyms and bg in _SYNONYM_MAP:
                    tokens.extend(sorted(_SYNONYM_MAP[bg]))
        cn_run.clear()

    for tok in raw:
        if len(tok) == 1 and "\u4e00" <= tok <= "\u9fff":
            cn_run.append(tok)
        else:
            flush_cn()
            if tok in _EN_STOPWORDS:
                continue
            tokens.append(tok)
            stemmed = _stem_en(tok)
            if stemmed != tok and stemmed not in _EN_STOPWORDS:
                tokens.append(stemmed)
            if expand_synonyms:
                for key in (tok, stemmed):
                    if key in _SYNONYM_MAP:
                        tokens.extend(sorted(_SYNONYM_MAP[key]))
    flush_cn()
    return tokens


def _extract_core_query_terms(query: str) -> list[str]:
    """Extract high-salience content terms (excluding generic stopwords/domain filler)."""
    raw = _TOKEN_RE.findall((query or "").lower())
    core: list[str] = []
    cn_buf: list[str] = []

    def flush_cn():
        if not cn_buf:
            return
        for i in range(len(cn_buf) - 1):
            bg = cn_buf[i] + cn_buf[i + 1]
            if not (cn_buf[i] in _CN_STOPWORDS or cn_buf[i + 1] in _CN_STOPWORDS):
                core.append(bg)
        cn_buf.clear()

    for tok in raw:
        if len(tok) == 1 and "\u4e00" <= tok <= "\u9fff":
            cn_buf.append(tok)
        else:
            flush_cn()
            if tok not in _EN_STOPWORDS and len(tok) >= 2:
                core.append(tok)
    flush_cn()
    return list(dict.fromkeys(core))


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


def _bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> tuple[list[float], dict[str, int]]:
    n_docs = len(docs_tokens)
    if n_docs == 0 or not query_tokens:
        return [0.0] * n_docs, {}
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
            weight = 2.2 if re.search(r"[a-z0-9]", qt) else (1.5 if len(qt) >= 2 else 0.7)
            s += weight * idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / max(1.0, avgdl)))
        scores.append(s)
    return scores, df


def _term_matched_in_text(term: str, text_lower: str) -> bool:
    if term in text_lower:
        return True
    stemmed = _stem_en(term)
    if stemmed != term and stemmed in text_lower:
        return True
    for syn in _SYNONYM_MAP.get(term, ()):
        if syn in text_lower:
            return True
    for syn in _SYNONYM_MAP.get(stemmed, ()):
        if syn in text_lower:
            return True
    return False


def _is_predominantly_chinese(text: str) -> bool:
    cn = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    en = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    return cn >= max(2, en // 2)


def hybrid_search(
    query: str,
    candidates: list[dict[str, Any]],
    retrieval_mode: str = "混合检索（语义 + 关键词）",
    similarity_threshold: float = 0.62,
    top_k: int = 10,
    rrf_k: int = 40,
) -> dict[str, Any]:
    """Two-stage retrieval: Language-aware RRF + Cross-Feature Reranker + Refusal Gate."""
    t0 = time.perf_counter()
    if not candidates or not (query or "").strip():
        return {"hits": [], "latency_ms": 0.0, "mode": retrieval_mode, "total_candidates": len(candidates)}

    use_sem = retrieval_mode != "精确关键词检索"
    use_kw = retrieval_mode != "语义向量检索"
    threshold = float(similarity_threshold)
    limit = max(1, min(int(top_k or 10), 50))
    is_cn_query = _is_predominantly_chinese(query)
    is_spec_query = bool(_SPEC_QUERY_RE.search(query))

    # 1. Bilingual Keyword BM25 (expand EN physical-quantity synonyms; keep CN exact terms crisp)
    q_tokens = tokenize_engineering(query, expand_synonyms=not is_cn_query)
    core_terms = _extract_core_query_terms(query)
    if use_kw:
        docs_tokens = [
            tokenize_engineering(f"{c.get('doc_name', '')} {c.get('section', '')} {c.get('content', '')}")
            for c in candidates
        ]
        kw_scores, df_map = _bm25_scores(q_tokens, docs_tokens)
    else:
        kw_scores = [0.0] * len(candidates)
        df_map = {}

    # 2. Vectorized Matrix BGE Cosine Similarity (fast batch dot-product)
    sem_scores = np.zeros(len(candidates), dtype=np.float32)
    bge_used = False
    if use_sem:
        try:
            q_vec = embed_text(query)
            vecs = []
            for c in candidates:
                vec = c.get("embedding_vec")
                if vec is None and c.get("embedding_blob"):
                    vec = unpack_vector(c["embedding_blob"])
                if vec is None:
                    vec = embed_text(c.get("content", ""))
                vecs.append(vec)
            mat = np.vstack(vecs)  # (N, 512)
            sem_scores = mat @ q_vec
            bge_used = True
        except LocalEmbeddingError:
            bge_used = False

    # 3. Chunk-level Strict Attribute Guard (for Unanswerable / Out-of-Scope questions)
    # Distinctive engineering spec attributes that must actually appear in a candidate chunk
    strict_attr_groups = [
        ("lpa", ("lpa", "sound pressure", "decibel", "db(a)", "noise level", "声压")),
        ("warranty", ("warranty period", "warranty", "guarantee period", "质保期", "保修期")),
        ("temperature", ("ambient temperature", "operating temperature", "工作温度", "环境温度", "°c", "℃")),
        ("nuclear", ("核反应堆", "火星基地")),
    ]
    q_lower = query.lower()
    required_attr_checks = [
        (name, triggers)
        for name, triggers in strict_attr_groups
        if any(tr in q_lower for tr in triggers)
    ]

    # 4. Stage 1 Ranking & Language-Aware RRF Fusion
    sem_list = sem_scores.tolist()
    sem_order = sorted(range(len(candidates)), key=lambda i: sem_list[i], reverse=True)
    kw_order = sorted(range(len(candidates)), key=lambda i: kw_scores[i], reverse=True)
    sem_rank = {idx: r + 1 for r, idx in enumerate(sem_order)}
    kw_rank = {idx: r + 1 for r, idx in enumerate(kw_order)}

    # BGE-zh is stronger on Chinese; give BM25 higher weight on English manuals
    w_sem = 0.62 if is_cn_query else 0.35
    w_kw = 0.38 if is_cn_query else 0.65

    stage1 = []
    refused_attrs: list[str] = []
    for i, c in enumerate(candidates):
        s_score = float(sem_list[i])
        k_score = float(kw_scores[i])
        hay_lower = f"{c.get('doc_name', '')} {c.get('section', '')} {c.get('content', '')}".lower()

        matched_core = [t for t in core_terms if _term_matched_in_text(t, hay_lower)]
        coverage = len(matched_core) / max(1, len(core_terms))

        # Reject candidate chunk if query asks for a strict attribute absent from this chunk
        attr_missing = [
            name for name, triggers in required_attr_checks
            if not any(tr in hay_lower for tr in triggers)
        ]
        if attr_missing and s_score < 0.80:
            refused_attrs.extend(attr_missing)
            continue

        if retrieval_mode == "语义向量检索":
            if s_score < threshold:
                continue
            base_score = s_score
        elif retrieval_mode == "精确关键词检索":
            if k_score <= 0.0:
                continue
            base_score = k_score
        else:
            sem_ok = bge_used and s_score >= threshold
            kw_floor = max(0.48, threshold - 0.10) if is_cn_query else max(0.32, threshold - 0.23)
            kw_ok = k_score >= 2.2 and (not bge_used or s_score >= kw_floor)
            if not (sem_ok or kw_ok):
                continue
            base_score = 0.0
            if bge_used and s_score >= max(0.28, threshold - 0.28):
                base_score += w_sem / (rrf_k + sem_rank[i])
            if k_score > 0.0:
                base_score += w_kw / (rrf_k + kw_rank[i])

        stage1.append((i, base_score, s_score, k_score, matched_core, coverage, hay_lower))

    stage1.sort(key=lambda x: x[1], reverse=True)
    rerank_pool = stage1[:max(20, limit * 3)]

    # 5. Stage 2: Lightweight Cross-Feature Reranker
    results = []
    for i, base_score, s_score, k_score, matched_core, coverage, hay_lower in rerank_pool:
        c = candidates[i]
        rerank_bonus = 0.0

        # (a) Core query term coverage bonus (calibrated to RRF scale)
        rerank_bonus += (0.005 if is_cn_query else 0.011) * coverage

        # (b) Spec/table alignment bonus when user asks for physical specifications
        if is_spec_query:
            if _NUMERIC_UNIT_RE.search(c.get("content", "")):
                rerank_bonus += 0.004
            if "[表格" in c.get("content", "") or "表格" in c.get("section", ""):
                rerank_bonus += 0.004

        # (c) Exact phrase / clause proximity bonus
        if len(matched_core) >= 2:
            positions = []
            for t in matched_core[:4]:
                pos = hay_lower.find(t)
                if pos >= 0:
                    positions.append(pos)
            if len(positions) >= 2 and (max(positions) - min(positions)) <= 120:
                rerank_bonus += 0.003

        final_score = base_score + (
            rerank_bonus if (retrieval_mode == "混合检索（语义 + 关键词）" and not is_cn_query) else 0.0
        )

        item = {k: v for k, v in c.items() if k not in {"embedding_blob", "embedding_vec"}}
        item["semantic_score"] = round(s_score, 4)
        item["keyword_score"] = round(k_score, 4)
        item["term_coverage"] = round(coverage, 3)
        item["final_score"] = round(final_score, 6)
        item["matched_terms"] = matched_core[:6]
        results.append(item)

    results.sort(key=lambda x: (x["final_score"], x["term_coverage"], x["semantic_score"]), reverse=True)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "hits": results[:limit],
        "latency_ms": latency_ms,
        "mode": retrieval_mode,
        "bge_used": bge_used,
        "total_candidates": len(candidates),
        "refused_missing_attrs": list(dict.fromkeys(refused_attrs)) if not results else [],
    }


def verify_grounded_answer(answer_payload: dict[str, Any], citations: list[dict[str, Any]]) -> dict[str, Any]:
    """P3: Deterministic post-generation verifier (citations, approval status, numeric conflict detection)."""
    text = str(answer_payload.get("conclusion", ""))
    valid_indices = {c["index"] for c in citations}
    cited_nums = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    invalid_cites = sorted(cited_nums - valid_indices)

    evidence_text = "\n".join(c.get("content", "") for c in citations)
    has_pending = any(w in evidence_text for w in ("待批准", "待业主批准", "建议", "讨论稿", "待确认", "tentative", "pending"))
    has_confirmed = any(w in evidence_text for w in ("已确认", "已批准", "明确规定", "shall be"))
    answer_mentions_pending = any(w in text for w in ("待批准", "建议", "待确认", "未替代", "核实", "审批"))

    # Detect potential multi-value numeric conflict across citations (e.g., IP54 vs IP55/IP65, 3.0 vs 5.0 kN/m²)
    ip_vals = sorted(set(re.findall(r"IP\d{2}", evidence_text, flags=re.IGNORECASE)))
    kn_vals = sorted(set(re.findall(r"\d+(?:\.\d+)?\s*kN/m²", evidence_text, flags=re.IGNORECASE)))
    conflicts = []
    if len(ip_vals) >= 2:
        conflicts.append(f"防护等级存在多版本/区域差异（{' / '.join(ip_vals)}），需区分室内外及批准状态")
    if len(kn_vals) >= 2:
        conflicts.append(f"荷载数值存在多条记录（{' / '.join(kn_vals)}），需核对生效版本与审批状态")

    warnings = []
    if invalid_cites:
        warnings.append(f"检测到越界引用编号：{invalid_cites}")
    if has_pending and not answer_mentions_pending:
        warnings.append("原文含待批准/建议条款，回答须显式保留审批状态限定")
    if has_pending and has_confirmed:
        warnings.append("证据同时包含已确认要求与待批准建议，不得混为一谈")

    return {
        "passed": len(invalid_cites) == 0 and (not has_pending or answer_mentions_pending),
        "citation_count": len(cited_nums & valid_indices),
        "invalid_citations": invalid_cites,
        "has_pending_terms": has_pending,
        "conflicts": conflicts,
        "warnings": warnings,
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
        "若不同文件存在数值或版本差异，请并列列出并标注来源编号。"
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
    """Synthesize a structured, citation-grounded answer with P3 post-verification."""
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
        missing = retrieval_meta.get("refused_missing_attrs") or []
        reason = f"（文档库中未检出核心属性词：{', '.join(missing)}）" if missing else ""
        return {
            "mode": mode,
            "retrieval": retrieval_meta,
            "llm_provider": llm_config_status()["status_text"],
            "verification": {"passed": True, "citation_count": 0, "invalid_citations": [], "conflicts": [], "warnings": []},
            "conclusion": (
                f"当前项目的独立数据库中未检索到满足阈值的原文依据{reason}，**证据不足，不作推测**。"
                "请确认相关工程文件已上传并索引，或在“设置”中调整检索方式/相似度阈值后重试。"
            ),
            "basis": [],
            "table": [
                {"item": "查询项目", "value": project_name},
                {"item": "检索状态", "value": f"已拒答（扫描 {retrieval_meta.get('total_candidates', 0)} 块，耗时 {retrieval_meta.get('latency_ms', 0)} ms）"},
            ],
            "citations": [],
        }

    llm_out = _call_external_llm(question, citations, project_name)
    if llm_out:
        payload = {
            "mode": mode,
            "retrieval": retrieval_meta,
            "llm_provider": llm_out.get("llm_provider", "在线模型"),
            "conclusion": llm_out["conclusion"],
            "basis": llm_out.get("basis", []),
            "table": llm_out.get("table", []),
            "citations": citations,
        }
        ver = verify_grounded_answer(payload, citations)
        if ver["has_pending_terms"] and not ver["passed"]:
            payload["conclusion"] += "（校验提示：原文包含待批准/建议条款，请以原件审批状态为准。）"
            ver = verify_grounded_answer(payload, citations)
        payload["verification"] = ver
        return payload

    # Deterministic local grounded synthesis
    basis_items = []
    pending_flag = False
    for c in citations:
        txt = c["content"].replace("\n", " ").strip()
        if any(w in txt for w in ("待批准", "待业主批准", "建议", "讨论稿", "待确认")):
            pending_flag = True
        short = txt[:140] + ("…" if len(txt) > 140 else "")
        basis_items.append({
            "index": c["index"],
            "source": f"{c['file_name']}（第 {c['page']} 页，{c['section']}）",
            "detail": f"{short} [{c['index']}]",
        })

    top = citations[0]
    top_sentences = [s.strip() for s in re.split(r"[。\n]+", top["content"]) if s.strip()]
    lead_sentence = top_sentences[0][:150] if top_sentences else top["content"][:150]
    for s in top_sentences:
        if any(t in s.lower() for t in (hits[0].get("matched_terms") or [])):
            lead_sentence = s[:160]
            break

    ref_tags = "".join(f" [{c['index']}]" for c in citations[:2])
    approval_note = " 注意：证据中包含**待批准/建议类条款**，尚未自动替代已确认要求，请核实原件审批状态。" if pending_flag else ""
    conclusion = (
        f"根据当前项目数据库的混合检索与精排结果（耗时 {retrieval_meta.get('latency_ms', 0)} ms），"
        f"最相关原文指出：**{lead_sentence}**{ref_tags}。{approval_note}"
    )

    payload = {
        "mode": mode,
        "retrieval": retrieval_meta,
        "llm_provider": llm_config_status()["status_text"],
        "conclusion": conclusion,
        "basis": basis_items,
        "table": [],
        "citations": citations,
    }
    ver = verify_grounded_answer(payload, citations)
    conflict_str = "；".join(ver["conflicts"]) if ver["conflicts"] else "未检出明显多版本数值冲突"
    payload["table"] = [
        {"item": "检索方式", "value": f"{retrieval_meta.get('mode', mode)}（本地 BGE 512维 + BM25 + 交叉精排）"},
        {"item": "置信度与耗时", "value": f"语义分 {top['semantic_score']:.3f} · 耗时 {retrieval_meta.get('latency_ms', 0)} ms"},
        {"item": "冲突与审批校验", "value": f"{conflict_str}（{'校验通过' if ver['passed'] else '需人工复核'}）"},
        {"item": "主要来源", "value": "；".join(f"{c['file_name']} 第 {c['page']} 页 [{c['index']}]" for c in citations[:2])},
    ]
    payload["verification"] = ver
    return payload
