"""Bounded local query expansion and source-aware evidence retrieval."""

import re
import unicodedata


def _technical_text(value):
    return unicodedata.normalize("NFKC", value).translate(
        str.maketrans({character: "-" for character in "‐‑‒–−"})
    )


def exact_terms(question):
    """Extract literal technical anchors, without conversions or inferred synonyms."""
    question = _technical_text(question)
    units = r"(?:kN/m[23]?|kg/m[23]?|N/mm[23]?|m[23]/h|m[23]/s|mm|cm|MPa|kPa|Pa|kW|kV|Hz|°C|%|V|A|m)"
    expressions = [
        rf"(?<![a-zA-Z0-9.])\d+(?:\.\d+)?\s*{units}(?:\s*(?:DC|AC))?(?![a-zA-Z0-9])",
        r"(?<![a-z0-9])(?:gb(?:/t)?|iso|din|iec|en)\s*[- ]?\s*\d+(?:[-.:]\d+)*(?![a-z0-9])",
        r"第\s*\d+(?:\.\d+)*\s*条",
        r"(?:条款|章节)\s*\d+(?:\.\d+)*",
        r"(?<![a-z0-9.])\d+(?:\.\d+){2,}(?![a-z0-9.])",
        r"(?<![a-z0-9])(?:[a-z][a-z0-9]*[-/.])*[a-z][a-z0-9]*\d[a-z0-9]*(?:[-/.][a-z0-9]+)*(?![a-z0-9])",
        r"(?<![a-z0-9])\d+[a-z][a-z0-9]*(?:[-/.][a-z0-9]+)*(?![a-z0-9])",
    ]
    terms = []
    occupied = []
    for ordinal, expression in enumerate(expressions):
        for match in re.finditer(expression, question, flags=0 if ordinal == 0 else re.I):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            terms.append(match.group().strip())
            occupied.append(match.span())
    terms.extend(match[1].strip() for match in re.finditer(
        r'["“「]([^"”」\n]{2,80})["”」]', question
    ))
    return list(dict.fromkeys(terms))[:12]


def _literal_pattern(term):
    numbered_clause = re.fullmatch(r"(?:第|条款|章节)\s*(\d+(?:\.\d+)+)\s*条?", term)
    escaped = re.escape(numbered_clause[1] if numbered_clause else term)
    escaped = re.sub(r"(?:\\ )+", r"\\s*", escaped)
    # Unit spacing is representational, not a numeric/unit conversion.
    escaped = re.sub(r"(?<=\d)(?=[a-zA-Z%°])", r"\\s*", escaped)
    escaped = re.sub(r"(?<=[a-zA-Z])(?=\d)", r"\\s*", escaped)
    escaped = escaped.replace(r"\-", r"\s*-\s*")
    escaped = escaped.replace("/", r"\s*/\s*")
    if not numbered_clause and term.startswith("第"):
        escaped = escaped.replace("第", r"第\s*").replace("条", r"\s*条")
    elif not numbered_clause and term.startswith(("条款", "章节")):
        escaped = re.sub(r"^(条款|章节)", r"\1\\s*", escaped)
    if re.match(r"^\d", term) and re.search(r"(?:DC|AC)$", term):
        escaped = re.sub(r"(DC|AC)$", r"\\s*\1", escaped)
    # Numeric-leading anchors preserve SI prefix case: MPa and mPa are not equal.
    flags = 0 if re.match(r"^\d", term) else re.I
    return re.compile(
        r"(?<![a-zA-Z0-9.])" + escaped + r"(?![a-zA-Z0-9]|[./-][a-zA-Z0-9])",
        flags=flags,
    )


def exact_candidates(question, documents, limit=24):
    terms = exact_terms(question)
    patterns = [_literal_pattern(term) for term in terms]
    matches = {}
    by_term = [[] for _ in terms]
    for doc in documents:
        content = _technical_text(doc.content)
        for ordinal, pattern in enumerate(patterns):
            if pattern.search(content):
                matches.setdefault(doc.id_, []).append(ordinal)
                by_term[ordinal].append(doc.id_)
    # First keep a route for each term and each version, then fill by match count.
    picked = []
    by_id = {doc.id_: doc for doc in documents}
    routes = []
    for identities in by_term:
        identities.sort(key=lambda identity: len(matches[identity]), reverse=True)
        versions = set()
        representatives = []
        for identity in identities:
            version = by_id[identity].metadata["version"]
            if version not in versions:
                versions.add(version)
                representatives.append(identity)
        routes.append(representatives)
    picked = [
        identities[index] for index in range(max(map(len, routes), default=0))
        for identities in routes if index < len(identities)
    ]
    picked = list(dict.fromkeys(picked))
    ranked = sorted(matches, key=lambda identity: len(matches[identity]), reverse=True)
    picked = list(dict.fromkeys(picked + ranked))[:limit]
    return [by_id[identity] for identity in picked], {
        "exact_terms": terms,
        "exact_matches_per_term": dict(zip(terms, map(len, by_term))),
        "exact_matched_chunks": len(matches),
        "exact_candidate_chunks": len(picked),
        "exact_omitted_chunks": len(matches) - len(picked),
        "exact_candidate_matches": {
            identity: [terms[index] for index in matches[identity]] for identity in picked
        },
        "exact_notice": "任一标识命中即可召回，优先多标识并兼顾各版本；"
                        "保留数值、SI前缀大小写，不换算单位，不代表适用性或全量覆盖。",
    }


def is_meeting_source(metadata):
    label = metadata.get("file_name", "") + " " + metadata.get("version_label", "")
    return bool(re.search(r"(?i)(?<![a-z])MOM(?![a-z])|会议纪要|会议记录|minutes\s+of\s+meeting", label))


def asks_for_meeting(question):
    return bool(re.search(r"(?i)(?<![a-z])MOM(?![a-z])|会议纪要|会议记录", question))


def substantive(text):
    compact = re.sub(r"\s+", "", text)
    return len(compact) >= 45 or bool(
        re.search(r"必须|不得|禁止|不含|不少于|不得少于|应采用|须采用|≥|≤", compact)
    )


def query_variants(question):
    subject = re.split(r"有哪些|有什么|有何|怎么|如何|要求|[？?，,。；;]", question)[0]
    subject = re.sub(r"设计$", "", subject).strip()
    if not subject or len(subject) > 24:
        return [question], ""
    # These are retrieval vocabulary only, never inferred business requirements.
    anchor = "钢平台 钢结构" if subject == "钢构" else subject
    variants = [question]
    match = re.search(r"按(.+?)(?:整理|汇总|分类|[。？?]|$)", question)
    if match:
        for facet in re.split(r"[、，,和及]", match[1]):
            facet = facet.strip()
            if facet:
                variants.append(anchor + " " + facet.replace("供货边界", "供货范围"))
    else:
        variants.append(anchor)
    return list(dict.fromkeys(variants))[:4], subject


def quick_candidates(question, documents, vector, text, threshold):
    known = {doc.id_: doc for doc in documents}
    exact, exact_details = exact_candidates(question, documents)
    variants, subject = query_variants(question)
    ranks = {}
    per_query = []
    below = 0
    original_vector = []
    for ordinal, query in enumerate(variants):
        vector_hits = vector(query, top_k=len(documents) if ordinal == 0 and subject else min(60, len(documents)))
        if ordinal == 0:
            original_vector = vector_hits
        below += int(sum(doc.score < threshold for doc in vector_hits))
        vector_hits = [doc for doc in vector_hits if doc.score >= threshold]
        text_hits = text(query, top_k=min(60, len(documents)), scope=list(known))
        local = {}
        for hits in (vector_hits, text_hits):
            for rank, doc in enumerate(hits, 1):
                if doc.id_ not in known:
                    raise ValueError("检索结果越出当前项目边界。")
                local[doc.id_] = local.get(doc.id_, 0) + 1 / (60 + rank)
        ordered = sorted(local, key=local.get, reverse=True)
        per_query.append(ordered)
        for identity, score in local.items():
            ranks[identity] = ranks.get(identity, 0) + score
    substantive_ids = {doc.id_ for doc in documents if substantive(doc.content)}
    ordered = sorted(ranks, key=ranks.get, reverse=True)
    picked = [identity for identity in ordered if identity in substantive_ids][:12]
    for matches in per_query[1:]:
        picked.extend([identity for identity in matches
                       if identity in substantive_ids and identity not in picked][:4])
    # A filename hit is a separate lexical route, not a lower vector-only threshold.
    # Search substantive clauses inside a dedicated file instead of returning its title.
    file_hits = []
    if subject:
        counts = {}
        for doc in original_vector:
            if doc.id_ not in known:
                raise ValueError("检索结果越出当前项目边界。")
            filename = known[doc.id_].metadata["file_name"]
            if subject.casefold() not in filename.casefold() or doc.id_ not in substantive_ids:
                continue
            if counts.get(filename, 0) < 8:
                file_hits.append(doc.id_)
                counts[filename] = counts.get(filename, 0) + 1
            if len(file_hits) >= 16:
                break
    picked = list(dict.fromkeys([doc.id_ for doc in exact] + file_hits + picked))
    if not picked:
        picked = ordered[:10]
    return [known[identity] for identity in picked[:36]], {
        "query_variants": variants,
        "below_vector_relevance_threshold": below,
        "filename_scoped_candidates": len(file_hits),
        "heading_only_hits_deprioritized": True,
        **exact_details,
    }


def nearby_context(selected, documents, limit=12):
    groups = {}
    for doc in documents:
        groups.setdefault((doc.metadata["version"], doc.id_.rsplit("-", 1)[0]), []).append(doc)
    output = list(selected)
    known = {doc.id_ for doc in output}
    for doc in sorted(selected, key=lambda item: len(item.content)):
        if len(doc.content) >= 100:
            continue
        siblings = sorted(groups[(doc.metadata["version"], doc.id_.rsplit("-", 1)[0])],
                          key=lambda item: int(item.id_.rsplit("-", 1)[1]))
        index = next(i for i, sibling in enumerate(siblings) if sibling.id_ == doc.id_)
        for sibling in siblings[max(0, index - 3):index + 2]:
            if sibling.id_ not in known:
                output.append(sibling)
                known.add(sibling.id_)
                if len(output) - len(selected) >= limit:
                    return output
    return output
