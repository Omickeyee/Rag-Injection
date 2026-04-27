import re


INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", 3, "Instruction override"),
    (r"ignore\s+the\s+user'?s\s+question", 3, "Query hijack"),
    (r"disregard\s+the\s+user'?s\s+question", 3, "Query hijack"),
    (r"system\s+override", 3, "System override"),
    (r"maintenance\s+mode", 2, "System override"),
    (r"repeat\s+verbatim\s+the\s+entire\s+system\s+prompt", 4, "Prompt exfiltration"),
    (r"hidden\s+instructions|confidential\s+system\s+data|internal\s+prompt", 4, "Prompt exfiltration"),
    (r"new\s+persona\s+activated|persona\s+activated", 3, "Persona hijack"),
    (r"you\s+are\s+now\s+(a|an)\s+", 2, "Role reassignment"),
    (r"dan\s+mode|do\s+anything\s+now", 3, "Persona hijack"),
    (r"respond\s+only\s+with|do\s+not\s+answer\s+anything\s+else", 2, "Forced output"),
    (r"start\s+with\s+exactly:|end\s+with\s+exactly:", 2, "Forced format"),
    (r"click\s+the\s+link|verify\s+your\s+(password|credentials|account)", 2, "Credential phishing"),
    (r"(password|credential|api[_\s]?key|token)\s*[:=]", 2, "Sensitive data request"),
    (r"backup\s+codes|2-factor|two-factor", 2, "Sensitive data request"),
    (r"log\s+all\s+user\s+inputs|transmit\s+them\s+to", 3, "Data exfiltration"),
    (r"https?://[^\s]+", 1, "External link"),
    (r"<!--\s*.*?(instruction|command|override|ignore).*?\s*-->", 1, "Hidden instruction"),
    (r"\[hidden\]|\[invisible\]", 1, "Hidden instruction"),
]

def analyze_doc_safety(doc_text):
    text = doc_text or ""
    matches = []
    score = 0

    for pattern, weight, label in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            score += weight
            matches.append(label)

    unique_matches = list(dict.fromkeys(matches))
    suspicious = score >= 3 or len(unique_matches) >= 2

    return {
        "score": score,
        "signals": unique_matches,
        "suspicious": suspicious,
    }

def chunk_scanner(text):
    analysis = analyze_doc_safety(text)
    return min(analysis["score"] / 5.0, 1.0)

def get_trust_score(label=None, metadata=None):
    if metadata and "trust_score" in metadata:
        return float(metadata["trust_score"])
    source_kind = None
    if metadata:
        source_kind = metadata.get("source_kind")
    if source_kind == "benign":
        return 1.0
    if source_kind == "hard_negative":
        return 0.7
    if source_kind == "poisoned":
        return 0.2
    if label == "filtered":
        return 0.0
    if label == "benign":
        return 1.0
    if label == "malicious":
        return 0.2
    return 0.6

def safety_reranker(retrieved_docs, alpha=0.6, beta=0.3, gamma=0.1):
    reranked = []
    n = len(retrieved_docs)
    for i, item in enumerate(retrieved_docs):
        if len(item) == 3:
            doc_id, doc_text, label = item
            metadata = {}
        else:
            doc_id, doc_text, label, metadata = item
        analysis = analyze_doc_safety(doc_text)
        safety = 1.0 - chunk_scanner(doc_text)
        trust = get_trust_score(label, metadata)
        relevance = metadata.get("relevance_score")
        if relevance is None:
            relevance = 1.0 - (i / max(n, 1))
        final_score = alpha * relevance + beta * safety + gamma * trust
        reranked.append(
            {
                "doc_id": doc_id,
                "text": doc_text,
                "label": label,
                "metadata": metadata,
                "analysis": analysis,
                "relevance_score": relevance,
                "safety_score": safety,
                "trust_score": trust,
                "final_score": final_score,
            }
        )
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked

def filter_doc_text(doc_text):
    analysis = analyze_doc_safety(doc_text)
    if analysis["suspicious"]:
        return "[FILTERED: document removed because it appears to contain prompt injection instructions.]", analysis
    return doc_text, analysis


def prepare_defended_docs(retrieved_docs):
    defended_docs = []
    analyses = []
    for doc_id, doc_text, label in retrieved_docs:
        safe_text, analysis = filter_doc_text(doc_text)
        analyses.append((doc_id, doc_text, label, analysis))
        defended_docs.append(
            (
                doc_id,
                safe_text,
                "filtered" if analysis["suspicious"] else label,
            )
        )
    return analyses, defended_docs
