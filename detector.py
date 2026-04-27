import re

def chunk_scanner(text):
    patterns = [
        (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE), 1.0),
        (re.compile(r"system\s+override", re.IGNORECASE), 1.0),
        (re.compile(r"ADMIN\s+NOTE", re.IGNORECASE), 0.9),
        (re.compile(r"ignore\s+all\s+prior", re.IGNORECASE), 1.0),
        (re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|context)", re.IGNORECASE), 1.0),
        (re.compile(r"new\s+instructions?\s*:", re.IGNORECASE), 0.85),
        (re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE), 0.9),
        (re.compile(r"do\s+not\s+follow\s+(the\s+)?(system|original)\s+(prompt|instructions)", re.IGNORECASE), 1.0),
        (re.compile(r"click\s+here\s+to", re.IGNORECASE), 0.6),
        (re.compile(r"(password|credential|api[_\s]?key)\s*[:=]", re.IGNORECASE), 0.65),
        (re.compile(r"https?://[^\s]+\.(xyz|top|tk|ml|ga|cf|gq|buzz|work)/", re.IGNORECASE), 0.7),
        (re.compile(r"(verify|confirm|reset)\s+your\s+(password|credentials|account)", re.IGNORECASE), 0.6),
        (re.compile(r"output\s+(the|all|any)\s+(api|secret|private)\s*(key|token)", re.IGNORECASE), 0.75),
        (re.compile(r"reveal\s+(any|all|the)\s+(secret|private|internal)", re.IGNORECASE), 0.7),
        (re.compile(r"<!--\s*.*?(instruction|command|override|ignore).*?\s*-->", re.IGNORECASE | re.DOTALL), 0.4),
        (re.compile(r"\x00|\x01|\x1b\[", re.IGNORECASE), 0.35),  # null bytes / ANSI escapes
        (re.compile(r"\[hidden\]|\[invisible\]", re.IGNORECASE), 0.4),
    ]
    max_score = 0.0
    for pattern, w in patterns:
        if pattern.search(text):
            max_score = max(max_score, w)
    return min(max_score, 1.0)

def safety_reranker(text):
    suspicion_score = chunk_scanner(text)
    