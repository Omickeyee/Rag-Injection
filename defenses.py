from __future__ import annotations
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any
from llama_index.core.schema import NodeWithScore
from settings import settings

logger = logging.getLogger(__name__)

SOURCE_TYPE_WEIGHTS = {
    "internal_docs": 0.9,
    "confluence": 0.8,
    "email": 0.7,
    "slack": 0.5,
}

ROLE_ACCESS = {
    "employee": {"public", "internal"},
    "manager": {"public", "internal", "confidential"},
    "executive": {"public", "internal", "confidential", "restricted"},
}

_HIGH_PATTERNS = [
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE), 1.0),
    (re.compile(r"system\s+override", re.IGNORECASE), 1.0),
    (re.compile(r"ADMIN\s+NOTE", re.IGNORECASE), 0.9),
    (re.compile(r"ignore\s+all\s+prior", re.IGNORECASE), 1.0),
    (re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|context)", re.IGNORECASE), 1.0),
    (re.compile(r"new\s+instructions?\s*:", re.IGNORECASE), 0.85),
    (re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE), 0.9),
    (re.compile(r"do\s+not\s+follow\s+(the\s+)?(system|original)\s+(prompt|instructions)", re.IGNORECASE), 1.0),
]

_MEDIUM_PATTERNS = [
    (re.compile(r"click\s+here\s+to", re.IGNORECASE), 0.6),
    (re.compile(r"(password|credential|api[_\s]?key)\s*[:=]", re.IGNORECASE), 0.65),
    (re.compile(r"https?://[^\s]+\.(xyz|top|tk|ml|ga|cf|gq|buzz|work)/", re.IGNORECASE), 0.7),
    (re.compile(r"(verify|confirm|reset)\s+your\s+(password|credentials|account)", re.IGNORECASE), 0.6),
    (re.compile(r"output\s+(the|all|any)\s+(api|secret|private)\s*(key|token)", re.IGNORECASE), 0.75),
    (re.compile(r"reveal\s+(any|all|the)\s+(secret|private|internal)", re.IGNORECASE), 0.7),
]

_LOW_PATTERNS = [
    (re.compile(r"<!--\s*.*?(instruction|command|override|ignore).*?\s*-->", re.IGNORECASE | re.DOTALL), 0.4),
    (re.compile(r"\x00|\x01|\x1b\[", re.IGNORECASE), 0.35),
    (re.compile(r"\[hidden\]|\[invisible\]", re.IGNORECASE), 0.4),
]

_W_SOURCE = 0.4
_W_META_TRUST = 0.4
_W_RECENCY = 0.2
_MAX_AGE_DAYS = 365

ALL_PATTERNS = _HIGH_PATTERNS + _MEDIUM_PATTERNS + _LOW_PATTERNS


class BaseDefense(ABC):
    @property
    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def apply(self, nodes, query, user_context):
        ...


def compute_suspicion_score(text):
    score = 0.0
    for pattern, weight in ALL_PATTERNS:
        if pattern.search(text):
            score = max(score, weight)
    return min(score, 1.0)


def _recency_score(created_at):
    if not created_at:
        return 0.5
    try:
        dt = datetime.fromisoformat(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(tz=timezone.utc) - dt).days
        return max(0.0, 1.0 - age_days / _MAX_AGE_DAYS)
    except (ValueError, TypeError):
        return 0.5

def compute_trust_score(metadata):
    composite = (
        _W_SOURCE * SOURCE_TYPE_WEIGHTS.get(metadata.get("source_type", ""), 0.5)
        + _W_META_TRUST * float(metadata.get("trust_score", 0.5))
        + _W_RECENCY * _recency_score(metadata.get("created_at"))
    )
    return max(0.0, min(1.0, composite))


class ChunkScanner(BaseDefense):
    def __init__(self, threshold = None, use_ml = False, warn_only = False):
        self._threshold = threshold if threshold is not None else settings.detector_threshold
        self._use_ml = use_ml
        self._warn_only = warn_only
        self._llm_defense = None
        if use_ml:
            self._try_load_llm_defense()

    @property
    def name(self):
        mode = "ML+Heuristic" if self._llm_defense is not None else "Heuristic"
        return f"ChunkScanner({mode})"

    def _try_load_llm_defense(self):
        model_path = settings.model_output_dir / 'LLM_defense'
        if not model_path.exists():
            logger.warning(
                "LLM defense requested but model not found at %s — "
                "falling back to heuristic-only mode.",
                model_path,
            )
            return
        try:
            from LLM_defense.model import LLMDefense
            self._llm_defense = LLMDefense.load(str(model_path))
            logger.info("Loaded LLM defense from %s", model_path)
        except Exception:
            logger.warning("Failed to load LLM defense — falling back to heuristic-only.", exc_info=True)

    def _ml_score(self, text):
        if self._llm_defense is None:
            return 0.0
        return float(self._llm_defense.predict(text))

    def score_node(self, node):
        text = node.node.get_content()
        score = compute_suspicion_score(text)
        if self._llm_defense is not None:
            score = max(score, self._ml_score(text))
        return score

    def apply(self, nodes, query, user_context):
        result = []
        for node in nodes:
            score = self.score_node(node)
            node.node.metadata["suspicion_score"] = score
            if score > self._threshold:
                if self._warn_only:
                    node.node.set_content(f"[WARNING: potentially injected content (score={score:.2f})] {node.node.get_content()}")
                    result.append(node)
                else:
                    logger.info(
                        "ChunkScanner: removed node (score=%.2f, threshold=%.2f): %.80s…",
                        score,
                        self._threshold,
                        node.node.get_content(),
                    )
                continue
            result.append(node)
        return result

class SourceTrustScorer(BaseDefense):
    def __init__(self, threshold = None):
        self._threshold = threshold if threshold is not None else settings.trust_score_threshold

    @property
    def name(self):
        return "SourceTrustScorer"

    def apply(self, nodes, query, user_context):
        scored = []
        for node in nodes:
            trust = compute_trust_score(node.node.metadata)
            node.node.metadata["computed_trust_score"] = trust
            if trust < self._threshold:
                logger.info(
                    "SourceTrustScorer: removed node (trust=%.2f, threshold=%.2f): %.80s…",
                    trust,
                    self._threshold,
                    node.node.get_content(),
                )
                continue
            scored.append((trust, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in scored]

class SafetyReranker(BaseDefense):
    def __init__(
        self, relevance_weight = None, safety_weight = None, trust_weight = None):
        self._w_rel = relevance_weight if relevance_weight is not None else settings.reranker_relevance_weight
        self._w_safe = safety_weight if safety_weight is not None else settings.reranker_safety_weight
        self._w_trust = trust_weight if trust_weight is not None else settings.reranker_trust_weight

    @property
    def name(self):
        return "SafetyReranker"

    def _compute_final_score(self, node):
        relevance = node.score if node.score is not None else 0.5
        suspicion = node.node.metadata.get("suspicion_score")
        if suspicion is None:
            suspicion = compute_suspicion_score(node.node.get_content())
            node.node.metadata["suspicion_score"] = suspicion
        trust = node.node.metadata.get("computed_trust_score")
        if trust is None:
            trust = compute_trust_score(node.node.metadata)
            node.node.metadata["computed_trust_score"] = trust
        return self._w_rel * relevance + self._w_safe * (1.0 - suspicion) + self._w_trust * trust

    def apply(self, nodes, query, user_context):
        scored = []
        for index, node in enumerate(nodes):
            final = self._compute_final_score(node)
            node.node.metadata["reranker_final_score"] = final
            node.score = final
            scored.append((final, index, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        logger.debug(
            "SafetyReranker: reordered %d nodes (top score=%.3f, bottom=%.3f)",
            len(scored),
            scored[0][0] if scored else 0.0,
            scored[-1][0] if scored else 0.0,
        )
        return [node for _, _, node in scored]

class PrivilegeFilter(BaseDefense):
    @property
    def name(self):
        return "PrivilegeFilter"

    def apply(self, nodes, query, user_context):
        role = user_context.get("role", "employee")
        allowed = ROLE_ACCESS.get(role, {"public"})
        result = []
        for node in nodes:
            access_level = node.node.metadata.get("access_level", "public")
            if access_level in allowed:
                result.append(node)
                continue
            logger.info(
                "PrivilegeFilter: blocked node (access_level=%s, role=%s): %.80s…",
                access_level,
                role,
                node.node.get_content(),
            )
        return result

__all__ = [
    "ALL_PATTERNS",
    "BaseDefense",
    "ChunkScanner",
    "PrivilegeFilter",
    "ROLE_ACCESS",
    "SOURCE_TYPE_WEIGHTS",
    "SafetyReranker",
    "SourceTrustScorer",
    "compute_suspicion_score",
    "compute_trust_score",
]
