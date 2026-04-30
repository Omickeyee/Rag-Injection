"""Core defense utilities and defense implementations."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from llama_index.core.schema import NodeWithScore

from config.settings import settings

logger = logging.getLogger(__name__)

SOURCE_TYPE_WEIGHTS: dict[str, float] = {
    "internal_docs": 0.9,
    "confluence": 0.8,
    "email": 0.7,
    "slack": 0.5,
}

ROLE_ACCESS: dict[str, set[str]] = {
    "employee": {"public", "internal"},
    "manager": {"public", "internal", "confidential"},
    "executive": {"public", "internal", "confidential", "restricted"},
}

_HIGH_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE), 1.0),
    (re.compile(r"system\s+override", re.IGNORECASE), 1.0),
    (re.compile(r"ADMIN\s+NOTE", re.IGNORECASE), 0.9),
    (re.compile(r"ignore\s+all\s+prior", re.IGNORECASE), 1.0),
    (re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|context)", re.IGNORECASE), 1.0),
    (re.compile(r"new\s+instructions?\s*:", re.IGNORECASE), 0.85),
    (re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE), 0.9),
    (re.compile(r"do\s+not\s+follow\s+(the\s+)?(system|original)\s+(prompt|instructions)", re.IGNORECASE), 1.0),
]

_MEDIUM_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"click\s+here\s+to", re.IGNORECASE), 0.6),
    (re.compile(r"(password|credential|api[_\s]?key)\s*[:=]", re.IGNORECASE), 0.65),
    (re.compile(r"https?://[^\s]+\.(xyz|top|tk|ml|ga|cf|gq|buzz|work)/", re.IGNORECASE), 0.7),
    (re.compile(r"(verify|confirm|reset)\s+your\s+(password|credentials|account)", re.IGNORECASE), 0.6),
    (re.compile(r"output\s+(the|all|any)\s+(api|secret|private)\s*(key|token)", re.IGNORECASE), 0.75),
    (re.compile(r"reveal\s+(any|all|the)\s+(secret|private|internal)", re.IGNORECASE), 0.7),
]

_LOW_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"<!--\s*.*?(instruction|command|override|ignore).*?\s*-->", re.IGNORECASE | re.DOTALL), 0.4),
    (re.compile(r"\x00|\x01|\x1b\[", re.IGNORECASE), 0.35),
    (re.compile(r"\[hidden\]|\[invisible\]", re.IGNORECASE), 0.4),
]

_W_SOURCE = 0.4
_W_META_TRUST = 0.4
_W_RECENCY = 0.2
_MAX_AGE_DAYS = 365

ALL_PATTERNS: list[tuple[re.Pattern[str], float]] = _HIGH_PATTERNS + _MEDIUM_PATTERNS + _LOW_PATTERNS


class BaseDefense(ABC):
    """Base class for defense mechanisms."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def apply(
        self,
        nodes: list[NodeWithScore],
        query: str,
        user_context: dict[str, Any],
    ) -> list[NodeWithScore]:
        ...


def compute_suspicion_score(text: str) -> float:
    """Return the maximum matched heuristic suspicion score."""
    score = 0.0
    for pattern, weight in ALL_PATTERNS:
        if pattern.search(text):
            score = max(score, weight)
    return min(score, 1.0)


def _recency_score(created_at: str | None) -> float:
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


def compute_trust_score(metadata: dict[str, Any]) -> float:
    composite = (
        _W_SOURCE * SOURCE_TYPE_WEIGHTS.get(metadata.get("source_type", ""), 0.5)
        + _W_META_TRUST * float(metadata.get("trust_score", 0.5))
        + _W_RECENCY * _recency_score(metadata.get("created_at"))
    )
    return max(0.0, min(1.0, composite))


class ChunkScanner(BaseDefense):
    """Regex + optional ML chunk-level injection scanner."""

    def __init__(
        self,
        threshold: float | None = None,
        use_ml: bool = False,
        warn_only: bool = False,
    ) -> None:
        self._threshold = threshold if threshold is not None else settings.detector_threshold
        self._use_ml = use_ml
        self._warn_only = warn_only
        self._ml_detector: Any | None = None

        if use_ml:
            self._try_load_ml_detector()

    @property
    def name(self) -> str:
        mode = "ML+Heuristic" if self._ml_detector is not None else "Heuristic"
        return f"ChunkScanner({mode})"

    def _try_load_ml_detector(self) -> None:
        model_path = settings.model_output_dir / "detector"
        if not model_path.exists():
            logger.warning(
                "ML detector requested but model not found at %s — "
                "falling back to heuristic-only mode.",
                model_path,
            )
            return
        try:
            from src.defenses.detector.model import InjectionDetector

            self._ml_detector = InjectionDetector.load(str(model_path))
            logger.info("Loaded ML injection detector from %s", model_path)
        except Exception:
            logger.warning("Failed to load ML detector — falling back to heuristic-only.", exc_info=True)

    def _ml_score(self, text: str) -> float:
        if self._ml_detector is None:
            return 0.0
        return float(self._ml_detector.predict(text))

    def score_node(self, node: NodeWithScore) -> float:
        text = node.node.get_content()
        score = compute_suspicion_score(text)
        if self._ml_detector is not None:
            score = max(score, self._ml_score(text))
        return score

    def apply(
        self,
        nodes: list[NodeWithScore],
        query: str,
        user_context: dict[str, Any],
    ) -> list[NodeWithScore]:
        result: list[NodeWithScore] = []
        for node in nodes:
            score = self.score_node(node)
            node.node.metadata["suspicion_score"] = score

            if score > self._threshold:
                if self._warn_only:
                    node.node.set_content(
                        f"[WARNING: potentially injected content (score={score:.2f})] {node.node.get_content()}"
                    )
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
    """Filters and sorts nodes by composite trust score."""

    def __init__(self, threshold: float | None = None) -> None:
        self._threshold = threshold if threshold is not None else settings.trust_score_threshold

    @property
    def name(self) -> str:
        return "SourceTrustScorer"

    def apply(
        self,
        nodes: list[NodeWithScore],
        query: str,
        user_context: dict[str, Any],
    ) -> list[NodeWithScore]:
        scored: list[tuple[float, NodeWithScore]] = []

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
    """Reorders nodes by a weighted combination of relevance, safety, and trust."""

    def __init__(
        self,
        relevance_weight: float | None = None,
        safety_weight: float | None = None,
        trust_weight: float | None = None,
    ) -> None:
        self._w_rel = relevance_weight if relevance_weight is not None else settings.reranker_relevance_weight
        self._w_safe = safety_weight if safety_weight is not None else settings.reranker_safety_weight
        self._w_trust = trust_weight if trust_weight is not None else settings.reranker_trust_weight

    @property
    def name(self) -> str:
        return "SafetyReranker"

    def _compute_final_score(self, node: NodeWithScore) -> float:
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

    def apply(
        self,
        nodes: list[NodeWithScore],
        query: str,
        user_context: dict[str, Any],
    ) -> list[NodeWithScore]:
        scored: list[tuple[float, int, NodeWithScore]] = []

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
    """Filters nodes by access level based on the user's role."""

    @property
    def name(self) -> str:
        return "PrivilegeFilter"

    def apply(
        self,
        nodes: list[NodeWithScore],
        query: str,
        user_context: dict[str, Any],
    ) -> list[NodeWithScore]:
        role = user_context.get("role", "employee")
        allowed = ROLE_ACCESS.get(role, {"public"})
        result: list[NodeWithScore] = []

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
