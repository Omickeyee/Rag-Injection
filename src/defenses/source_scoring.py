"""Compatibility wrapper for source-trust scoring."""

from src.defenses.core import SOURCE_TYPE_WEIGHTS, SourceTrustScorer, compute_trust_score

__all__ = ["SOURCE_TYPE_WEIGHTS", "SourceTrustScorer", "compute_trust_score"]
