"""Compatibility wrapper for chunk-scanning defenses."""

from src.defenses.core import ALL_PATTERNS, ChunkScanner, compute_suspicion_score

__all__ = ["ALL_PATTERNS", "ChunkScanner", "compute_suspicion_score"]
