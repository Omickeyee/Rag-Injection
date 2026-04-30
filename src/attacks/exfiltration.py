"""Compatibility wrapper for the exfiltration attack."""

from src.attacks.core import ExfiltrationAttack, _EXFILTRATION_TARGET_QUERIES

_TARGET_QUERIES = _EXFILTRATION_TARGET_QUERIES

__all__ = ["ExfiltrationAttack", "_TARGET_QUERIES"]
