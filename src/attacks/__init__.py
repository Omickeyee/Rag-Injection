"""Attack implementations for indirect prompt injection scenarios."""

from src.attacks.core import (
    ALL_ATTACKS,
    Attack,
    ExfiltrationAttack,
    GoalHijackingAttack,
    PatternMatchingAttack,
    PhishingAttack,
    PrivilegeEscalationAttack,
)

__all__ = [
    "ALL_ATTACKS",
    "Attack",
    "ExfiltrationAttack",
    "GoalHijackingAttack",
    "PatternMatchingAttack",
    "PhishingAttack",
    "PrivilegeEscalationAttack",
]
