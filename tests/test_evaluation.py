"""Tests for evaluation metric computation functions.

Imports metrics directly from the module to avoid pulling in heavy
dependencies (chromadb, llama_index) through the package __init__.
"""

from __future__ import annotations

import sys

import pytest


def _import_metrics():
    """Import src.evaluation.metrics while mocking out heavy transitive deps.

    The evaluation __init__ imports runner -> rag.pipeline -> chromadb etc.
    We only need the metrics module, so we replace the evaluation package
    entry in sys.modules with a lightweight stand-in that has the correct
    __path__ so Python can locate the metrics submodule on disk.
    """
    import importlib
    import types
    from pathlib import Path

    # Determine the real filesystem path of the evaluation package
    eval_dir = str(Path(__file__).resolve().parent.parent / "src" / "evaluation")

    if "src.evaluation" not in sys.modules:
        pkg = types.ModuleType("src.evaluation")
        pkg.__path__ = [eval_dir]  # type: ignore[attr-defined]
        pkg.__package__ = "src.evaluation"
        sys.modules["src.evaluation"] = pkg
    else:
        # Already imported (maybe by conftest); replace to avoid __init__ side effects
        existing = sys.modules["src.evaluation"]
        if not hasattr(existing, "__path__") or not existing.__path__:
            existing.__path__ = [eval_dir]  # type: ignore[attr-defined]

    import src.evaluation.metrics as _m
    return _m


_metrics = _import_metrics()
attack_success_rate = _metrics.attack_success_rate
defense_block_rate = _metrics.defense_block_rate
false_positive_rate = _metrics.false_positive_rate
mean_reciprocal_rank = _metrics.mean_reciprocal_rank


# ---------------------------------------------------------------------------
# Attack Success Rate
# ---------------------------------------------------------------------------

class TestAttackSuccessRate:
    def test_all_succeed(self):
        results = [{"success": True}, {"success": True}, {"success": True}]
        assert attack_success_rate(results) == pytest.approx(1.0)

    def test_none_succeed(self):
        results = [{"success": False}, {"success": False}]
        assert attack_success_rate(results) == pytest.approx(0.0)

    def test_partial(self):
        results = [
            {"success": True},
            {"success": False},
            {"success": True},
            {"success": False},
        ]
        assert attack_success_rate(results) == pytest.approx(0.5)

    def test_empty(self):
        assert attack_success_rate([]) == pytest.approx(0.0)

    def test_missing_key_treated_as_false(self):
        """Result dicts without a 'success' key should count as failure."""
        results = [{"success": True}, {"query": "something"}]
        assert attack_success_rate(results) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Defense Block Rate
# ---------------------------------------------------------------------------

class TestDefenseBlockRate:
    def test_all_blocked(self):
        no_defense = [
            {"query": "q1", "success": True},
            {"query": "q2", "success": True},
        ]
        with_defense = [
            {"query": "q1", "success": False},
            {"query": "q2", "success": False},
        ]
        assert defense_block_rate(no_defense, with_defense) == pytest.approx(1.0)

    def test_none_blocked(self):
        no_defense = [
            {"query": "q1", "success": True},
            {"query": "q2", "success": True},
        ]
        with_defense = [
            {"query": "q1", "success": True},
            {"query": "q2", "success": True},
        ]
        assert defense_block_rate(no_defense, with_defense) == pytest.approx(0.0)

    def test_partial_block(self):
        no_defense = [
            {"query": "q1", "success": True},
            {"query": "q2", "success": True},
            {"query": "q3", "success": False},  # didn't succeed w/o defense
        ]
        with_defense = [
            {"query": "q1", "success": False},  # blocked
            {"query": "q2", "success": True},   # still succeeds
            {"query": "q3", "success": False},
        ]
        # Only q1 and q2 succeeded without defense; q1 was blocked => 1/2
        assert defense_block_rate(no_defense, with_defense) == pytest.approx(0.5)

    def test_no_successful_attacks(self):
        no_defense = [{"query": "q1", "success": False}]
        with_defense = [{"query": "q1", "success": False}]
        assert defense_block_rate(no_defense, with_defense) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# False Positive Rate
# ---------------------------------------------------------------------------

class TestFalsePositiveRate:
    def test_no_false_positives(self):
        benign = [
            {"response": "Here is a detailed answer about the project timeline and milestones."},
            {"response": "The deployment procedure involves several steps as documented."},
        ]
        assert false_positive_rate(benign) == pytest.approx(0.0)

    def test_all_false_positives(self):
        benign = [
            {"response": ""},  # empty
            {"response": "N/A"},  # < 20 chars
        ]
        assert false_positive_rate(benign) == pytest.approx(1.0)

    def test_content_blocked_flag(self):
        benign = [
            {"response": "A full response with plenty of useful information.", "content_blocked": False},
            {"response": "Another good response here.", "content_blocked": True},
        ]
        assert false_positive_rate(benign) == pytest.approx(0.5)

    def test_empty_list(self):
        assert false_positive_rate([]) == pytest.approx(0.0)

    def test_short_response_counted(self):
        """Response shorter than 20 chars is considered a false positive."""
        benign = [{"response": "Short."}]
        assert false_positive_rate(benign) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Mean Reciprocal Rank
# ---------------------------------------------------------------------------

class TestMeanReciprocalRank:
    def test_first_position(self):
        results = [{"retrieved_node_ids": ["doc1", "doc2", "doc3"]}]
        relevant = {"doc1"}
        assert mean_reciprocal_rank(results, relevant) == pytest.approx(1.0)

    def test_second_position(self):
        results = [{"retrieved_node_ids": ["doc2", "doc1", "doc3"]}]
        relevant = {"doc1"}
        assert mean_reciprocal_rank(results, relevant) == pytest.approx(0.5)

    def test_third_position(self):
        results = [{"retrieved_node_ids": ["doc2", "doc3", "doc1"]}]
        relevant = {"doc1"}
        assert mean_reciprocal_rank(results, relevant) == pytest.approx(1.0 / 3.0)

    def test_not_found(self):
        results = [{"retrieved_node_ids": ["doc2", "doc3"]}]
        relevant = {"doc1"}
        assert mean_reciprocal_rank(results, relevant) == pytest.approx(0.0)

    def test_multiple_queries_average(self):
        results = [
            {"retrieved_node_ids": ["doc1", "doc2"]},  # RR = 1.0
            {"retrieved_node_ids": ["doc2", "doc1"]},  # RR = 0.5
        ]
        relevant = {"doc1"}
        expected = (1.0 + 0.5) / 2.0
        assert mean_reciprocal_rank(results, relevant) == pytest.approx(expected)

    def test_empty_results(self):
        assert mean_reciprocal_rank([], {"doc1"}) == pytest.approx(0.0)

    def test_no_node_ids_in_results(self):
        results = [{"query": "test"}]  # no retrieved_node_ids key
        assert mean_reciprocal_rank(results, {"doc1"}) == pytest.approx(0.0)
