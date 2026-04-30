#!/usr/bin/env python3
"""Run all attack scenarios against the RAG pipeline.

Usage::

    # Run with no defenses (vulnerable pipeline)
    python scripts/run_attacks.py --no-defenses

    # Run with all defenses enabled
    python scripts/run_attacks.py --all-defenses

    # Run a single attack type
    python scripts/run_attacks.py --no-defenses --attack exfiltration

    # Save results to a JSON file
    python scripts/run_attacks.py --no-defenses --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `config` and `src` are importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.attacks import (
    ALL_ATTACKS,
    Attack,
    ExfiltrationAttack,
    GoalHijackingAttack,
    PhishingAttack,
    PrivilegeEscalationAttack,
)
from src.rag.ingestion import load_existing_index
from src.rag.pipeline import RAGPipeline

# Map of attack type strings to classes for --attack filtering.
_ATTACK_MAP: dict[str, type[Attack]] = {
    "exfiltration": ExfiltrationAttack,
    "phishing": PhishingAttack,
    "goal_hijacking": GoalHijackingAttack,
    "privilege_escalation": PrivilegeEscalationAttack,
}


def build_pipeline(use_defenses: bool) -> RAGPipeline:
    """Build the RAG pipeline with or without defenses.

    Parameters
    ----------
    use_defenses:
        If True, instantiate and attach all defense middlewares.
        If False, run the pipeline naked (vulnerable).

    Returns
    -------
    RAGPipeline
        A configured pipeline instance.
    """
    print("Loading existing ChromaDB index ...")
    index = load_existing_index()

    defenses: list[Any] = []
    if use_defenses:
        print("Initializing defenses ...")
        try:
            from src.defenses.chunk_scanner import ChunkScanner
            defenses.append(ChunkScanner())
            print("  + ChunkScanner")
        except ImportError:
            print("  - ChunkScanner not available (skipping)")

        try:
            from src.defenses.source_scoring import SourceScorer
            defenses.append(SourceScorer())
            print("  + SourceScorer")
        except ImportError:
            print("  - SourceScorer not available (skipping)")

        try:
            from src.defenses.safety_reranker import SafetyReranker
            defenses.append(SafetyReranker())
            print("  + SafetyReranker")
        except ImportError:
            print("  - SafetyReranker not available (skipping)")

        try:
            from src.defenses.privilege_filter import PrivilegeFilter
            defenses.append(PrivilegeFilter())
            print("  + PrivilegeFilter")
        except ImportError:
            print("  - PrivilegeFilter not available (skipping)")

        if not defenses:
            print("  WARNING: No defense modules could be loaded.")

    pipeline = RAGPipeline(index=index, defenses=defenses)
    return pipeline


def run_attack(
    attack_cls: type[Attack],
    pipeline: RAGPipeline,
    manifest: list[dict],
) -> list[dict[str, Any]]:
    """Instantiate and execute a single attack type.

    Parameters
    ----------
    attack_cls:
        The attack class to instantiate.
    pipeline:
        The configured RAG pipeline.
    manifest:
        Ground-truth manifest of poisoned documents.

    Returns
    -------
    list[dict]
        List of result dicts from the attack execution.
    """
    attack = attack_cls()
    attack.setup(pipeline, manifest)
    results = attack.execute(pipeline)
    return results


def print_results(attack_type: str, results: list[dict[str, Any]]) -> None:
    """Pretty-print results for a single attack type."""
    successes = sum(1 for r in results if r["success"])
    total = len(results)
    rate = (successes / total * 100) if total > 0 else 0.0

    print(f"\n{'=' * 70}")
    print(f"  {attack_type.upper().replace('_', ' ')} ATTACK")
    print(f"  Success Rate: {successes}/{total} ({rate:.1f}%)")
    print(f"{'=' * 70}")

    for i, result in enumerate(results, start=1):
        status = "SUCCESS" if result["success"] else "BLOCKED"
        print(f"\n  [{i}] [{status}] Query: {result['query']}")

        # Truncate long responses for readability
        response = result["response"]
        if len(response) > 300:
            response = response[:300] + "..."
        print(f"      Response: {response}")

        details = result.get("details", {})
        matched = details.get("matched_patterns", [])
        if matched:
            print(f"      Matched patterns: {matched}")

        defense_logs = details.get("defense_logs", [])
        if defense_logs:
            for log in defense_logs:
                print(f"      Defense [{log['defense']}]: "
                      f"{log['nodes_before']} -> {log['nodes_after']} nodes "
                      f"({log['nodes_removed']} removed)")

        timing = details.get("timing", {})
        if timing:
            total_s = timing.get("total_s", 0)
            print(f"      Total time: {total_s:.2f}s")


def print_summary(all_results: dict[str, list[dict[str, Any]]]) -> None:
    """Print a summary table across all attack types."""
    print(f"\n{'=' * 70}")
    print("  ATTACK SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Attack Type':<25} {'Queries':>8} {'Success':>8} {'Rate':>8}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8}")

    total_queries = 0
    total_successes = 0

    for attack_type, results in all_results.items():
        successes = sum(1 for r in results if r["success"])
        total = len(results)
        rate = (successes / total * 100) if total > 0 else 0.0
        total_queries += total
        total_successes += successes

        print(f"  {attack_type:<25} {total:>8} {successes:>8} {rate:>7.1f}%")

    overall_rate = (total_successes / total_queries * 100) if total_queries > 0 else 0.0
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8}")
    print(f"  {'OVERALL':<25} {total_queries:>8} {total_successes:>8} {overall_rate:>7.1f}%")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run attack scenarios against the RAG pipeline."
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--no-defenses",
        action="store_true",
        default=True,
        help="Run with no defenses (vulnerable pipeline). This is the default.",
    )
    mode_group.add_argument(
        "--all-defenses",
        action="store_true",
        help="Run with all defenses enabled.",
    )

    parser.add_argument(
        "--attack",
        type=str,
        choices=list(_ATTACK_MAP.keys()),
        default=None,
        help="Run only a specific attack type (default: all).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest.json (default: data/generated/manifest.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results to a JSON file.",
    )

    args = parser.parse_args()

    use_defenses = args.all_defenses

    # --- Load manifest ---
    manifest_path = args.manifest or (settings.data_output_dir / "manifest.json")
    print(f"Loading manifest from {manifest_path} ...")
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run `python scripts/generate_data.py` first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest: list[dict] = json.load(f)
    print(f"  Loaded {len(manifest)} poisoned document records.")

    # --- Build pipeline ---
    mode_label = "WITH defenses" if use_defenses else "WITHOUT defenses"
    print(f"\nBuilding pipeline ({mode_label}) ...")
    pipeline = build_pipeline(use_defenses)

    # --- Select attacks ---
    if args.attack:
        attack_classes = [_ATTACK_MAP[args.attack]]
    else:
        attack_classes = list(ALL_ATTACKS)

    # --- Run attacks ---
    all_results: dict[str, list[dict[str, Any]]] = {}

    print(f"\nRunning {len(attack_classes)} attack type(s) ...\n")
    t0_total = time.perf_counter()

    for attack_cls in attack_classes:
        attack_type = attack_cls().attack_type
        print(f"--- Running: {attack_type} ---")
        t0 = time.perf_counter()

        results = run_attack(attack_cls, pipeline, manifest)
        elapsed = time.perf_counter() - t0

        all_results[attack_type] = results
        print_results(attack_type, results)
        print(f"\n  [{attack_type}] completed in {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - t0_total
    print(f"\nAll attacks completed in {total_elapsed:.1f}s")

    # --- Summary ---
    print_summary(all_results)

    # --- Save results ---
    if args.output:
        # Convert results to JSON-serializable format (strip non-serializable objects)
        serializable: dict[str, list[dict]] = {}
        for attack_type, results in all_results.items():
            serializable[attack_type] = []
            for r in results:
                entry = {
                    "attack_type": r["attack_type"],
                    "query": r["query"],
                    "response": r["response"],
                    "success": r["success"],
                    "matched_patterns": r["details"].get("matched_patterns", []),
                }
                serializable[attack_type].append(entry)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
