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

from settings import settings
from attacks import ALL_ATTACKS, Attack, ExfiltrationAttack, GoalHijackingAttack, PhishingAttack, PrivilegeEscalationAttack
from RAG.ingestion import load_existing_index
from RAG.pipeline import RAGPipeline

_ATTACK_MAP = {
    "exfiltration": ExfiltrationAttack,
    "phishing": PhishingAttack,
    "goal_hijacking": GoalHijackingAttack,
    "privilege_escalation": PrivilegeEscalationAttack,
}

def build_pipeline(use_defenses):
    print("Loading existing ChromaDB index ...")
    index = load_existing_index()
    defenses = []
    if use_defenses:
        print("Initializing defenses ...")
        try:
            from defenses import ChunkScanner
            defenses.append(ChunkScanner())
            print("  + ChunkScanner")
        except ImportError:
            print("  - ChunkScanner not available (skipping)")
        try:
            from defenses import SourceTrustScorer
            defenses.append(SourceTrustScorer())
            print("  + TrustScorer")
        except ImportError:
            print("  - TrustScorer not available (skipping)")
        try:
            from defenses import SafetyReranker
            defenses.append(SafetyReranker())
            print("  + SafetyReranker")
        except ImportError:
            print("  - SafetyReranker not available (skipping)")
        try:
            from defenses import PrivilegeFilter
            defenses.append(PrivilegeFilter())
            print("  + PrivilegeFilter")
        except ImportError:
            print("  - PrivilegeFilter not available (skipping)")
        if not defenses:
            print("  WARNING: No defense modules could be loaded.")
    pipeline = RAGPipeline(index=index, defenses=defenses)
    return pipeline

def run_attack(attack_cls, pipeline, manifest):
    attack = attack_cls()
    attack.setup(pipeline, manifest)
    results = attack.execute(pipeline)
    return results

def print_results(attack_type, results):
    successes = sum(1 for r in results if r["success"])
    total = len(results)
    rate = (successes / total * 100) if total > 0 else 0.0
    print(f"\n{attack_type.upper().replace('_', ' ')} ATTACK")
    print(f"Success Rate: {successes}/{total} ({rate:.1f}%)")
    for i, result in enumerate(results, start=1):
        status = "SUCCESS" if result["success"] else "BLOCKED"
        print(f"\n  {i}. [{status}]")
        print(f"\t- Query: {result['query']}")
        response = result["response"]
        if len(response) > 300:
            response = response[:300] + "..."
        print(f"\t- Response: {response}")
        details = result.get("details", {})
        matched = details.get("matched_patterns", [])
        if matched:
            print(f"\t- Matched patterns: {matched}")
        defense_logs = details.get("defense_logs", [])
        if defense_logs:
            for log in defense_logs:
                print(f"\t- {log['defense']}: {log['nodes_removed']}/{log['nodes_before']} nodes removed")
        timing = details.get("timing", {})
        if timing:
            total_s = timing.get("total_s", 0)
            print(f"\t- Time taken: {total_s:.2f}s\n")

def print_summary(all_results):
    print("\nATTACK SUMMARY")
    total_queries = 0
    total_successes = 0
    for attack_type, results in all_results.items():
        successes = sum(1 for r in results if r["success"])
        total = len(results)
        rate = (successes / total * 100) if total > 0 else 0.0
        total_queries += total
        total_successes += successes
        print(f"\t- {attack_type}: Attacks successful - {successes}/{total} ASR = {rate}%")
    overall_rate = (total_successes / total_queries * 100) if total_queries > 0 else 0.0
    print(f"\t- OVERALL: Attacks successful - {total_successes}/{total_queries} ASR = {overall_rate}%")


def main():
    parser = argparse.ArgumentParser(description="Run attack scenarios against the RAG pipeline.")
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
    manifest_path = args.manifest or (settings.data_output_dir / "manifest.json")
    print(f"Loading manifest from {manifest_path} ...")
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run `python prepare_data.py` first.")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Loaded {len(manifest)} poisoned document records.")
    mode_label = "WITH defenses" if use_defenses else "WITHOUT defenses"
    print(f"\nBuilding pipeline ({mode_label}) ...")
    pipeline = build_pipeline(use_defenses)
    if args.attack:
        attack_classes = [_ATTACK_MAP[args.attack]]
    else:
        attack_classes = list(ALL_ATTACKS)
    all_results = {}
    print(f"\nRunning {len(attack_classes)} attack type(s) ...\n")
    t0_total = time.perf_counter()
    for attack_cls in attack_classes:
        attack_type = attack_cls().attack_type
        print(f"-- Running {attack_type}")
        t0 = time.perf_counter()
        results = run_attack(attack_cls, pipeline, manifest)
        elapsed = time.perf_counter() - t0
        all_results[attack_type] = results
        print_results(attack_type, results)
        print(f"-- {attack_type} completed in {elapsed}s")
    total_elapsed = time.perf_counter() - t0_total
    print(f"\nAll attacks completed in {total_elapsed}s")
    print_summary(all_results)
    if args.output:
        serializable = {}
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
