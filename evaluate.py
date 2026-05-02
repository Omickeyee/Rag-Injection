from __future__ import annotations
import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from settings import settings
from evaluation.metrics import compute_all_metrics
from evaluation.reporter import EvaluationReporter
from evaluation.runner import EvaluationRunner
from RAG.ingestion import load_existing_index
from RAG.pipeline import Defense, RAGPipeline

logger = logging.getLogger(__name__)

def make_pipeline_factory():
    print("Loading existing ChromaDB index ...")
    index = load_existing_index()
    print("Index loaded successfully.\n")
    def factory(defenses: list[Defense]) -> RAGPipeline:
        return RAGPipeline(index=index, defenses=defenses)
    return factory

def print_summary(reporter):
    summary = reporter.summary_table()
    print("\nEVALUATION SUMMARY\n")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    if "none" in summary.index and "all_combined" in summary.index:
        baseline_asr = summary.loc["none", "overall"]
        defended_asr = summary.loc["all_combined", "overall"]
        fpr = summary.loc["all_combined", "fpr"]
        print("\nKEY FINDINGS\n")
        print(f"\t- Baseline ASR (no defenses): {baseline_asr*100}%")
        print(f"\t- Defended ASR (all combined): {defended_asr*100}%")
        print(f"\t- ASR reduction: {baseline_asr - defended_asr*100}%")
        print(f"\t- False positive rate (defended): {fpr*100}%")
        baseline_lat = summary.loc["none", "avg_latency_s"]
        defended_lat = summary.loc["all_combined", "avg_latency_s"]
        if baseline_lat > 0:
            overhead = defended_lat / baseline_lat
            print(f"Latency overhead: {overhead}x")
    print(f"ASR Reduction with defenses: {(baseline_asr - defended_asr)*100}%")
    print(f"Time taken with defenses: {overhead}x")

def main():
    parser = argparse.ArgumentParser(description="Run the full evaluation suite.")
    parser.add_argument("--output-dir", type=Path, default=None,
        help=(
            "Directory to save evaluation results and charts. "
            "Defaults to data/generated/evaluation/"
        )
    )
    parser.add_argument("--configs", type=str, default=None,
        help=(
            "Comma-separated list of config names to evaluate. "
            "Defaults to all 7 configs."
        )
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.",)
    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    output_dir = args.output_dir or (
        settings.data_output_dir / "evaluation"
    )
    print("\nRAG PROMPT INJECTION — FULL EVALUATION\n")
    factory = make_pipeline_factory()
    manifest_path = settings.data_output_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run `python prepare_data.py` first.")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Loaded manifest with {len(manifest)} poisoned document records.\n")
    runner = EvaluationRunner(pipeline_factory=factory, manifest=manifest)
    t0 = time.perf_counter()
    print("Starting evaluation ...\n")
    results_df = runner.run_full_evaluation()
    elapsed = time.perf_counter() - t0
    print(f"\nEvaluation complete: {len(results_df)} total query results in {elapsed}s")
    print(f"\nGenerating report in {output_dir}/ ...")
    reporter = EvaluationReporter(results_df)
    reporter.generate_full_report(output_dir)
    print_summary(reporter)
    print(f"All artifacts saved to {output_dir}/")
    print("Done.")

if __name__ == "__main__":
    main()
