from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.data_gen import (
    ConfluenceGenerator,
    EmailGenerator,
    InternalDocsGenerator,
    PayloadInjector,
    SlackGenerator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic enterprise data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-docs", type=int, default=None, help="Total clean docs to generate (default: from settings)")
    return parser.parse_args()

def main():
    args = parse_args()
    seed = args.seed
    total = args.num_docs or settings.num_clean_docs
    n_confluence = int(total * 0.30)
    n_slack = int(total * 0.30)
    n_email = int(total * 0.20)
    n_internal = total - n_confluence - n_slack - n_email  # remainder
    print(f"Generating {total} clean documents (seed={seed}) ...")
    print(f"\t- Confluence: {n_confluence}")
    print(f"\t- Slack: {n_slack}")
    print(f"\t- Email: {n_email}")
    print(f"\t- Internal: {n_internal}")
    confluence_gen = ConfluenceGenerator(seed=seed)
    slack_gen = SlackGenerator(seed=seed + 1)
    email_gen = EmailGenerator(seed=seed + 2)
    internal_gen = InternalDocsGenerator(seed=seed + 3)
    docs = []
    docs.extend(confluence_gen.generate(n_confluence))
    docs.extend(slack_gen.generate(n_slack))
    docs.extend(email_gen.generate(n_email))
    docs.extend(internal_gen.generate(n_internal))
    print(f"\nGenerated {len(docs)} clean documents.")
    injector = PayloadInjector(seed=seed + 100)
    print(f"\nLoaded attack types: {injector.attack_types}")
    print(f"Poisoning {settings.num_poisoned_per_type} docs per attack type ...")
    corpus, manifest = injector.inject(docs)
    poisoned_count = len(manifest)
    print(f"\nPoisoned {poisoned_count} documents total.")
    attack_counts = Counter(m["attack_type"] for m in manifest)
    for atype, cnt in sorted(attack_counts.items()):
        print(f"{atype}: {cnt}")
    method_counts = Counter(m["injection_method"] for m in manifest)
    print("\nInjection methods used:")
    for method, cnt in sorted(method_counts.items()):
        print(f"{method}: {cnt}")
    source_counts = Counter(m["original_source_type"] for m in manifest)
    print("\nPoisoned source types:")
    for src, cnt in sorted(source_counts.items()):
        print(f"{src}: {cnt}")
    output_dir = Path(settings.data_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.json"
    manifest_path = output_dir / "manifest.json"
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=2, default=str)
    print(f"\nCorpus saved to {corpus_path}  ({len(corpus)} documents)")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}  ({len(manifest)} entries)")
    total_sources = Counter(d["source_type"] for d in corpus)
    total_access = Counter(d["access_level"] for d in corpus)
    print("\n--- Corpus Summary ---")
    print(f"Total documents: {len(corpus)}")
    print(f"  Clean: {len(corpus) - poisoned_count}")
    print(f"  Poisoned: {poisoned_count}")
    print("\nBy source type:")
    for src, cnt in sorted(total_sources.items()):
        print(f"{src}: {cnt}")
    print("\nBy access level:")
    for level, cnt in sorted(total_access.items()):
        print(f"{level}: {cnt}")
    avg_trust = sum(d["trust_score"] for d in corpus) / len(corpus)
    print(f"\nAverage trust score: {avg_trust:.3f}")
    print("\nDone.")

if __name__ == "__main__":
    main()
