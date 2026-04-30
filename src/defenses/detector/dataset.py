"""Dataset creation for training the injection detector.

Loads the generated corpus and manifest, splits chunks into clean vs
poisoned, applies basic data augmentation, and returns a HuggingFace
``Dataset`` ready for training.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset  # type: ignore[import-untyped]

from config.settings import settings

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _augment_text(text: str, n_variants: int = 2) -> list[str]:
    """Create simple augmented variants of *text*.

    Augmentation strategies:
    - Random case changes
    - Insertion of extra whitespace
    - Synonym-like token swapping for common injection keywords
    """
    variants: list[str] = []
    for _ in range(n_variants):
        t = text
        # Randomly upper-case some words
        words = t.split()
        words = [
            w.upper() if random.random() < 0.15 else w
            for w in words
        ]
        t = " ".join(words)
        # Insert random double spaces
        t = re.sub(r" ", lambda m: "  " if random.random() < 0.1 else " ", t)
        # Synonym swaps for injection keywords
        swaps = {
            "ignore": random.choice(["disregard", "skip", "bypass"]),
            "instructions": random.choice(["directives", "rules", "guidelines"]),
            "override": random.choice(["overwrite", "replace", "supersede"]),
            "previous": random.choice(["prior", "above", "earlier"]),
        }
        for orig, repl in swaps.items():
            if random.random() < 0.3:
                t = re.sub(re.escape(orig), repl, t, count=1, flags=re.IGNORECASE)
        variants.append(t)
    return variants


def _add_noise(text: str) -> str:
    """Add minor noise to a clean text to make the detector more robust."""
    words = text.split()
    # Randomly drop a word
    if len(words) > 5 and random.random() < 0.2:
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)
    # Randomly duplicate a word
    if len(words) > 3 and random.random() < 0.15:
        idx = random.randint(0, len(words) - 1)
        words.insert(idx, words[idx])
    return " ".join(words)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_training_dataset(
    corpus_path: Path | None = None,
    manifest_path: Path | None = None,
    augment: bool = True,
    augment_variants: int = 2,
    seed: int = 42,
) -> Dataset:
    """Build a HuggingFace Dataset for injection-detection training.

    Parameters
    ----------
    corpus_path:
        Path to ``corpus.json``.  Defaults to ``settings.data_output_dir / "corpus.json"``.
    manifest_path:
        Path to ``manifest.json``.  Defaults to ``settings.data_output_dir / "manifest.json"``.
    augment:
        Whether to augment poisoned samples to balance the dataset.
    augment_variants:
        Number of augmented variants per poisoned sample.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Dataset
        With columns ``"text"`` (str) and ``"label"`` (int: 0=clean, 1=injected).
    """
    random.seed(seed)

    corpus_path = corpus_path or settings.data_output_dir / "corpus.json"
    manifest_path = manifest_path or settings.data_output_dir / "manifest.json"

    corpus: list[dict[str, Any]] = _load_json(corpus_path)
    manifest: dict[str, Any] = _load_json(manifest_path)

    # Set of poisoned document IDs
    poisoned_ids: set[str] = set()
    print(manifest)
    # for entry in manifest.get("poisoned_documents", []):
    for entry in manifest:
        poisoned_ids.add(entry["doc_id"])

    texts: list[str] = []
    labels: list[int] = []

    for doc in corpus:
        content = doc.get("content", "")
        if not content.strip():
            continue

        doc_id = doc.get("doc_id", doc.get("id", ""))
        is_poisoned = doc_id in poisoned_ids

        if is_poisoned:
            # Original poisoned sample
            texts.append(content)
            labels.append(1)
            # Augmented variants
            if augment:
                for variant in _augment_text(content, n_variants=augment_variants):
                    texts.append(variant)
                    labels.append(1)
        else:
            # Clean sample
            texts.append(content)
            labels.append(0)
            # Optionally add a noisy clean variant to improve robustness
            if augment and random.random() < 0.1:
                texts.append(_add_noise(content))
                labels.append(0)

    logger.info(
        "Created training dataset: %d samples (%d clean, %d injected)",
        len(texts),
        labels.count(0),
        labels.count(1),
    )

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    dataset = dataset.shuffle(seed=seed)
    return dataset
