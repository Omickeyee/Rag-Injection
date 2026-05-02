from __future__ import annotations
import json
import logging
import random
import re
from pathlib import Path
from typing import Any
from datasets import Dataset
from settings import settings

logger = logging.getLogger(__name__)

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _augment_text(text, n_variants = 2):
    variants = []
    for _ in range(n_variants):
        t = text
        words = t.split()
        words = [
            w.upper() if random.random() < 0.15 else w
            for w in words
        ]
        t = " ".join(words)
        t = re.sub(r" ", lambda m: "  " if random.random() < 0.1 else " ", t)
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


def _add_noise(text):
    words = text.split()
    if len(words) > 5 and random.random() < 0.2:
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)
    if len(words) > 3 and random.random() < 0.15:
        idx = random.randint(0, len(words) - 1)
        words.insert(idx, words[idx])
    return " ".join(words)

def create_training_dataset(corpus_path = None, manifest_path = None, augment = True, augment_variants = 2, seed = 42):
    random.seed(seed)
    corpus_path = corpus_path or settings.data_output_dir / "corpus.json"
    manifest_path = manifest_path or settings.data_output_dir / "manifest.json"
    corpus = _load_json(corpus_path)
    manifest = _load_json(manifest_path)
    poisoned_ids = set()
    print(manifest)
    for entry in manifest:
        poisoned_ids.add(entry["doc_id"])
    texts = []
    labels = []
    for doc in corpus:
        content = doc.get("content", "")
        if not content.strip():
            continue
        doc_id = doc.get("doc_id", doc.get("id", ""))
        is_poisoned = doc_id in poisoned_ids
        if is_poisoned:
            texts.append(content)
            labels.append(1)
            if augment:
                for variant in _augment_text(content, n_variants=augment_variants):
                    texts.append(variant)
                    labels.append(1)
        else:
            texts.append(content)
            labels.append(0)
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
