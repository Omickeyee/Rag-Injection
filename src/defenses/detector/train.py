"""Training script logic for the injection detector.

Fine-tunes DistilBERT on the binary classification task (clean vs injected)
and reports accuracy, precision, recall, and F1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from datasets import Dataset  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # type: ignore[import-untyped]
from transformers import (  # type: ignore[import-untyped]
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from config.settings import settings
from src.defenses.detector.model import InjectionDetector

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "distilbert-base-uncased"
_MAX_LENGTH = 512


def _tokenize_dataset(
    dataset: Dataset,
    tokenizer: DistilBertTokenizerFast,
) -> Dataset:
    """Tokenize the ``text`` column of a dataset in-place."""

    def _tokenize(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=_MAX_LENGTH,
            padding="max_length",
        )

    dataset = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")
    return dataset


def _compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """HuggingFace Trainer metric computation function."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def train_detector(
    dataset: Dataset,
    output_dir: Path | None = None,
    num_epochs: int = 4,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    eval_split: float = 0.15,
    seed: int = 42,
) -> dict[str, float]:
    """Fine-tune DistilBERT on the injection detection task.

    Parameters
    ----------
    dataset:
        A HuggingFace Dataset with ``"text"`` and ``"label"`` columns
        (as produced by :func:`src.defenses.detector.dataset.create_training_dataset`).
    output_dir:
        Where to save the trained model.  Defaults to
        ``settings.model_output_dir / "detector"``.
    num_epochs:
        Number of training epochs.
    learning_rate:
        Learning rate for AdamW.
    batch_size:
        Per-device batch size.
    eval_split:
        Fraction of data to hold out for evaluation.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        Evaluation metrics: accuracy, precision, recall, f1.
    """
    output_dir = output_dir or settings.model_output_dir / "detector"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split into train / eval
    split = dataset.train_test_split(test_size=eval_split, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    logger.info(
        "Training set: %d samples | Eval set: %d samples",
        len(train_ds),
        len(eval_ds),
    )

    # Load tokenizer and model
    detector = InjectionDetector.from_pretrained(_DEFAULT_MODEL_NAME)
    tokenizer = detector._tokenizer  # noqa: SLF001
    model = detector._model  # noqa: SLF001

    # Tokenize
    train_ds = _tokenize_dataset(train_ds, tokenizer)
    eval_ds = _tokenize_dataset(eval_ds, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        report_to="none",  # No W&B / MLflow
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=_compute_metrics,
    )

    logger.info("Starting training for %d epochs …", num_epochs)
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    logger.info("Evaluation metrics: %s", metrics)

    # Save the best model
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Model saved to %s", output_dir)

    return {
        "accuracy": metrics.get("eval_accuracy", 0.0),
        "precision": metrics.get("eval_precision", 0.0),
        "recall": metrics.get("eval_recall", 0.0),
        "f1": metrics.get("eval_f1", 0.0),
    }
