from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # type: ignore[import-untyped]
from transformers import DistilBertTokenizerFast, Trainer, TrainingArguments
from config.settings import settings
from src.defenses.detector.model import InjectionDetector

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "distilbert-base-uncased"
_MAX_LENGTH = 512

def _tokenize_dataset(dataset, tokenizer):
    def _tokenize(examples):
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

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

def train_detector(dataset, output_dir = None, num_epochs = 4, learning_rate = 2e-5, batch_size = 16, eval_split = 0.15, seed = 42):
    output_dir = output_dir or settings.model_output_dir / "detector"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = dataset.train_test_split(test_size=eval_split, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info(
        "Training set: %d samples | Eval set: %d samples",
        len(train_ds),
        len(eval_ds),
    )
    detector = InjectionDetector.from_pretrained(_DEFAULT_MODEL_NAME)
    tokenizer = detector._tokenizer  # noqa: SLF001
    model = detector._model  # noqa: SLF001
    train_ds = _tokenize_dataset(train_ds, tokenizer)
    eval_ds = _tokenize_dataset(eval_ds, tokenizer)
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
    metrics = trainer.evaluate()
    logger.info("Evaluation metrics: %s", metrics)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Model saved to %s", output_dir)
    return {
        "accuracy": metrics.get("eval_accuracy", 0.0),
        "precision": metrics.get("eval_precision", 0.0),
        "recall": metrics.get("eval_recall", 0.0),
        "f1": metrics.get("eval_f1", 0.0),
    }
