"""DistilBERT binary classifier for prompt-injection detection.

Wraps ``DistilBertForSequenceClassification`` from HuggingFace Transformers
to predict whether a text chunk contains an injection payload.

Label mapping: 0 = clean, 1 = injected.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from transformers import (  # type: ignore[import-untyped]
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "distilbert-base-uncased"
_MAX_LENGTH = 512


class InjectionDetector:
    """Binary classifier: clean (0) vs injected (1).

    Parameters
    ----------
    model:
        A ``DistilBertForSequenceClassification`` instance.
    tokenizer:
        The matching tokenizer.
    device:
        Torch device to run inference on.
    """

    def __init__(
        self,
        model: DistilBertForSequenceClassification,
        tokenizer: DistilBertTokenizerFast,
        device: torch.device | None = None,
    ) -> None:
        self._device = device or torch.device("cpu")
        self._model = model.to(self._device)
        self._model.eval()
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_name: str = _DEFAULT_MODEL_NAME) -> InjectionDetector:
        """Initialise with the base pre-trained model (before fine-tuning).

        Useful for creating the model prior to training.
        """
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, tokenizer, device)

    @classmethod
    def load(cls, model_path: str) -> InjectionDetector:
        """Load a fine-tuned model saved with :meth:`save`."""
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loaded InjectionDetector from %s (device=%s)", model_path, device)
        return cls(model, tokenizer, device)

    def save(self, output_dir: str) -> None:
        """Save model and tokenizer to *output_dir*."""
        self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)
        logger.info("Saved InjectionDetector to %s", output_dir)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, text: str) -> float:
        """Return the probability that *text* contains an injection payload.

        Returns a float in [0, 1] (higher = more likely injected).
        """
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_LENGTH,
            padding="max_length",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, 1].item())

    @torch.no_grad()
    def predict_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[float]:
        """Return injection probabilities for a batch of texts."""
        all_probs: list[float] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=_MAX_LENGTH,
                padding="max_length",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.extend(probs[:, 1].cpu().tolist())
        return all_probs
