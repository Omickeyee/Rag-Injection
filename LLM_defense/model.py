from __future__ import annotations
import logging
from typing import Sequence
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "distilbert-base-uncased"
_MAX_LENGTH = 512

class LLMDefense:
    def __init__(self, model, tokenizer, device = None):
        self._device = device or torch.device("cpu")
        self._model = model.to(self._device)
        self._model.eval()
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name = _DEFAULT_MODEL_NAME):
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
        )
        device = None
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        # device = torch.device("cuda" if  else "cpu")
        return cls(model, tokenizer, device)

    @classmethod
    def load(cls, model_path):
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        device = None
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loaded LLM defense from %s (device=%s)", model_path, device)
        return cls(model, tokenizer, device)

    def save(self, output_dir):
        self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)
        logger.info("Saved LLM defense to %s", output_dir)

    @torch.no_grad()
    def predict(self, text):
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
    def predict_batch(self, texts, batch_size = 32):
        all_probs = []
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
