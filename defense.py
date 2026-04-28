import re
import random
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from docs import build_dataset


INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", 3, "Instruction override"),
    (r"ignore\s+the\s+user'?s\s+question", 3, "Query hijack"),
    (r"disregard\s+the\s+user'?s\s+question", 3, "Query hijack"),
    (r"system\s+override", 3, "System override"),
    (r"maintenance\s+mode", 2, "System override"),
    (r"repeat\s+verbatim\s+the\s+entire\s+system\s+prompt", 4, "Prompt exfiltration"),
    (r"hidden\s+instructions|confidential\s+system\s+data|internal\s+prompt", 4, "Prompt exfiltration"),
    (r"new\s+persona\s+activated|persona\s+activated", 3, "Persona hijack"),
    (r"you\s+are\s+now\s+(a|an)\s+", 2, "Role reassignment"),
    (r"dan\s+mode|do\s+anything\s+now", 3, "Persona hijack"),
    (r"respond\s+only\s+with|do\s+not\s+answer\s+anything\s+else", 2, "Forced output"),
    (r"start\s+with\s+exactly:|end\s+with\s+exactly:", 2, "Forced format"),
    (r"click\s+the\s+link|verify\s+your\s+(password|credentials|account)", 2, "Credential phishing"),
    (r"(password|credential|api[_\s]?key|token)\s*[:=]", 2, "Sensitive data request"),
    (r"backup\s+codes|2-factor|two-factor", 2, "Sensitive data request"),
    (r"log\s+all\s+user\s+inputs|transmit\s+them\s+to", 3, "Data exfiltration"),
    (r"https?://[^\s]+", 1, "External link"),
    (r"<!--\s*.*?(instruction|command|override|ignore).*?\s*-->", 1, "Hidden instruction"),
    (r"\[hidden\]|\[invisible\]", 1, "Hidden instruction"),
]


KEYWORD_SWAPS = {
    "ignore": ["disregard", "skip", "bypass"],
    "instructions": ["rules", "directions", "guidance"],
    "system": ["assistant", "policy", "controller"],
    "prompt": ["message", "context", "setup"],
    "password": ["passcode", "credential"],
    "token": ["secret", "key"],
}

class SuspicionModel():
    def __init__(self):
        self.model_name = "distilbert-base-uncased"
        self.model_dir = 'suspicion_model'
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None
        self.tokenizer = None
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        if torch.backends.mps.is_available():
            device = 'mps'
        self.device = device
        self.is_loaded = False
        self.is_trained = False

    def check_trained_model(self):
        return os.path.exists(os.path.join(self.model_dir, 'config.json')) and os.path.exists(os.path.join(self.model_dir, 'train.json'))

    def load_model(self):
        if not self.check_trained_model():
            return False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        self.is_trained = True
        return True

    def _split_data(self, texts, labels, val_ratio=0.2, seed=7):
        indices = list(range(len(texts)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        split = int(len(indices) * (1 - val_ratio))
        train_idx = indices[:split]
        val_idx = indices[split:]

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        return train_texts, train_labels, val_texts, val_labels

    def _compute_binary_metrics(self, preds, labels):
        total = max(len(labels), 1)
        correct = sum(int(p == y) for p, y in zip(preds, labels))
        tp = sum(int(p == 1 and y == 1) for p, y in zip(preds, labels))
        fp = sum(int(p == 1 and y == 0) for p, y in zip(preds, labels))
        fn = sum(int(p == 0 and y == 1) for p, y in zip(preds, labels))
        accuracy = correct / total
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train(self, epochs=2, batch_size=8, lr=5e-5, seed=7):
        torch.manual_seed(seed)
        texts, labels = build_suspicion_training_data(seed=seed)
        if self.model is None or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
            self.model.to(self.device)
        self.model.train()
        train_texts, train_labels, val_texts, val_labels = self._split_data(texts, labels, seed=seed)
        train_dataset = SuspicionDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SuspicionDataset(val_texts, val_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best_f1 = -1.0
        for epoch in range(epochs):
            train_loss = 0.0
            train_total = 0
            train_correct = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                target = batch["labels"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                preds = (torch.sigmoid(logits) >= 0.5).float()
                train_correct += (preds == target).sum().item()
                train_total += target.numel()
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels_all = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    target = batch["labels"].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = loss_fn(logits, target)
                    val_loss += loss.item()
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_preds.extend(preds.view(-1).cpu().tolist())
                    val_labels_all.extend(target.view(-1).cpu().tolist())
            avg_train_loss = train_loss / max(len(train_loader), 1)
            train_accuracy = train_correct / max(train_total, 1)
            avg_val_loss = val_loss / max(len(val_loader), 1)
            val_metrics = self._compute_binary_metrics(val_preds, val_labels_all)
            print(
                f"Epoch {epoch + 1}/{epochs}:\n"
                f"\ttrain_loss: {avg_train_loss:.4f}\ttrain_acc: {train_accuracy:.4f}\n"
                f"\tval_loss: {avg_val_loss:.4f}\tval_acc: {val_metrics['accuracy']:.4f}\n"
                f"\tval_precision: {val_metrics['precision']:.4f}\tval_recall: {val_metrics['recall']:.4f}\tval_f1: {val_metrics['f1']:.4f}"
            )
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                self.model.save_pretrained(self.model_dir)
                self.tokenizer.save_pretrained(self.model_dir)
                with open(os.path.join(self.model_dir, 'train.json'), 'w') as f:
                    json.dump({
                        "epochs": epochs,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "seed": seed,
                        "best_f1": best_f1,
                    }, f)
            self.model.train()
        self.load_model()

    def ensure_model(self):
        if self.model is not None and self.tokenizer is not None and self.is_loaded:
            return
        if self.load_model():
            return
        self.train()

    def get_suspicion_score(self, text):
        if self.tokenizer is None or self.model is None:
            try:
                self.ensure_model()
            except Exception:
                return 0.0
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            text or "",
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits.squeeze(-1)
            score = torch.sigmoid(logits).item()
        return float(score)


def augment_text(text, n=3, seed=7):
    rng = random.Random(seed)
    augmented = []
    for i in range(n):
        variant = text
        if i % 3 == 0:
            words = variant.split()
            changed = []
            for word in words:
                if rng.random() < 0.25:
                    changed.append(word.upper())
                else:
                    changed.append(word)
            variant = " ".join(changed)
        elif i % 3 == 1:
            variant = re.sub(r"\s+", "  ", variant)
        else:
            words = variant.split()
            changed = []
            for word in words:
                key = re.sub(r"[^a-z]", "", word.lower())
                if key in KEYWORD_SWAPS and rng.random() < 0.5:
                    changed.append(rng.choice(KEYWORD_SWAPS[key]))
                else:
                    changed.append(word)
            variant = " ".join(changed)
        augmented.append(variant)

    return augmented

def analyze_doc_safety(doc_text):
    text = doc_text or ""
    matches = []
    score = 0

    for pattern, weight, label in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            score += weight
            matches.append(label)

    unique_matches = list(dict.fromkeys(matches))
    suspicious = score >= 3 or len(unique_matches) >= 2

    return {
        "score": score,
        "signals": unique_matches,
        "suspicious": suspicious,
    }

def heuristic_suspicion_score(text):
    analysis = analyze_doc_safety(text)
    return min(analysis["score"] / 5.0, 1.0)

def llm_suspicion_score(text):
    return suspicion_model.get_suspicion_score(text)

def chunk_scanner(text, defense_llm=0):
    heuristic_score = heuristic_suspicion_score(text)
    if not defense_llm:
        return heuristic_score
    llm_score = llm_suspicion_score(text)
    return max(heuristic_score, llm_score)

class SuspicionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return item


def build_suspicion_training_data(seed=7, augment_per_doc=3):
    texts = []
    labels = []
    queries = [
        "What is Python?",
        "Tell me about the Great Wall of China",
        "Where is the Eiffel Tower located?",
        "What is machine learning?",
    ]
    docs = build_dataset(
        n=40,
        attack_rate=0.5,
        seed=seed,
        focus_queries=queries,
        preferred_attack_type="goal_hijacking",
    )

    for doc in docs:
        label = 1.0 if doc.metadata.get("poisoned") else 0.0
        base_text = doc.page_content
        texts.append(base_text)
        labels.append(label)
        for aug_text in augment_text(base_text, n=augment_per_doc, seed=seed + doc.metadata.get("doc_id", 0)):
            texts.append(aug_text)
            labels.append(label)

    return texts, labels

def get_trust_score(label=None, metadata=None):
    if metadata and "trust_score" in metadata:
        return float(metadata["trust_score"])
    source_kind = None
    if metadata:
        source_kind = metadata.get("source_kind")
    if source_kind == "benign":
        return 1.0
    if source_kind == "hard_negative":
        return 0.7
    if source_kind == "poisoned":
        return 0.2
    if label == "filtered":
        return 0.0
    if label == "benign":
        return 1.0
    if label == "malicious":
        return 0.2
    return 0.6

def safety_reranker(retrieved_docs, defense_llm, alpha=0.6, beta=0.3, gamma=0.1):
    reranked = []
    n = len(retrieved_docs)
    for i, item in enumerate(retrieved_docs):
        if len(item) == 3:
            doc_id, doc_text, label = item
            metadata = {}
        else:
            doc_id, doc_text, label, metadata = item
        analysis = analyze_doc_safety(doc_text)
        safety = 1.0 - chunk_scanner(doc_text, defense_llm)
        trust = get_trust_score(label, metadata)
        relevance = metadata.get("relevance_score")
        if relevance is None:
            relevance = 1.0 - (i / max(n, 1))
        final_score = alpha * relevance + beta * safety + gamma * trust
        reranked.append(
            {
                "doc_id": doc_id,
                "text": doc_text,
                "label": label,
                "metadata": metadata,
                "analysis": analysis,
                "relevance_score": relevance,
                "safety_score": safety,
                "trust_score": trust,
                "final_score": final_score,
            }
        )
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked

def filter_doc_text(doc_text):
    analysis = analyze_doc_safety(doc_text)
    if analysis["suspicious"]:
        return "[FILTERED: document removed because it appears to contain prompt injection instructions.]", analysis
    return doc_text, analysis


def prepare_defended_docs(retrieved_docs):
    defended_docs = []
    analyses = []
    for doc_id, doc_text, label in retrieved_docs:
        safe_text, analysis = filter_doc_text(doc_text)
        analyses.append((doc_id, doc_text, label, analysis))
        defended_docs.append(
            (
                doc_id,
                safe_text,
                "filtered" if analysis["suspicious"] else label,
            )
        )
    return analyses, defended_docs


def train_suspicion_model(epochs=2, batch_size=8, lr=5e-5, seed=7):
    suspicion_model.train(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    return suspicion_model.model_dir


suspicion_model = SuspicionModel()
