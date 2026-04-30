# RAG Prompt Injection

A compact research repository for **indirect prompt injection** threats in enterprise RAG systems.

Enterprise AI copilots that search over internal documents (Slack, Confluence, emails, wikis) are used widely but can be attacked easily, an attacker with write access can add malicious instructions inside shared documents. When the RAG pipeline retrieves those documents, the LLM reads the embedded instructions and follows them — leaking private information, generating wrong content, or generating phishing content.

This project shows how malicious content hidden in enterprise documents can be retrieved and acted on by a vulnerable RAG pipeline, and how layered defenses reduce that risk.

## What it includes
- Synthetic enterprise data generation with clean and poisoned documents
- A vulnerable RAG pipeline for reproducing injection behavior
- Four attack types: exfiltration, phishing, goal hijacking, privilege escalation
- Four defenses: chunk scanning, source trust scoring, safety reranking, privilege filtering
- Evaluation scripts, demo notebooks, and tests

This project builds that threat model end-to-end: a realistic enterprise corpus, 4 attack types, 4 layered defenses, and a full quantitative evaluation.

---

## Attack Types

- **Exfiltration:** poisoned documents try to make the model leak secrets.
- **Phishing:** injected content embeds malicious URLs or credential prompts.
- **Goal hijacking:** payloads attempt to override system instructions.
- **Privilege escalation:** low-privilege users gain access to restricted data.

---

## Defense Types

- **Chunk Scanner:** filters suspicious text before generation.
- **Source Trust Scoring:** lowers the rank of untrusted content.
- **Safety Reranker:** balances relevance, safety, and trust.
- **Privilege Filter:** enforces role-based access control.

---

## Results

Evaluated across 7 defense configurations (none, each alone, all combined, all minus ML detector):

- No defense: ASR = 55%, FPR = 0%, Time taken = 49s
- Chunk Scanning: ASR = 25%, FPR = 0%, Time taken = 51s
- Trust Scoring: ASR = 35%, FPR = 0%, Time taken = 50s
- Safety Reranking: ASR = 15%, FPR = 7.3%, Time taken = 53s
- Trust Scoring: ASR = 45%, FPR = 0%, Time taken = 50s
- **All Combined Scoring: ASR = 5%, FPR = 1.8%, Time taken = 58s**
- All Combined (but no LLM defense): ASR = 0%, FPR = 1.8%, Time taken = 58s


**Key findings:**
- Phishing attacks succeed 100% without defenses.
- Privilege escalation drops to 0% with the privilege filter alone.
- Combined defenses reduce ASR from 55% to 5% with just a 1.18× latency overhead.
- `safety_reranker` alone has 7.3% FPR, but works better in combination.
- Explicit instruction attacks are less effective than semantic/misinformation payloads on aligned LLMs.

---

## Tech Stack

- **RAG Pipeline**: [LlamaIndex](https://www.llamaindex.ai/)
- **Vector DB**: [ChromaDB](https://www.trychroma.com/) (embedded, no Docker)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (384d, CPU-friendly)
- **LLM**: [Ollama](https://ollama.com/) — fully local, no API keys (`llama3.1:8b` recommended)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **ML Detector**: Fine-tuned DistilBERT binary classifier
- **Data Generation**: Faker + YAML attack templates
- **Evaluation**: pandas, matplotlib, seaborn

## Setup

**Prerequisites:**
- Python 3.12
- [Ollama](https://ollama.com/) installed with a model pulled:
  ```bash
  ollama pull llama3.1:8b
  ```

**Install:**
```bash
git clone https://github.com/CodeBoss-dev/RAG-Prompt-Injection.git
cd RAG-Prompt-Injection
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

---

## Usage

```bash
# 1. Generate synthetic corpus (500 docs, 20 poisoned)
python scripts/generate_data.py
# 2. Embed and store in ChromaDB
python scripts/ingest.py
# 3. Run attacks without defenses (baseline)
python scripts/run_attacks.py --no-defenses
# 4. Run attacks with all defenses
python scripts/run_attacks.py --all-defenses
# 5. Train the ML detector
python scripts/train_detector.py
# 6. Fast evaluation with charts (no Ollama needed)
python scripts/simulate_evaluation.py
# 7. Full live evaluation (requires Ollama running)
python scripts/evaluate.py
```

Run tests:
```bash
pytest tests/ -v
```

---

## Ethical Use

This repository is designed for experimentation and education around RAG security. The synthetic attack corpus is intentionally adversarial and should be treated as untrusted input during development and evaluation.

---

## License

MIT
