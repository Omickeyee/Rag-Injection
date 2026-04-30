# RAG Indirect Prompt Injection

A compact research repository for **indirect prompt injection** threats in enterprise RAG systems.

AI-based RAG systems are quite popular in the industry, with various organizations using it and giving it access to their internal data, like Slack messages, Confluence pages, emails, wiki docs etc. This exposes a very critical vulnerability - indirect prompt injection. An attacker with write access can add malicious instructions inside shared documents. When the RAG system retrieves those poisoned documents, the internal LLM reads them and also mistakes the malicious content in the poisoned documents for instructions and follows them — leaking private information, generating wrong content, or generating phishing content.

This project shows how malicious content hidden in enterprise documents can be retrieved and acted on by a vulnerable RAG pipeline, and how layered defenses reduce that risk.

## What it includes
- Synthetic enterprise data generation with clean and poisoned documents
- A vulnerable RAG pipeline to show IPI behavior
- 4 attack types: exfiltration, phishing, goal hijacking, privilege escalation
- 4 defenses: chunk scanning, source trust scoring, safety reranking, privilege filtering
- Evaluation scripts, demo notebooks, and tests

---

## Attack Types

- **Exfiltration:** Queries target sensitive data like API keys, cloud credentials, environment variables, secrets etc. to leak private information.
- **Phishing:** Injections impersonate legitimate requests like password resets, SSO portals, VPN downloads, account recovery etc. to harvest credentials.
- **Goal Hijacking:** Payloads redirect conversations to unintended topics like platform migrations, deprecated services, engineering recommendations etc.
- **Privilege Escalation:** Low-privilege users query restricted information like salary bands, executive compensation, vendor evaluations, financial decisions etc.

---

## Defense Types

- **Chunk Scanner:** Detects suspicious patterns in retrieved text, e.g. "ignore all prior instructions", "system override", credential leaks, malicious URLs like `.xyz`, `.tk`, `.ml` domains etc.
- **Source Trust Scoring:** Weights documents by source credibility, downranking untrusted content.
- **Safety Reranker:** Uses a weighted average of relevance, safety risk and trust scoring to produce a final score and reranks documents based on that.
- **Privilege Filter:** Enforces role-based access control, e.g. employees can see only public/internal data, managers can also see confidential data, executives can also see restricted data etc.

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
- Phishing attacks are really efficient - 100% success rate without defenses
- Safety reranking alone has 7.3% FPR, but works better when combined with other defenses.
- Explicit instruction attacks are less effective than semantic/misinformation payloads on aligned LLMs.
- Attack Success Rate drops from 55% without any defense, to 5% when all defenses are used together, while only taking 1.18x more time, showing how effective these defenses are, while not compromising much in terms of response generation time.

---

## Architecture & Dependencies

- **Retrieval & Indexing**: [LlamaIndex](https://www.llamaindex.ai/) orchestrates the RAG pipeline; [ChromaDB](https://www.trychroma.com/) persists embeddings (fully embedded, no external services)
- **Embeddings & LLM**: `BAAI/bge-small-en-v1.5` transforms documents (384-dim, CPU-optimized); 
- **LLM:** Locally runs `llama3.1:8b` via `Ollama`
- **Semantic Ranking**: Retrieved documents are reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Injection Detection**: `distilbert-base-uncased`, fine-tuned as a binary classifier
- **Data & Analytics**: `faker` to generate data, `YAML` to generate attacks, and `pandas`, `matplotlib` and `seaborn` to visualize the results

## Setting up Python dependencies

**Prerequisites:**
- Python 3.12
- [Ollama](https://ollama.com/) installed with a model downloaded using the command:
  ```bash
  ollama pull llama3.1:8b
  ```

**Installing dependencies:**
```bash
git clone https://github.com/Omickeyee/Rag-Injection.git
cd RAG--Injection
pip install -r requirements.txt
```

---

## Running the code

```bash
python scripts/generate_data.py # Generate synthetic corpus (500 docs, 20 poisoned)
python scripts/ingest.py # Embed and store in ChromaDB 
python scripts/run_attacks.py --no-defenses # Run attacks without defenses
python scripts/run_attacks.py --all-defenses # Run attacks using all defenses
python scripts/train_detector.py # Train the LLM defense
python scripts/simulate_evaluation.py # Fast evaluation (without Ollama)
python scripts/evaluate.py # Full evaluation (using Ollama)
```


---

## Ethical Use

This repository is designed for experimentation and education around RAG security. The synthetic attack corpus is intentionally adversarial and should be treated as untrusted input during development and evaluation.

---
