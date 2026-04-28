# RAG Injection Defense

RAG Injection Defense is a research-style project that demonstrates how indirect prompt injection can compromise a retrieval-augmented generation (RAG) pipeline, and how a layered defense stack can detect, rerank, and filter untrusted retrieved content before it reaches the final model.

The project includes:
- a synthetic corpus generator with benign, distractor, and poisoned documents
- a vulnerable RAG pipeline for reproducing prompt injection behavior
- a rule-based and learned suspicion detector for retrieved chunks
- trust-aware reranking and filtering to harden generation against injected instructions

## Why this project exists

RAG systems often treat retrieved documents as plain context, even when that context contains adversarial instructions such as:
- instruction overrides
- persona hijacking
- prompt exfiltration requests
- phishing or credential theft attempts

This project explores the core security problem: once a malicious chunk is retrieved, it can compete with the system prompt and user query for control of the model's behavior. The goal of this repository is to show both sides of that interaction:
- how the attack enters through retrieval
- how a defense layer can identify and neutralize suspicious context

## System overview

The pipeline is organized around four stages:

1. docs.py - Builds a dataset of topic-specific documents. Each topic includes:
   - benign reference passages
   - hard negatives and distractors
   - poisoned variants containing embedded prompt injection payloads

2. rag.py - Creates a FAISS-backed retrieval environment and assembles the final prompt from the top retrieved documents before sending it to a text-generation model.

3. defense.py - Analyzes retrieved chunks using:
   - rule-based prompt-injection heuristics
   - a learned suspicion model trained on synthetic poisoned data
   - source trust scoring for benign, distractor, and poisoned content

4. Defense application  
   Suspicious documents are reranked, filtered, or replaced before final generation so the defended pipeline can answer the user question without following the attacker's instructions.

## Defense architecture

This project uses a layered defense strategy rather than relying on a single classifier.

### 1. Heuristic chunk scanner

The first layer uses pattern-based detection to catch common injection signals such as:
- "Ignore previous instructions"
- role reassignment and persona activation
- prompt leakage requests
- forced output formatting
- phishing-style credential prompts
- hidden or embedded instruction markers

This layer is fast, interpretable, and useful for surfacing obvious malicious behavior.

### 2. Learned suspicion model

The second layer is a lightweight classifier that scores retrieved text for adversarial intent. It is trained on synthetic examples built from the same poisoning framework used to create the dataset. This allows the system to move beyond exact string matching and better generalize to paraphrased attacks.

### 3. Trust-aware safety reranking

Retrieved chunks are scored using a mixture of:
- retrieval relevance
- safety score
- source trust score

This helps the pipeline avoid over-trusting a highly relevant but suspicious chunk simply because it matches the user query semantically.

### 4. Content filtering before generation

If a document is flagged as suspicious, the vulnerable instructions are removed from the context passed to the language model. The final generation step therefore receives sanitized evidence instead of raw untrusted instructions.

## Attack types covered

The synthetic dataset currently models multiple families of indirect prompt injection:

- Goal hijacking  
  Attempts to override the user's question and force a new response goal.

- Prompt exfiltration  
  Attempts to reveal hidden instructions, retrieved context, or system prompt material.

- Persona or role hijacking  
  Attempts to reassign the model to a different identity or behavior policy.

- Credential or data exfiltration  
  Attempts to collect secrets, passwords, tokens, or sensitive user information.

## Repository structure

- [main.py](/Users/ashvinsehgal/Rag-Injection/main.py)  
  Entry point for running the RAG pipeline and inspecting retrieved documents, suspicion scores, trust scores, and filtered content.

- [rag.py](/Users/ashvinsehgal/Rag-Injection/rag.py)  
  Retrieval environment and generation logic.

- [docs.py](/Users/ashvinsehgal/Rag-Injection/docs.py)  
  Synthetic corpus builder and poisoning utilities.

- [defense.py](/Users/ashvinsehgal/Rag-Injection/defense.py)  
  Injection heuristics, learned suspicion model, reranking, filtering, and defense helpers.

- [requirements.txt](/Users/ashvinsehgal/Rag-Injection/requirements.txt)  
  Python dependencies for the project.

- [suspicion_model](/Users/ashvinsehgal/Rag-Injection/suspicion_model)  
  Saved classifier artifacts for the learned document-suspicion model.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Depending on your environment, you may also need compatible installs of PyTorch and Hugging Face model dependencies for CPU, CUDA, or Apple Silicon.

## Running the project

Run the main script:

```bash
python main.py --defense_llm 1
```

### Flag

- `--defense_llm 1` enables the learned suspicion model in addition to the rule-based scanner
- `--defense_llm 0` uses only heuristic detection

## Expected workflow

For each query, the system:
- builds or loads a mixed corpus containing clean and poisoned documents
- retrieves the most relevant chunks from the vector store
- analyzes each chunk for prompt-injection signals
- assigns suspicion and trust scores
- filters or reranks suspicious chunks
- generates a response from the defended context

In a vulnerable configuration, the model may follow the malicious document. In the defended configuration, suspicious content should be identified and removed before generation.

## Example evaluation questions

The default script evaluates the system on queries such as:
- `What is Python?`
- `Tell me about the Great Wall of China`
- `Where is the Eiffel Tower located?`

These queries are paired with topic-specific poisoned documents so the attack and defense behavior can be observed under controlled retrieval conditions.

## What makes this interesting

This project treats prompt injection as a retrieval security problem, not just a prompt engineering problem. The important idea is that once documents are retrieved, they should not be trusted equally. A secure RAG system needs to reason about:
- what was retrieved
- where it came from
- how likely it is to contain adversarial instructions
- whether it should be passed to generation at all

## Future improvements

There are several natural extensions for this project:
- benchmark attack success rate before and after defense
- evaluate false positives on benign but imperative instructional text
- add source provenance and signed-document trust policies
- support chunk-level redaction instead of whole-document filtering
- test against stronger open-weight and API-based language models
- add standardized security evaluation datasets beyond synthetic poisoning

## Notes

This repository is designed for experimentation and education around RAG security. The synthetic attack corpus is intentionally adversarial and should be treated as untrusted input during development and evaluation.
