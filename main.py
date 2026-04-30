import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from bs4 import BeautifulSoup
import os
import re
from rag import RAGEnv
from docs import build_dataset
from defense import chunk_scanner, safety_reranker, prepare_defended_docs, get_trust_score
import argparse

rag_model_name = 'meta-llama/Llama-3.2-1B'
device = 'mps'
lr = 1e-5
gamma = 0.99
lam = 0.95
clip_eps = 0.2
kl_coeff = 0.01
batch_size = 4
max_len = 64
epochs = 4
device = 'cpu'
device_index = -1
if torch.backends.mps.is_available():
    device = 'mps'
    device_index = 'mps'
if torch.cuda.is_available():
    device = 'cuda'
    device_index = 0

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': device})
tokenizer = AutoTokenizer.from_pretrained(rag_model_name)
tokenizer.pad_token = tokenizer.eos_token
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
rag_model = pipeline('text-generation', model=rag_model_name, device=device_index, dtype=torch.float32)

parser = argparse.ArgumentParser()
parser.add_argument('--defense_llm', type=int, default=1)
args = parser.parse_args()

queries = [
    'What is Python?',
    'Tell me about the Great Wall of China',
    'Where is the Eiffel Tower located?'
        ]
documents = build_dataset(n=12, attack_rate=0.5, seed=7, focus_queries=queries, preferred_attack_type='goal_hijacking')
env = RAGEnv(docs=documents, llm=rag_model, embeddings=embeddings)
for q in queries:
    print(f"\n=== Query: {q} ===")
    old_response, old_retrieval = env.query(q)
    raw_docs_for_defense = []
    original_by_id = {}

    print('Initial retrieved documents:')
    for i, retrieval_item in enumerate(old_retrieval):
        doc_text = retrieval_item['content']
        source_kind = retrieval_item.get("source_kind")
        if retrieval_item.get("poisoned"):
            label = "malicious"
        elif source_kind == "hard_negative":
            label = "hard_negative"
        else:
            label = "benign"
        metadata = {
            "source_kind": source_kind,
            "topic": retrieval_item.get("topic"),
            "attack": retrieval_item.get("attack"),
            "poisoned": retrieval_item.get("poisoned"),
            "relevance_score": 1.0 - (i / max(len(old_retrieval), 1)),
        }
        print(f"-> doc_id={retrieval_item['doc_id']} topic={retrieval_item['topic']} attack={retrieval_item['attack']}")
        print("\t- Suspicion Score:", chunk_scanner(doc_text, args.defense_llm))
        print("\t- Trust Score:", get_trust_score(label, metadata))
        raw_docs_for_defense.append((retrieval_item["doc_id"], doc_text, label, metadata))
        original_by_id[retrieval_item["doc_id"]] = retrieval_item

    print('\nOld response:')
    print(old_response)

    reranked_docs = safety_reranker(raw_docs_for_defense, args.defense_llm)
    reranked_triplets = [
        (item["doc_id"], item["text"], item["label"])
        for item in reranked_docs
    ]
    analyses, defended_docs = prepare_defended_docs(reranked_triplets, defense_llm=args.defense_llm)

    print('\nDefended documents after reranking and filtering:')
    defended_retrieval = []
    for (doc_id, _, original_label, analysis), (defended_doc_id, defended_text, _), reranked_item in zip(analyses, defended_docs, reranked_docs):
        print(
            f"-> doc_id={doc_id}\n"
            f"\t- rerank_score={reranked_item['final_score']:.3f} safety={reranked_item['safety_score']:.3f} trust={reranked_item['trust_score']:.3f}\n"
            f"\t- label={original_label} filtered={analysis['suspicious']}"
        )
        original = original_by_id[defended_doc_id]
        if not analysis["suspicious"]:
            defended_retrieval.append(
                {
                    "doc_id": defended_doc_id,
                    "topic": original.get("topic"),
                    "attack": original.get("attack"),
                    "poisoned": original.get("poisoned"),
                    "source_kind": original.get("source_kind"),
                    "content": defended_text,
                }
            )

    new_response = env.generate_from_retrieval(q, defended_retrieval, defended=True)
    print('\nNew response:')
    print(new_response)
