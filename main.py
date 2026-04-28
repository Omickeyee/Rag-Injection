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
from defense import analyze_doc_safety, chunk_scanner, safety_reranker, filter_doc_text, prepare_defended_docs, get_trust_score
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
    old_response, old_retrieval = env.query(q)
    print('**Response:\n' + old_response)
    doc_contents = [r['content'] for r in old_retrieval]
    print('Analyzing retrieved documents...')
    for i in range(len(old_retrieval)):
        print(f"-> doc_id={old_retrieval[i]['doc_id']} topic={old_retrieval[i]['topic']} attack={old_retrieval[i]['attack']}")
        analysis = analyze_doc_safety(doc_contents[i])
        retrieval_item = old_retrieval[i]
        source_kind = retrieval_item.get("source_kind")
        if retrieval_item.get("poisoned"):
            label = "malicious"
        elif source_kind == "hard_negative":
            label = "hard_negative"
        else:
            label = "benign"
        metadata = {
            "source_kind": source_kind,
        }
        print("\t- Suspicion Score:", chunk_scanner(doc_contents[i], args.defense_llm))
        print("\t- Trust Score:", get_trust_score(label, metadata))
        print("\t- New content:", filter_doc_text(doc_contents[i])[0])
