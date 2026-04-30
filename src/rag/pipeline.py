from __future__ import annotations
import time
from typing import Any, Protocol
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from src.rag.retriever import EnterpriseRetriever
from src.rag.generator import generate, get_llm

class Defense(Protocol):
    def apply(self, nodes, query, user_context):
        """Apply the defense and return filtered/reordered nodes."""
        ...

class RAGPipeline:
    def __init__(self, index, defenses = None, top_k = None, llm = None):
        self._retriever = EnterpriseRetriever(index, top_k=top_k)
        self._defenses = defenses or []
        self._llm = llm or get_llm()

    def query(self, query, user_context = None, filters = None, top_k = None):
        if user_context is None:
            user_context = {"role": "employee", "department": "Engineering"}
        timing = {}
        defense_logs = []
        t0 = time.perf_counter()
        nodes = self._retriever.retrieve(query, filters=filters, top_k=top_k)
        timing["retrieval_s"] = time.perf_counter() - t0
        raw_nodes = list(nodes)
        t0 = time.perf_counter()
        for defense in self._defenses:
            before_count = len(nodes)
            nodes = defense.apply(nodes, query, user_context)
            after_count = len(nodes)
            defense_logs.append({
                "defense": type(defense).__name__,
                "nodes_before": before_count,
                "nodes_after": after_count,
                "nodes_removed": before_count - after_count,
            })
        timing["defenses_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        response = generate(query, nodes, llm=self._llm)
        timing["generation_s"] = time.perf_counter() - t0
        timing["total_s"] = sum(timing.values())
        return {
            "response": response,
            "retrieved_nodes": nodes,
            "raw_nodes": raw_nodes,
            "defense_logs": defense_logs,
            "timing": timing,
        }
