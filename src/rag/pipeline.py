"""End-to-end RAG pipeline orchestrator with defense middleware support."""

from __future__ import annotations

import time
from typing import Any, Protocol

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama

from src.rag.retriever import EnterpriseRetriever
from src.rag.generator import generate, get_llm


class Defense(Protocol):
    """Protocol that all defense middleware must satisfy.

    Each defense receives the current list of nodes, the original query,
    and user context, and returns a (possibly filtered/reordered) list of
    nodes along with a log dict describing what it did.
    """

    def apply(
        self,
        nodes: list[NodeWithScore],
        query: str,
        user_context: dict[str, Any],
    ) -> list[NodeWithScore]:
        """Apply the defense and return filtered/reordered nodes."""
        ...


class RAGPipeline:
    """End-to-end RAG orchestrator.

    Query flow::

        query -> retrieve -> [defense_1 -> defense_2 -> ...] -> generate -> result

    When *defenses* is empty the pipeline runs "naked" (vulnerable).

    Parameters
    ----------
    index:
        The VectorStoreIndex to retrieve from.
    defenses:
        Ordered list of defense middlewares.  Each must implement the
        :class:`Defense` protocol.
    top_k:
        Number of chunks to retrieve.  Defaults to settings value.
    llm:
        Optional pre-configured Ollama instance.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        defenses: list[Defense] | None = None,
        top_k: int | None = None,
        llm: Ollama | None = None,
    ) -> None:
        self._retriever = EnterpriseRetriever(index, top_k=top_k)
        self._defenses: list[Defense] = defenses or []
        self._llm = llm or get_llm()

    def query(
        self,
        query: str,
        user_context: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Run the full RAG pipeline and return a structured result.

        Parameters
        ----------
        query:
            The user's natural-language question.
        user_context:
            Information about the requesting user.  At minimum should
            contain ``{"role": "employee", "department": "Engineering"}``.
        filters:
            Optional metadata filters passed to the retriever (e.g.
            ``{"access_level": "public"}``).

        Returns
        -------
        dict
            Keys:
            - ``response`` (str): The generated answer.
            - ``retrieved_nodes`` (list[NodeWithScore]): Nodes after
              defense filtering.
            - ``raw_nodes`` (list[NodeWithScore]): Nodes before defense
              filtering.
            - ``defense_logs`` (list[dict]): Per-defense log entries.
            - ``timing`` (dict): Timing breakdown in seconds.
        """
        if user_context is None:
            user_context = {"role": "employee", "department": "Engineering"}

        timing: dict[str, float] = {}
        defense_logs: list[dict[str, Any]] = []

        # --- Retrieve ---
        t0 = time.perf_counter()
        nodes = self._retriever.retrieve(query, filters=filters, top_k=top_k)
        timing["retrieval_s"] = time.perf_counter() - t0

        raw_nodes = list(nodes)  # snapshot before defenses

        # --- Apply defenses ---
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

        # --- Generate ---
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
