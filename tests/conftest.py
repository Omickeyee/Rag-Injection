"""Shared test fixtures and helpers.

All tests mock llama_index and external dependencies so they can run
without Ollama, ChromaDB, or a GPU.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Stub out llama_index imports before any src modules are loaded
# ---------------------------------------------------------------------------

def _ensure_llama_index_stubs() -> None:
    """Insert lightweight stubs for llama_index packages if not installed."""
    modules_to_stub = [
        "llama_index",
        "llama_index.core",
        "llama_index.core.schema",
        "llama_index.core.node_parser",
        "llama_index.llms",
        "llama_index.llms.ollama",
        "llama_index.embeddings",
        "llama_index.embeddings.huggingface",
        "llama_index.vector_stores",
        "llama_index.vector_stores.chroma",
    ]
    for mod_name in modules_to_stub:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    # Provide real-ish TextNode / NodeWithScore so tests can construct them
    schema_mod = sys.modules.get("llama_index.core.schema")
    if isinstance(schema_mod, MagicMock):
        # Replace with a tiny module that has concrete classes
        real_schema = ModuleType("llama_index.core.schema")

        class TextNode:
            """Minimal stand-in for llama_index TextNode."""

            def __init__(self, text: str = "", metadata: dict | None = None, **kwargs):
                self.text = text
                self.metadata = metadata if metadata is not None else {}
                self.id_ = kwargs.get("id_", "")

            def get_content(self) -> str:
                return self.text

            def set_content(self, text: str) -> None:
                self.text = text

        class NodeWithScore:
            """Minimal stand-in for llama_index NodeWithScore."""

            def __init__(self, node: TextNode | None = None, score: float = 0.0):
                self.node = node or TextNode()
                self.score = score

        real_schema.TextNode = TextNode
        real_schema.NodeWithScore = NodeWithScore
        sys.modules["llama_index.core.schema"] = real_schema
        # Also patch on the core mock so `from llama_index.core.schema import ...` works
        core_mod = sys.modules.get("llama_index.core")
        if core_mod is not None:
            core_mod.schema = real_schema


_ensure_llama_index_stubs()


# ---------------------------------------------------------------------------
# Re-usable helper
# ---------------------------------------------------------------------------

def make_node(text: str, metadata: dict | None = None, score: float = 0.8):
    """Create a NodeWithScore with the given text, metadata, and score."""
    from llama_index.core.schema import NodeWithScore, TextNode

    node = TextNode(text=text, metadata=metadata or {})
    return NodeWithScore(node=node, score=score)


@pytest.fixture
def node_factory():
    """Fixture that returns the make_node helper."""
    return make_node
