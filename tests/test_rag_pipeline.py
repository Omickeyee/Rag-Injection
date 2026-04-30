"""Tests for the RAG pipeline (mock LLM, embeddings, and vector store).

The pipeline module imports from llama_index and ollama, which may not be
installed.  The conftest stubs guarantee that llama_index.core.schema is
available with real TextNode / NodeWithScore classes, while the remaining
llama_index subpackages are MagicMocks.

We also need to ensure ``src.rag.pipeline`` can be imported without pulling
in chromadb (via ``src.rag.__init__``).  We do this by inserting a lightweight
``src.rag`` package stub in sys.modules *before* importing the pipeline.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.conftest import make_node

# ---------------------------------------------------------------------------
# Ensure src.rag can be imported without triggering its __init__.py
# (which would pull in chromadb, llama_index vector stores, etc.)
# ---------------------------------------------------------------------------

_rag_dir = str(Path(__file__).resolve().parent.parent / "src" / "rag")

if "src.rag" not in sys.modules:
    _pkg = types.ModuleType("src.rag")
    _pkg.__path__ = [_rag_dir]  # type: ignore[attr-defined]
    _pkg.__package__ = "src.rag"
    sys.modules["src.rag"] = _pkg

# Stub modules that pipeline.py imports at the top level
for _mod_name in (
    "src.rag.retriever",
    "src.rag.generator",
    "src.rag.embeddings",
    "src.rag.vector_store",
    "src.rag.ingestion",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# Now we can safely import the pipeline module
import src.rag.pipeline as pipeline_mod  # noqa: E402

RAGPipeline = pipeline_mod.RAGPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_nodes(n: int = 3):
    """Return a list of mock NodeWithScore objects."""
    return [
        make_node(
            f"chunk text {i}",
            metadata={"source_type": "confluence"},
            score=0.9 - i * 0.1,
        )
        for i in range(n)
    ]


class FakeDefense:
    """A simple defense that records calls and passes nodes through."""

    def __init__(self, name: str = "FakeDefense"):
        self._name = name
        self.calls: list[dict] = []

    def apply(self, nodes, query, user_context):
        self.calls.append({
            "nodes_count": len(nodes),
            "query": query,
            "user_context": user_context,
        })
        return nodes


class FilteringDefense:
    """A defense that removes the first node."""

    def apply(self, nodes, query, user_context):
        return nodes[1:] if len(nodes) > 1 else nodes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    """Tests for the RAGPipeline orchestrator.

    All tests patch the retriever, generator, and LLM so that no external
    services are needed.
    """

    @patch.object(pipeline_mod, "generate", return_value="mocked answer")
    @patch.object(pipeline_mod, "get_llm", return_value=MagicMock())
    @patch.object(pipeline_mod, "EnterpriseRetriever")
    def test_pipeline_query_flow(self, MockRetriever, mock_get_llm, mock_generate):
        """Verify pipeline calls retrieve -> defenses -> generate in order."""
        mock_index = MagicMock()
        mock_retriever_instance = MockRetriever.return_value
        mock_retriever_instance.retrieve.return_value = _mock_nodes(3)

        pipeline = RAGPipeline(index=mock_index, defenses=[])
        result = pipeline.query("test question")

        mock_retriever_instance.retrieve.assert_called_once()
        mock_generate.assert_called_once()
        assert result["response"] == "mocked answer"

    @patch.object(pipeline_mod, "generate", return_value="answer")
    @patch.object(pipeline_mod, "get_llm", return_value=MagicMock())
    @patch.object(pipeline_mod, "EnterpriseRetriever")
    def test_pipeline_applies_defenses_in_order(self, MockRetriever, mock_get_llm, mock_generate):
        """Create 2 mock defenses and verify both are called sequentially."""
        mock_index = MagicMock()
        MockRetriever.return_value.retrieve.return_value = _mock_nodes(3)

        defense_a = FakeDefense("A")
        defense_b = FakeDefense("B")

        pipeline = RAGPipeline(index=mock_index, defenses=[defense_a, defense_b])
        result = pipeline.query("test", user_context={"role": "employee", "department": "Eng"})

        assert len(defense_a.calls) == 1
        assert len(defense_b.calls) == 1
        assert defense_a.calls[0]["query"] == "test"
        assert defense_b.calls[0]["query"] == "test"

        assert len(result["defense_logs"]) == 2
        assert result["defense_logs"][0]["defense"] == "FakeDefense"
        assert result["defense_logs"][1]["defense"] == "FakeDefense"

    @patch.object(pipeline_mod, "generate", return_value="response text")
    @patch.object(pipeline_mod, "get_llm", return_value=MagicMock())
    @patch.object(pipeline_mod, "EnterpriseRetriever")
    def test_pipeline_returns_structured_result(self, MockRetriever, mock_get_llm, mock_generate):
        """Verify result dict has all required keys."""
        mock_index = MagicMock()
        MockRetriever.return_value.retrieve.return_value = _mock_nodes(2)

        pipeline = RAGPipeline(index=mock_index)
        result = pipeline.query("what is X?")

        required_keys = {"response", "retrieved_nodes", "raw_nodes", "defense_logs", "timing"}
        assert required_keys.issubset(set(result.keys()))
        assert isinstance(result["timing"], dict)
        assert "retrieval_s" in result["timing"]
        assert "generation_s" in result["timing"]
        assert "total_s" in result["timing"]

    @patch.object(pipeline_mod, "generate", return_value="naked answer")
    @patch.object(pipeline_mod, "get_llm", return_value=MagicMock())
    @patch.object(pipeline_mod, "EnterpriseRetriever")
    def test_pipeline_no_defenses(self, MockRetriever, mock_get_llm, mock_generate):
        """Verify naked pipeline works -- no defenses, no defense logs."""
        mock_index = MagicMock()
        nodes = _mock_nodes(3)
        MockRetriever.return_value.retrieve.return_value = nodes

        pipeline = RAGPipeline(index=mock_index, defenses=[])
        result = pipeline.query("naked query")

        assert result["defense_logs"] == []
        assert result["response"] == "naked answer"
        assert len(result["raw_nodes"]) == len(result["retrieved_nodes"])

    @patch.object(pipeline_mod, "generate", return_value="answer")
    @patch.object(pipeline_mod, "get_llm", return_value=MagicMock())
    @patch.object(pipeline_mod, "EnterpriseRetriever")
    def test_load_corpus_creates_documents(self, MockRetriever, mock_get_llm, mock_generate):
        """Mock a corpus and verify Document-like objects flow through the pipeline."""
        mock_index = MagicMock()
        MockRetriever.return_value.retrieve.return_value = _mock_nodes(2)

        pipeline = RAGPipeline(index=mock_index)
        result = pipeline.query("test")

        assert len(result["retrieved_nodes"]) == 2
        for nws in result["retrieved_nodes"]:
            assert hasattr(nws, "node")
            assert hasattr(nws.node, "get_content")
            assert isinstance(nws.node.get_content(), str)
