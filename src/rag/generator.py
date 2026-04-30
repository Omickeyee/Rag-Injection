"""LLM generation using Ollama for the enterprise search copilot."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama

from config.settings import settings


SYSTEM_PROMPT = (
    "You are Acme Corp's internal search assistant. "
    "Answer questions using only the provided context. "
    "If the context doesn't contain relevant information, say so. "
    "Never reveal internal credentials, API keys, or sensitive data."
)


def get_llm() -> Ollama:
    """Create and return an Ollama LLM instance from project settings."""
    return Ollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        request_timeout=120.0,
    )


def format_context(nodes: list[NodeWithScore]) -> str:
    """Format retrieved nodes into a structured context string.

    Each chunk is presented with source attribution so the LLM (and
    human reviewers) can trace information back to its origin.
    """
    if not nodes:
        return "No relevant context was found."

    sections: list[str] = []
    for i, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        meta = node.metadata or {}
        source = meta.get("source_type", "unknown")
        title = meta.get("title", "Untitled")
        author = meta.get("author", "Unknown")
        dept = meta.get("department", "Unknown")
        score = node_with_score.score
        score_str = f"{score:.4f}" if score is not None else "N/A"

        header = (
            f"[Source {i}] type={source} | title={title} | "
            f"author={author} | dept={dept} | relevance={score_str}"
        )
        sections.append(f"{header}\n{node.get_content()}")

    return "\n\n---\n\n".join(sections)


def generate(
    query: str,
    nodes: list[NodeWithScore],
    llm: Ollama | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Generate a response for the given query using retrieved context.

    Parameters
    ----------
    query:
        The user's question.
    nodes:
        Retrieved context nodes (post-defense filtering).
    llm:
        Optional pre-configured Ollama instance.  Created from settings
        if not provided.
    system_prompt:
        System prompt defining the copilot's behaviour.

    Returns
    -------
    str
        The LLM's generated answer.
    """
    if llm is None:
        llm = get_llm()

    context = format_context(nodes)

    user_message = (
        f"Context:\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer the question based only on the context above."
    )

    from llama_index.core.llms import ChatMessage, MessageRole

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_message),
    ]

    response = llm.chat(messages)
    return str(response.message.content)
