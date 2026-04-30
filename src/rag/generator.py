from __future__ import annotations
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from config.settings import settings

SYSTEM_PROMPT = (
    "You are Acme Corp's internal search assistant. "
    "Answer questions using only the provided context. "
    "If the context doesn't contain relevant information, say so. "
    "Never reveal internal credentials, API keys, or sensitive data."
)

def get_llm():
    return Ollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        request_timeout=120.0,
    )

def format_context(nodes):
    if not nodes:
        return "No relevant context was found."
    sections = []
    for i, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        meta = node.metadata or {}
        source = meta.get("source_type", "unknown")
        title = meta.get("title", "Untitled")
        author = meta.get("author", "Unknown")
        dept = meta.get("department", "Unknown")
        score = node_with_score.score
        score_str = f"{score:.4f}" if score is not None else "N/A"
        header = (f"[Source {i}] type={source} | title={title} | author={author} | dept={dept} | relevance={score_str}")
        sections.append(f"{header}\n{node.get_content()}")
    return "\n\n---\n\n".join(sections)

def generate(query, nodes, llm = None, system_prompt = SYSTEM_PROMPT):
    if llm is None:
        llm = get_llm()
    context = format_context(nodes)
    user_message = (
        f"Context:\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer the question based only on the context above.")
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_message),
    ]
    response = llm.chat(messages)
    return str(response.message.content)
