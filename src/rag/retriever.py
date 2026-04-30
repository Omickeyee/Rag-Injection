"""Custom retriever with optional metadata filtering."""

from __future__ import annotations

from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

from config.settings import settings


class EnterpriseRetriever:
    """Wraps a VectorStoreIndex retriever with metadata filtering support.

    Parameters
    ----------
    index:
        The underlying vector store index to query.
    top_k:
        Number of results to retrieve.  Defaults to ``settings.top_k``.
    """

    def __init__(self, index: VectorStoreIndex, top_k: int | None = None) -> None:
        self._index = index
        self._top_k = top_k or settings.top_k

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[NodeWithScore]:
        """Retrieve the most relevant nodes for a query.

        Parameters
        ----------
        query:
            The user's natural-language question.
        filters:
            Optional metadata filters as ``{field: value}`` pairs.
            Supported fields include ``access_level``, ``source_type``,
            ``department``, etc.  Each filter is applied as an equality
            match.

        Returns
        -------
        list[NodeWithScore]
            Ranked list of retrieved nodes with similarity scores and
            full metadata.
        """
        metadata_filters = None
        if filters:
            filter_list = [
                MetadataFilter(key=key, value=value, operator=FilterOperator.EQ)
                for key, value in filters.items()
            ]
            metadata_filters = MetadataFilters(filters=filter_list)

        retriever = self._index.as_retriever(
            similarity_top_k=top_k or self._top_k,
            filters=metadata_filters,
        )

        return retriever.retrieve(query)
