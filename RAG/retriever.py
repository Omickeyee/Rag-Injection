from __future__ import annotations
from typing import Any
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from settings import settings

class EnterpriseRetriever:
    def __init__(self, index, top_k = None):
        self._index = index
        self._top_k = top_k or settings.top_k

    def retrieve(self, query, filters = None, top_k = None):
        metadata_filters = None
        if filters:
            filter_list = [
                MetadataFilter(key=key, value=value, operator=FilterOperator.EQ)
                for key, value in filters.items()
            ]
            metadata_filters = MetadataFilters(filters=filter_list)
        retriever = self._index.as_retriever(similarity_top_k=top_k or self._top_k, filters=metadata_filters,)
        return retriever.retrieve(query)
