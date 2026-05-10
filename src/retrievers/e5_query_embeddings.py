"""Обёртка Embeddings для E5: к запросу добавляется префикс «query: ».

Префикс «passage: » для индексации задаётся в pipelines/indexing.
"""
from typing import List

from langchain_core.embeddings import Embeddings


class E5QueryEmbeddings(Embeddings):
    """Делегирует в base; для `embed_query` добавляет префикс E5."""
    def __init__(self, base_embeddings: Embeddings):
        self.base_embeddings = base_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Документы векторизуются без изменений и мы добавляем префикс "passage:" на этапе индексации
        return self.base_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        # К поисковому запросу добавляется префикс
        return self.base_embeddings.embed_query(f"query: {text}")