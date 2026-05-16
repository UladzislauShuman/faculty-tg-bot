"""Обёртка Embeddings для E5: к запросу добавляется префикс «query: ».

Префикс «passage: » для индексации задаётся в pipelines/indexing.
"""
from typing import List

from langchain_core.embeddings import Embeddings


class E5QueryEmbeddings(Embeddings):
    """Делегирует в base; для `embed_query` добавляет префикс E5, для `embed_documents` — passage."""
    def __init__(self, base_embeddings: Embeddings):
        self.base_embeddings = base_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Добавляем префикс "passage:" перед отправкой в модель (текст в БД остается чистым)
        prefixed_texts = [f"passage: {t}" for t in texts]
        return self.base_embeddings.embed_documents(prefixed_texts)

    def embed_query(self, text: str) -> List[float]:
        # К поисковому запросу добавляется префикс
        return self.base_embeddings.embed_query(f"query: {text}")