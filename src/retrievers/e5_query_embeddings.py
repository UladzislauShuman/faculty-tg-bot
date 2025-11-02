from typing import List
from langchain_core.embeddings import Embeddings

class E5QueryEmbeddings(Embeddings):
    """
    Класс-обертка для моделей e5, который автоматически добавляет
    префикс 'query: ' к каждому запросу перед векторизацией.
    Это необходимо для правильной работы моделей семейства E5.
    """
    def __init__(self, base_embeddings: Embeddings):
        self.base_embeddings = base_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Документы векторизуются без изменений (но мы добавляем префикс "passage:" на этапе индексации)
        return self.base_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        # К поисковому запросу добавляется специальный префикс
        return self.base_embeddings.embed_query(f"query: {text}")