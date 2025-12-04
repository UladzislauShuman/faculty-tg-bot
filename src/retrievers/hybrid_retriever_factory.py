import os
import pickle
from typing import Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from .e5_query_embeddings import E5QueryEmbeddings


class HybridRetrieverFactory:
  """
  Класс-фабрика, который инкапсулирует логику создания двух разных
  типов ретриверов (векторного и по ключевым словам) и их объединения
  в гибрид.
  """

  def __init__(self, config: Dict[str, Any]):
    self.config = config

  def _create_vector_retriever(self) -> BaseRetriever:
    """Создает ретривер для векторного поиска на основе ChromaDB."""
    print("Инициализация векторного ретривера (ChromaDB)...")

    model_name = self.config['embedding_model']['name']
    device = self.config['embedding_model']['device']

    base_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    if "e5" in model_name:
      print(
        "Обнаружена модель e5. Используем обертку для добавления префикса 'query: ' к запросам.")
      embeddings_for_query = E5QueryEmbeddings(base_embeddings)
    else:
      embeddings_for_query = base_embeddings

    vector_store = Chroma(
        persist_directory=self.config['retrievers']['vector_store']['db_path'],
        embedding_function=embeddings_for_query
    )

    return vector_store.as_retriever(
        search_kwargs={
          "k": self.config['retrievers']['vector_store']['search_k']}
    )

  def _create_bm25_retriever(self) -> BaseRetriever:
    """Загрузка BM25 индекса из файла."""
    print("Инициализация BM25 ретривера...")
    bm25_index_path = self.config['retrievers']['bm25']['index_path']
    if not os.path.exists(bm25_index_path):
      raise FileNotFoundError(
        f"BM25 индекс не найден: {bm25_index_path}. Запустите индексацию.")

    with open(bm25_index_path, "rb") as f:
      bm25_data = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(documents=bm25_data['docs'])
    bm25_retriever.k = self.config['retrievers']['vector_store']['search_k']
    return bm25_retriever

  def create(self) -> BaseRetriever:
    """
    Основной метод, который создает и объединяет ретриверы
    """
    chroma_retriever = self._create_vector_retriever()
    bm25_retriever = self._create_bm25_retriever()

    # Получаем веса из конфига с дефолтными значениями
    weights_config = self.config['retrievers'].get('hybrid_weights', {})
    vector_weight = weights_config.get('vector', 0.7)
    keyword_weight = weights_config.get('keyword', 0.3)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[vector_weight, vector_weight]
    )

    print("✅ Гибридный ретривер (Ensemble) успешно создан.")
    return ensemble_retriever


def create_hybrid_retriever(config: dict) -> BaseRetriever:
  """
  Функция-обертка для удобного вызова фабрики.
  """
  factory = HybridRetrieverFactory(config)
  return factory.create()