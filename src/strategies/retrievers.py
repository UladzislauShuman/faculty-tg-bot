import os
import pickle
from typing import Dict, Any, List
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings


class E5QueryEmbeddings(Embeddings):
  """
  Класс-обертка для моделей e5, который автоматически добавляет
  префикс 'query: ' к каждому запросу перед векторизацией.
  """

  def __init__(self, base_embeddings: Embeddings):
    self.base_embeddings = base_embeddings

  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    return self.base_embeddings.embed_documents(texts)

  def embed_query(self, text: str) -> List[float]:
    return self.base_embeddings.embed_query(f"query: {text}")

class HybridRetrieverFactory:
  """
  Этот класс инкапсулирует логику создания двух разных типов ретриверов
  (векторного и по ключевым словам) и их объединения в одно целое
  """

  def __init__(self, config: Dict[str, Any]):
    """
    Args:
        config: Словарь с конфигурацией, загруженный из config.yaml.
    """
    self.config = config

  def _create_vector_retriever(self) -> BaseRetriever:
    """
    Создает ретривер для векторного поиска.
    """
    print("Инициализация векторного ретривера (ChromaDB)...")

    # --- ML-компоненты ---
    # 1. HuggingFaceEmbeddings: Класс для создания векторов (эмбеддингов) из текста.
    #    'model_name' - указывает, какую модель с Hugging Face использовать.
    #    'model_kwargs' - позволяет передать параметры для модели, например,
    #                     на каком устройстве ее запускать ('cpu' или 'cuda').
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
    """
    Создает и возвращает ретривер для BM25.
    """
    print("Инициализация BM25 ретривера...")

    bm25_index_path = self.config['retrievers']['bm25']['index_path']
    if not os.path.exists(bm25_index_path):
      raise FileNotFoundError(
        f"BM25 индекс не найден: {bm25_index_path}. Запустите индексацию.")

    # Загружаем сохраненный индекс и документы
    with open(bm25_index_path, "rb") as f:
      bm25_data = pickle.load(f)

    # --- ML-компоненты ---
    # 1. BM25Retriever: Класс-обертка в LangChain для работы с индексом BM25.
    #    .from_documents() - удобный метод для его создания из списка документов.
    #                       Под капотом он выполняет токенизацию и индексацию.
    bm25_retriever = BM25Retriever.from_documents(
        documents=bm25_data['docs']
    )
    # Устанавливаем количество возвращаемых документов.
    bm25_retriever.k = self.config['retrievers']['vector_store']['search_k']

    return bm25_retriever

  def create(self) -> BaseRetriever:
    """
    Основной метод, который создает и объединяет ретриверы в гибридный ансамбль.
    """
    chroma_retriever = self._create_vector_retriever()
    bm25_retriever = self._create_bm25_retriever()

    # --- ML-компоненты ---
    # EnsembleRetriever: Специальный ретривер в LangChain, который объединяет
    # результаты нескольких других ретриверов.
    # 'retrievers' - список ретриверов для использования.
    # 'weights' - список весов, определяющих "важность" каждого ретривера.
    #            [0.5, 0.5] означает, что оба поиска равноправны.
    #            [0.7, 0.3] отдаст предпочтение векторному поиску.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    print("✅ Гибридный ретривер (Ensemble) успешно создан.")
    return ensemble_retriever


def create_hybrid_retriever(config: dict) -> BaseRetriever:
  """
  Создает экземпляр фабрики и возвращает готовый гибридный ретривер.
  """
  factory = HybridRetrieverFactory(config)
  return factory.create()