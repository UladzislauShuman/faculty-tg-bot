"""Qdrant hybrid retriever: dense HuggingFace + sparse FastEmbed (BM25-like).

Имена векторов dense/sparse должны совпадать с индексацией в pipelines/indexing.
"""
import logging
import os
from typing import Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from src.util.hf_embeddings import huggingface_embedding_model_kwargs

from .e5_query_embeddings import E5QueryEmbeddings
from .hyde_retriever import HyDEQueryEmbeddings

logger = logging.getLogger(__name__)


def create_qdrant_retriever(
  config: dict,
  hyde_llm: Optional[BaseLanguageModel] = None,
) -> BaseRetriever:
  """Hybrid Qdrant: dense query embeddings, sparse по исходному тексту запроса.

  Шаги: embeddings → опционально HyDE → QdrantVectorStore HYBRID → as_retriever(k).
  """
  logger.info("Qdrant hybrid: инициализация")
  qdrant_config = config['retrievers']['qdrant']

  # 1. Dense Embeddings (E5)
  emb_cfg = config['embedding_model']
  model_name = emb_cfg['name']
  base_embeddings = HuggingFaceEmbeddings(
      model_name=model_name,
      model_kwargs=huggingface_embedding_model_kwargs(emb_cfg),
      encode_kwargs={'normalize_embeddings': True}
  )
  embeddings_for_query = E5QueryEmbeddings(
    base_embeddings) if "e5" in model_name.lower() else base_embeddings

  hyde_cfg = config.get("hyde") or {}
  if hyde_llm is not None:
    logger.info("HyDE включён для dense Qdrant")
    embeddings_for_query = HyDEQueryEmbeddings(
      embeddings_for_query,
      hyde_llm,
      timeout_seconds=float(hyde_cfg.get("timeout_seconds", 8)),
      num_hypotheses=int(hyde_cfg.get("num_hypotheses", 1)),
      verbose_console=bool(hyde_cfg.get("verbose_console", False)),
    )

  # 2. Sparse Embeddings (BM25-like) через FastEmbed
  sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

  # 3. Подключение
  host = os.getenv("QDRANT_HOST", qdrant_config.get('host', 'localhost'))
  port = qdrant_config.get('port', 6333)

  client = QdrantClient(host=host, port=port)
  logger.info(
      "Qdrant клиент: %s:%s коллекция=%s",
      host,
      port,
      qdrant_config['collection_name'],
  )

  # 4. Инициализация Hybrid Store
  # vector_name/sparse_vector_name должны совпадать с именами, использованными при индексации
  qdrant_store = QdrantVectorStore(
      client=client,
      collection_name=qdrant_config['collection_name'],
      embedding=embeddings_for_query,
      sparse_embedding=sparse_embeddings,
      retrieval_mode=RetrievalMode.HYBRID,
      vector_name="dense",
      sparse_vector_name="sparse",
  )

  return qdrant_store.as_retriever(
      search_kwargs={"k": qdrant_config.get('search_k', 10)}
  )