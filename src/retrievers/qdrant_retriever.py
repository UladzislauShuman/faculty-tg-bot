import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from .e5_query_embeddings import E5QueryEmbeddings


def create_qdrant_retriever(config: dict) -> BaseRetriever:
  print("Инициализация Hybrid Qdrant ретривера (Dense + Sparse)...")
  qdrant_config = config['retrievers']['qdrant']

  # 1. Dense Embeddings (E5)
  model_name = config['embedding_model']['name']
  device = config['embedding_model']['device']
  base_embeddings = HuggingFaceEmbeddings(
      model_name=model_name,
      model_kwargs={'device': device},
      encode_kwargs={'normalize_embeddings': True}
  )
  embeddings_for_query = E5QueryEmbeddings(
    base_embeddings) if "e5" in model_name else base_embeddings

  # 2. Sparse Embeddings (BM25-like) через FastEmbed
  sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

  # 3. Подключение
  host = os.getenv("QDRANT_HOST", qdrant_config.get('host', 'localhost'))
  port = qdrant_config.get('port', 6333)

  client = QdrantClient(host=host, port=port)

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