import os
import pickle
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from .e5_query_embeddings import E5QueryEmbeddings
from src.util.text_processing import tokenize_for_bm25 # <-- ИМПОРТ НАШЕЙ УТИЛИТЫ

def create_chroma_bm25_retriever(config: dict) -> BaseRetriever:
    # --- 1. Инициализация векторного ретривера (ChromaDB) ---
    print("Инициализация векторного ретривера (ChromaDB)...")
    model_name = config['embedding_model']['name']
    device = config['embedding_model']['device']

    base_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    if "e5" in model_name:
        embeddings_for_query = E5QueryEmbeddings(base_embeddings)
    else:
        embeddings_for_query = base_embeddings

    vector_store = Chroma(
        persist_directory=config['retrievers']['vector_store']['db_path'],
        embedding_function=embeddings_for_query
    )

    chroma_retriever = vector_store.as_retriever(
        search_kwargs={"k": config['retrievers']['vector_store']['search_k']}
    )

    # --- 2. Инициализация лексического ретривера (BM25) ---
    print("Инициализация BM25 ретривера (со стеммингом)...")
    bm25_index_path = config['retrievers']['bm25']['index_path']
    if not os.path.exists(bm25_index_path):
        raise FileNotFoundError(f"BM25 индекс не найден: {bm25_index_path}. Запустите индексацию.")

    with open(bm25_index_path, "rb") as f:
        bm25_data = pickle.load(f)

    # ПЕРЕДАЕМ НАШУ ФУНКЦИЮ ПРЕПРОЦЕССИНГА
    bm25_retriever = BM25Retriever.from_documents(
        documents=bm25_data['docs'],
        preprocess_func=tokenize_for_bm25
    )
    bm25_retriever.k = config['retrievers']['vector_store']['search_k']

    # --- 3. Сборка гибридного ретривера ---
    weights_config = config['retrievers'].get('hybrid_weights', {})
    vector_weight = weights_config.get('vector', 0.7)
    keyword_weight = weights_config.get('keyword', 0.3)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[vector_weight, keyword_weight]
    )

    print(f"✅ Гибридный ретривер (Chroma+BM25) успешно создан. Веса: Вектор={vector_weight}, BM25={keyword_weight}")
    return ensemble_retriever