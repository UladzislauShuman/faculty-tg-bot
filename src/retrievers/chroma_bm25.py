"""Гибридный ретривер: Chroma (dense) + BM25 (sparse), RRF через AsyncEnsembleRetriever.

Опционально HyDE оборачивает только dense-эмбеддинги запроса; BM25 идёт по исходному тексту.
"""
import logging
import os
import pickle
from typing import Optional

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from src.retrievers.async_ensemble_retriever import AsyncEnsembleRetriever
from src.util.hf_embeddings import huggingface_embedding_model_kwargs
from src.util.text_processing import tokenize_for_bm25

from .e5_query_embeddings import E5QueryEmbeddings
from .hyde_retriever import HyDEQueryEmbeddings

logger = logging.getLogger(__name__)


def create_chroma_bm25_retriever(
    config: dict,
    hyde_llm: Optional[BaseLanguageModel] = None,
) -> BaseRetriever:
    """Собирает ensemble Chroma + BM25.

    Шаги: (1) HF embeddings + E5 query prefix + опционально HyDE;
    (2) Chroma retriever с k из vector_store;
    (3) BM25 из pickle, тот же k;
    (4) Ensemble с весами hybrid_weights.
    """
    logger.info("Chroma+BM25: инициализация dense (Chroma)")
    emb_cfg = config['embedding_model']
    model_name = emb_cfg['name']

    base_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=huggingface_embedding_model_kwargs(emb_cfg),
        encode_kwargs={'normalize_embeddings': True}
    )

    if "e5" in model_name:
        embeddings_for_query = E5QueryEmbeddings(base_embeddings)
    else:
        embeddings_for_query = base_embeddings

    hyde_cfg = config.get("hyde") or {}
    if hyde_llm is not None:
        logger.info("HyDE включён для dense Chroma")
        embeddings_for_query = HyDEQueryEmbeddings(
            embeddings_for_query,
            hyde_llm,
            timeout_seconds=float(hyde_cfg.get("timeout_seconds", 8)),
            num_hypotheses=int(hyde_cfg.get("num_hypotheses", 1)),
            verbose_console=bool(hyde_cfg.get("verbose_console", False)),
        )

    vector_store = Chroma(
        persist_directory=config['retrievers']['vector_store']['db_path'],
        embedding_function=embeddings_for_query
    )

    chroma_retriever = vector_store.as_retriever(
        search_kwargs={"k": config['retrievers']['vector_store']['search_k']}
    )

    logger.info("Chroma+BM25: загрузка BM25 pickle")
    bm25_index_path = config['retrievers']['bm25']['index_path']
    if not os.path.exists(bm25_index_path):
        raise FileNotFoundError(
            f"BM25 индекс не найден: {bm25_index_path}. Запустите индексацию."
        )

    with open(bm25_index_path, "rb") as f:
        bm25_data = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(
        documents=bm25_data['docs'],
        preprocess_func=tokenize_for_bm25
    )
    bm25_retriever.k = config['retrievers']['vector_store']['search_k']

    weights_config = config['retrievers'].get('hybrid_weights', {})
    vector_weight = weights_config.get('vector', 0.7)
    keyword_weight = weights_config.get('keyword', 0.3)

    ensemble_retriever = AsyncEnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[vector_weight, keyword_weight]
    )

    logger.info(
        "Chroma+BM25 готов: vector=%.2f keyword=%.2f k=%s",
        vector_weight,
        keyword_weight,
        config['retrievers']['vector_store']['search_k'],
    )
    return ensemble_retriever
