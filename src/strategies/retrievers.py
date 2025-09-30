import os
import pickle
from rank_bm25 import BM25Okapi
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever

# Примечание: Эта реализация показывает, как собрать EnsembleRetriever.
# Re-ranker и Query Expansion будут добавлены поверх него в RAG-пайплайне.

def create_hybrid_retriever(config: dict):
    db_dir = config['retrievers']['vector_store']['db_path']
    model_name = config['embedding_model']['name']
    bm25_index_path = config['retrievers']['bm25']['index_path']
    
    # 1. Инициализация векторного ретривера
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    vector_store = Chroma(persist_directory=db_dir, embedding_function=embeddings_model)
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 2. Инициализация BM25 ретривера
    if not os.path.exists(bm25_index_path):
        raise FileNotFoundError(f"BM25 индекс не найден по пути {bm25_index_path}. Запустите индексацию.")
        
    with open(bm25_index_path, "rb") as f:
        bm25_data = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(
        documents=bm25_data['docs']
    )
    bm25_retriever.k = 5

    # 3. Создание ансамбля
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.5, 0.5] # Можно играть с весами
    )
    
    print("✅ Гибридный ретривер (Ensemble) успешно создан.")
    return ensemble_retriever