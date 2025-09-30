# src/pipelines/indexing.py

import os
import pickle
import requests
import textwrap 
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.strategies.chunkers import HTMLContextChunker

def run_indexing(config: dict):
    """Полный пайплайн индексации: загрузка, чанкинг, создание двух индексов и сохранение чанков в txt."""
    
    # 1. Загрузка и парсинг
    url = config['data_source']['url']
    print(f"Загрузка данных со страницы: {url}")
    response = requests.get(url, verify=False)
    raw_document = Document(page_content=response.text, metadata={"source": url})

    # 2. Гибридный чанкинг
    chunker = HTMLContextChunker()
    chunks = chunker.chunk(raw_document)
    print(f"✅ Документ разделен на {len(chunks)} структурных чанков.")

    if not chunks:
        print("Нет чанков для индексации.")
        return

    output_dir = config['paths']['output_dir']
    output_filename = config['paths']['indexing']
    output_file_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(f"Исходный документ: {url}\n")
        f.write(f"Всего чанков: {len(chunks)}\n")
        f.write("="*80 + "\n\n")

        for i, chunk in enumerate(chunks):
            f.write(f"--- ЧАНК #{i+1} ---\n")
            f.write(f"Метаданные: {chunk.metadata}\n")
            f.write("-" * 20 + "\n")
            wrapped_text = textwrap.fill(chunk.page_content, width=100)
            f.write(wrapped_text)
            f.write("\n\n" + "="*80 + "\n\n")
            
    print(f"✅ Все чанки сохранены в файл для анализа: {output_file_path}")

    # 3. Создание и сохранение BM25 индекса
    tokenized_corpus = [doc.page_content.split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_index_path = config['retrievers']['bm25']['index_path']
    os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)
    with open(bm25_index_path, "wb") as f:
        pickle.dump({'bm25': bm25, 'docs': chunks}, f)
    print(f"✅ BM25 индекс сохранен в {bm25_index_path}")

    # 4. Создание и сохранение векторного индекса
    db_dir = config['retrievers']['vector_store']['db_path']
    model_name = config['embedding_model']['name']
    device = config['embedding_model']['device']
    
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})
    db = Chroma.from_documents(chunks, embeddings_model, persist_directory=db_dir)
    print(f"✅ Векторная база данных сохранена в {db_dir}")

    print("\n🎉 Индексация успешно завершена!")