import os
import pickle
import requests
import textwrap
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# BM25Okapi - это классический алгоритм поиска по ключевым словам.
# Он создает индекс, который позволяет быстро находить документы,
# содержащие те же слова, что и в запросе.
from rank_bm25 import BM25Okapi
from src.strategies.chunkers import HTMLContextChunker


def _load_raw_document(url: str) -> Document:
  """
  Загружает HTML-содержимое по URL
  и возвращает его в виде единого объекта Document.
  """
  print(f"Загрузка данных со страницы: {url}")
  try:
    response = requests.get(url, verify=False, timeout=10)
    response.raise_for_status()
    return Document(page_content=response.text, metadata={"source": url})
  except requests.RequestException as e:
    print(f"❌ Ошибка при загрузке URL {url}: {e}")
    return None


def _save_chunks_to_file(chunks: List[Document], url: str,
    config: Dict[str, Any]):
  """
  Сохраняет отформатированный список чанков в текстовый файл для отладки.
  """
  output_dir = config['paths']['output_dir']
  output_filename = config['paths']['indexing']
  output_file_path = os.path.join(output_dir, output_filename)

  os.makedirs(output_dir, exist_ok=True)

  with open(output_file_path, "w", encoding="utf-8") as f:
    f.write(f"Исходный документ: {url}\n")
    f.write(f"Всего чанков: {len(chunks)}\n")
    f.write("=" * 80 + "\n\n")

    for i, chunk in enumerate(chunks):
      f.write(f"--- ЧАНК #{i + 1} ---\n")
      f.write(f"Метаданные: {chunk.metadata}\n")
      f.write("-" * 20 + "\n")
      wrapped_text = textwrap.fill(chunk.page_content, width=100)
      f.write(wrapped_text)
      f.write("\n\n" + "=" * 80 + "\n\n")

  print(f"✅ Все чанки сохранены в файл для анализа: {output_file_path}")


def _create_bm25_index(chunks: List[Document], config: Dict[str, Any]):
  """
  Создает и сохраняет на диск индекс BM25 для поиска по ключевым словам.
  """
  print("Создание BM25 индекса...")
  # 1. Токенизация: разбиваем текст каждого чанка на список слов.
  tokenized_corpus = [doc.page_content.split() for doc in chunks]

  # 2. Обучение индекса: создаем объект BM25Okapi, передавая ему токенизированный корпус.
  bm25 = BM25Okapi(tokenized_corpus)

  # 3. Сохранение: мы сохраняем как сам обученный объект bm25, так и
  #    связанные с ним документы (chunks), чтобы ретривер мог их вернуть.
  bm25_index_path = config['retrievers']['bm25']['index_path']
  os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)
  with open(bm25_index_path, "wb") as f:
    pickle.dump({'bm25': bm25, 'docs': chunks}, f)

  print(f"✅ BM25 индекс сохранен в {bm25_index_path}")


def _create_vector_store(chunks: List[Document], config: Dict[str, Any]):
  """
  Создает и сохраняет на диск векторную базу данных (ChromaDB).
  """
  print("Создание векторной базы данных...")
  db_dir = config['retrievers']['vector_store']['db_path']
  model_name = config['embedding_model']['name']
  device = config['embedding_model']['device']

  # 1. Инициализация embedding-модели.
  # HuggingFaceEmbeddings - это класс-обертка в LangChain, который
  # упрощает использование любой embedding-модели с сайта Hugging Face.
  embeddings_model = HuggingFaceEmbeddings(model_name=model_name,
                                           model_kwargs={'device': device})

  # 2. Создание и сохранение базы.
  # Chroma.from_documents - это удобный метод, который делает все за нас:
  # - Проходит по каждому чанку.
  # - Вызывает embeddings_model для получения вектора.
  # - Сохраняет текст, метаданные и вектор в базу на диске (persist_directory).
  db = Chroma.from_documents(chunks, embeddings_model, persist_directory=db_dir)

  print(f"✅ Векторная база данных сохранена в {db_dir}")


def run_indexing(config: Dict[str, Any]):
  """
  Главная функция-оркестратор для полного пайплайна индексации.
  """
  # Загрузка "сырого" документа
  url = config['data_source']['url']
  raw_document = _load_raw_document(url)
  if not raw_document:
    return

  # Разделение документа на чанки
  chunker = HTMLContextChunker()
  chunks = chunker.chunk(raw_document)
  print(f"✅ Документ разделен на {len(chunks)} структурных чанков.")
  if not chunks:
    print("Нет чанков для индексации.")
    return

  # Сохраняем чанки в файл
  _save_chunks_to_file(chunks, url, config)

  # Создание индекса для поиска по ключевым словам
  _create_bm25_index(chunks, config)

  # Создание индекса для семантического поиска
  _create_vector_store(chunks, config)

  print("\n🎉 Индексация успешно завершена!")