import os
import pickle
import textwrap
import hashlib
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

from src.interfaces.data_processor_interfaces import DataSourceProcessor
from src.pipelines.indexing.crawlers.website_crawler import WebsiteCrawler
from src.util.yaml_parser import TestSetLoader


# --- ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ ---

def _prepare_embeddings(config: Dict[str, Any]):
  """Инициализация модели эмбеддингов с общими параметрами."""
  model_name = config['embedding_model']['name']
  device = config['embedding_model']['device']
  return HuggingFaceEmbeddings(
      model_name=model_name,
      model_kwargs={'device': device},
      encode_kwargs={'normalize_embeddings': True}
  )


def _apply_e5_passage_prefix(chunks: List[Document], model_name: str) -> List[
  Document]:
  """
  Добавляет префикс 'passage: ' к контенту чанков, если используется модель E5.
  Это критически важно для качества векторного поиска.
  """
  if "e5" in model_name:
    return [
      Document(
          page_content=f"passage: {chunk.page_content}",
          metadata=chunk.metadata
      )
      for chunk in chunks
    ]
  return chunks


def _save_chunks_to_file(chunks: List[Document], config: Dict[str, Any]):
  """Сохраняет отформатированный список чанков в текстовый файл для отладки."""
  output_dir = config['paths']['output_dir']
  output_filename = config['paths']['indexing']
  output_file_path = os.path.join(output_dir, output_filename)
  os.makedirs(output_dir, exist_ok=True)

  with open(output_file_path, "w", encoding="utf-8") as f:
    f.write(f"Всего чанков со всех страниц: {len(chunks)}\n")
    f.write("=" * 80 + "\n\n")
    for i, chunk in enumerate(chunks):
      f.write(f"--- ЧАНК #{i + 1} ---\n")
      f.write(f"Метаданные: {chunk.metadata}\n")
      f.write("-" * 20 + "\n")
      wrapped_text = textwrap.fill(chunk.page_content, width=100)
      f.write(wrapped_text)
      f.write("\n\n" + "=" * 80 + "\n\n")
  print(f"✅ Все чанки сохранены в файл для анализа: {output_file_path}")


# --- РЕАЛИЗАЦИИ ИНДЕКСАЦИИ (STORAGE SPECIFIC) ---

def _index_chroma_bm25(chunks: List[Document], config: Dict[str, Any]):
  """Логика индексации для связки ChromaDB + BM25 (Pickle)."""

  # 1. Подготовка документов для BM25
  print("Сохранение документов для BM25 индекса...")
  bm25_index_path = config['retrievers']['bm25']['index_path']
  os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)
  with open(bm25_index_path, "wb") as f:
    # Сохраняем только документы. Индекс построится при инициализации ретривера.
    pickle.dump({'docs': chunks}, f)
  print(f"✅ Документы для BM25 сохранены в {bm25_index_path}")

  # 2. Индексация в ChromaDB
  print("Создание векторной базы данных ChromaDB...")
  db_dir = config['retrievers']['vector_store']['db_path']
  embeddings_model = _prepare_embeddings(config)

  # Применяем префиксы E5 перед векторизацией
  chunks_to_embed = _apply_e5_passage_prefix(chunks,
                                             config['embedding_model']['name'])

  Chroma.from_documents(
      documents=chunks_to_embed,
      embedding=embeddings_model,
      persist_directory=db_dir
  )
  print(f"✅ Векторная база данных Chroma сохранена в {db_dir}")

def _index_qdrant(chunks: List[Document], config: Dict[str, Any]):
  print("Создание Hybrid индекса в Qdrant (Dense + Sparse)...")
  qdrant_config = config['retrievers']['qdrant']
  host = os.getenv("QDRANT_HOST", qdrant_config.get('host', 'localhost'))
  port = qdrant_config.get('port', 6333)
  collection_name = qdrant_config['collection_name']

  embeddings_model = _prepare_embeddings(config)
  # Важно: для Sparse векторов тоже нужен препроцессинг (тот же, что мы делали для BM25)
  # Но FastEmbed/Qdrant сделают базовую токенизацию сами.

  sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
  chunks_to_embed = _apply_e5_passage_prefix(chunks,
                                             config['embedding_model']['name'])

  # Загружаем в Qdrant с поддержкой Hybrid векторов
  # vector_name/sparse_vector_name должны совпадать с именами в qdrant_retriever.py
  QdrantVectorStore.from_documents(
      chunks_to_embed,
      embeddings_model,
      sparse_embedding=sparse_embeddings,
      url=f"http://{host}:{port}",
      collection_name=collection_name,
      retrieval_mode=RetrievalMode.HYBRID,
      vector_name="dense",
      sparse_vector_name="sparse",
      force_recreate=True,
  )
  print(f"✅ Гибридная коллекция {collection_name} создана.")


# --- ОСНОВНОЙ ПАЙПЛАЙН ---

def run_indexing(config: dict, processor: DataSourceProcessor, mode: str):
  """
  Главная функция запуска процесса индексации.
  Оркестрирует сбор данных, чанкинг и выбор хранилища.
  """
  urls_to_process = []

  # 1. Определение списка URL
  if mode == 'test':
    print("--- Режим 'test': загрузка URL из qa-test-set.yaml ---")
    loader = TestSetLoader(config['paths']['qa_test_set'])
    urls_to_process = loader.get_test_urls()
    if not urls_to_process:
      print("❌ Ошибка: В qa-test-set.yaml нет активных URL для индексации.")
      return
  elif mode == 'full':
    print("--- Режим 'full': запуск полного обхода сайта ---")
    base_url = config['data_source']['url']
    max_depth = config['data_source'].get('max_depth', 2)
    crawler = WebsiteCrawler(base_url=base_url, max_depth=max_depth)
    urls_to_process = crawler.crawl()

  # 2. Сбор и чанкинг контента
  all_chunks = []
  for url in urls_to_process:
    chunks_from_url = processor.process(url)
    all_chunks.extend(chunks_from_url)

  if not all_chunks:
    print("⚠️ Нет чанков для индексации. Процесс остановлен.")
    return

  print(
    f"\n✅ Обработка источников завершена. Получено {len(all_chunks)} чанков.")

  # 3. Генерация уникальных ID (для дедупликации и отслеживания)
  print("Генерация уникальных ID для каждого чанка...")
  for doc in all_chunks:
    unique_string = f"{doc.metadata['source']}{doc.page_content}"
    chunk_id = hashlib.md5(unique_string.encode('utf-8')).hexdigest()
    doc.metadata['chunk_id'] = chunk_id

  # 4. Сохранение текстового дампа для отладки
  _save_chunks_to_file(all_chunks, config)

  # 5. Маршрутизация в выбранное хранилище
  active_retriever_type = config['retrievers'].get('active_type', 'chroma_bm25')

  if active_retriever_type == 'chroma_bm25':
    _index_chroma_bm25(all_chunks, config)
  elif active_retriever_type == 'qdrant':
    _index_qdrant(all_chunks, config)
  else:
    print(
      f"❌ Ошибка: Неизвестный тип ретривера в конфиге: {active_retriever_type}")
    return

  print("\n🎉 Индексация успешно завершена!")