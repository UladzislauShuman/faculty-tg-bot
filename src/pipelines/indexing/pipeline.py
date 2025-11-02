import os
import pickle
import textwrap
import hashlib
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from src.interfaces.data_processor_interfaces import DataSourceProcessor
from src.pipelines.indexing.crawlers.website_crawler import WebsiteCrawler


# --- БЛОК УТИЛИТАРНЫХ ФУНКЦИЙ ---
# Эти функции являются "помощниками". Они выполняют конкретные,
# атомарные задачи и не зависят от бизнес-логики.

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


def _create_bm25_index(chunks: List[Document], config: Dict[str, Any]):
  """Создает и сохраняет на диск индекс BM25 для поиска по ключевым словам."""
  print("Создание BM25 индекса...")
  tokenized_corpus = [doc.page_content.split() for doc in chunks]
  bm25 = BM25Okapi(tokenized_corpus)
  bm25_index_path = config['retrievers']['bm25']['index_path']
  os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)
  with open(bm25_index_path, "wb") as f:
    pickle.dump({'bm25': bm25, 'docs': chunks}, f)
  print(f"✅ BM25 индекс сохранен в {bm25_index_path}")


def _create_vector_store(chunks: List[Document], config: Dict[str, Any]):
  """Создает и сохраняет на диск векторную базу данных (ChromaDB)."""
  print("Создание векторной базы данных...")
  db_dir = config['retrievers']['vector_store']['db_path']
  model_name = config['embedding_model']['name']
  device = config['embedding_model']['device']
  embeddings_model = HuggingFaceEmbeddings(
      model_name=model_name,
      model_kwargs={'device': device},
      encode_kwargs={'normalize_embeddings': True}
  )

  # Специальная обработка для моделей e5, которые требуют префикс "passage:"
  if "e5" in model_name:
    chunks_to_embed = [
      Document(page_content=f"passage: {chunk.page_content}",
               metadata=chunk.metadata)
      for chunk in chunks
    ]
  else:
    chunks_to_embed = chunks

  db = Chroma.from_documents(chunks_to_embed, embeddings_model,
                             persist_directory=db_dir)
  print(f"✅ Векторная база данных сохранена в {db_dir}")


# --- ГЛАВНАЯ ФУНКЦИЯ-ОРКЕСТРАТОР ---
def run_indexing(config: dict, processor: DataSourceProcessor, mode: str):
  """
  Главная функция-оркестратор пайплайна индексации.

  Args:
      config (dict): Словарь с конфигурацией из config.yaml.
      processor (DataSourceProcessor): Объект для обработки данных.
      mode (str): Режим работы - 'full' или 'test'.
  """
  urls_to_process = []

  # --- ЛОГИКА ВЫБОРА URL В ЗАВИСИМОСТИ ОТ РЕЖИМА ---
  if mode == 'test':
    print("--- Режим 'test': используются URL из `test_urls` в config.yaml ---")
    urls_to_process = config['data_source'].get('test_urls', [])
    if not urls_to_process:
      print("❌ Ошибка: Список 'test_urls' в config.yaml пуст.")
      return
  elif mode == 'full':
    print("--- Режим 'full': запуск полного обхода сайта ---")
    base_url = config['data_source']['url']
    max_depth = config['data_source'].get('max_depth', 2)
    crawler = WebsiteCrawler(base_url=base_url, max_depth=max_depth)
    urls_to_process = crawler.crawl()

  # --- ОБЩАЯ ЛОГИКА ОБРАБОТКИ И ИНДЕКСАЦИИ ---
  all_chunks = []
  for url in urls_to_process:
    chunks_from_url = processor.process(url)
    all_chunks.extend(chunks_from_url)

  print(
    f"\n✅ Обработка всех источников завершена. Получено {len(all_chunks)} чанков.")

  if not all_chunks:
    print("Нет чанков для индексации. Процесс остановлен.")
    return

  print("Генерация уникальных ID для каждого чанка...")
  for doc in all_chunks:
    unique_string = f"{doc.metadata['source']}{doc.page_content}"
    chunk_id = hashlib.md5(unique_string.encode('utf-8')).hexdigest()
    doc.metadata['chunk_id'] = chunk_id
  print("✅ ID успешно сгенерированы.")

  _save_chunks_to_file(all_chunks, config)
  _create_bm25_index(all_chunks, config)
  _create_vector_store(all_chunks, config)

  print("\n🎉 Индексация успешно завершена!")