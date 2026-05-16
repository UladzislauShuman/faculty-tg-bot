"""Оффлайн-индексация: сбор URL, чанкинг, запись Chroma+BM25 или Qdrant hybrid.

Точка входа — run_indexing; ветка хранилища задаётся config.retrievers.active_type.
"""
import hashlib
import json
import logging
import os
import pickle
import textwrap
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from src.interfaces.data_processor_interfaces import DataSourceProcessor
from src.pipelines.indexing.crawlers.website_crawler import WebsiteCrawler
from src.util.hf_embeddings import huggingface_embedding_model_kwargs
from src.util.yaml_parser import TestSetLoader

logger = logging.getLogger(__name__)


def configure_indexing_loggers() -> None:
  """Поднимает уровень до INFO для этапов индексации (см. RAG_INDEX_LOGS).

  После замены print() на logging без этого не было видно логов
  ``ConfigurableProcessor`` / чанкеров (пакет ``src.parsing_and_chunking``).
  """
  if os.environ.get("RAG_INDEX_LOGS", "1").lower() in ("0", "false", "no"):
    return
  for name in (
      "src.pipelines.indexing.pipeline",
      "src.pipelines.indexing.crawlers.website_crawler",
      "src.parsing_and_chunking",
  ):
    logging.getLogger(name).setLevel(logging.INFO)


# --- ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ ---

def _prepare_embeddings(config: Dict[str, Any]):
  """Создаёт HuggingFaceEmbeddings по имени модели и device из config.embedding_model."""
  emb_cfg = config['embedding_model']
  model_name = emb_cfg['name']
  hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
  cache_model_dir = os.path.join(hf_home, "hub", f"models--{model_name.replace('/', '--')}")
  try:
    emb = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=huggingface_embedding_model_kwargs(emb_cfg),
        encode_kwargs={'normalize_embeddings': True},
    )
    logger.info("Модель эмбеддингов загружена: %s", model_name)
    return emb
  except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
    extra = ""
    if isinstance(e, json.JSONDecodeError):
      extra = " Похоже на битый JSON в кеше (часто tokenizer_config.json после обрыва загрузки)."
    offline = bool(emb_cfg.get("local_files_only", False))
    hint = (
        f"Не удалось загрузить эмбеддинг-модель {model_name!r} ({e!s}).{extra} "
        "Частые причины: нет DNS/интернета или прерванная загрузка (битый кеш). "
        f"Проверьте сеть из контейнера; при необходимости удалите каталог "
        f"{cache_model_dir} и повторите индексацию."
    )
    if offline:
      hint += (
          " С embedding_model.local_files_only=true модель должна быть полностью "
          "в HF-кеше; иначе временно выставьте local_files_only: false."
      )
    logger.error("%s", hint)
    raise RuntimeError(hint) from e


def _apply_e5_passage_prefix(chunks: List[Document], model_name: str) -> List[
  Document]:
  """
  Для E5 к тексту чанка добавляется префикс passage: (согласовано с запросами query:).
  """
  if "e5" in model_name.lower():
    return [
      Document(
          page_content=f"passage: {chunk.page_content}",
          metadata=chunk.metadata
      )
      for chunk in chunks
    ]
  return chunks


def _save_chunks_to_file(chunks: List[Document], config: Dict[str, Any]):
  """Пишет человекочитаемый дамп чанков в paths.output_dir + paths.indexing."""
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
  logger.info("Дамп чанков записан: %s (%s шт.)", output_file_path, len(chunks))


# --- РЕАЛИЗАЦИИ ИНДЕКСАЦИИ (STORAGE SPECIFIC) ---

def _index_chroma_bm25(chunks: List[Document], config: Dict[str, Any]):
  """BM25 pickle + векторизация в Chroma по тем же чанкам."""

  # 1. Подготовка документов для BM25
  logger.info("BM25: сохранение списка документов в pickle")
  bm25_index_path = config['retrievers']['bm25']['index_path']
  os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)
  with open(bm25_index_path, "wb") as f:
    # Сохраняем только документы. Индекс построится при инициализации ретривера.
    pickle.dump({'docs': chunks}, f)
  logger.info("BM25 pickle готов: %s", bm25_index_path)

  # 2. Индексация в ChromaDB
  db_dir = config['retrievers']['vector_store']['db_path']
  logger.info("Chroma: векторизация и persist в %s", db_dir)
  embeddings_model = _prepare_embeddings(config)

  # Применяем префиксы E5 перед векторизацией
  chunks_to_embed = _apply_e5_passage_prefix(chunks,
                                             config['embedding_model']['name'])

  Chroma.from_documents(
      documents=chunks_to_embed,
      embedding=embeddings_model,
      persist_directory=db_dir
  )
  logger.info("Chroma индекс записан: %s", db_dir)

def _index_qdrant(chunks: List[Document], config: Dict[str, Any]):
  """Hybrid Qdrant: dense (HF) + sparse (FastEmbed); коллекция пересоздаётся."""
  logger.info("Qdrant: hybrid индексация (dense+sparse)")
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
  logger.info("Qdrant коллекция готова: %s @ %s:%s", collection_name, host, port)


# --- ОСНОВНОЙ ПАЙПЛАЙН ---

def run_indexing(config: dict, processor: DataSourceProcessor, mode: str):
  """Оркестратор: URL из test/full → чанки → parent docstore → дамп → Chroma или Qdrant.

  mode: 'test' — URL из qa-test-set; 'full' — краулер с data_source.
  """
  configure_indexing_loggers()
  urls_to_process = []

  # 1. Определение списка URL
  if mode == 'test':
    logger.info("Индексация: режим test, читаем URL из qa-test-set")
    loader = TestSetLoader(config['paths']['qa_test_set'])
    urls_to_process = loader.get_test_urls()
    if not urls_to_process:
      logger.error("В qa-test-set.yaml нет активных URL для индексации")
      return
  elif mode == 'full':
    logger.info("Индексация: режим full, краулер с %s", config['data_source']['url'])
    base_url = config['data_source']['url']
    max_depth = config['data_source'].get('max_depth', 2)
    crawler = WebsiteCrawler(base_url=base_url, max_depth=max_depth)
    urls_to_process = crawler.crawl()

  # 2. Сбор и чанкинг контента
  all_chunks = []
  all_parents = []
  for url in urls_to_process:
    if hasattr(processor, "process_with_parents"):
      chunks_from_url, parents_from_url = processor.process_with_parents(url)
      all_chunks.extend(chunks_from_url)
      all_parents.extend(parents_from_url)
    else:
      chunks_from_url = processor.process(url)
      all_chunks.extend(chunks_from_url)

  if not all_chunks:
    logger.warning("Нет чанков для индексации — выход")
    return

  logger.info(
      "Обработка источников завершена: чанков=%s, родителей=%s",
      len(all_chunks),
      len(all_parents),
  )

  # 2.5 Сохранение родителей (если включен Parent Document Retrieval)
  parent_config = config.get('parent_document', {})
  if parent_config.get('enabled', False) and all_parents:
    docstore_path = parent_config.get('docstore_path', 'data/parent_docstore.pkl')
    os.makedirs(os.path.dirname(docstore_path), exist_ok=True)
    
    # Преобразуем список в словарь {doc_id: Document}
    docstore = {}
    for parent in all_parents:
      doc_id = parent.metadata.get('doc_id')
      if doc_id:
        docstore[doc_id] = parent
        
    with open(docstore_path, "wb") as f:
      pickle.dump(docstore, f)
    logger.info("Parent docstore: %s записей → %s", len(docstore), docstore_path)

  # 3. Генерация уникальных ID (для дедупликации и отслеживания)
  logger.debug("Генерация chunk_id (md5) для каждого чанка")
  for doc in all_chunks:
    unique_string = f"{doc.metadata['source']}{doc.page_content}"
    chunk_id = hashlib.md5(unique_string.encode('utf-8')).hexdigest()
    doc.metadata['chunk_id'] = chunk_id

  # 4. Сохранение текстового дампа для отладки
  _save_chunks_to_file(all_chunks, config)

  # 5. Маршрутизация в выбранное хранилище
  active_retriever_type = config['retrievers'].get('active_type', 'chroma_bm25')
  logger.info("Запись индекса: active_type=%s", active_retriever_type)

  if active_retriever_type == 'chroma_bm25':
    _index_chroma_bm25(all_chunks, config)
  elif active_retriever_type == 'qdrant':
    _index_qdrant(all_chunks, config)
  else:
    logger.error("Неизвестный retrievers.active_type: %s", active_retriever_type)
    return

  logger.info("Индексация успешно завершена")