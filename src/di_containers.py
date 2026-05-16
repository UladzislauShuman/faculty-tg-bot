"""Сборка зависимостей: один composition root для бота, RAG, retrieval и eval.

Сюда сводятся фабрики из config.yaml. Точки входа (main, server-startup) создают
контейнер и подставляют конфиг — бизнес-код не должен инстанцировать сервисы
вручную.
"""
import logging

from dependency_injector import containers, providers

from src.retrievers.chroma_bm25 import create_chroma_bm25_retriever
from src.retrievers.qdrant_retriever import create_qdrant_retriever
from src.retrievers.rerankers import create_reranker

# Импорты RAG пайплайна
from src.pipelines.rag.pipeline import (
  create_rag_chain,
  create_chat_only_chain,
  create_final_retriever,
  create_search_only_chain,
  create_generation_chain,
  get_llm_from_config,
)
from src.evaluation.metrics import (
    FaithfulnessEvaluator,
    ReferenceSimilarityEvaluator,
    RelevanceEvaluator,
)

# Импорты Чанкеров
from src.parsing_and_chunking.chunkers.semantic_html_chunker import \
  SemanticHTMLChunker
from src.parsing_and_chunking.markdown_processor import MarkdownProcessor
from src.parsing_and_chunking.unstructured_processor import \
  UnstructuredProcessor
from src.parsing_and_chunking.configurable_processor import \
  ConfigurableProcessor
from src.parsing_and_chunking.chunkers.parent_child_chunker import ParentChildHTMLChunker
from src.retrievers.parent_document_retriever import ParentDocumentRetriever
from src.pipelines.routing.router import create_semantic_routing_service

# Импорты Бота
from src.tg_bot.repositories.implementations import UserRepository, \
  AnswerRepository, SessionRepository
from src.tg_bot.services.implementations import UserService, AnswerService, \
  SessionService
from src.tg_bot.services.summarizer import SummarizerService

logger = logging.getLogger(__name__)


def _hyde_llm_from_config(hyde_cfg: object):
  """Возвращает LLM для HyDE только если в конфиге включён hyde.enabled."""
  if not isinstance(hyde_cfg, dict) or not hyde_cfg.get("enabled", False):
    return None
  llm_cfg = hyde_cfg.get("llm")
  if not llm_cfg:
    return None
  return get_llm_from_config(dict(llm_cfg))


def _wrap_with_parent_retriever(base_retriever, parent_cfg):
  """Оборачивает базовый ретривер в ParentDocumentRetriever при включённом parent_document.

  Шаги: проверить флаг → при необходимости загрузить docstore и вернуть обёртку.
  """
  if not isinstance(parent_cfg, dict) or not parent_cfg.get("enabled", False):
    return base_retriever
  docstore_path = parent_cfg.get("docstore_path", "data/parent_docstore.pkl")
  logger.info("Parent document retrieval: обёртка над базовым ретривером, docstore=%s", docstore_path)
  return ParentDocumentRetriever.from_config(
      base_retriever=base_retriever,
      docstore_path=docstore_path
  )

class Container(containers.DeclarativeContainer):
  """DI-контейнер: конфиг, процессоры, бот, retrieval, цепочки RAG, eval."""
  config = providers.Configuration()

  # --- Processors ---
  semantic_chunker = providers.Factory(SemanticHTMLChunker)

  # Именованные процессоры (для CLI)
  markdown_processor = providers.Factory(MarkdownProcessor)
  unstructured_processor = providers.Factory(UnstructuredProcessor)
  semantic_processor = providers.Factory(ConfigurableProcessor,
                                         chunker=semantic_chunker)

  parent_chunker = providers.Factory(
      ParentChildHTMLChunker,
      child_chunk_size=config.parent_document.child_chunk_size,
      parent_chunk_size=config.parent_document.parent_chunk_size
  )
  parent_processor = providers.Factory(ConfigurableProcessor, chunker=parent_chunker)

  # Default (для обратной совместимости)
  data_processor = providers.Factory(markdown_processor)

  # --- Bot ---
  bot_user_repo = providers.Singleton(UserRepository)
  bot_answer_repo = providers.Singleton(AnswerRepository)
  bot_session_repo = providers.Singleton(SessionRepository)
  bot_user_service = providers.Factory(UserService, user_repo=bot_user_repo)
  bot_answer_service = providers.Factory(AnswerService,
                                         answer_repo=bot_answer_repo)
  bot_session_service = providers.Factory(SessionService,
                                          session_repo=bot_session_repo)

  # --- Retrieval Components ---
  hyde_llm = providers.Callable(
      _hyde_llm_from_config,
      hyde_cfg=config.hyde,
  )

  # 1. Провайдеры базовых ретриверов
  chroma_bm25_retriever = providers.Factory(
      create_chroma_bm25_retriever,
      config=config,
      hyde_llm=hyde_llm,
  )
  qdrant_retriever = providers.Factory(
      create_qdrant_retriever,
      config=config,
      hyde_llm=hyde_llm,
  )

  # Динамический выбор ретривера на основе конфига
  raw_base_retriever = providers.Selector(
      config.retrievers.active_type,
      chroma_bm25=chroma_bm25_retriever,
      qdrant=qdrant_retriever
  )

  base_retriever = providers.Callable(
      _wrap_with_parent_retriever,
      base_retriever=raw_base_retriever,
      parent_cfg=config.parent_document
  )

  # 2. Провайдер реранкера (вернет объект или None)
  reranker = providers.Factory(create_reranker, config=config)

  # 3. Финальная сборка ретривера (Поиск + Реранкер)
  final_retriever = providers.Factory(
      create_final_retriever,
      base_retriever=base_retriever,
      reranker=reranker,
      config=config,
  )

  # --- RAG Chains ---
  retrieval_chain = providers.Factory(
      create_search_only_chain,
      config=config,
      retriever=final_retriever
  )

  generation_chain = providers.Factory(
      create_generation_chain,
      config=config
  )

  # --- Smart Memory (Sprint 3) ---
  summary_llm = providers.Factory(
      get_llm_from_config,
      provider_config=config.memory.summary_llm,
  )
  summarizer_service = providers.Factory(
      SummarizerService,
      llm=summary_llm,
      timeout=config.memory.summary_timeout_seconds,
  )

  rag_chain = providers.Factory(
    create_rag_chain,
    config=config,
    retriever=final_retriever,
    answer_repo=bot_answer_repo,
    session_repo=bot_session_repo,
    summarizer=summarizer_service,
  )

  chat_only_chain = providers.Factory(
      create_chat_only_chain,
      config=config,
      answer_repo=bot_answer_repo,
      session_repo=bot_session_repo,
      summarizer=summarizer_service,
  )

  semantic_routing_service = providers.Factory(
      create_semantic_routing_service,
      config=config,
  )

  # --- Evaluation (LLM-as-a-Judge) ---
  judge_llm = providers.Factory(
      get_llm_from_config,
      provider_config=config.evaluation_metrics.judge_llm,
  )
  faithfulness_evaluator = providers.Factory(
      FaithfulnessEvaluator,
      llm=judge_llm,
      timeout=config.evaluation_metrics.timeout_seconds,
  )
  relevance_evaluator = providers.Factory(
      RelevanceEvaluator,
      llm=judge_llm,
      timeout=config.evaluation_metrics.timeout_seconds,
  )
  reference_similarity_evaluator = providers.Factory(
      ReferenceSimilarityEvaluator,
      llm=judge_llm,
      timeout=config.evaluation_metrics.timeout_seconds,
  )