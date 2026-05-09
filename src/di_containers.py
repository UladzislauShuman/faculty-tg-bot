from dependency_injector import containers, providers

from src.retrievers.chroma_bm25 import create_chroma_bm25_retriever
from src.retrievers.qdrant_retriever import create_qdrant_retriever # НОВЫЙ ИМПОРТ
from src.retrievers.rerankers import create_reranker

# Импорты RAG пайплайна
from src.pipelines.rag.pipeline import (
  create_rag_chain,
  create_final_retriever,
  create_search_only_chain,
  create_generation_chain,
  get_llm_from_config,
)
from src.evaluation.metrics import FaithfulnessEvaluator, RelevanceEvaluator

# Импорты Ретриверов и Реранкеров
from src.retrievers.chroma_bm25 import create_chroma_bm25_retriever
from src.retrievers.rerankers import create_reranker

# Импорты Чанкеров
from src.parsing_and_chunking.chunkers.semantic_html_chunker import \
  SemanticHTMLChunker
from src.parsing_and_chunking.markdown_processor import MarkdownProcessor
from src.parsing_and_chunking.unstructured_processor import \
  UnstructuredProcessor
from src.parsing_and_chunking.configurable_processor import \
  ConfigurableProcessor

# Импорты Бота
from src.tg_bot.repositories.implementations import UserRepository, \
  AnswerRepository, SessionRepository
from src.tg_bot.services.implementations import UserService, AnswerService, \
  SessionService
from src.tg_bot.services.summarizer import SummarizerService


class Container(containers.DeclarativeContainer):
  config = providers.Configuration()

  # --- Processors ---
  semantic_chunker = providers.Factory(SemanticHTMLChunker)

  # Именованные процессоры (для CLI)
  markdown_processor = providers.Factory(MarkdownProcessor)
  unstructured_processor = providers.Factory(UnstructuredProcessor)
  semantic_processor = providers.Factory(ConfigurableProcessor,
                                         chunker=semantic_chunker)

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
  # 1. Провайдеры базовых ретриверов
  chroma_bm25_retriever = providers.Factory(create_chroma_bm25_retriever,
                                            config=config)
  qdrant_retriever = providers.Factory(create_qdrant_retriever, config=config)

  # Динамический выбор ретривера на основе конфига
  base_retriever = providers.Selector(
      config.retrievers.active_type,
      chroma_bm25=chroma_bm25_retriever,
      qdrant=qdrant_retriever
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