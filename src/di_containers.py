from dependency_injector import containers, providers

from src.retrievers.chroma_bm25 import create_chroma_bm25_retriever
from src.retrievers.qdrant_retriever import create_qdrant_retriever # НОВЫЙ ИМПОРТ
from src.retrievers.rerankers import create_reranker

# Импорты RAG пайплайна
from src.pipelines.rag.pipeline import (
  create_rag_chain,
  create_final_retriever,
  create_search_only_chain,
  create_generation_chain
)

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
      reranker=reranker
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

  rag_chain = providers.Factory(
      create_rag_chain,
      config=config,
      retriever=final_retriever,
      answer_repo=bot_answer_repo
  )