from dependency_injector import containers, providers

# Импорты
from src.pipelines.rag.pipeline import create_rag_chain, create_final_retriever, \
  create_retrieval_chain, create_generation_chain
from src.parsing_and_chunking.chunkers.semantic_html_chunker import \
  SemanticHTMLChunker
from src.parsing_and_chunking.markdown_processor import MarkdownProcessor
from src.parsing_and_chunking.unstructured_processor import \
  UnstructuredProcessor
from src.parsing_and_chunking.configurable_processor import \
  ConfigurableProcessor

from src.tg_bot.repositories.implementations import UserRepository, \
  AnswerRepository
from src.tg_bot.services.implementations import UserService, AnswerService


class Container(containers.DeclarativeContainer):
  config = providers.Configuration()

  # --- RAG Components ---
  final_retriever = providers.Factory(create_final_retriever, config=config)

  # Раздельные компоненты для тестов
  retrieval_chain = providers.Factory(create_retrieval_chain, config=config,
                                      retriever=final_retriever)
  generation_chain = providers.Factory(create_generation_chain, config=config)

  # Единый компонент для Бота
  rag_chain = providers.Factory(create_rag_chain, config=config,
                                retriever=final_retriever)

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
  bot_user_service = providers.Factory(UserService, user_repo=bot_user_repo)
  bot_answer_service = providers.Factory(AnswerService,
                                         answer_repo=bot_answer_repo)