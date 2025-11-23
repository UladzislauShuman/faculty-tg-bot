from dependency_injector import containers, providers

from src.pipelines.rag.pipeline import create_rag_chain, create_final_retriever, create_retrieval_chain
from src.parsing_and_chunking.chunkers.advanced_html_chunker import AdvancedHTMLChunker

from src.parsing_and_chunking.markdown_processor import MarkdownProcessor
from src.parsing_and_chunking.unstructured_processor import UnstructuredProcessor

from src.parsing_and_chunking.configurable_processor import ConfigurableProcessor
from src.parsing_and_chunking.chunkers.semantic_html_chunker import SemanticHTMLChunker
from src.parsing_and_chunking.chunkers.html_context_chunker import HTMLContextChunker

from src.tg_bot.repositories.interfaces import IUserRepository, IAnswerRepository
from src.tg_bot.repositories.implementations import UserRepository, AnswerRepository
from src.tg_bot.services.interfaces import IUserService, IAnswerService
from src.tg_bot.services.implementations import UserService, AnswerService

class Container(containers.DeclarativeContainer):
  """
  DI-контейнер, который собирает все компоненты приложения.
  Является местом для переключения реализаций.
  """
  config = providers.Configuration()

  # --- Провайдеры для RAG-цепочки ---
  final_retriever = providers.Factory(create_final_retriever, config=config)

  retrieval_chain = providers.Factory(
      create_retrieval_chain,
      config=config,
      retriever=final_retriever,
  )

  rag_chain = providers.Factory(create_rag_chain, config=config,
                                retriever=final_retriever)


  # --- Секция выбора стратегий обработки данных ---

  # --- Определяем доступные ЧАНКЕРЫ ---
  # Мы можем создать фабрики для каждого из наших старых чанкеров.
  semantic_chunker = providers.Factory(SemanticHTMLChunker)
  html_context_chunker = providers.Factory(HTMLContextChunker)
  advanced_html_chunker = providers.Factory(AdvancedHTMLChunker)

  # --- Определяем доступные ПРОЦЕССОРЫ ---

  # 2.1 "Монолитные" процессоры (как раньше)
  markdown_processor = providers.Factory(MarkdownProcessor)
  unstructured_processor = providers.Factory(UnstructuredProcessor)

  # 2.2 Новый "конфигурируемый" процессор, который СОБИРАЕТСЯ из чанкера.
  #     Мы говорим: "Создай ConfigurableProcessor и передай ему в конструктор
  #     объект, созданный фабрикой semantic_chunker".
  configurable_semantic_processor = providers.Factory(
      ConfigurableProcessor,
      chunker=semantic_chunker,
  )

  # --- ГЛАВНЫЙ ПЕРЕКЛЮЧАТЕЛЬ ---
  # Здесь вы можете выбрать ЛЮБОЙ из определенных выше процессоров.
  #
  # ВАРИАНТЫ:
  # 1. markdown_processor: Надежный модульный пайплайн через Markdown. (РЕКОМЕНДУЕТСЯ)
  # 2. unstructured_processor: Мощный парсер "из коробки".
  # 3. configurable_semantic_processor: Гибкий процессор, использующий SemanticHTMLChunker.

  data_processor = providers.Factory(
    markdown_processor
    # configurable_semantic_processor
    # unstructured_processor
  )

  # --- Компоненты Telegram-бота ---

  # Репозитории для бота
  bot_user_repo = providers.Singleton(UserRepository)
  bot_answer_repo = providers.Singleton(AnswerRepository)

  # Сервисы для бота
  bot_user_service = providers.Factory(
      UserService,
      user_repo=bot_user_repo
  )
  bot_answer_service = providers.Factory(
      AnswerService,
      answer_repo=bot_answer_repo
  )