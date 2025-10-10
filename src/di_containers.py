from dependency_injector import containers, providers
from src.pipelines.rag import create_rag_chain, create_final_retriever, \
  create_full_retrieval_chain


class Container(containers.DeclarativeContainer):
  """
  DI-контейнер, который собирает все компоненты приложения.
  """
  config = providers.Configuration()

  # Фабрика для создания финального ретривера (гибридный + Re-ranker)
  final_retriever = providers.Factory(
      create_final_retriever,
      config=config,
  )

  full_retrieval_chain = providers.Factory(
      create_full_retrieval_chain,
      config=config,
      retriever=final_retriever,
  )

  # Фабрика для создания полной RAG-цепочки
  rag_chain = providers.Factory(
      create_rag_chain,
      config=config,
      retriever=final_retriever,
  )
