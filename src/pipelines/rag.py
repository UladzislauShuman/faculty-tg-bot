import os
import yaml
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms import YandexGPT
from langchain_ollama import OllamaLLM as Ollama

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

from src.strategies.retrievers import create_hybrid_retriever

load_dotenv()

#  промпт для ответа на вопрос
PROMPT_TEMPLATE = """
Ты — вежливый и полезный ассистент-помощник. Твоя задача — отвечать на вопросы, основываясь ТОЛЬКО на предоставленном ниже контексте.
Если в контексте нет информации для ответа, честно скажи: "К сожалению, я не нашел информации по вашему вопросу в своей базе знаний."
Не придумывай ничего от себя. Отвечай развернуто, используя всю релевантную информацию из контекста. Указывай полное ФИО и должности, если они есть.

Контекст:
{context}

Вопрос:
{question}

Ответ:
"""

# Промпт для расширения исходного запроса пользователя
QUERY_EXPANSION_TEMPLATE = """
Ты — полезный AI-ассистент. Твоя задача — переписать вопрос пользователя в 3-х различных вариантах, чтобы улучшить поиск в векторной базе.
Сохраняй исходный смысл. Не добавляй лишней информации. Выведи каждую версию на новой строке. Не нумеруй их.

Оригинальный вопрос: {question}

Твои версии:
"""

def get_llm_from_config(provider_config: dict):
    """
    Создает и возвращает экземпляр LLM на основе конфигурации.
    """
    provider_type = provider_config.get("type")

    if provider_type == "ollama":
        print(f"Инициализация LLM от Ollama с моделью: {provider_config.get('model')}")
        return Ollama(model=provider_config.get("model"))
    
    elif provider_type == "yandex_gpt":
        print(f"Инициализация LLM от YandexGPT с моделью: {provider_config.get('model')}")
        secret_key = provider_config.get("secret")
        if not secret_key or secret_key == "YOUR_YANDEX_SECRET_KEY_HERE":
             raise NotImplementedError(
                "YandexGPT интеграция настроена, но требует реального API ключа. "
                "Пожалуйста, вставьте ваш IAM-токен или API-ключ в поле 'secret' файла config.yaml."
            )
        return YandexGPT(api_key=secret_key)
    
    else:
        raise ValueError(f"Неизвестный тип провайдера LLM: {provider_type}")


def create_final_retriever(config: dict):
    """
    Собирает и возвращает финальный ретривер (гибридный + Re-ranker).
    """
    hybrid_retriever = create_hybrid_retriever(config)
    
    reranker_model = config['retrievers']['reranker']['model']
    reranker_top_n = config['retrievers']['reranker']['top_n']
    
    cross_encoder_model = HuggingFaceCrossEncoder(model_name=reranker_model)
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=reranker_top_n)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=hybrid_retriever
    )
    print("✅ Финальный ретривер с Re-ranker'ом успешно создан.")
    return compression_retriever


def create_rag_chain(config: dict, retriever):
  """
  Собирает и возвращает финальный ретривер (гибридный + Re-ranker).
  """
  provider_name = os.getenv("LLM_PROVIDER")
  provider_config = config.get('providers', {}).get(provider_name)
  llm = get_llm_from_config(provider_config)

  # --- QUERY EXPANSION ---
  query_expansion_prompt = PromptTemplate.from_template(
    QUERY_EXPANSION_TEMPLATE)
  # Используем ту же LLM для генерации синонимов
  query_expansion_chain = query_expansion_prompt | llm | StrOutputParser()

  def expand_and_retrieve(query: str):
    """
    Функция, которая сначала расширяет запрос, а потом ищет по всем версиям
    """
    print(f"\n--- Расширение запроса: '{query}' ---")
    expanded_queries_str = query_expansion_chain.invoke({"question": query})
    all_queries = [query] + expanded_queries_str.strip().split('\n')
    # Убираем пустые строки на всякий случай
    all_queries = [q for q in all_queries if q]
    print(f"Всего запросов для поиска: {all_queries}")

    all_docs = []
    for q in all_queries:
      # Для каждого запроса используем наш финальный ретривер (с Re-ranker'ом)
      all_docs.extend(retriever.invoke(q))

    # Убираем дубликаты, сохраняя порядок
    unique_docs_dict = {}
    for doc in all_docs:
      if doc.page_content not in unique_docs_dict:
        unique_docs_dict[doc.page_content] = doc

    unique_docs = list(unique_docs_dict.values())
    print(
      f"Найдено {len(unique_docs)} уникальных и переранжированных документов.")
    return unique_docs

  prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

  rag_chain = (
      {"context": expand_and_retrieve, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )

  print("✅ Полная гибридная RAG-цепочка c Query Expansion успешно создана.")
  return rag_chain