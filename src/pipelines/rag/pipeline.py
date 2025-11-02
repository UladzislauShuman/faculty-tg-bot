import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from langchain_community.llms import YandexGPT
from langchain_ollama import OllamaLLM as Ollama

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from src.retrievers.hybrid_retriever_factory import create_hybrid_retriever

load_dotenv()

#  Промпт для финальной генерации ответа на основе найденного контекста
PROMPT_TEMPLATE = """
Ты — официальный, вежливый и очень точный ассистент-помощник Факультета прикладной математики и информатики (ФПМИ) БГУ.
Твоя задача — давать исчерпывающие ответы на вопросы, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном ниже контексте.

### ОСНОВНЫЕ ПРАВИЛА:
1.  **СТРОГО ПО КОНТЕКСТУ:** Не придумывай ничего от себя. Не используй свои общие знания. Вся информация для ответа должна быть взята из текста в блоке 'Контекст'.
2.  **ЕСЛИ ОТВЕТА НЕТ:** Если в контексте нет информации для ответа на вопрос, твой единственный правильный ответ должен быть: "К сожалению, в предоставленных материалах я не нашел ответа на ваш вопрос." Не пытайся угадать.
3.  **ИСПОЛЬЗУЙ ВСЮ ИНФОРМАЦИЮ:** Если для полного ответа нужно собрать информацию из нескольких фрагментов контекста, сделай это. Объедини факты в связный и логичный ответ.

### СТИЛЬ ОТВЕТА:
- Отвечай развернуто, по существу, но без лишней "воды".
- Если в тексте есть полное ФИО (Фамилия Имя Отчество) и должности, всегда используй их.
- При перечислении документов, требований или списков используй Markdown-форматирование (маркированные списки) для лучшей читаемости.
- Всегда отвечай на русском языке.

---
КОНТЕКСТ:
{context}
---
ВОПРОС:
{question}
---
ОТВЕТ:
"""

# Промпт для расширения исходного запроса пользователя с целью улучшения поиска
QUERY_EXPANSION_TEMPLATE = """
Ты — эксперт по поиску, ассистент для базы знаний Факультета прикладной математики и информатики (ФПМИ).
Твоя задача — помочь пользователю найти информацию, создав 3 альтернативные, семантически близкие версии его вопроса для улучшения поиска в векторной базе.

### ПРИМЕРЫ
Оригинал: кто был первым деканом?
Версии:
кем руководил факультет при его основании
кого можно считать первоначальным лидером ФПМИ
кому принадлежал титул первого директора факультета прикладной математики и информатики

Оригинал: для чего был создан ФПМИ?
Версии:
какова цель создания ФПМИ
миссия и задачи факультета прикладной математики и информатики
причины основания факультета

### СТРОГИЕ ПРАВИЛА ФОРМАТИРОВАНИЯ
- Только 3 версии.
- Каждая версия на новой строке.
- Никаких номеров, маркеров, кавычек или слов вроде "Вариант:".
- В ответе должен быть ТОЛЬКО текст трех вопросов и ничего больше.

Оригинал: {question}
Версии:
"""


def get_llm_from_config(provider_config: dict):
  """Создает и возвращает экземпляр LLM на основе конфигурации."""
  provider_type = provider_config.get("type")

  if provider_type == "ollama":
    print(
      f"Инициализация LLM от Ollama с моделью: {provider_config.get('model')}")
    return Ollama(model=provider_config.get("model"))
  elif provider_type == "yandex_gpt":
    print(
      f"Инициализация LLM от YandexGPT с моделью: {provider_config.get('model')}")
    secret_key = provider_config.get("secret")
    if not secret_key or secret_key == "YOUR_YANDEX_SECRET_KEY_HERE":
      raise NotImplementedError("YandexGPT требует API-ключ в config.yaml.")
    return YandexGPT(api_key=secret_key)
  else:
    raise ValueError(f"Неизвестный тип провайдера LLM: {provider_type}")


def create_final_retriever(config: dict) -> BaseRetriever:
  """
  Собирает и возвращает финальный ретривер, который включает:
  1. Гибридный поиск (BM25 + Vector Search).
  2. Переранжирование с помощью Cross-Encoder для повышения точности.
  """
  hybrid_retriever = create_hybrid_retriever(config)

  reranker_model = config['retrievers']['reranker']['model']
  reranker_top_n = config['retrievers']['reranker']['top_n']

  cross_encoder_model = HuggingFaceCrossEncoder(model_name=reranker_model)
  compressor = CrossEncoderReranker(model=cross_encoder_model,
                                    top_n=reranker_top_n)

  compression_retriever = ContextualCompressionRetriever(
      base_compressor=compressor, base_retriever=hybrid_retriever
  )
  print("✅ Финальный ретривер с Re-ranker'ом успешно создан.")
  return compression_retriever


def _expand_query_and_retrieve(query: str, retriever: BaseRetriever,
    config: dict):
  """
  Приватная функция, которая расширяет запрос, выполняет поиск по всем версиям
  и надежно удаляет дубликаты по chunk_id.
  """
  print(f"\n--- Расширение запроса: '{query}' ---")

  provider_name = os.getenv("LLM_PROVIDER", "ollama")
  provider_config = config.get('providers', {}).get(provider_name)
  llm = get_llm_from_config(provider_config)

  query_expansion_prompt = PromptTemplate.from_template(
    QUERY_EXPANSION_TEMPLATE)
  query_expansion_chain = query_expansion_prompt | llm | StrOutputParser()

  expanded_queries_str = query_expansion_chain.invoke({"question": query})
  all_queries = [query] + [q.strip() for q in
                           expanded_queries_str.strip().split('\n') if
                           q.strip()]
  print(f"Всего запросов для поиска: {all_queries}")

  all_docs = []
  for q in all_queries:
    # Используем .invoke() для ContextualCompressionRetriever
    all_docs.extend(retriever.invoke(q))

  # Удаляем дубликаты по chunk_id, сохраняя порядок
  unique_docs_dict = {}
  for doc in all_docs:
    chunk_id = doc.metadata.get('chunk_id')
    if chunk_id not in unique_docs_dict:
      unique_docs_dict[chunk_id] = doc

  unique_docs = list(unique_docs_dict.values())

  print(
    f"Найдено {len(unique_docs)} уникальных и переранжированных документов.")
  return unique_docs

def create_retrieval_chain(config: dict,
    retriever: BaseRetriever) -> Runnable:
  """
  Создает и возвращает только ту часть цепочки, которая отвечает за ретривинг
  (расширение запроса и получение документов).
  """

  # Создаем "замыкание" (closure), чтобы передать retriever и config в RunnableLambda
  def retrieval_closure(query: str):
    return _expand_query_and_retrieve(query, retriever, config)

  return RunnableLambda(retrieval_closure)


def create_rag_chain(config: dict, retriever: BaseRetriever):
  """
  Собирает и возвращает полную RAG-цепочку.
  Теперь эта функция будет использовать `create_retrieval_chain`.
  """
  prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

  provider_name = os.getenv("LLM_PROVIDER", "ollama")
  provider_config = config.get('providers', {}).get(provider_name)
  llm = get_llm_from_config(provider_config)

  # 1. Создаем шаг ретривинга как отдельную, именованную цепочку.
  retrieval_chain = create_retrieval_chain(config, retriever)

  # 2. Собираем финальную RAG-цепочку, используя эту именованную часть.
  rag_chain = (
      {
        "context": retrieval_chain,
        "question": RunnablePassthrough()
      }
      | prompt
      | llm
      | StrOutputParser()
  )

  print("✅ Полная гибридная RAG-цепочка c Query Expansion успешно создана.")
  return rag_chain