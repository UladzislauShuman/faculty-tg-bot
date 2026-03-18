import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, \
  RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from langchain_ollama import OllamaLLM as Ollama
from langchain_community.llms import YandexGPT

from langchain.retrievers.contextual_compression import \
  ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from src.retrievers.hybrid_retriever_factory import create_hybrid_retriever

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.tg_bot.db.history import ReadOnlyPostgresHistory

load_dotenv()

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Учитывая историю чата и последний вопрос пользователя, \
который может ссылаться на контекст в истории чата, сформулируй самостоятельный вопрос, \
который можно понять без истории чата. НЕ отвечай на вопрос, просто переформулируй его, если нужно, \
иначе верни как есть."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

QA_SYSTEM_PROMPT = """Ты — официальный ассистент ФПМИ БГУ. Отвечай ТОЛЬКО на основе контекста ниже.

КОНТЕКСТ:
{context}

ИНСТРУКЦИЯ:
1. Если в контексте нет точного ответа, ответь: "К сожалению, в базе знаний нет информации по вашему вопросу."
2. Не придумывай факты, телефоны или имена.
3. После каждого факта указывай источник в скобках.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

PROMPT_TEMPLATE = """
Ты — официальный ассистент ФПМИ БГУ. Отвечай ТОЛЬКО на основе контекста ниже.

КОНТЕКСТ:
{context}

ИНСТРУКЦИЯ:
1. Если в контексте нет точного ответа, ответь: "К сожалению, в базе знаний нет информации по вашему вопросу."
2. Не придумывай факты, телефоны или имена.
3. После каждого факта указывай источник в скобках, например: (Источник: Деканат).

ВОПРОС: {question}
ОТВЕТ:
"""

def create_retrieval_chain_test(config: dict,
    retriever: BaseRetriever) -> Runnable:
  """Оставляем для тестов (rag-cli retrieve)"""
  return RunnableLambda(lambda q: retriever.invoke(q))

def get_llm_from_config(provider_config: dict):
  provider_type = provider_config.get("type")
  if provider_type == "ollama":
    return Ollama(
        model=provider_config.get("model"),
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
  elif provider_type == "yandex_gpt":
    secret_key = os.getenv("YANDEX_GPT_SECRET")
    if not secret_key:
      secret_key = provider_config.get("secret")
    if not secret_key or secret_key == "YOUR_YANDEX_SECRET_KEY_HERE":
      raise ValueError(
          "❌ Ошибка: Не найден API-ключ YandexGPT. "
          "Добавьте YANDEX_GPT_SECRET в файл .env"
      )
    return YandexGPT(api_key=secret_key)
  else:
    raise ValueError(f"Unknown provider: {provider_type}")


def create_final_retriever(config: dict) -> BaseRetriever:
  hybrid_retriever = create_hybrid_retriever(config)

  # Reranker
  reranker_conf = config['retrievers']['reranker']
  cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_conf['model'])
  compressor = CrossEncoderReranker(model=cross_encoder,
                                    top_n=reranker_conf['top_n'])

  return ContextualCompressionRetriever(
      base_compressor=compressor, base_retriever=hybrid_retriever
  )


# --- ЦЕПОЧКИ ---

def create_search_only_chain(config: dict, retriever: BaseRetriever) -> Runnable:
  """Только поиск документов (используется в тестах)."""
  return RunnableLambda(lambda q: retriever.invoke(q))


def create_generation_chain(config: dict) -> Runnable:
  """
  Только генерация.
  Принимает dict: {"context": str, "question": str}
  """
  provider_name = os.getenv("LLM_PROVIDER", "ollama")
  provider_config = config.get('providers', {}).get(provider_name)
  llm = get_llm_from_config(provider_config)

  prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
  return prompt | llm | StrOutputParser()


def create_rag_chain(config: dict, retriever: BaseRetriever, answer_repo):
  """Полный Conversational RAG (для бота)."""

  provider_name = os.getenv("LLM_PROVIDER", "ollama")
  provider_config = config.get('providers', {}).get(provider_name)
  llm = get_llm_from_config(provider_config)

  # 1. Умный ретривер (переформулирует вопрос с учетом истории)
  history_aware_retriever = create_history_aware_retriever(
      llm, retriever, contextualize_q_prompt
  )

  # 2. Цепочка ответов (генерирует ответ по найденным документам)
  question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

  # 3. Общая RAG-цепочка (Поиск + Ответ)
  rag_chain = create_retrieval_chain(history_aware_retriever,
                                     question_answer_chain)

  # 4. Функция для получения истории из нашей БД
  def get_session_history(session_id: str):
    return ReadOnlyPostgresHistory(session_id=session_id,
                                   answer_repo=answer_repo)

  # 5. Оборачиваем в менеджер памяти
  conversational_rag_chain = RunnableWithMessageHistory(
      rag_chain,
      get_session_history,
      input_messages_key="input",
      history_messages_key="chat_history",
      output_messages_key="answer",
  )

  print("✅ Полная Conversational RAG-цепочка создана.")
  return conversational_rag_chain