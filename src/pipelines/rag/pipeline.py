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
from langchain_openai import ChatOpenAI

from src.retrievers.hybrid_retriever_factory import create_hybrid_retriever

load_dotenv()

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

def create_retrieval_chain(config: dict, retriever: BaseRetriever) -> Runnable:
  """Только поиск документов."""
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


def create_rag_chain(config: dict, retriever: BaseRetriever):
  """Полный RAG (для бота)."""
  gen_chain = create_generation_chain(config)

  rag_chain = (
      {
        "context": lambda x: "\n\n".join(
            [d.page_content for d in retriever.invoke(x)]),
        "question": RunnablePassthrough()
      }
      | gen_chain
  )
  print("✅ Полная RAG-цепочка (для Бота) создана.")
  return rag_chain