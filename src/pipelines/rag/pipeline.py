import logging
import os
import time
from typing import Any, Optional
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.config import RunnableConfig
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import BaseDocumentCompressor

from langchain_ollama import OllamaLLM as Ollama
from langchain_community.llms import YandexGPT

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from src.pipelines.rag.timed_wrappers import (
    TimedContextualCompressionRetriever,
    TimedRetrievalOnlyRetriever,
    stage_timing_logs_enabled,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.tg_bot.db.history import ReadOnlyPostgresHistory

load_dotenv()

logger = logging.getLogger(__name__)

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

def create_retrieval_chain_test(config: dict, retriever: BaseRetriever) -> Runnable:
    """Оставляем для тестов (rag-cli retrieve)"""
    return RunnableLambda(lambda q: retriever.invoke(q))

def get_llm_from_config(provider_config: dict):
    if not isinstance(provider_config, dict):
        provider_config = dict(provider_config)
    provider_type = provider_config.get("type")
    if provider_type == "ollama":
        return Ollama(
            model=provider_config.get("model"),
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            temperature=provider_config.get("temperature", 0.7),
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


def create_final_retriever(
    base_retriever: BaseRetriever,
    reranker: Optional[BaseDocumentCompressor] = None,
    config: Optional[dict] = None,
) -> BaseRetriever:
    """
    Оборачивает базовый поиск в реранкер (если он передан).
    Sprint 4: при включённом stage_timing логирует длительности retrieval / reranker.
    """
    timing = stage_timing_logs_enabled(config)
    if not reranker:
        print("✅ Используется базовый поиск (без переранжирования).")
        if timing:
            return TimedRetrievalOnlyRetriever(base_retriever=base_retriever)
        return base_retriever

    print("✅ Reranker успешно подключен к пайплайну.")
    if timing:
        return TimedContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )


def _create_history_aware_retriever_with_timing(
    llm: Any,
    retriever: BaseRetriever,
    prompt: Any,
) -> Any:
    """Эквивалент create_history_aware_retriever + явные [TIMING] для reformulation (Sprint 4)."""
    reformulate_chain = prompt | llm | StrOutputParser()

    def _log_skip_reformulation(x: dict) -> str:
        logger.info(
            "[TIMING] stage=reformulation elapsed=0.00s (skipped; empty chat_history)"
        )
        return x["input"]

    def _timed_reform_sync(x: dict, config: RunnableConfig) -> str:
        t0 = time.perf_counter()
        try:
            return reformulate_chain.invoke(x, config)
        finally:
            logger.info(
                "[TIMING] stage=reformulation elapsed=%.2fs",
                time.perf_counter() - t0,
            )

    async def _timed_reform_async(x: dict, config: RunnableConfig) -> str:
        t0 = time.perf_counter()
        try:
            return await reformulate_chain.ainvoke(x, config)
        finally:
            logger.info(
                "[TIMING] stage=reformulation elapsed=%.2fs",
                time.perf_counter() - t0,
            )

    timed_reformulate = RunnableLambda(_timed_reform_sync, afunc=_timed_reform_async)
    direct_path = RunnableLambda(_log_skip_reformulation) | retriever
    history_path = timed_reformulate | retriever

    return RunnableBranch(
        (lambda x: not x.get("chat_history", False), direct_path),
        history_path,
    ).with_config(run_name="chat_retriever_chain")

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

def create_rag_chain(
    config: dict,
    retriever: BaseRetriever,
    answer_repo: Any,
    session_repo: Any = None,
    summarizer: Any = None,
):
    """Полный Conversational RAG (для бота)."""
    provider_name = os.getenv("LLM_PROVIDER", "ollama")
    provider_config = config.get('providers', {}).get(provider_name)
    llm = get_llm_from_config(provider_config)

    timing = stage_timing_logs_enabled(config)

    # 1. Умный ретривер (переформулирует вопрос с учетом истории)
    if timing:
        history_aware_retriever = _create_history_aware_retriever_with_timing(
            llm, retriever, contextualize_q_prompt
        )
    else:
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

    # 2. Цепочка ответов (генерирует ответ по найденным документам)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Общая RAG-цепочка (Поиск + Ответ)
    if timing:
        def _gen_sync(inputs: dict, config: RunnableConfig) -> str:
            t0 = time.perf_counter()
            try:
                return question_answer_chain.invoke(inputs, config)
            finally:
                logger.info(
                    "[TIMING] stage=generation elapsed=%.2fs",
                    time.perf_counter() - t0,
                )

        async def _gen_async(inputs: dict, config: RunnableConfig) -> str:
            t0 = time.perf_counter()
            try:
                return await question_answer_chain.ainvoke(inputs, config)
            finally:
                logger.info(
                    "[TIMING] stage=generation elapsed=%.2fs",
                    time.perf_counter() - t0,
                )

        timed_answer = RunnableLambda(_gen_sync, afunc=_gen_async)
        rag_chain = (
            RunnablePassthrough.assign(
                context=history_aware_retriever.with_config(
                    run_name="retrieve_documents",
                ),
            ).assign(answer=timed_answer)
        ).with_config(run_name="retrieval_chain")
    else:
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

    # 4. Функция для получения истории из нашей БД
    mem = config.get("memory") or {}

    def get_session_history(session_id: str):
        return ReadOnlyPostgresHistory(
            session_id=session_id,
            answer_repo=answer_repo,
            session_repo=session_repo,
            summarizer=summarizer,
            window_size=int(mem.get("window_size", 5)),
            summarization_threshold=int(mem.get("summarization_threshold", 4)),
            memory_enabled=bool(mem.get("enabled", False)),
        )

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