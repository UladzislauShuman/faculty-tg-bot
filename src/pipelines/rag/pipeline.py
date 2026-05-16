"""RAG pipeline: LLM из конфига, retrieval, история чата, сборка LCEL-цепочек.

Основной сценарий для бота — create_rag_chain + RunnableWithMessageHistory
и ReadOnlyPostgresHistory для подгрузки диалога из БД.
"""
import logging
import os
import time
from typing import Any, Optional
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
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

_OLD_CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Учитывая историю чата и последний вопрос пользователя, \
который может ссылаться на контекст в истории чата, сформулируй самостоятельный вопрос, \
который можно понять без истории чата. НЕ отвечай на вопрос, просто переформулируй его, если нужно, \
иначе верни как есть."""

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Ты — модуль подготовки поисковой строки для справочной системы ФПМИ БГУ.

ЦЕЛЬ: по истории и последней реплике пользователя вернуть ОДНУ строку для поиска по базе знаний. Строка должна звучать как фрагмент официального текста (тема, заголовок, сухое описание), а не как разговорный вопрос в Telegram.

ЭТО НЕ ОТВЕТ ПОЛЬЗОВАТЕЛЮ. Не утверждай факты, которых нет в вопросе и истории. Не добавляй цифры, имена, даты, адреса из головы. Не рассуждай и не объясняй.

КАК ФОРМУЛИРОВАТЬ:
1. Сними разговорные вводные и «рамку вопроса», если смысл от этого не теряется: «подскажи», «а», «скажите», «где находится» → оставь смысловой остов (часто в родительном/именительном падеже: «деканат ФПМИ», «кабинет декана», «проходной балл …»).
2. Замени местоимения и отсылки (он, она, они, там, тогда, в этом году, на платное и т.п.) на сущности и формулировки из истории чата.
3. Раскрой эллипсис: короткая реплика должна стать однозначной темой для поиска («а на платное?» + контекст про специальность и год → явно повторить специальность и год из истории).
4. Предпочитай стиль близкий к тому, как информация может быть изложена на сайте: тематическая цепочка существительных и уточнений, без лишних вопросительных слов в начале, если без них запрос всё ещё однозначен.
5. Если без вопросительной формы смысл размывается — допустима короткая вопросительная формулировка, но без воды и без «А где …?» ради звука.
6. Одна строка, без преамбул и кавычек. Не дублируй «Переформулировано:» и подобное.

ПРИМЕРЫ:

История:
Human: Кто декан ФПМИ?
AI: Деканом является Орлович Юрий Леонидович.
Human: А в каком он кабинете?
Строка для поиска: кабинет декана ФПМИ Орлович Юрий Леонидович расположение

История:
Human: Какой проходной балл на Прикладную информатику в 2025?
AI: 391 балл.
Human: А на платное?
Строка для поиска: проходной балл платная форма обучения Прикладная информатика 2025

История:
Human: Когда был основан факультет?
AI: 1 апреля 1970 года.
Human: Сколько ему сейчас лет?
Строка для поиска: возраст ФПМИ дата основания 1 апреля 1970

История:
Human: Кто заведующий кафедрой ТП?
AI: ...
Human: А кафедра МСС?
Строка для поиска: заведующий кафедрой МСС многопроцессорных систем и сетей ФПМИ

История: (пустая)
Human: Кто декан факультета?
Строка для поиска: декан факультета ФПМИ БГУ"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

_OLD_QA_SYSTEM_PROMPT = """Ты — официальный ассистент ФПМИ БГУ. Отвечай ТОЛЬКО на основе контекста ниже.

КОНТЕКСТ:
{context}

ИНСТРУКЦИЯ:
1. Если в контексте нет точного ответа, ответь: "К сожалению, в базе знаний нет информации по вашему вопросу."
2. Не придумывай факты, телефоны или имена.
3. После каждого факта указывай источник в скобках.
"""

QA_SYSTEM_PROMPT = """Ты — официальный справочный ассистент ФПМИ БГУ (Факультет прикладной математики и информатики Белорусского государственного университета).

ИСТОЧНИК ДАННЫХ:
{context}

ПРАВИЛА ОТВЕТА:
1. Отвечай ТОЛЬКО на основе блока «ИСТОЧНИК ДАННЫХ» выше. Не используй внешние знания.
2. Если в источнике есть прямой ответ — дай его кратко и по делу (1–3 предложения, факты без воды).
3. Если в источнике есть ЧАСТЬ ответа — приведи ту часть, что нашёл, и явно укажи, чего не хватает: «В материалах нет точных данных о X, но известно, что Y».
4. Если ответа нет совсем — ответь дословно: «К сожалению, в базе знаний нет информации по вашему вопросу.»
5. Никогда не придумывай имена, телефоны, даты, номера кабинетов, ссылки. При сомнении — пункт 4.
6. После каждого утверждения указывай источник в скобках: (Источник: <короткое название раздела или URL из метаданных>).
7. Имена пиши без сокращений и без искажений (никаких «Кадуринина» вместо «Кадурина»).
8. Если вопрос ссылается на предыдущие реплики («он», «она», «там») — используй контекст истории чата ниже, чтобы понять референт.

ФОРМАТ:
- Без приветствий, без «Ответ:», без markdown-заголовков.
- Списки — только если в источнике явно список (например, состав комиссии).
- Ссылки — только реальные URL из источника, без выдуманных."""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

CHAT_ONLY_SMALLTALK_SYSTEM = """Ты — дружелюбный ассистент ФПМИ БГУ в Telegram. Отвечай по-русски, кратко и естественно.
Опирайся на историю диалога: продолжай тему, используй факты, которые пользователь уже видел в этой беседе (например посчитай разницу лет, если год основания или другая дата уже обсуждались).
Не придумывай новые конкретные факты о факультете (имена, даты с сайта), если их не было в переписке — тогда предложи задать отдельный вопрос про факультет, чтобы подключился поиск по базе знаний."""

CHAT_ONLY_DIRECT_SYSTEM = """Ты — ассистент ФПМИ БГУ. Пользователь просит ссылку или где что посмотреть.
Учитывай историю диалога. Официальный сайт: https://fpmi.bsu.by — кратко направь к разделам (расписание, новости, поступление). Если ссылку уже давали — ответ можно короче."""

PROMPT_TEMPLATE = """
Ты — официальный справочный ассистент ФПМИ БГУ. Отвечай ТОЛЬКО на основе контекста ниже.

КОНТЕКСТ:
{context}

ПРАВИЛА:
1. Отвечай только по контексту. Не используй внешние знания.
2. При прямом ответе — 1–3 предложения, после факта укажи источник: (Источник: …).
3. При частичных данных опиши, что есть, и чего не хватает.
4. Если ответа нет — дословно: «К сожалению, в базе знаний нет информации по вашему вопросу.»
5. Не придумывай имена, телефоны, даты, ссылки.

ВОПРОС: {question}
ОТВЕТ:
"""

def create_retrieval_chain_test(config: dict, retriever: BaseRetriever) -> Runnable:
    """Оставляем для тестов (rag-cli retrieve)"""
    return RunnableLambda(lambda q: retriever.invoke(q))

def get_llm_from_config(provider_config: dict):
    """Строит инстанс LLM по YAML-блоку провайдера (ollama / yandex_gpt).

    URL Ollama и секрет Yandex при необходимости читаются из окружения.
    """
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
    """Сборка финального ретривера: базовый поиск ± ContextualCompression (reranker).

    Шаги: (1) проверить включён ли stage_timing; (2) без reranker — optional timed wrapper;
    (3) с reranker — обёртка compressor+retriever (с таймингом или без).
    """
    timing = stage_timing_logs_enabled(config)
    if not reranker:
        logger.info("Финальный ретривер: без реранкера, только базовый поиск")
        if timing:
            return TimedRetrievalOnlyRetriever(base_retriever=base_retriever)
        return base_retriever

    logger.info("Финальный ретривер: с реранкером (ContextualCompression)")
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
    *,
    log_rewrites: bool = False,
) -> Any:
    """Ветка history-aware при наличии истории; иначе прямой retriever + лог reformulation skipped.

    Шаги: собрать reformulate chain → RunnableBranch по пустой/непустой chat_history.
    """
    reformulate_chain = prompt | llm | StrOutputParser()

    def _log_skip_reformulation(x: dict) -> str:
        logger.info(
            "[TIMING] stage=reformulation elapsed=0.00s (skipped; empty chat_history)"
        )
        return x["input"]

    def _timed_reform_sync(
        x: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        t0 = time.perf_counter()
        try:
            out = reformulate_chain.invoke(x, config, **kwargs)
            if log_rewrites:
                _log_reformulation(x.get("input", ""), out)
            return out
        finally:
            logger.info(
                "[TIMING] stage=reformulation elapsed=%.2fs",
                time.perf_counter() - t0,
            )

    async def _timed_reform_async(
        x: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        t0 = time.perf_counter()
        try:
            out = await reformulate_chain.ainvoke(x, config, **kwargs)
            if log_rewrites:
                _log_reformulation(x.get("input", ""), out)
            return out
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


def _create_passthrough_query_retriever(retriever: BaseRetriever) -> Runnable:
    """Поиск по последнему `input` без LLM-переформулирования (история в ретривер не подмешивается)."""

    def _query_from_input(x: dict) -> str:
        return x["input"]

    return (RunnableLambda(_query_from_input) | retriever).with_config(
        run_name="chat_retriever_chain",
    )


def _create_passthrough_query_retriever_with_timing(retriever: BaseRetriever) -> Runnable:
    """То же, что passthrough retriever, с одной строкой [TIMING] вместо этапа reformulation."""

    def _skip_reformulation(x: dict) -> str:
        logger.info(
            "[TIMING] stage=reformulation elapsed=0.00s "
            "(skipped; reformulator disabled)",
        )
        return x["input"]

    return (RunnableLambda(_skip_reformulation) | retriever).with_config(
        run_name="chat_retriever_chain",
    )


def _log_reformulation(user_input: str, rewritten: str) -> None:
    """Превью оригинала и результата rewrite для отладки."""
    inp = user_input if len(user_input) <= 200 else user_input[:197] + "..."
    out = rewritten if len(rewritten) <= 200 else rewritten[:197] + "..."
    logger.info("[reformulation] input=%r rewritten=%r", inp, out)

# --- ЦЕПОЧКИ ---

def create_search_only_chain(config: dict, retriever: BaseRetriever) -> Runnable:
    """Цепочка только retrieval: синхронный invoke ретривера по строке запроса."""
    return RunnableLambda(lambda q: retriever.invoke(q))

def create_generation_chain(config: dict) -> Runnable:
    """Генерация ответа только по переданному context + question (без retrieval).

    Выбирает LLM через env LLM_PROVIDER и блок providers в config.
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
    """Собирает conversational RAG: history-aware retrieve → stuff documents → ответ.

    Шаги: (1) LLM для reformulation/answer из LLM_PROVIDER + config.providers;
    (2) history-aware или ветка без reformulation при пустой истории;
    (3) обёртка RunnableWithMessageHistory с историей из БД через get_session_history.
    """
    logger.info(
        "Сборка RAG-цепочки: LLM_PROVIDER=%s",
        os.getenv("LLM_PROVIDER", "ollama"),
    )
    provider_name = os.getenv("LLM_PROVIDER", "ollama")
    provider_config = config.get('providers', {}).get(provider_name)
    llm = get_llm_from_config(provider_config)

    rag_cfg = config.get("rag_pipeline") or {}
    reformulator_cfg = rag_cfg.get("reformulator") or {}
    reformulation_on = bool(reformulator_cfg.get("enabled", False))
    reformulator_llm = llm
    if reformulation_on:
        r_llm_cfg = reformulator_cfg.get("llm")
        if isinstance(r_llm_cfg, dict) and r_llm_cfg.get("model"):
            reformulator_llm = get_llm_from_config(r_llm_cfg)
            logger.info(
                "Reformulator LLM: model=%s temperature=%s",
                r_llm_cfg.get("model"),
                r_llm_cfg.get("temperature"),
            )
        else:
            logger.warning(
                "rag_pipeline.reformulator.enabled но llm не задан — используется основная LLM",
            )
    log_rewrites = bool(reformulator_cfg.get("log_rewrites", False))

    timing = stage_timing_logs_enabled(config)

    # 1. Ретривер: при enabled — history-aware LLM-переформулирование; иначе только последний input
    if reformulation_on:
        if timing:
            history_aware_retriever = _create_history_aware_retriever_with_timing(
                reformulator_llm,
                retriever,
                contextualize_q_prompt,
                log_rewrites=log_rewrites,
            )
        else:
            history_aware_retriever = create_history_aware_retriever(
                reformulator_llm, retriever, contextualize_q_prompt
            )
    else:
        logger.info(
            "RAG: reformulator выключен (rag_pipeline.reformulator.enabled=false) — "
            "retrieval без LLM-переформулирования",
        )
        if timing:
            history_aware_retriever = _create_passthrough_query_retriever_with_timing(
                retriever,
            )
        else:
            history_aware_retriever = _create_passthrough_query_retriever(retriever)

    # 2. Цепочка ответов (генерирует ответ по найденным документам)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Общая RAG-цепочка (Поиск + Ответ)
    if timing:
        def _gen_sync(
            inputs: dict,
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
        ) -> str:
            t0 = time.perf_counter()
            try:
                return question_answer_chain.invoke(inputs, config, **kwargs)
            finally:
                logger.info(
                    "[TIMING] stage=generation elapsed=%.2fs",
                    time.perf_counter() - t0,
                )

        async def _gen_async(
            inputs: dict,
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
        ) -> str:
            t0 = time.perf_counter()
            try:
                return await question_answer_chain.ainvoke(inputs, config, **kwargs)
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

    logger.info("RAG-цепочка готова: RunnableWithMessageHistory подключён")
    return conversational_rag_chain


def _inject_non_rag_system_prompt(input_dict: dict) -> dict:
  """Убирает служебный non_rag_label, подставляет system_prompt для chat-only ветки."""
  label = input_dict.get("non_rag_label") or "smalltalk"
  system = (
      CHAT_ONLY_DIRECT_SYSTEM if label == "direct_link" else CHAT_ONLY_SMALLTALK_SYSTEM
  )
  out = {k: v for k, v in input_dict.items() if k != "non_rag_label"}
  out["system_prompt"] = system
  return out


def create_chat_only_chain(
    config: dict,
    answer_repo: Any,
    session_repo: Any = None,
    summarizer: Any = None,
):
  """Диалог с тем же PostgresHistory/summary, что и RAG, но без retrieval по документам."""
  provider_name = os.getenv("LLM_PROVIDER", "ollama")
  provider_config = config.get('providers', {}).get(provider_name)
  llm = get_llm_from_config(provider_config)

  chat_prompt = ChatPromptTemplate.from_messages([
      ("system", "{system_prompt}"),
      MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
  ])
  llm_branch = chat_prompt | llm | StrOutputParser()
  core = (
      RunnableLambda(_inject_non_rag_system_prompt)
      | RunnablePassthrough.assign(answer=llm_branch)
  )

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

  return RunnableWithMessageHistory(
      core,
      get_session_history,
      input_messages_key="input",
      history_messages_key="chat_history",
      output_messages_key="answer",
  )
