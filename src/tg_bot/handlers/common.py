"""Хендлеры Aiogram: старт, сброс сессии, основной вопрос с роутингом и RAG.

Поток вопроса: user/session → typing → semantic_routing → либо быстрый ответ,
либо RAG (ainvoke, при AttributeError — sync invoke, блокирует event loop).
"""
import logging

from aiogram import Router
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from langchain_core.runnables import Runnable

from src.util.callbacks import ProfilingCallbackHandler
from src.tg_bot.services.interfaces import IUserService, IAnswerService, ISessionService
from src.pipelines.routing import SemanticRoutingPort

logger = logging.getLogger(__name__)

common_router = Router()


@common_router.message(CommandStart())
async def start_handler(
    message: Message, user_service: IUserService, session_service: ISessionService
):
  """Регистрация пользователя и активной сессии; приветствие."""
  await user_service.get_or_create_user(
      user_id=message.from_user.id,
      first_name=message.from_user.first_name,
      username=message.from_user.username
  )

  await session_service.get_or_create_active_session(message.from_user.id)

  await message.answer(
      f"Привет, {message.from_user.first_name}!\n\n"
      "Я RAG-бот, готовый отвечать на вопросы о факультете ФПМИ. "
      "Просто задай мне свой вопрос.\n\n"
      "🔄 Если хочешь начать диалог с чистого листа (сбросить контекст), нажми /newchat"
  )


@common_router.message(Command("newchat"))
async def newchat_handler(message: Message, session_service: ISessionService):
  """Закрывает активную сессию и создаёт новую (команда /newchat)."""
  await session_service.start_new_session(message.from_user.id)
  await message.answer("🔄 Контекст очищен! Начат новый диалог. О чем хочешь спросить?")


@common_router.message()
async def question_handler(
    message: Message,
    user_service: IUserService,
    answer_service: IAnswerService,
    session_service: ISessionService,
    rag_chain: Runnable,
    chat_only_chain: Runnable,
    semantic_routing_service: SemanticRoutingPort,
):
  """Обычное сообщение: роутинг → RAG или заготовка; ответ и запись в БД.

  Для не-текстовых апдейтов message.text может быть None — передаём в RAG пустую строку
  для роутера; сохраняем как есть (осторожно с NOT NULL в Answer).
  """
  text = message.text or ""
  uid = message.from_user.id

  await user_service.get_or_create_user(
      user_id=uid,
      first_name=message.from_user.first_name,
      username=message.from_user.username
  )

  session_id = await session_service.get_or_create_active_session(uid)
  logger.info(
      "Сообщение user_id=%s session_id=%s has_text=%s",
      uid,
      session_id,
      bool(message.text),
  )

  await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

  decision = await semantic_routing_service.route(text)
  if not decision.use_rag:
    if decision.use_llm_for_reply:
      profiler_nc = ProfilingCallbackHandler()
      chain_config_nc = {
          "configurable": {"session_id": session_id},
          "callbacks": [profiler_nc],
      }
      try:
        response_nc = await chat_only_chain.ainvoke(
            {
                "input": text,
                "non_rag_label": decision.non_rag_label or "smalltalk",
            },
            config=chain_config_nc,
        )
        bot_answer = response_nc["answer"]
      except AttributeError:
        response_nc = chat_only_chain.invoke(
            {
                "input": text,
                "non_rag_label": decision.non_rag_label or "smalltalk",
            },
            config=chain_config_nc,
        )
        bot_answer = response_nc["answer"]
      except Exception as exc:
        logger.warning(
            "chat_only_chain ошибка session_id=%s: %s — шаблон из конфига",
            session_id,
            exc,
        )
        bot_answer = (decision.answer or "").strip() or (
            "Задайте вопрос о факультете — я отвечу по базе знаний."
        )
    else:
      bot_answer = (decision.answer or "").strip() or (
          "Задайте вопрос о факультете — я отвечу по базе знаний."
      )
    logger.info(
        "Маршрут: без RAG (%s) session_id=%s llm=%s",
        decision.non_rag_label,
        session_id,
        decision.use_llm_for_reply,
    )
    await message.answer(bot_answer)
    await answer_service.save_answer(
        session_id=session_id,
        question=message.text,
        bot_answer=bot_answer,
    )
    return

  profiler = ProfilingCallbackHandler()

  chain_config = {
    "configurable": {"session_id": session_id},
    "callbacks": [profiler]
  }

  try:
    response = await rag_chain.ainvoke(
        {"input": text},
        config=chain_config
    )
    bot_answer = response["answer"]
    logger.info("RAG ответ получен (ainvoke) session_id=%s", session_id)
  except AttributeError:
    logger.warning(
        "rag_chain без ainvoke — fallback на sync invoke (блокирует event loop) session_id=%s",
        session_id,
    )
    response = rag_chain.invoke(
        {"input": text},
        config=chain_config
    )
    bot_answer = response["answer"]

  await message.answer(bot_answer)

  await answer_service.save_answer(
      session_id=session_id,
      question=message.text,
      bot_answer=bot_answer
  )
