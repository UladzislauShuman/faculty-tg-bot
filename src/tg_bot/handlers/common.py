from aiogram import Router
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from langchain_core.runnables import Runnable

from src.util.callbacks import ProfilingCallbackHandler
from src.tg_bot.services.interfaces import IUserService, IAnswerService, ISessionService

common_router = Router()

@common_router.message(CommandStart())
async def start_handler(message: Message, user_service: IUserService, session_service: ISessionService):
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
    # Закрываем старую сессию и начинаем новую
    await session_service.start_new_session(message.from_user.id)
    await message.answer("🔄 Контекст очищен! Начат новый диалог. О чем хочешь спросить?")

@common_router.message()
async def question_handler(
    message: Message,
    user_service: IUserService,
    answer_service: IAnswerService,
    session_service: ISessionService,
    rag_chain: Runnable
):
    # Убедимся, что пользователь есть в базе
    await user_service.get_or_create_user(
        user_id=message.from_user.id,
        first_name=message.from_user.first_name,
        username=message.from_user.username
    )

    # 1. Получаем ID активной сессии для этого пользователя
    session_id = await session_service.get_or_create_active_session(message.from_user.id)

    # Отправляем "печатает..."
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    # Получаем ответ от RAG-цепочки (она уже инициализирована с конфигом)
    # invoke блокирующий, но aiogram запускает хендлеры в тредах, так что для MVP ок.
    # В идеале rag_chain.ainvoke, если поддерживается.
    profiler = ProfilingCallbackHandler()

    # 2. Формируем конфиг для LangChain: передаем session_id и коллбеки
    chain_config = {
      "configurable": {"session_id": session_id},
      "callbacks": [profiler]
    }

    # 3. Вызываем умную цепочку (передаем словарь с ключом "input", как мы указали в pipeline.py)
    try:
      response = await rag_chain.ainvoke(
          {"input": message.text},
          config=chain_config
      )
      bot_answer = response[
        "answer"]
    except AttributeError:
      response = rag_chain.invoke(
          {"input": message.text},
          config=chain_config
      )
      bot_answer = response["answer"]

    # 4. Отправляем ответ пользователю
    await message.answer(bot_answer)

    # 5. Сохраняем вопрос и ответ в БД, привязывая к ТЕКУЩЕЙ СЕССИИ
    await answer_service.save_answer(
        session_id=session_id,
        question=message.text,
        bot_answer=bot_answer
    )