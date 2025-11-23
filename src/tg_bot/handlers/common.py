from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message
from langchain_core.runnables import Runnable

from src.tg_bot.services.interfaces import IUserService, IAnswerService

common_router = Router()

@common_router.message(CommandStart())
async def start_handler(message: Message, user_service: IUserService):
    await user_service.get_or_create_user(
        user_id=message.from_user.id,
        first_name=message.from_user.first_name,
        username=message.from_user.username
    )
    await message.answer(
        f"Привет, {message.from_user.first_name}!\n\n"
        "Я RAG-бот, готовый отвечать на вопросы о факультете ФПМИ. "
        "Просто задай мне свой вопрос."
    )

@common_router.message()
async def question_handler(
    message: Message,
    user_service: IUserService,
    answer_service: IAnswerService,
    rag_chain: Runnable
):
    # Убедимся, что пользователь есть в базе
    await user_service.get_or_create_user(
        user_id=message.from_user.id,
        first_name=message.from_user.first_name,
        username=message.from_user.username
    )

    # Отправляем "печатает..."
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    # Получаем ответ от RAG-цепочки (она уже инициализирована с конфигом)
    # invoke блокирующий, но aiogram запускает хендлеры в тредах, так что для MVP ок.
    # В идеале rag_chain.ainvoke, если поддерживается.
    try:
        bot_answer = await rag_chain.ainvoke(message.text)
    except AttributeError:
        # Если цепочка не поддерживает асинхронность нативно
        bot_answer = rag_chain.invoke(message.text)

    # Отправляем ответ пользователю
    await message.answer(bot_answer)

    # Сохраняем вопрос и ответ в БД
    await answer_service.save_answer(
        user_id=message.from_user.id,
        question=message.text,
        bot_answer=bot_answer
    )