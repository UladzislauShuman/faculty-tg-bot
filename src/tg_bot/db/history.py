from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.tg_bot.repositories.interfaces import IAnswerRepository

class ReadOnlyPostgresHistory(BaseChatMessageHistory):
    """
    Адаптер истории для LangChain.
    Он только читает историю из БД. Сохранение мы контролируем вручную в хендлерах бота.
    """
    def __init__(self, session_id: str, answer_repo: IAnswerRepository, limit: int = 5):
        self.session_id = session_id
        self._repo = answer_repo
        self.limit = limit

    @property
    def messages(self) -> List[BaseMessage]:
        raise NotImplementedError("Синхронное чтение не поддерживается. Используется aget_messages().")

    async def aget_messages(self) -> List[BaseMessage]:
        """LangChain вызывает этот метод перед генерацией ответа, чтобы получить контекст."""
        answers = await self._repo.get_session_answers(self.session_id, self.limit)
        chat_messages =[]
        for ans in answers:
            chat_messages.append(HumanMessage(content=ans.question))
            chat_messages.append(AIMessage(content=ans.bot_answer))
        return chat_messages

    def add_message(self, message: BaseMessage) -> None:
        pass

    async def aadd_messages(self, messages: List[BaseMessage]) -> None:
        """Заглушка. Мы сохраняем сообщения сами в tg_bot/handlers/common.py"""
        pass

    def clear(self) -> None:
        pass