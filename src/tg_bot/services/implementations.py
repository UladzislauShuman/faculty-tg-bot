from typing import List
from src.tg_bot.repositories.interfaces import IUserRepository, IAnswerRepository, ISessionRepository
from .interfaces import IUserService, IAnswerService, ISessionService
from src.tg_bot.models import User

class UserService(IUserService):
    def __init__(self, user_repo: IUserRepository):
        self._repo = user_repo

    async def get_or_create_user(self, user_id: int, first_name: str, username: str | None) -> User:
        defaults = {"first_name": first_name, "username": username}
        return await self._repo.get_or_create(user_id, defaults)

    async def get_all_user_ids(self) -> List[int]:
        users = await self._repo.get_all_users()
        return [user.id for user in users]


class SessionService(ISessionService):
  def __init__(self, session_repo: ISessionRepository):
    self._repo = session_repo

  async def get_or_create_active_session(self, user_id: int) -> str:
    """Возвращает ID активной сессии. Если ее нет - создает новую."""
    session = await self._repo.get_active_session(user_id)
    if not session:
      session = await self._repo.create_session(user_id)
    return session.id

  async def start_new_session(self, user_id: int) -> str:
    """Закрывает старую сессию и начинает новую (для команды /newchat)."""
    await self._repo.close_active_session(user_id)
    new_session = await self._repo.create_session(user_id)
    return new_session.id


class AnswerService(IAnswerService):
  def __init__(self, answer_repo: IAnswerRepository):
    self._repo = answer_repo

  async def save_answer(self, session_id: str, question: str, bot_answer: str):
    await self._repo.create(session_id, question, bot_answer)