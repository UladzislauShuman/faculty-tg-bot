from typing import List
from src.tg_bot.repositories.interfaces import IUserRepository, IAnswerRepository
from .interfaces import IUserService, IAnswerService
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

class AnswerService(IAnswerService):
    def __init__(self, answer_repo: IAnswerRepository):
        self._repo = answer_repo

    async def save_answer(self, user_id: int, question: str, bot_answer: str):
        await self._repo.create(user_id, question, bot_answer)