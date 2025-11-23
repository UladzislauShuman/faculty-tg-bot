from abc import abstractmethod
from typing import Protocol, List
from src.tg_bot.models import User, Answer

class IUserRepository(Protocol):
    @abstractmethod
    async def get_or_create(self, user_id: int, defaults: dict) -> User: ...
    @abstractmethod
    async def get_all_users(self) -> List[User]: ...

class IAnswerRepository(Protocol):
    @abstractmethod
    async def create(self, user_id: int, question: str, bot_answer: str) -> Answer: ...