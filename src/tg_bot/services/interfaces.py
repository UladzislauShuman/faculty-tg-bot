from abc import abstractmethod
from typing import Protocol, List
from src.tg_bot.models import User

class IUserService(Protocol):
    @abstractmethod
    async def get_or_create_user(self, user_id: int, first_name: str, username: str | None) -> User: ...
    @abstractmethod
    async def get_all_user_ids(self) -> List[int]: ...

class IAnswerService(Protocol):
    @abstractmethod
    async def save_answer(self, user_id: int, question: str, bot_answer: str): ...