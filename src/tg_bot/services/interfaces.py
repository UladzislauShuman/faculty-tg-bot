from abc import abstractmethod
from typing import Protocol, List
from src.tg_bot.models import User, UserSession

class IUserService(Protocol):
    @abstractmethod
    async def get_or_create_user(self, user_id: int, first_name: str, username: str | None) -> User: ...
    @abstractmethod
    async def get_all_user_ids(self) -> List[int]: ...


class ISessionService(Protocol):
  @abstractmethod
  async def get_or_create_active_session(self, user_id: int) -> str: ...

  @abstractmethod
  async def start_new_session(self, user_id: int) -> str: ...


class IAnswerService(Protocol):
  @abstractmethod
  async def save_answer(self, session_id: str, question: str,
      bot_answer: str): ...