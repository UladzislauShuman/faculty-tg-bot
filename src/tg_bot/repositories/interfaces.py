from abc import abstractmethod
from typing import Protocol, List
from src.tg_bot.models import User, Answer, UserSession

class IUserRepository(Protocol):
    @abstractmethod
    async def get_or_create(self, user_id: int, defaults: dict) -> User: ...
    @abstractmethod
    async def get_all_users(self) -> List[User]: ...


class ISessionRepository(Protocol):
  @abstractmethod
  async def get_active_session(self, user_id: int) -> UserSession | None: ...

  @abstractmethod
  async def create_session(self, user_id: int) -> UserSession: ...

  @abstractmethod
  async def close_active_session(self, user_id: int) -> None: ...

class IAnswerRepository(Protocol):
  @abstractmethod
  async def create(self, session_id: str, question: str,
      bot_answer: str) -> Answer: ...

  @abstractmethod
  async def get_session_answers(self, session_id: str, limit: int = 5) -> List[
    Answer]: ...