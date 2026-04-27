from typing import List, Optional
from sqlalchemy import update, desc
from sqlalchemy.future import select
from src.tg_bot.db import asession
from src.tg_bot.models import User, Answer, UserSession
from .interfaces import IUserRepository, IAnswerRepository, ISessionRepository


class SessionRepository(ISessionRepository):
  async def get_active_session(self, user_id: int) -> UserSession | None:
    async with asession() as session:
      stmt = select(UserSession).where(
          UserSession.user_id == user_id,
          UserSession.is_active == True
      )
      result = await session.execute(stmt)
      return result.scalar_one_or_none()

  async def create_session(self, user_id: int) -> UserSession:
    async with asession.begin() as session:
      new_session = UserSession(user_id=user_id, is_active=True)
      session.add(new_session)
      return new_session

  async def close_active_session(self, user_id: int) -> None:
    async with asession.begin() as session:
      stmt = update(UserSession).where(
          UserSession.user_id == user_id,
          UserSession.is_active == True
      ).values(is_active=False)
      await session.execute(stmt)

  async def get_summary(self, session_id: str) -> Optional[str]:
    async with asession() as session:
      stmt = select(UserSession).where(UserSession.id == session_id)
      result = await session.execute(stmt)
      row = result.scalar_one_or_none()
      return row.summary if row else None

  async def update_summary(self, session_id: str, summary: str) -> None:
    async with asession.begin() as session:
      stmt = (
          update(UserSession)
          .where(UserSession.id == session_id)
          .values(summary=summary)
      )
      await session.execute(stmt)

class UserRepository(IUserRepository):
    async def get_or_create(self, user_id: int, defaults: dict) -> User:
        async with asession.begin() as session:
            stmt = select(User).where(User.id == user_id)
            result = await session.execute(stmt)
            instance = result.scalar_one_or_none()
            if instance:
                return instance
            else:
                instance = User(id=user_id, **defaults)
                session.add(instance)
                return instance

    async def get_all_users(self) -> List[User]:
        async with asession() as session:
            stmt = select(User)
            result = await session.execute(stmt)
            return result.scalars().all()

class AnswerRepository(IAnswerRepository):
  async def create(self, session_id: str, question: str,
      bot_answer: str) -> Answer:
    async with asession.begin() as session:
      answer = Answer(session_id=session_id, question=question,
                      bot_answer=bot_answer)
      session.add(answer)
      return answer

  async def get_session_answers(self, session_id: str, limit: int = 5) -> List[
    Answer]:
    async with asession() as session:
      stmt = select(Answer).where(Answer.session_id == session_id).order_by(
        desc(Answer.created_at)).limit(limit)
      result = await session.execute(stmt)
      answers = result.scalars().all()
      return list(reversed(answers))