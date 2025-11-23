from typing import List
from sqlalchemy.future import select
from src.tg_bot.db import asession
from src.tg_bot.models import User, Answer
from .interfaces import IUserRepository, IAnswerRepository

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
    async def create(self, user_id: int, question: str, bot_answer: str) -> Answer:
        async with asession.begin() as session:
            answer = Answer(user_id=user_id, question=question, bot_answer=bot_answer)
            session.add(answer)
            return answer