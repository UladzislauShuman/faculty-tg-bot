import datetime
from sqlalchemy import BigInteger, String, TIMESTAMP, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.tg_bot.db.base import Base
import typing

if typing.TYPE_CHECKING:
    from .user import User

class Answer(Base):
    __tablename__ = "rag_bot_answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("rag_bot_users.id"))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    bot_answer: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP, default=datetime.datetime.utcnow
    )

    user: Mapped["User"] = relationship(back_populates="answers")