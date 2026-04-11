import datetime
import uuid
from sqlalchemy import BigInteger, String, TIMESTAMP, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.tg_bot.db.base import Base
import typing

if typing.TYPE_CHECKING:
  from .user import User
  from .answer import Answer


class UserSession(Base):
  __tablename__ = "rag_bot_sessions"

  id: Mapped[str] = mapped_column(String, primary_key=True,
                                  default=lambda: str(uuid.uuid4()))
  user_id: Mapped[int] = mapped_column(BigInteger,
                                       ForeignKey("rag_bot_users.id"))

  is_active: Mapped[bool] = mapped_column(Boolean, default=True)

  created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP,
                                                        default=datetime.datetime.utcnow)

  user: Mapped["User"] = relationship(back_populates="sessions")
  answers: Mapped[list["Answer"]] = relationship(back_populates="session")