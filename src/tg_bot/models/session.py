import datetime
import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, Boolean, ForeignKey, String, Text, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.tg_bot.db.base import Base

if TYPE_CHECKING:
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

  summary: Mapped[Optional[str]] = mapped_column(
      Text, nullable=True, default=None
  )

  user: Mapped["User"] = relationship(back_populates="sessions")
  answers: Mapped[list["Answer"]] = relationship(back_populates="session")