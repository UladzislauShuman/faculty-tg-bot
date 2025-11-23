from sqlalchemy import BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.tg_bot.db import Base
import typing

if typing.TYPE_CHECKING:
    from .answer import Answer

class User(Base):
    __tablename__ = "rag_bot_users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    first_name: Mapped[str] = mapped_column(String, nullable=False)
    username: Mapped[str | None] = mapped_column(String, nullable=True)

    answers: Mapped[list["Answer"]] = relationship(back_populates="user")