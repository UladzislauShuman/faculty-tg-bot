import inspect
from contextlib import asynccontextmanager
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from src.tg_bot.db.orm import asession


@asynccontextmanager
async def get_session(provided_session: Optional[AsyncSession] = None):
  if provided_session:
    yield provided_session
  else:
    async with asession.begin() as new_session:
      yield new_session


# use for python earlier than 3.10
if not hasattr(inspect, 'asyncgen'):
  inspect.asyncgen = inspect.isasyncgen


