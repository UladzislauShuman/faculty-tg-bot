from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
)
from src.tg_bot.repositories.interfaces import IAnswerRepository

if TYPE_CHECKING:
  from src.tg_bot.repositories.interfaces import ISessionRepository
  from src.tg_bot.services.summarizer import SummarizerService


class ReadOnlyPostgresHistory(BaseChatMessageHistory):
  """
  Адаптер истории для LangChain.
  Только чтение из БД; summary кешируется в rag_bot_sessions.summary (Sprint 3).
  """
  def __init__(
    self,
    session_id: str,
    answer_repo: IAnswerRepository,
    session_repo: Optional["ISessionRepository"] = None,
    summarizer: Optional["SummarizerService"] = None,
    window_size: int = 5,
    summarization_threshold: int = 4,
    memory_enabled: bool = False,
  ) -> None:
    self.session_id = session_id
    self._repo = answer_repo
    self._session_repo = session_repo
    self._summarizer = summarizer
    self._window_size = window_size
    self._summarization_threshold = summarization_threshold
    self._memory_enabled = memory_enabled

  @property
  def messages(self) -> List[BaseMessage]:
    raise NotImplementedError(
        "Синхронное чтение не поддерживается. Используется aget_messages()."
    )

  @staticmethod
  def _qa_pairs_to_messages(answers) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for ans in answers:
      out.append(HumanMessage(content=ans.question))
      out.append(AIMessage(content=ans.bot_answer))
    return out

  async def aget_messages(self) -> List[BaseMessage]:
    """Контекст для RAG: при memory.enabled — summary + последнее окно Q/A."""
    memory_on = (
        self._memory_enabled
        and self._session_repo is not None
        and self._summarizer is not None
    )

    if not memory_on:
      answers = await self._repo.get_session_answers(
          self.session_id, self._window_size
      )
      return self._qa_pairs_to_messages(answers)

    all_answers = await self._repo.get_session_answers(
        self.session_id, limit=100
    )

    if len(all_answers) < self._summarization_threshold:
      recent = all_answers[-self._window_size :]
      return self._qa_pairs_to_messages(recent)

    messages: List[BaseMessage] = []
    cached_summary = await self._session_repo.get_summary(self.session_id)
    if not cached_summary:
      raw_msgs: List[BaseMessage] = []
      for ans in all_answers:
        raw_msgs.append(HumanMessage(content=ans.question))
        raw_msgs.append(AIMessage(content=ans.bot_answer))
      cached_summary = await self._summarizer.summarize(raw_msgs)
      if cached_summary:
        await self._session_repo.update_summary(self.session_id, cached_summary)

    if cached_summary:
      messages.append(
          SystemMessage(
              content=f"Краткое содержание диалога: {cached_summary}"
          )
      )

    recent_answers = all_answers[-self._window_size :]
    for ans in recent_answers:
      messages.append(HumanMessage(content=ans.question))
      messages.append(AIMessage(content=ans.bot_answer))
    return messages

  def add_message(self, message: BaseMessage) -> None:
    pass

  async def aadd_messages(self, messages: List[BaseMessage]) -> None:
    """Заглушка. Сохранение — в хендлерах бота."""
    pass

  def clear(self) -> None:
    pass
