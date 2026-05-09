"""Контекст для связки событий HyDE с трассировкой evaluation (контур ContextVar для async/to_thread)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Dict, List, Optional

_sink: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "hyde_trace_sink",
    default=None,
)


def hyde_trace_begin() -> Token:
    """Начало одного пользовательского «запроса» (один retrieval / один rag invocation)."""
    buf: List[Dict[str, Any]] = []
    return _sink.set(buf)


def hyde_trace_end(token: Token) -> List[Dict[str, Any]]:
    """Читает и сбрасывает буфер; возвращает копию списка событий."""
    buf = _sink.get()
    try:
        return list(buf) if buf else []
    finally:
        _sink.reset(token)


def hyde_trace_append(event: Dict[str, Any]) -> None:
    """Дописать событие (если нет активного буфера — no-op)."""
    buf = _sink.get()
    if buf is not None:
        buf.append(event)

