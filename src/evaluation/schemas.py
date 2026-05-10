"""Pydantic-схемы для ответа LLM-judge в evaluation."""

from pydantic import BaseModel, Field


class EvalScore(BaseModel):
  """Оценка судьи: число 0–1 и краткое обоснование на русском."""

  score: float = Field(
      ge=0.0,
      le=1.0,
      description="Оценка качества от 0.0 до 1.0",
  )
  reason: str = Field(
      description="Краткое обоснование оценки на русском",
  )
