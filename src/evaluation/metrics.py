import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from src.evaluation.prompts import FAITHFULNESS_SYSTEM_PROMPT, RELEVANCE_SYSTEM_PROMPT
from src.evaluation.schemas import EvalScore

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _eval_score_from_llm_string(raw: str) -> EvalScore:
    """
    Парсит JSON в EvalScore: сначала вся строка, затем извлечение из code fence и по скобкам.
    """
    s = (raw or "").strip()
    try:
        return EvalScore.model_validate_json(s)
    except (json.JSONDecodeError, ValidationError, ValueError):
        pass

    fence = _JSON_FENCE.search(s)
    if fence:
        inner = fence.group(1).strip()
        try:
            return EvalScore.model_validate_json(inner)
        except (json.JSONDecodeError, ValidationError, ValueError):
            s = inner

    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        snippet = s[start : end + 1]
        return EvalScore.model_validate_json(snippet)

    raise json.JSONDecodeError("No valid EvalScore JSON", raw or "", 0)


class FaithfulnessEvaluator:
    def __init__(self, llm: BaseLanguageModel, timeout: int = 60) -> None:
        self._chain = (
            PromptTemplate.from_template(FAITHFULNESS_SYSTEM_PROMPT)
            | llm
            | StrOutputParser()
        )
        self._timeout = timeout

    async def aevaluate(self, answer: str, context: str) -> EvalScore:
        """
        Оценивает, основан ли answer исключительно на фактах из context.
        Возвращает EvalScore(score=0.0, reason="...") при любой ошибке.
        """
        try:
            raw: Any = await asyncio.wait_for(
                self._chain.ainvoke({"context": context, "answer": answer}),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Faithfulness evaluation timeout after %s s", self._timeout
            )
            return EvalScore(
                score=0.0,
                reason="Timeout: LLM did not respond in time",
            )
        except Exception as e:  # noqa: BLE001 — контракт: не пробрасывать наружу
            logger.warning("Faithfulness evaluation failed: %s", e)
            return EvalScore(
                score=0.0,
                reason=f"Error: {type(e).__name__}: {e!s}"[:500],
            )

        raw_s = raw if isinstance(raw, str) else str(raw)
        try:
            return _eval_score_from_llm_string(raw_s)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            logger.warning(
                "Faithfulness parse error: %s; raw (truncated)=%r",
                e,
                (raw_s[:200] if raw_s else ""),
            )
            return EvalScore(
                score=0.0,
                reason=f"Parse error: {raw_s[:100]}",
            )


class RelevanceEvaluator:
    def __init__(self, llm: BaseLanguageModel, timeout: int = 60) -> None:
        self._chain = (
            PromptTemplate.from_template(RELEVANCE_SYSTEM_PROMPT)
            | llm
            | StrOutputParser()
        )
        self._timeout = timeout

    async def aevaluate(self, question: str, answer: str) -> EvalScore:
        """
        Оценивает, отвечает ли answer на question.
        Возвращает EvalScore(score=0.0, reason="...") при любой ошибке.
        """
        try:
            raw: Any = await asyncio.wait_for(
                self._chain.ainvoke({"question": question, "answer": answer}),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Relevance evaluation timeout after %s s", self._timeout
            )
            return EvalScore(
                score=0.0,
                reason="Timeout: LLM did not respond in time",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Relevance evaluation failed: %s", e)
            return EvalScore(
                score=0.0,
                reason=f"Error: {type(e).__name__}: {e!s}"[:500],
            )

        raw_s = raw if isinstance(raw, str) else str(raw)
        try:
            return _eval_score_from_llm_string(raw_s)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            logger.warning(
                "Relevance parse error: %s; raw (truncated)=%r",
                e,
                (raw_s[:200] if raw_s else ""),
            )
            return EvalScore(
                score=0.0,
                reason=f"Parse error: {raw_s[:100]}",
            )
