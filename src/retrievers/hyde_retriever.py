"""HyDE (Hypothetical Document Embeddings) — обёртка над Embeddings для dense-поиска.

LLM строит гипотетический фрагмент ответа, он эмбеддится через inner; события пишутся в
``hyde_trace_append`` для evaluation/trace.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent import futures
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from src.retrievers.hyde_trace_context import hyde_trace_append

from src.config.prompts import HYDE_PROMPT

logger = logging.getLogger(__name__)

_PREVIEW_Q = 300
_PREVIEW_H = 900

_HYDE_PROMPT_TEMPLATE = PromptTemplate(
    template=HYDE_PROMPT,
    input_variables=["question"],
)


def _coerce_llm_text(result: Any) -> str:
    """Текст из ответа LLM: str, атрибут content или str(result)."""
    if isinstance(result, str):
        return result.strip()
    content = getattr(result, "content", None)
    if content is not None:
        return str(content).strip()
    return str(result).strip()


def _mean_l2_normalize(vectors: List[List[float]]) -> List[float]:
    """Покомпонентное среднее векторов и L2-нормировка (несколько гипотез HyDE)."""
    if not vectors:
        return []
    if len(vectors) == 1:
        return list(vectors[0])
    arr = np.asarray(vectors, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.astype(np.float64).tolist()


class HyDEQueryEmbeddings(Embeddings):
    """
    Для запросов: LLM → гипотетический фрагмент → embed_query(inner).
    Для документов: только inner (без LLM).
    """

    def __init__(
        self,
        inner: Embeddings,
        llm: BaseLanguageModel,
        *,
        timeout_seconds: float = 8.0,
        num_hypotheses: int = 1,
        verbose_console: bool = False,
    ) -> None:
        self._inner = inner
        self._llm = llm
        self._timeout_seconds = max(0.001, float(timeout_seconds))
        self._num_hypotheses = max(1, int(num_hypotheses))
        self._verbose_console = bool(verbose_console)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Индексация документов: тот же inner, без гипотез."""
        return self._inner.embed_documents(texts)

    def _emit_sink(
        self,
        *,
        user_query_preview: str,
        status: str,
        hypotheses_raw: Sequence[str],
        elapsed_hypothesis_s: float,
        embed_phase_note: Optional[str],
        hypo_notes: Sequence[str],
    ) -> None:
        """Собрать событие HyDE (превью гипотез) и добавить в trace; при verbose — лог."""
        hypo_store: List[str] = []
        for h in hypotheses_raw:
            if len(h) <= _PREVIEW_H:
                hypo_store.append(h)
            else:
                hypo_store.append(h[: _PREVIEW_H - 3] + "...")
        evt: Dict[str, Any] = {
            "user_query_preview": user_query_preview,
            "status": status,
            "hypotheses": hypo_store,
            "num_hypotheses_config": self._num_hypotheses,
            "elapsed_hypothesis_llm_s": round(elapsed_hypothesis_s, 3),
            "hypothesis_issues": list(hypo_notes),
            "embed_phase_note": embed_phase_note,
        }
        hyde_trace_append(evt)

        if self._verbose_console:
            logger.info(
                "[HyDE] status=%s hypos=%s llm_elapsed=%.2fs preview=%s",
                status,
                len(hypo_store),
                elapsed_hypothesis_s,
                user_query_preview[:120].replace("\n", " "),
            )

    def embed_query(self, text: str) -> List[float]:
        """Эмбеддинг запроса: гипотезы → inner.embed_query; при ошибке/пустоте — исходный текст."""
        hypo_notes: List[str] = []
        hypo_elapsed_total = 0.0

        preview = (
            text if len(text) <= _PREVIEW_Q else text[: _PREVIEW_Q - 3] + "..."
        )

        try:
            t_hyp0 = time.perf_counter()
            hypotheses, hypo_notes = self._generate_hypotheses_traced(text)
            hypo_elapsed_total = time.perf_counter() - t_hyp0

            if not hypotheses:
                logger.info("HyDE fallback to base embedding — no hypothesis")
                self._emit_sink(
                    user_query_preview=preview,
                    status="fallback_no_hypothesis",
                    hypotheses_raw=[],
                    elapsed_hypothesis_s=hypo_elapsed_total,
                    embed_phase_note=None,
                    hypo_notes=hypo_notes or ["empty_or_all_failed"],
                )
                return self._inner.embed_query(text)

            embeddings = [
                self._inner.embed_query(hyp)
                for hyp in hypotheses
                if hyp.strip()
            ]
            if not embeddings:
                logger.info("HyDE fallback to base embedding — empty embeddings")
                self._emit_sink(
                    user_query_preview=preview,
                    status="fallback_inner_embed_failed",
                    hypotheses_raw=hypotheses,
                    elapsed_hypothesis_s=hypo_elapsed_total,
                    embed_phase_note="inner.embed_query yielded no vectors",
                    hypo_notes=hypo_notes,
                )
                return self._inner.embed_query(text)

            self._emit_sink(
                user_query_preview=preview,
                status="ok",
                hypotheses_raw=hypotheses,
                elapsed_hypothesis_s=hypo_elapsed_total,
                embed_phase_note="hypothetical text embedded for dense retrieval",
                hypo_notes=hypo_notes,
            )
            return _mean_l2_normalize(embeddings)

        except Exception as exc:
            logger.exception("HyDE fallback to base embedding — exception")
            self._emit_sink(
                user_query_preview=preview,
                status="fallback_exception",
                hypotheses_raw=[],
                elapsed_hypothesis_s=hypo_elapsed_total,
                embed_phase_note=None,
                hypo_notes=list(hypo_notes) + [f"{type(exc).__name__}: {exc}"],
            )
            return self._inner.embed_query(text)

    async def aembed_query(self, text: str) -> List[float]:
        """Асинхронная обёртка над ``embed_query`` (выполнение в thread pool)."""
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Асинхронная обёртка над ``embed_documents``."""
        return await asyncio.to_thread(self.embed_documents, texts)

    def _generate_hypotheses_traced(
        self,
        question: str,
    ) -> Tuple[List[str], List[str]]:
        """Несколько независимых слотов гипотез; заметки с префиксами h1:, h2:, …."""
        out: List[str] = []
        notes: List[str] = []
        for hi in range(self._num_hypotheses):
            hyp, note = self._one_hypothesis_with_timeout(question, slot=hi)
            if hyp:
                out.append(hyp)
            if note:
                notes.append(f"h{hi + 1}:{note}")
        return out, notes

    def _one_hypothesis_with_timeout(
        self,
        question: str,
        *,
        slot: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Один вызов LLM в отдельном потоке с таймаутом; возвращает (текст, заметка_ошибки)."""
        prompt_text = _HYDE_PROMPT_TEMPLATE.format(question=question)

        def _invoke() -> Tuple[str, Optional[str]]:
            raw = self._llm.invoke(prompt_text)
            coerced = _coerce_llm_text(raw)
            if not coerced:
                return "", "empty_llm_body"
            return coerced, None

        try:
            with futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke)
                try:
                    body, invoke_note = future.result(timeout=self._timeout_seconds)
                    if invoke_note == "empty_llm_body":
                        return None, "empty_llm_body"
                    return body, None
                except futures.TimeoutError:
                    logger.warning(
                        "HyDE LLM timed out after %.2fs (slot=%s)",
                        self._timeout_seconds,
                        slot + 1,
                    )
                    return None, "timeout"
        except Exception as exc:
            logger.exception(
                "HyDE LLM invocation error (slot=%s)", slot + 1,
            )
            return None, f"{type(exc).__name__}: {exc!s}"
