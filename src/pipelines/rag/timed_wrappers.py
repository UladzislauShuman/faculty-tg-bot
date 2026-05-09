"""Sprint 4: per-stage timing for RAG (retrieval vs rerank vs generation, reformulation)."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, cast

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

logger = logging.getLogger(__name__)


def stage_timing_logs_enabled(config: Optional[dict]) -> bool:
    if not config:
        return True
    sec = config.get("rag_pipeline") or {}
    timing = sec.get("stage_timing_logs") or {}
    return bool(timing.get("enabled", True))


class TimedContextualCompressionRetriever(ContextualCompressionRetriever):
    """Logs ``retrieval`` (base retriever) and ``reranker`` (compressor) durations."""

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        t0 = time.perf_counter()
        docs = await self.base_retriever.ainvoke(
            query, config={"callbacks": run_manager.get_child()}, **kwargs
        )
        logger.info(
            "[TIMING] stage=retrieval elapsed=%.2fs",
            time.perf_counter() - t0,
        )
        if not docs:
            return []
        t1 = time.perf_counter()
        compressed_docs = await self.base_compressor.acompress_documents(
            docs, query, callbacks=run_manager.get_child()
        )
        logger.info(
            "[TIMING] stage=reranker elapsed=%.2fs",
            time.perf_counter() - t1,
        )
        return list(compressed_docs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        t0 = time.perf_counter()
        docs = self.base_retriever.invoke(
            query, config={"callbacks": run_manager.get_child()}, **kwargs
        )
        logger.info(
            "[TIMING] stage=retrieval elapsed=%.2fs",
            time.perf_counter() - t0,
        )
        if not docs:
            return []
        t1 = time.perf_counter()
        compressed_docs = self.base_compressor.compress_documents(
            docs, query, callbacks=run_manager.get_child()
        )
        logger.info(
            "[TIMING] stage=reranker elapsed=%.2fs",
            time.perf_counter() - t1,
        )
        return list(compressed_docs)


class TimedRetrievalOnlyRetriever(BaseRetriever):
    """When reranker is off: log a single ``retrieval`` stage around the base retriever."""

    base_retriever: BaseRetriever
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        t0 = time.perf_counter()
        docs = await self.base_retriever.ainvoke(
            query,
            config={"callbacks": run_manager.get_child()},
            **kwargs,
        )
        logger.info(
            "[TIMING] stage=retrieval elapsed=%.2fs",
            time.perf_counter() - t0,
        )
        return cast(list[Document], docs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        t0 = time.perf_counter()
        docs = self.base_retriever.invoke(
            query,
            config={"callbacks": run_manager.get_child()},
            **kwargs,
        )
        logger.info(
            "[TIMING] stage=retrieval elapsed=%.2fs",
            time.perf_counter() - t0,
        )
        return docs
