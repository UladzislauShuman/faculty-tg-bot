from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, cast

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import patch_config


class AsyncEnsembleRetriever(EnsembleRetriever):
    """EnsembleRetriever with parallel sync ``rank_fusion`` (invoke path)."""

    def rank_fusion(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        def _run_one(idx: int) -> List[Document]:
            r = self.retrievers[idx]
            docs = r.invoke(
                query,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(tag=f"retriever_{idx + 1}"),
                ),
            )
            return [
                Document(page_content=cast(str, doc)) if isinstance(doc, str) else doc
                for doc in docs
            ]

        max_workers = max(1, len(self.retrievers))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            retriever_docs = list(pool.map(_run_one, range(len(self.retrievers))))

        return self.weighted_reciprocal_rank(retriever_docs)
