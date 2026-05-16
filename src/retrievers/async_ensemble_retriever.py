"""Обертка EnsembleRetriever: параллельный sync invoke двух ретриверов и RRF."""
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

    def weighted_reciprocal_rank(self, doc_lists: List[List[Document]]) -> List[Document]:
        """
        Слияние результатов по алгоритму RRF.
        Переопределено для дедупликации по chunk_id (если есть) или по хешу контента,
        чтобы избежать дубликатов из-за мелких различий в тексте (например, префиксов).
        """
        c = 60
        rrf_score: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                # Используем chunk_id для надежной дедупликации
                doc_key = doc.metadata.get("chunk_id")
                if not doc_key:
                    doc_key = str(hash(doc.page_content))
                
                if doc_key not in rrf_score:
                    rrf_score[doc_key] = 0.0
                    doc_map[doc_key] = doc
                
                rrf_score[doc_key] += weight / (rank + c)

        # Сортируем по убыванию RRF score
        sorted_keys = sorted(rrf_score.keys(), key=lambda k: rrf_score[k], reverse=True)
        return [doc_map[k] for k in sorted_keys]
