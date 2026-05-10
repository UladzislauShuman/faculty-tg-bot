"""Parent document retrieval: поиск по дочерним чанкам, в ответ — родительский контекст.

Docstore — pickle {parent_id: Document}, путь из config.parent_document.docstore_path.
"""
import logging
import os
import pickle
from typing import Dict, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class ParentDocumentRetriever(BaseRetriever):
    """Обертка над base_retriever: подмена child → parent из docstore, дедуп по parent_id."""

    base_retriever: BaseRetriever
    docstore: Dict[str, Document]

    @classmethod
    def from_config(cls, base_retriever: BaseRetriever, docstore_path: str) -> "ParentDocumentRetriever":
        """Загружает pickle docstore; при ошибке или отсутствии файла — пустой docstore."""
        docstore = {}
        if os.path.exists(docstore_path):
            try:
                with open(docstore_path, "rb") as f:
                    docstore = pickle.load(f)
                logger.info("Parent docstore загружен: %s записей из %s", len(docstore), docstore_path)
            except Exception as e:
                logger.exception("Не удалось прочитать parent docstore %s: %s", docstore_path, e)
        else:
            logger.warning("Parent docstore не найден: %s (будет fallback на child)", docstore_path)

        return cls(base_retriever=base_retriever, docstore=docstore)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync: invoke base → маппинг parent_id → уникальные parents либо child fallback."""
        child_docs = self.base_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
        return self._children_to_parents(child_docs)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Async: ainvoke base → та же подмена и дедуп."""
        child_docs = await self.base_retriever.ainvoke(query, config={"callbacks": run_manager.get_child()})
        return self._children_to_parents(child_docs)

    def _children_to_parents(self, child_docs: List[Document]) -> List[Document]:
        """Собирает родителей по parent_id; без родителя оставляет child с дедупом по hash контента."""
        parent_docs: List[Document] = []
        seen_parent_ids = set()

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")

            if parent_id and parent_id in self.docstore:
                if parent_id not in seen_parent_ids:
                    parent_docs.append(self.docstore[parent_id])
                    seen_parent_ids.add(parent_id)
            else:
                child_hash = hash(child.page_content)
                if child_hash not in seen_parent_ids:
                    parent_docs.append(child)
                    seen_parent_ids.add(child_hash)

        return parent_docs
