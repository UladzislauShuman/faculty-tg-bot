import os
import pickle
from typing import List, Dict

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class ParentDocumentRetriever(BaseRetriever):
    """
    Ретривер, который ищет по мелким чанкам (через base_retriever),
    а затем подменяет их на крупные родительские документы из docstore.
    """
    
    base_retriever: BaseRetriever
    docstore: Dict[str, Document]

    @classmethod
    def from_config(cls, base_retriever: BaseRetriever, docstore_path: str) -> "ParentDocumentRetriever":
        """Фабричный метод для загрузки docstore из файла."""
        docstore = {}
        if os.path.exists(docstore_path):
            try:
                with open(docstore_path, "rb") as f:
                    docstore = pickle.load(f)
            except Exception as e:
                print(f"❌ Ошибка при загрузке docstore: {e}")
        else:
            print(f"⚠️ Docstore не найден по пути: {docstore_path}")
            
        return cls(base_retriever=base_retriever, docstore=docstore)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. Ищем дочерние чанки
        child_docs = self.base_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
        
        # 2. Подменяем на родителей с дедупликацией
        parent_docs = []
        seen_parent_ids = set()
        
        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            
            if parent_id and parent_id in self.docstore:
                if parent_id not in seen_parent_ids:
                    parent_docs.append(self.docstore[parent_id])
                    seen_parent_ids.add(parent_id)
            else:
                # Fallback: если родитель не найден, возвращаем самого ребенка
                # (чтобы не терять информацию)
                # Для дедупликации детей без родителя используем их контент
                child_hash = hash(child.page_content)
                if child_hash not in seen_parent_ids:
                    parent_docs.append(child)
                    seen_parent_ids.add(child_hash)
                    
        return parent_docs
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. Ищем дочерние чанки асинхронно
        child_docs = await self.base_retriever.ainvoke(query, config={"callbacks": run_manager.get_child()})
        
        # 2. Подменяем на родителей с дедупликацией
        parent_docs = []
        seen_parent_ids = set()
        
        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            
            if parent_id and parent_id in self.docstore:
                if parent_id not in seen_parent_ids:
                    parent_docs.append(self.docstore[parent_id])
                    seen_parent_ids.add(parent_id)
            else:
                # Fallback
                child_hash = hash(child.page_content)
                if child_hash not in seen_parent_ids:
                    parent_docs.append(child)
                    seen_parent_ids.add(child_hash)
                    
        return parent_docs
