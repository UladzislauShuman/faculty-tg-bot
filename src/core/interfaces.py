from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class ChunkerInterface(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Document]:
        """Разделяет один большой документ на список чанков."""
        pass
