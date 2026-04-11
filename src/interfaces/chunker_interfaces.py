from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class ChunkerInterface(ABC):
  """
    Интерфейс стратегии чанкинга.
    Принимает на вход HTML страницу и возвращает список чанков
    """

  @abstractmethod
  def chunk(self, document: Document) -> List[Document]:
    pass