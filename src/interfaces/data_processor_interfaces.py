from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class DataSourceProcessor(ABC):
  """
  Интерфейс для обработки источника данных.
  Загружает данные по URL (или пути) и возвращает список чанков.
  """

  @abstractmethod
  def process(self, source: str) -> List[Document]:
    pass