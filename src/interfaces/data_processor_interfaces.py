from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class DataSourceProcessor(ABC):
  """
  Абстрактный класс (интерфейс) для обработчиков источников данных.

  Определяет "контракт", которому должны следовать все классы-стратегии
  по обработке данных. Его задача — взять на вход один источник (URL)
  и вернуть список готовых к индексации документов (чанков).
  """

  @abstractmethod
  def process(self, source: str) -> List[Document]:
    """
    Основной метод, который должен быть реализован в каждой стратегии.

    Args:
        source: Строка, идентифицирующая источник (например, "https://...").

    Returns:
        Список документов (чанков), полученных из этого источника.
    """
    pass