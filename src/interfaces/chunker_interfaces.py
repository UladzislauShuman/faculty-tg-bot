from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class ChunkerInterface(ABC):
  """
  Абстрактный класс (интерфейс) для всех стратегий чанкинга.

  Задача чанкера — взять один большой документ (например, целую HTML-страницу)
  и разбить его на список более мелких, семантически связанных документов (чанков).
  """

  @abstractmethod
  def chunk(self, document: Document) -> List[Document]:
    """
    Основной метод, который должен быть реализован в каждой стратегии чанкинга.

    Args:
        document: Один большой документ, который нужно разбить.

    Returns:
        Список маленьких документов (чанков).
    """
    pass