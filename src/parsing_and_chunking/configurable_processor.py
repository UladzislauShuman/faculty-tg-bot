from typing import List
import requests
from langchain_core.documents import Document

from src.interfaces.data_processor_interfaces import DataSourceProcessor
from src.interfaces.chunker_interfaces import ChunkerInterface


class ConfigurableProcessor(DataSourceProcessor):
  """
  Гибкий обработчик, который делегирует задачу чанкинга
  внешнему объекту-чанкеру.

  Эта реализация следует принципу композиции: ее поведение
  определяется тем, какой объект-чанкер ей передали.
  """

  def __init__(self, chunker: ChunkerInterface):
    """
    Конструктор, который принимает зависимость (чанкер).

    Args:
        chunker: Экземпляр класса, реализующего ChunkerInterface.
    """
    self.chunker = chunker
    print(
      f"⚙️ ConfigurableProcessor инициализирован с чанкером: {chunker.__class__.__name__}")

  def process(self, source: str) -> List[Document]:
    """
    Шаг 1: Загружает сырой HTML-контент.
    Шаг 2: Передает его в виде объекта Document в self.chunker.
    """
    print(f"⚙️ Обработка {source} с помощью ConfigurableProcessor...")
    try:
      response = requests.get(source, headers={"User-Agent": "Mozilla/5.0"},
                              timeout=10, verify=False)
      response.raise_for_status()

      # Создаем один большой документ из всей страницы
      full_document = Document(
          page_content=response.text,
          metadata={"source": source}
      )

      # Делегируем разбиение на чанки нашему чанкеру
      chunks = self.chunker.chunk(full_document)

      print(
        f"  - ✅ Чанкер {self.chunker.__class__.__name__} создал {len(chunks)} чанков.")
      return chunks

    except Exception as e:
      print(f"  - ❌ Ошибка при обработке {source}: {e}")
      return []