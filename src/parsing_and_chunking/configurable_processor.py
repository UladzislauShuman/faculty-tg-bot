from typing import List
import requests
from langchain_core.documents import Document

from src.interfaces.data_processor_interfaces import DataSourceProcessor
from src.interfaces.chunker_interfaces import ChunkerInterface


class ConfigurableProcessor(DataSourceProcessor):
  """
  Процессор, который отделяет логику скачивания от логики нарезки.
  Использует принцип композиции: чанкер передается в конструктор.
  """

  def __init__(self, chunker: ChunkerInterface):
    self.chunker = chunker
    print(
      f"⚙️ ConfigurableProcessor инициализирован с чанкером: {chunker.__class__.__name__}")

  def process(self, source: str) -> List[Document]:
    chunks, _ = self.process_with_parents(source)
    return chunks

  def process_with_parents(self, source: str) -> tuple[List[Document], List[Document]]:
    print(f"⚙️ Обработка {source} с помощью ConfigurableProcessor...")
    try:
      # 1. Скачиваем HTML
      response = requests.get(source, headers={"User-Agent": "Mozilla/5.0"},
                              timeout=10, verify=False)
      response.raise_for_status()

      # 2. Оборачиваем в Document
      full_document = Document(
          page_content=response.text,
          metadata={"source": source}
      )

      # 3. Делегируем нарезку чанкеру
      if hasattr(self.chunker, "chunk_with_parents"):
        children, parents = self.chunker.chunk_with_parents(full_document)
      else:
        children = self.chunker.chunk(full_document)
        parents = []

      print(
        f"  - ✅ Чанкер {self.chunker.__class__.__name__} создал {len(children)} чанков (и {len(parents)} родителей).")
      return children, parents

    except Exception as e:
      print(f"  - ❌ Ошибка при обработке {source}: {e}")
      return [], []