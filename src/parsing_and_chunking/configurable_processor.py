"""Скачивание HTML и делегирование чанкеру; поддержка parent+children через chunk_with_parents."""
import logging
from typing import List

from langchain_core.documents import Document

from src.interfaces.data_processor_interfaces import DataSourceProcessor
from src.interfaces.chunker_interfaces import ChunkerInterface
from src.util.http_fetch import create_indexing_session

logger = logging.getLogger(__name__)


class ConfigurableProcessor(DataSourceProcessor):
  """HTTP GET → Document → chunker.chunk или chunk_with_parents."""

  def __init__(self, chunker: ChunkerInterface):
    self.chunker = chunker
    self._http = create_indexing_session()
    logger.info("ConfigurableProcessor: чанкер=%s", chunker.__class__.__name__)

  def process(self, source: str) -> List[Document]:
    chunks, _ = self.process_with_parents(source)
    return chunks

  def process_with_parents(self, source: str) -> tuple[List[Document], List[Document]]:
    """Шаги: запрос URL → обёртка в Document → нарезка (с родителями или без)."""
    logger.info("ConfigurableProcessor: загрузка %s", source)
    try:
      # 1. Скачиваем HTML
      response = self._http.get(
          source,
          headers={"User-Agent": "Mozilla/5.0"},
          timeout=10,
      )
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

      logger.info(
          "Чанкер %s: children=%s parents=%s",
          self.chunker.__class__.__name__,
          len(children),
          len(parents),
      )
      return children, parents

    except Exception as e:
      logger.warning("Ошибка обработки %s: %s", source, e)
      return [], []