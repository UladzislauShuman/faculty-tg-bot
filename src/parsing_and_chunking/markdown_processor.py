"""HTML → Markdown → сплит по заголовкам; метаданные из title и source."""
import logging
from typing import List

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

from src.interfaces.data_processor_interfaces import DataSourceProcessor
from src.util.http_fetch import create_indexing_session

logger = logging.getLogger(__name__)


class MarkdownProcessor(DataSourceProcessor):
  """Парсинг основного блока страницы, markdownify, MarkdownHeaderTextSplitter."""

  def __init__(self) -> None:
    self._http = create_indexing_session()

  def process(self, source: str) -> List[Document]:
    logger.info("MarkdownProcessor: %s", source)
    try:
      response = self._http.get(
          source,
          headers={"User-Agent": "Mozilla/5.0"},
          timeout=10,
      )
      response.raise_for_status()
      soup = BeautifulSoup(response.text, 'lxml')

      # Шаг 1: Извлечение метаданных
      title = soup.find('title').get_text(strip=True) if soup.find(
        'title') else "No Title"
      base_metadata = {"source": source, "title": title}

      # Шаг 2: Изоляция и очистка контента
      content_block = soup.find('div', id='block-famcs-content') or soup.find(
        'article')
      if not content_block:
        logger.warning("Не найден основной блок контента: %s", source)
        return []

      # Чистим
      for tag in content_block.find_all(
          ['script', 'form', 'nav', 'header', 'footer']):
        tag.decompose()

      # Шаг 3: Конвертация в Markdown
      markdown_content = md(str(content_block), heading_style='ATX')

      # Шаг 4: Чанкинг по заголовкам Markdown и обогащение метаданными
      headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
      markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on)
      md_header_splits = markdown_splitter.split_text(markdown_content)

      for chunk in md_header_splits:
        chunk.metadata.update(base_metadata)

      logger.info("MarkdownProcessor: чанков=%s", len(md_header_splits))
      return md_header_splits
    except Exception as e:
      logger.warning("Ошибка %s: %s", source, e)
      return []