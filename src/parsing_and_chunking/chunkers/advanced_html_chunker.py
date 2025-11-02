from typing import List
from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document
from src.interfaces.chunker_interfaces import ChunkerInterface


class AdvancedHTMLChunker(ChunkerInterface):
  """
  Стратегии этого чанкера:
  1. Pre-cleaning: Очищает HTML от навигации, футеров, скриптов до начала парсинга.
  2. Hierarchical Grouping: Группирует абзацы и списки под их ближайшим родительским заголовком.
  3. Context-Enriched: Обогащает каждый чанк информацией о заголовке.
  """

  # Теги, которые считаются заголовками и начинают новый чанк.
  HEADER_TAGS: List[str] = ["h1", "h2", "h3", "h4", "h5", "h6"]
  # Теги, которые мы удаляем из всего документа перед обработкой.
  TAGS_TO_REMOVE: List[str] = ["nav", "header", "footer", "script", "style",
                               "aside"]

  def __init__(self, min_chunk_size: int = 100):
    """
    Args:
        min_chunk_size: Минимальная длина текста для сохранения чанка.
    """
    self.min_chunk_size = min_chunk_size

  def _clean_html(self, soup: BeautifulSoup):
    """Удаляет из HTML все ненужные секции."""
    for tag_name in self.TAGS_TO_REMOVE:
      for tag in soup.find_all(tag_name):
        tag.decompose()

  def chunk(self, document: Document) -> List[Document]:
    """
    Основной метод, который выполняет разделение документа на чанки.
    """
    soup = BeautifulSoup(document.page_content, 'lxml')
    self._clean_html(soup)

    # Ищем основной контент. Если не находим, работаем со всем <body>.
    body = soup.find('div', id='block-famcs-content') or soup.find(
      'article') or soup.find("body")
    if not body:
      return []

    chunks = []
    current_heading = ""
    current_chunk_text = []

    # Идем по всем дочерним элементам основного контента.
    for element in body.find_all(recursive=False):
      if not isinstance(element, Tag):
        continue

      # Если встречаем заголовок, "закрываем" предыдущий чанк и начинаем новый.
      if element.name in self.HEADER_TAGS:
        # 1. Сохраняем накопленный текст, если он есть.
        if current_chunk_text:
          full_text = "\n".join(current_chunk_text).strip()
          if len(full_text) >= self.min_chunk_size:
            metadata = document.metadata.copy()
            metadata['heading'] = current_heading
            chunks.append(Document(page_content=full_text, metadata=metadata))

        # 2. Начинаем новый чанк.
        current_heading = element.get_text(strip=True)
        current_chunk_text = [
          current_heading]  # Заголовок становится частью нового чанка.

      # Если это не заголовок, просто добавляем его текст к текущему чанку.
      else:
        text = element.get_text(strip=True)
        if text:
          current_chunk_text.append(text)

    # Не забываем сохранить последний накопленный чанк после выхода из цикла.
    if current_chunk_text:
      full_text = "\n".join(current_chunk_text).strip()
      if len(full_text) >= self.min_chunk_size:
        metadata = document.metadata.copy()
        metadata['heading'] = current_heading
        chunks.append(Document(page_content=full_text, metadata=metadata))

    return chunks