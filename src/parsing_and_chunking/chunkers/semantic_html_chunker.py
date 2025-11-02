from typing import List, Dict
from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document
from src.interfaces.chunker_interfaces import ChunkerInterface


class SemanticHTMLChunker(ChunkerInterface):
  """
  Семантический чанкер, который решает проблему "грязной" верстки.
  Стратегии:
  1.  Рекурсивный обход: ищет теги на любой глубине вложенности.
  2.  Иерархия заголовков: отслеживает последний h1, h2, h3 и т.д. для создания точного контекста.
  3.  Гранулярность: создает чанки из небольших смысловых единиц (p, li, tr).
  4.  Обогащение контекстом: добавляет иерархию заголовков в метаданные.
  5.  Специальная обработка таблиц.
  """
  HEADER_TAGS: List[str] = ["h1", "h2", "h3", "h4", "h5", "h6"]
  CONTENT_TAGS: List[str] = ["p", "li"]
  TAGS_TO_REMOVE: List[str] = ["nav", "header", "footer", "script", "style",
                               "aside"]

  def __init__(self, min_chunk_size: int = 50):
    self.min_chunk_size = min_chunk_size

  def _clean_html(self, soup: BeautifulSoup):
    """Удаляет из HTML все ненужные секции."""
    for tag_name in self.TAGS_TO_REMOVE:
      for tag in soup.find_all(tag_name):
        tag.decompose()

  def _get_heading_text(self, headings_stack: Dict[str, str]) -> str:
    """Собирает иерархию заголовков в одну строку, например "Главная / О нас"."""
    return " / ".join(
      filter(None, [headings_stack.get(f"h{i}", "") for i in range(1, 7)]))

  def _process_table(self, tag: Tag, base_metadata: dict, heading_text: str) -> \
  List[Document]:
    """Обрабатывает таблицу, создавая чанк для каждой значимой строки."""
    table_chunks = []
    # Используем separator=' ', чтобы <br> превращался в пробел, а не исчезал
    header_cells = [cell.get_text(strip=True, separator=' ') for cell in
                    tag.find_all('th')]

    for row in tag.find_all('tr'):
      row_cells = [cell.get_text(strip=True, separator=' ') for cell in
                   row.find_all(['td', 'th'])]

      # Пропускаем пустые или "мусорные" строки
      if not "".join(row_cells).strip():
        continue

      row_text = " | ".join(filter(None, row_cells))

      if len(row_text) < self.min_chunk_size:
        continue

      metadata = base_metadata.copy()
      metadata['heading_hierarchy'] = heading_text

      # Обогащаем контент, чтобы LLM было проще понять, что это таблица
      enriched_content = f"Раздел: {heading_text}\nТаблица: {row_text}"
      table_chunks.append(
        Document(page_content=enriched_content, metadata=metadata))

    return table_chunks

  def chunk(self, document: Document) -> List[Document]:
    """Основной метод, который выполняет разделение документа на чанки."""
    soup = BeautifulSoup(document.page_content, 'lxml')
    self._clean_html(soup)

    body = soup.find('div', id='block-famcs-content') or soup.find(
      'article') or soup.find("body")
    if not body:
      return []

    chunks = []
    # Словарь для хранения текущих заголовков по уровням
    headings_stack = {tag: "" for tag in self.HEADER_TAGS}

    # Находим все интересующие нас теги в порядке их следования в документе
    all_tags = body.find_all(self.HEADER_TAGS + self.CONTENT_TAGS + ["table"])

    for tag in all_tags:
      # 1. Если это заголовок - обновляем стек
      if tag.name in self.HEADER_TAGS:
        level = int(tag.name[1])
        headings_stack[tag.name] = tag.get_text(strip=True)
        # Сбрасываем все заголовки более низкого уровня
        for i in range(level + 1, 7):
          headings_stack[f"h{i}"] = ""
        continue

      current_heading_text = self._get_heading_text(headings_stack)

      # 2. Если это таблица - обрабатываем ее отдельно
      if tag.name == 'table':
        chunks.extend(
          self._process_table(tag, document.metadata, current_heading_text))
        continue

      # 3. Если это обычный контент (p, li)
      text = tag.get_text(strip=True)
      if text and len(text) >= self.min_chunk_size:
        metadata = document.metadata.copy()
        metadata['heading_hierarchy'] = current_heading_text

        # Обогащаем контент для лучшего понимания LLM
        enriched_content = f"Раздел: {current_heading_text}\n\n{text}"
        chunks.append(
          Document(page_content=enriched_content, metadata=metadata))

    return chunks