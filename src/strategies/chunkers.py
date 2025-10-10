from typing import List, Dict, Any
from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document
from src.core.interfaces import ChunkerInterface

class HTMLContextChunker(ChunkerInterface):
  """
  Гибридный чанкер, реализующий две стратегии:
  1. Document-Specific Chunking: Разделяет HTML-документ на основе
     структуры тегов (заголовки, абзацы, списки, таблицы).
  2. Context-Enriched Chunking: Обогащает каждый чанк информацией
     о его родительском заголовке для улучшения контекста при поиске.
  """
  # Список тегов, которые мы считаем заголовками.
  DEFAULT_HEADER_TAGS: List[str] = ["h1", "h2", "h3", "h4", "h5", "h6"]
  # Список тегов, которые мы считаем основным контентом для создания чанков.
  DEFAULT_CONTENT_TAGS: List[str] = ["p", "li", "div","table"]

  def __init__(self,
      content_selector: str = "#block-famcs-content, article, [role='main']",
      min_chunk_size: int = 30):
    """
    Конструктор класса.
    Args:
        content_selector: CSS-селектор для поиска основного блока с контентом.
        min_chunk_size: Минимальная длина текста для создания чанка.
    """
    self.content_selector = content_selector
    self.min_chunk_size = min_chunk_size

  def _extract_main_content(self, soup: BeautifulSoup) -> Tag:
    """
    Находит и возвращает основной блок с контентом на странице.
    """
    content_element = soup.select_one(self.content_selector)

    # Перед извлечением текста, удалим все ненужные элементы из контента.
    if content_element:
      # Находим все элементы, которые хотим удалить (скрипты, стили, формы и т.д.)
      elements_to_remove = content_element.find_all(
          ['script', 'style', 'form', 'nav'])
      for element in elements_to_remove:
        # .decompose() - полностью удаляет тег из HTML-дерева.
        element.decompose()

    return content_element

  def _process_table_tag(self, tag: Tag) -> str:
    """
    Специальный обработчик для тега <table>.
    Превращает каждую строку таблицы в осмысленное предложение.
    """
    rows_text = []
    # Находим все строки (<tr>) в таблице
    for row in tag.find_all('tr'):
      # Находим все ячейки (<td> или <th>) в строке и извлекаем их текст
      cells_text = [cell.get_text(strip=True) for cell in
                    row.find_all(['td', 'th'])]
      # Объединяем текст ячеек в одну строку, "Иванов | Декан | 1970-1980"
      row_text = " | ".join(filter(None, cells_text))
      if row_text:
        rows_text.append(row_text)
    # Объединяем все строки в один текстовый блок, разделяя их переносом строки.
    return "\n".join(rows_text)

  def _create_chunk(self, text: str, base_metadata: Dict[str, Any],
      current_heading: str) -> Document:
    """
    Создает объект Document для одного чанка, обогащая его метаданными.
    """
    metadata = base_metadata.copy()
    metadata['heading'] = current_heading
    return Document(page_content=text, metadata=metadata)

  def chunk(self, document: Document) -> List[Document]:
    """
    Основной метод, который выполняет разделение документа на чанки.
    """
    soup = BeautifulSoup(document.page_content, 'lxml')

    content_element = self._extract_main_content(soup)
    if not content_element:
      print(
        f"⚠️  Предупреждение: Не удалось найти основной контент на странице {document.metadata.get('source')}.")
      return []

    chunks = []
    current_heading = ""

    tags_to_process = content_element.find_all(
      self.DEFAULT_HEADER_TAGS + self.DEFAULT_CONTENT_TAGS)

    for tag in tags_to_process:
      text = ""
      # Если тег является заголовком, мы обновляем наш "текущий заголовок".
      if tag.name in self.DEFAULT_HEADER_TAGS:
        current_heading = tag.get_text(strip=True)
        text = current_heading  # Заголовки тоже могут быть важными чанками

      elif tag.name == 'table':
        text = self._process_table_tag(tag)

      # Если это обычный контентный тег
      elif tag.name in self.DEFAULT_CONTENT_TAGS:
        text = tag.get_text(strip=True)

      # Создаем чанк, если текст не пустой и проходит проверку на длину
      if text and len(text) >= self.min_chunk_size:
        new_chunk = self._create_chunk(text, document.metadata, current_heading)
        chunks.append(new_chunk)

    return chunks


class AdvancedHTMLChunker(ChunkerInterface):
  """
  Стратегии:
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


class SemanticHTMLChunker(ChunkerInterface):
  """
  Семантический чанкер, который решает проблему "грязной" верстки.
  Стратегии:
  1.  Рекурсивный обход: ищет теги на любой глубине вложенности.
  2.  Иерархия заголовков: отслеживает последний h1, h2, h3 и т.д. для создания точного контекста.
  3.  Гранулярность: создает чанки из небольших смысловых единиц (p, li, tr).
  4.  Обогащение контекстом: добавляет иерархию заголовков в метаданные и в начало текста чанка.
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
    """Собирает иерархию заголовков в одну строку."""
    return " / ".join(
      filter(None, [headings_stack.get(f"h{i}", "") for i in range(1, 7)]))

  def _process_table(self, tag: Tag, base_metadata: dict, heading_text: str) -> \
  List[Document]:
    """Обрабатывает таблицу, создавая чанк для каждой значимой строки."""
    table_chunks = []
    # Используем separator=' ', чтобы <br> превращался в пробел
    header_cells = [cell.get_text(strip=True, separator=' ') for cell in
                    tag.find_all('th')]

    for row in tag.find_all('tr'):
      # Используем separator=' ' и здесь
      row_cells = [cell.get_text(strip=True, separator=' ') for cell in
                   row.find_all(['td', 'th'])]

      # Пропускаем пустые или "мусорные" строки
      row_text_for_check = "".join(row_cells).lower()
      if not row_text_for_check or row_text_for_check.count(
          'изображение') == len(row_cells):
        continue

      row_text = " | ".join(filter(None, row_cells))

      if header_cells and len(header_cells) == len(row_cells):
        row_text = ", ".join(
            [f"{h}: {c}" for h, c in zip(header_cells, row_cells) if c])

      if len(row_text) < self.min_chunk_size:
        continue

      metadata = base_metadata.copy()
      metadata['heading'] = heading_text

      enriched_content = f"Раздел: {heading_text}\nТаблица: {row_text}"

      table_chunks.append(
        Document(page_content=enriched_content, metadata=metadata))

    return table_chunks

  def chunk(self, document: Document) -> List[Document]:
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
        metadata['heading'] = current_heading_text

        enriched_content = f"Раздел: {current_heading_text}\n\n{text}"

        chunks.append(
          Document(page_content=enriched_content, metadata=metadata))

    return chunks