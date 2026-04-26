from typing import List, Dict, Any
from bs4 import BeautifulSoup, Tag, NavigableString
from langchain_core.documents import Document
from src.interfaces.chunker_interfaces import ChunkerInterface


class SemanticHTMLChunker(ChunkerInterface):
  """
  Реализация Element-Based Chunking для HTML.

  Стратегия:
  1. Разбор HTML на структурные элементы (Заголовки, Параграфы, Списки, Таблицы).
  2. Сохранение контекста (иерархия заголовков) для каждого элемента.
  3. Агрегация мелких элементов (p, li) в чанки разумного размера.
  4. Особая обработка таблиц (сохранение связи "Колонка-Значение").
  """

  def __init__(self, min_chunk_size: int = 50, max_chunk_size: int = 1000):
    self.min_chunk_size = min_chunk_size
    self.max_chunk_size = max_chunk_size
    self.headers_stack = {}

  def _clean_element(self, element: Tag):
    """Удаляет 'шум' (boilerplate) из HTML."""
    for tag in element.find_all(
        ['script', 'style', 'nav', 'footer', 'iframe', 'video', 'form',
         'noscript', 'aside'], recursive=True):
      tag.decompose()

  def _get_context_string(self) -> str:
    """Возвращает строку-хлебные крошки: 'О факультете > История'."""
    levels = sorted(self.headers_stack.keys())
    return " > ".join([self.headers_stack[l] for l in levels])

  def _process_table(self, table: Tag, source: str) -> List[Document]:
    """
    Превращает HTML-таблицу в набор документов.
    Каждая строка таблицы становится отдельным чанком с контекстом заголовков.
    """
    chunks = []
    headers = []

    # 1. Попытка найти заголовки
    thead = table.find('thead')
    if thead:
      headers = [th.get_text(" ", strip=True) for th in thead.find_all('th')]

    rows = table.find_all('tr')
    if not headers and rows:
      # Эвристика: первая строка жирная или th?
      first_cells = rows[0].find_all(['td', 'th'])
      if all(c.name == 'th' or c.find('strong') for c in first_cells):
        headers = [c.get_text(" ", strip=True) for c in first_cells]
        rows = rows[1:]

    context = self._get_context_string()

    for row in rows:
      cells = row.find_all(['td', 'th'])
      if not cells: continue

      row_data = []
      for i, cell in enumerate(cells):
        cell_text = cell.get_text(" ", strip=True)
        if not cell_text: continue

        col_name = headers[i] if i < len(headers) else ""
        if col_name:
          row_data.append(f"{col_name}: {cell_text}")
        else:
          row_data.append(cell_text)

      if row_data:
        content = f"Контекст: {context}\nДанные: {'; '.join(row_data)}" if headers else f"Контекст: {context}\nСтрока: {' | '.join(row_data)}"
        chunks.append(Document(
            page_content=content,
            metadata={"source": source, "type": "table_row", "context": context}
        ))
    return chunks

  def chunk(self, document: Document) -> List[Document]:
    soup = BeautifulSoup(document.page_content, 'html.parser')

    # 1. Умный поиск контента (Drupal specific + Fallback)
    candidates = [
      soup.select_one('article'),
      soup.select_one('div.field--name-body'),
      soup.select_one('div#block-famcs-content'),
      soup.select_one('main'),
      soup.body
    ]
    main_content = next((c for c in candidates if c), soup)
    self._clean_element(main_content)

    # Поиск H1 страницы
    page_title = soup.find('h1', class_='page-title')
    self.headers_stack = {
      1: page_title.get_text(strip=True)} if page_title else {}

    final_chunks = []
    current_text_buffer = []
    current_metadata = document.metadata.copy()

    # 2. Линейный обход элементов (Element-Based Traversal)
    tags_to_parse = ['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'table', 'div',
                     'dl', 'blockquote']

    for element in main_content.find_all(tags_to_parse):
      if not element.get_text(strip=True): continue

      # --- Заголовки: Меняют контекст и сбрасывают буфер ---
      if element.name in ['h1', 'h2', 'h3', 'h4']:
        # Если накопился текст - сохраняем его перед сменой темы
        if current_text_buffer:
          text = "\n".join(current_text_buffer)
          if len(text) > self.min_chunk_size:
            final_chunks.append(Document(
                page_content=f"Контекст: {self._get_context_string()}\n{text}",
                metadata=current_metadata
            ))
          current_text_buffer = []

        # Обновляем стек заголовков
        level = int(element.name[1])
        self.headers_stack[level] = element.get_text(" ", strip=True)
        self.headers_stack = {k: v for k, v in self.headers_stack.items() if
                              k <= level}

      # --- Таблицы: Обрабатываются отдельно (Atomic Chunking) ---
      elif element.name == 'table':
        if element.find_parent('table'): continue
        # Сбрасываем буфер перед таблицей
        if current_text_buffer:
          text = "\n".join(current_text_buffer)
          if len(text) > self.min_chunk_size:
            final_chunks.append(Document(
                page_content=f"Контекст: {self._get_context_string()}\n{text}",
                metadata=current_metadata
            ))
          current_text_buffer = []

        final_chunks.extend(
          self._process_table(element, document.metadata.get('source', '')))

      # --- Текст и Списки: Агрегируются ---
      elif element.name in ['p', 'div', 'blockquote', 'ul', 'ol', 'dl']:
        # Пропускаем контейнеры
        if element.name == 'div' and element.find(
            ['p', 'table', 'h2']): continue
        if element.name in ['ul', 'ol'] and element.find_parent(
            ['ul', 'ol']): continue

        # Формируем текст элемента
        text_part = ""
        if element.name in ['ul', 'ol']:
          items = [li.get_text(" ", strip=True) for li in
                   element.find_all('li')]
          text_part = "\n".join([f"- {item}" for item in items if item])
        elif element.name == 'dl':
          dl_parts = []
          for dt in element.find_all('dt'):
            dd = dt.find_next_sibling('dd')
            if dd: dl_parts.append(
              f"{dt.get_text(strip=True)}: {dd.get_text(strip=True)}")
          text_part = "\n".join(dl_parts)
        else:
          text_part = element.get_text(" ", strip=True)

        # Добавляем в буфер
        if text_part:
          current_text_buffer.append(text_part)

        # Проверка размера буфера (Soft Limit)
        current_len = sum(len(t) for t in current_text_buffer)
        if current_len > self.max_chunk_size:
          text = "\n".join(current_text_buffer)
          final_chunks.append(Document(
              page_content=f"Контекст: {self._get_context_string()}\n{text}",
              metadata=current_metadata
          ))
          current_text_buffer = []

    # Сохраняем остаток буфера
    if current_text_buffer:
      text = "\n".join(current_text_buffer)
      if len(text) > self.min_chunk_size:
        final_chunks.append(Document(
            page_content=f"Контекст: {self._get_context_string()}\n{text}",
            metadata=current_metadata
        ))

    return final_chunks