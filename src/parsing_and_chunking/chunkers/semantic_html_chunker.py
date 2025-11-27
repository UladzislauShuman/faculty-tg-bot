from typing import List
from bs4 import BeautifulSoup, NavigableString, Tag
from langchain_core.documents import Document
from src.interfaces.chunker_interfaces import ChunkerInterface


class SemanticHTMLChunker(ChunkerInterface):
  """
  Продвинутый чанкер для HTML.
  - Сохраняет иерархию заголовков (H1 > H2).
  - Умно парсит таблицы (Колонка: Значение).
  """

  def __init__(self, min_chunk_size: int = 50):
    self.min_chunk_size = min_chunk_size
    self.headers_stack = {}

  def _clean_element(self, element: Tag):
    """Удаляет мусорные теги внутри конкретного элемента."""
    for tag in element.find_all(
        ['script', 'style', 'nav', 'footer', 'iframe', 'video', 'aside',
         'form'], recursive=True):
      tag.decompose()

  def _get_context_string(self) -> str:
    levels = sorted(self.headers_stack.keys())
    return " > ".join([self.headers_stack[l] for l in levels])

  def _process_table(self, table: Tag, source: str) -> List[Document]:
    chunks = []
    headers = []

    # Попытка найти заголовки
    thead = table.find('thead')
    if thead:
      headers = [th.get_text(strip=True) for th in thead.find_all('th')]

    rows = table.find_all('tr')
    if not headers and rows:
      # Эвристика: первая строка жирная или th?
      first_row_cells = rows[0].find_all(['td', 'th'])
      if all(c.name == 'th' for c in first_row_cells) or len(rows) > 1:
        headers = [c.get_text(strip=True) for c in first_row_cells]
        rows = rows[1:]

    context = self._get_context_string()

    for row in rows:
      cells = row.find_all(['td', 'th'])
      if not cells: continue

      row_data = []
      for i, cell in enumerate(cells):
        cell_text = cell.get_text(" ",
                                  strip=True)  # separator=" " чтобы не слипалось
        if not cell_text: continue

        col_name = headers[i] if i < len(headers) else ""
        if col_name:
          row_data.append(f"{col_name}: {cell_text}")
        else:
          row_data.append(cell_text)

      if row_data:
        # Если заголовков нет, просто соединяем через пайп
        if headers:
          content = f"Контекст: {context}\nДанные: {'; '.join(row_data)}"
        else:
          content = f"Контекст: {context}\nСтрока таблицы: {' | '.join(row_data)}"

        chunks.append(Document(
            page_content=content,
            metadata={"source": source, "type": "table_row", "context": context}
        ))
    return chunks

  def chunk(self, document: Document) -> List[Document]:
    # Используем html.parser, он иногда прощает ошибки верстки лучше lxml
    soup = BeautifulSoup(document.page_content, 'html.parser')

    # Стратегия поиска контента (по убыванию специфичности)
    candidates = [
      soup.select_one('article'),
      soup.select_one('div.field--name-body'),
      soup.select_one('div#block-famcs-content'),
      soup.select_one('main'),
      soup.select_one('div.region-content'),
      soup.body
    ]

    main_content = None
    for c in candidates:
      if c:
        main_content = c
        break

    if not main_content:
      # Если совсем ничего не нашли - берем весь суп
      main_content = soup

    # Чистим найденный контент от мусора
    self._clean_element(main_content)

    chunks = []
    self.headers_stack = {}

    # Ищем H1 заголовки страницы, если они вне content блока (иногда бывает)
    page_title = soup.find('h1', class_='page-title')
    if page_title:
      self.headers_stack[1] = page_title.get_text(strip=True)

    # Итерируемся
    for element in main_content.find_all(
        ['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'table', 'div']):

      # Игнорируем пустые элементы
      if not element.get_text(strip=True):
        continue

      if element.name in ['h1', 'h2', 'h3', 'h4']:
        level = int(element.name[1])
        self.headers_stack[level] = element.get_text(strip=True)
        self.headers_stack = {k: v for k, v in self.headers_stack.items() if
                              k <= level}

      elif element.name == 'table':
        chunks.extend(
          self._process_table(element, document.metadata.get('source', '')))

      elif element.name in ['p', 'div']:
        # Для div проверяем, что это текстовый блок, а не контейнер
        if element.name == 'div' and element.find(['p', 'table', 'h2']):
          continue  # Это контейнер, пропускаем, зайдем внутрь на след итерации

        text = element.get_text(separator=' ', strip=True)
        # Фильтр короткого мусора (менюшки, даты)
        if len(text) > 40:
          context = self._get_context_string()
          content = f"Контекст: {context}\nТекст: {text}"
          chunks.append(Document(
              page_content=content,
              metadata={**document.metadata, "context": context}
          ))

      elif element.name in ['ul', 'ol']:
        # Списки обрабатываем целиком
        items = [li.get_text(strip=True) for li in element.find_all('li')]
        if items:
          text = "\n".join([f"- {item}" for item in items])
          if len(text) > 30:
            context = self._get_context_string()
            content = f"Контекст: {context}\nСписок:\n{text}"
            chunks.append(Document(
                page_content=content,
                metadata={**document.metadata, "context": context}
            ))

    return chunks