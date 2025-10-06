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