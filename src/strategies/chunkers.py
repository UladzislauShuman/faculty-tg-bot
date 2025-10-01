from typing import List, Dict, Any
from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document
from src.core.interfaces import ChunkerInterface

class HTMLContextChunker(ChunkerInterface):
  """
  Гибридный чанкер, реализующий две стратегии:
  1. Document-Specific Chunking: Разделяет HTML-документ на основе
     структуры тегов (заголовки, абзацы, списки).
  2. Context-Enriched Chunking: Обогащает каждый чанк информацией
     о его родительском заголовке для улучшения контекста при поиске.
  """
  # Список тегов, которые мы считаем заголовками.
  DEFAULT_HEADER_TAGS: List[str] = ["h1", "h2", "h3", "h4", "h5", "h6"]
  # Список тегов, которые мы считаем контентом для создания чанков.
  DEFAULT_CONTENT_TAGS: List[str] = ["p", "li", "div"]

  def __init__(self,
      content_selector: str = "article, [role='main']",
      min_chunk_size: int = 50):
    """
    Конструктор класса.

    Args:
        content_selector: CSS-селектор для поиска основного блока с контентом на странице.
                          По умолчанию ищет тег <article> или любой тег с атрибутом role="main".
        min_chunk_size: Минимальная длина текста в символах для создания чанка.
                        Это помогает отфильтровать пустые или бессмысленные теги.
    """
    self.content_selector = content_selector
    self.min_chunk_size = min_chunk_size

  def _extract_main_content(self, soup: BeautifulSoup) -> Tag:
    """
    Находит и возвращает основной блок с контентом на странице.
    """
    # soup.select_one() - это метод BeautifulSoup для поиска первого элемента,
    # который соответствует CSS-селектору. Это более гибко
    # позволяет использовать сложные селекторы, например, "div#content, main".
    content_element = soup.select_one(self.content_selector)
    return content_element

  def _create_chunk(self, text: str, base_metadata: Dict[str, Any],
      current_heading: str) -> Document:
    """
    Создает объект Document для одного чанка, обогащая его метаданными.
    """
    # Копируем базовые метаданные (например, 'source' URL).
    metadata = base_metadata.copy()
    # Обогащаем контекстом - добавляем текущий заголовок.
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
        f"⚠️  Предупреждение: Не удалось найти основной контент на странице {document.metadata.get('source')} с помощью селектора '{self.content_selector}'.")
      return []

    chunks = []
    current_heading = ""

    # находит все теги, соответствующие списку.
    tags_to_process = content_element.find_all(
      self.DEFAULT_HEADER_TAGS + self.DEFAULT_CONTENT_TAGS)

    for tag in tags_to_process:
      # извлекает весь текст из тега,
      # очищая его от лишних пробелов и переносов строк.
      text = tag.get_text(strip=True)

      # Если тег является заголовком, мы обновляем наш "текущий заголовок".
      if tag.name in self.DEFAULT_HEADER_TAGS:
        current_heading = text

      # Если тег является контентом и проходит проверку на минимальную длину...
      elif tag.name in self.DEFAULT_CONTENT_TAGS and len(
          text) >= self.min_chunk_size:
        # ...мы создаем новый чанк.
        new_chunk = self._create_chunk(text, document.metadata, current_heading)
        chunks.append(new_chunk)

    return chunks