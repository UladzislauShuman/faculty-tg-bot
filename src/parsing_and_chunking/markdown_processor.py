from typing import List
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

from src.interfaces.data_processor_interfaces import DataSourceProcessor


class MarkdownProcessor(DataSourceProcessor):
  """
  Реализация обработчика, которая использует надежный 4-шаговый пайплайн
  с предварительной конвертацией контента в Markdown.

  Эта стратегия дает максимальный контроль и использует легкие зависимости.
  """

  def process(self, source: str) -> List[Document]:
    """
    Загружает HTML, извлекает метаданные, конвертирует основной контент
    в Markdown, а затем разбивает его на чанки по заголовкам.
    """
    print(f"⚙️ Обработка {source} с помощью MarkdownProcessor...")
    try:
      response = requests.get(source, headers={"User-Agent": "Mozilla/5.0"},
                              timeout=10, verify=False)
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
        print(
          f"  - ⚠️ Предупреждение: Не найден основной блок контента на {source}.")
        return []

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

      print(f"  - ✅ Успешно создано {len(md_header_splits)} чанков.")
      return md_header_splits
    except Exception as e:
      print(f"  - ❌ Ошибка при обработке {source}: {e}")
      return []