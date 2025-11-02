from typing import List
from langchain_core.documents import Document
from unstructured.partition.html import partition_html

from src.interfaces.data_processor_interfaces import DataSourceProcessor


class UnstructuredProcessor(DataSourceProcessor):
  """
  Реализация обработчика, которая использует библиотеку Unstructured
  для извлечения контента и его чанкинга "из коробки".

  Эта стратегия требует меньше кода и лучше справляется со сложными
  элементами вроде таблиц, но добавляет тяжелую зависимость.
  """

  def process(self, source: str) -> List[Document]:
    """
    Использует `partition_html` для автоматического извлечения
    структурированных элементов со страницы по URL.
    """
    print(f"⚙️ Обработка {source} с помощью UnstructuredProcessor...")
    try:
      elements = partition_html(
          url=source,
          ssl_verify=False,
          headers={"User-Agent": "Mozilla/5.0"},
          chunking_strategy="by_title",
          max_characters=2000,
          combine_text_under_n_chars=500,
          infer_table_structure=True
      )

      chunks = []
      for el in elements:
        metadata = {"source": source, "category": el.category}
        if hasattr(el.metadata, 'title'):
          metadata['title'] = el.metadata.title

        content = el.metadata.text_as_html if el.category == "Table" else el.text
        chunks.append(Document(page_content=content, metadata=metadata))

      print(f"  - ✅ Успешно создано {len(chunks)} чанков.")
      return chunks
    except Exception as e:
      print(f"  - ❌ Ошибка при обработке {source}: {e}")
      return []