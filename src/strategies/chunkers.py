import textwrap
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from src.core.interfaces import ChunkerInterface

class HTMLContextChunker(ChunkerInterface):
    """
    Гибридный чанкер:
    1. Document-Specific: Парсит HTML по тегам h1-h6, p, li.
    2. Context-Enriched: Добавляет в метаданные каждого чанка его родительский заголовок.
    """
    def chunk(self, document: Document) -> list[Document]:
        soup = BeautifulSoup(document.page_content, 'lxml')
        
        content_element = soup.find('div', id='block-famcs-content') or soup.find('article')
        if not content_element:
            return []

        chunks = []
        current_heading = ""
        # Собираем все значащие теги
        tags_to_process = content_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li'])

        for tag in tags_to_process:
            text = tag.get_text(strip=True)
            
            # Если тег - заголовок, обновляем текущий заголовок
            if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                current_heading = text
            
            # Если тег - контент, создаем чанк с контекстом
            elif text and len(text) > 30: # Фильтруем слишком короткие строки
                metadata = document.metadata.copy() # Копируем исходные метаданные
                metadata['heading'] = current_heading # Обогащаем контекстом
                chunks.append(Document(page_content=text, metadata=metadata))

        return chunks