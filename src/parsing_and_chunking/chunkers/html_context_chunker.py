from typing import List, Dict, Any
from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document
from src.interfaces.chunker_interfaces import ChunkerInterface

class HTMLContextChunker(ChunkerInterface):
    """
    Гибридный чанкер, разделяющий HTML на основе структуры тегов
    и обогащающий чанки информацией о родительском заголовке.
    """
    DEFAULT_HEADER_TAGS: List[str] = ["h1", "h2", "h3", "h4", "h5", "h6"]
    DEFAULT_CONTENT_TAGS: List[str] = ["p", "li", "div", "table"]

    def __init__(self, content_selector: str = "#block-famcs-content, article, [role='main']", min_chunk_size: int = 30):
        self.content_selector = content_selector
        self.min_chunk_size = min_chunk_size

    def _extract_main_content(self, soup: BeautifulSoup) -> Tag:
        content_element = soup.select_one(self.content_selector)
        if content_element:
            for element in content_element.find_all(['script', 'style', 'form', 'nav']):
                element.decompose()
        return content_element

    def _process_table_tag(self, tag: Tag) -> str:
        rows_text = []
        for row in tag.find_all('tr'):
            cells_text = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            row_text = " | ".join(filter(None, cells_text))
            if row_text:
                rows_text.append(row_text)
        return "\n".join(rows_text)

    def _create_chunk(self, text: str, base_metadata: Dict[str, Any], current_heading: str) -> Document:
        metadata = base_metadata.copy()
        metadata['heading'] = current_heading
        return Document(page_content=text, metadata=metadata)

    def chunk(self, document: Document) -> List[Document]:
        soup = BeautifulSoup(document.page_content, 'lxml')
        content_element = self._extract_main_content(soup)
        if not content_element:
            return []

        chunks = []
        current_heading = ""
        tags_to_process = content_element.find_all(self.DEFAULT_HEADER_TAGS + self.DEFAULT_CONTENT_TAGS)

        for tag in tags_to_process:
            text = ""
            if tag.name in self.DEFAULT_HEADER_TAGS:
                current_heading = tag.get_text(strip=True)
                text = current_heading
            elif tag.name == 'table':
                text = self._process_table_tag(tag)
            elif tag.name in self.DEFAULT_CONTENT_TAGS:
                text = tag.get_text(strip=True)

            if text and len(text) >= self.min_chunk_size:
                new_chunk = self._create_chunk(text, document.metadata, current_heading)
                chunks.append(new_chunk)
        return chunks