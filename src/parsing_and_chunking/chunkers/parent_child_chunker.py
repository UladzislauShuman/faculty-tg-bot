from typing import List, Tuple
import uuid

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from markdownify import markdownify as md

from src.interfaces.chunker_interfaces import ChunkerInterface

# Legacy parent strategy (semantic blocks on raw HTML):
# from src.parsing_and_chunking.chunkers.semantic_html_chunker import SemanticHTMLChunker


class ParentChildHTMLChunker(ChunkerInterface):
    """
    Двухуровневый чанкер:
    1. Parent: тот же флоу, что у MarkdownProcessor — HTML → Markdown → нарезка по заголовкам.
    2. Child: мелкие фрагменты для поиска (RecursiveCharacterTextSplitter).

    Ранее родительский уровень строился через SemanticHTMLChunker (код сохранён в комментариях).
    """

    def __init__(self, child_chunk_size: int = 300, parent_chunk_size: int = 1200):
        self.child_chunk_size = child_chunk_size
        # Оставлен для совместимости с DI/config; в текущем markdown-parent флоу не используется.
        self.parent_chunk_size = parent_chunk_size

        # --- Legacy: родители через SemanticHTMLChunker ---
        # self.parent_chunker = SemanticHTMLChunker(
        #     min_chunk_size=50,
        #     max_chunk_size=self.parent_chunk_size,
        # )

        self._markdown_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def chunk(self, document: Document) -> List[Document]:
        """
        Возвращает только children (для обратной совместимости с ChunkerInterface).
        """
        children, _ = self.chunk_with_parents(document)
        return children

    def _html_to_parent_documents(self, document: Document) -> List[Document]:
        """
        Повторяет изоляцию контента и чанкинг MarkdownProcessor, но без HTTP:
        вход — HTML в document.page_content.
        """
        html = document.page_content
        source = document.metadata.get("source", "")
        soup = BeautifulSoup(html, "lxml")
        title = (
            soup.find("title").get_text(strip=True)
            if soup.find("title")
            else "No Title"
        )
        base_metadata = {**dict(document.metadata), "title": title}

        content_block = soup.find("div", id="block-famcs-content") or soup.find(
            "article"
        )
        if not content_block:
            label = source or "(inline html)"
            print(
                "  - ⚠️ Предупреждение: не найден основной блок контента "
                f"(parent-child markdown parents): {label}."
            )
            return []

        for tag in content_block.find_all(
            ["script", "form", "nav", "header", "footer"]
        ):
            tag.decompose()

        markdown_content = md(str(content_block), heading_style="ATX")
        parents = self._markdown_header_splitter.split_text(markdown_content)
        for p in parents:
            p.metadata.update(base_metadata)
        return list(parents)

    def chunk_with_parents(self, document: Document) -> Tuple[List[Document], List[Document]]:
        """
        Возвращает кортеж (children, parents).
        Каждый child содержит 'parent_id' в metadata, указывающий на 'doc_id' родителя.
        """
        # Legacy:
        # parents = self.parent_chunker.chunk(document)
        parents = self._html_to_parent_documents(document)

        all_children: List[Document] = []
        all_parents: List[Document] = []

        for parent in parents:
            parent_id = parent.metadata.get("doc_id")
            if not parent_id:
                parent_id = str(uuid.uuid4())
                parent.metadata["doc_id"] = parent_id

            all_parents.append(parent)

            child_texts = self.child_splitter.split_text(parent.page_content)

            for child_text in child_texts:
                child_metadata = parent.metadata.copy()
                child_metadata["parent_id"] = parent_id
                if "doc_id" in child_metadata:
                    del child_metadata["doc_id"]

                all_children.append(
                    Document(page_content=child_text, metadata=child_metadata)
                )

        return all_children, all_parents
