import os
import pickle
import requests
import textwrap
from typing import List, Dict, Any, Set
from urllib.parse import urljoin, urlparse
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from src.strategies.chunkers import HTMLContextChunker

class WebsiteCrawler:
  """
  Класс для обхода веб-сайта, сбора контента с его страниц и сохранения карты сайта.
  """

  def __init__(self, base_url: str, max_depth: int = 2):
    parsed_url = urlparse(base_url)
    self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    self.domain = parsed_url.netloc
    self.visited_urls: Set[str] = set()
    self.max_depth = max_depth
    self.headers = {
      "User-Agent": "Mozilla/5.0 (Windows NT 1.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

  def _is_valid_url(self, url: str) -> bool:
    """
    Проверяет, является ли ссылка внутренней и подходящей для парсинга (не файл, не якорь).
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ['http', 'https']:
      return False
    if parsed_url.netloc and parsed_url.netloc != self.domain:
      return False
    if any(url.endswith(ext) for ext in
           ['.pdf', '.jpg', '.png', '.zip', '.docx', '.xlsx', '.pptx', '.mp4',
            '.doc']):
      return False
    if '/admin' in parsed_url.path or '/edit' in parsed_url.path or '/feed' in parsed_url.path:
      return False
    return True

  def crawl(self) -> List[Document]:
    """
    Запускает процесс обхода сайта с ограничением по глубине.
    """
    urls_to_visit = [(self.base_url, 0)]
    all_documents: List[Document] = []

    while urls_to_visit:
      current_url, current_depth = urls_to_visit.pop(0)
      current_url = urljoin(current_url, urlparse(current_url).path)

      if current_url in self.visited_urls or current_depth > self.max_depth:
        if current_depth > self.max_depth:
          print(
            f"🌀 Достигнута максимальная глубина ({self.max_depth}). Пропускаем: {current_url}")
        continue

      print(f"🕸️  [Глубина {current_depth}] Обход страницы: {current_url}")
      self.visited_urls.add(current_url)

      try:
        response = requests.get(current_url, headers=self.headers, verify=False,
                                timeout=10)
        response.raise_for_status()
      except requests.RequestException as e:
        print(f"❌ Не удалось загрузить {current_url}: {e}")
        continue

      raw_document = Document(page_content=response.text,
                              metadata={"source": current_url})
      all_documents.append(raw_document)

      soup = BeautifulSoup(response.text, 'lxml')
      for link in soup.find_all('a', href=True):
        href = link['href']
        absolute_url = urljoin(self.base_url, href)

        if self._is_valid_url(
            absolute_url) and absolute_url not in self.visited_urls:
          if absolute_url not in [item[0] for item in urls_to_visit]:
            urls_to_visit.append((absolute_url, current_depth + 1))

    print(f"\n✅ Обход сайта завершен. Собрано {len(all_documents)} страниц.")
    return all_documents

  def _build_url_tree(self) -> Dict:
    """Строит вложенный словарь (дерево) из плоского списка URL."""
    tree = {}
    for url in sorted(list(self.visited_urls)):
      path = urlparse(url).path
      parts = path.strip('/').split('/')

      current_level = tree
      for part in parts:
        if not part: continue
        if part not in current_level:
          current_level[part] = {}
        current_level = current_level[part]
    return tree

  def _print_tree_recursive(self, tree: Dict, prefix: str, file):
    """Рекурсивно печатает дерево URL в файл, создавая красивую структуру."""
    keys = sorted(tree.keys())
    for i, key in enumerate(keys):
      is_last = (i == len(keys) - 1)
      connector = "└── " if is_last else "├── "
      file.write(f"{prefix}{connector}{key}\n")

      new_prefix = prefix + ("    " if is_last else "│   ")
      self._print_tree_recursive(tree[key], new_prefix, file)

  def save_sitemap_to_file(self, config: Dict[str, Any]):
    """Сохраняет дерево посещенных URL в текстовый файл."""
    output_dir = config['paths']['output_dir']
    sitemap_filename = config['paths']['sitemap']
    sitemap_path = os.path.join(output_dir, sitemap_filename)
    os.makedirs(output_dir, exist_ok=True)

    url_tree = self._build_url_tree()

    with open(sitemap_path, 'w', encoding='utf-8') as f:
      f.write(f"Карта сайта для: {self.base_url}\n")
      f.write(f"Всего посещено страниц: {len(self.visited_urls)}\n\n")
      f.write(f"{self.domain}\n")
      self._print_tree_recursive(url_tree, "", f)

    print(f"✅ Карта посещенных сайтов сохранена в: {sitemap_path}")

def _save_chunks_to_file(chunks: List[Document], base_url: str,
    config: Dict[str, Any]):
  """
  Сохраняет отформатированный список чанков в текстовый файл для отладки.
  """
  output_dir = config['paths']['output_dir']
  output_filename = config['paths']['indexing']
  output_file_path = os.path.join(output_dir, output_filename)

  os.makedirs(output_dir, exist_ok=True)

  with open(output_file_path, "w", encoding="utf-8") as f:
    f.write(f"Исходный сайт: {base_url}\n")
    f.write(f"Всего чанков со всех страниц: {len(chunks)}\n")
    f.write("=" * 80 + "\n\n")

    for i, chunk in enumerate(chunks):
      f.write(f"--- ЧАНК #{i + 1} ---\n")
      f.write(f"Метаданные: {chunk.metadata}\n")
      f.write("-" * 20 + "\n")
      wrapped_text = textwrap.fill(chunk.page_content, width=100)
      f.write(wrapped_text)
      f.write("\n\n" + "=" * 80 + "\n\n")

  print(f"✅ Все чанки сохранены в файл для анализа: {output_file_path}")


def _create_bm25_index(chunks: List[Document], config: Dict[str, Any]):
  """
  Создает и сохраняет на диск индекс BM25 для поиска по ключевым словам.
  """
  print("Создание BM25 индекса...")
  tokenized_corpus = [doc.page_content.split() for doc in chunks]
  bm25 = BM25Okapi(tokenized_corpus)

  bm25_index_path = config['retrievers']['bm25']['index_path']
  os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)
  with open(bm25_index_path, "wb") as f:
    pickle.dump({'bm25': bm25, 'docs': chunks}, f)

  print(f"✅ BM25 индекс сохранен в {bm25_index_path}")


def _create_vector_store(chunks: List[Document], config: Dict[str, Any]):
  """
  Создает и сохраняет на диск векторную базу данных (ChromaDB).
  """
  print("Создание векторной базы данных...")
  db_dir = config['retrievers']['vector_store']['db_path']
  model_name = config['embedding_model']['name']
  device = config['embedding_model']['device']

  embeddings_model = HuggingFaceEmbeddings(model_name=model_name,
                                           model_kwargs={'device': device})
  db = Chroma.from_documents(chunks, embeddings_model, persist_directory=db_dir)

  print(f"✅ Векторная база данных сохранена в {db_dir}")


def run_indexing(config: Dict[str, Any]):
  """
  Главная функция-оркестратор для полного пайплайна индексации всего сайта.
  """
  base_url = config['data_source']['url']
  max_depth = config['data_source'].get('max_depth', 2)
  sitemap_only = config['data_source'].get('sitemap_only', False)

  # Шаг 1: Обход сайта и сбор всех "сырых" документов
  crawler = WebsiteCrawler(base_url, max_depth=max_depth)
  raw_documents = crawler.crawl()

  # Шаг 1.1: Сохраняем карту сайта после обхода
  crawler.save_sitemap_to_file(config)

  if sitemap_only:
    print("\n✅ Режим 'sitemap_only' активен. Процесс индексации пропущен.")
    return

  if not raw_documents:
    print("Не найдено ни одного документа для индексации.")
    return

  # Шаг 2: Чанкинг каждого документа из списка
  chunker = HTMLContextChunker()
  all_chunks = []
  for doc in raw_documents:
    chunks_from_doc = chunker.chunk(doc)
    all_chunks.extend(chunks_from_doc)
  print(f"✅ Все документы разделены на {len(all_chunks)} структурных чанков.")

  if not all_chunks:
    print("Нет чанков для индексации после разделения.")
    return
  _save_chunks_to_file(all_chunks, base_url, config)

  # Шаг 3: Создание индекса BM25 по ВСЕМ чанкам
  _create_bm25_index(all_chunks, config)

  # Шаг 4: Создание векторной базы по ВСЕМ чанкам
  _create_vector_store(all_chunks, config)

  print("\n🎉 Индексация всего сайта успешно завершена!")