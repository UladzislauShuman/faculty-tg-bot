"""Обход сайта по ссылкам в пределах домена (BFS). Только сбор URL, без чанкинга."""
import logging
from typing import List, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.util.http_fetch import create_indexing_session

logger = logging.getLogger(__name__)


class WebsiteCrawler:
  """Собирает уникальные URL в рамках домена до max_depth."""

  def __init__(self, base_url: str, max_depth: int = 2):
    parsed_url = urlparse(base_url)
    self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    self.domain = parsed_url.netloc
    self.visited_urls: Set[str] = set()
    self.max_depth = max_depth
    self.headers = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    self._http = create_indexing_session()

  def _is_valid_url(self, url: str) -> bool:
    """Только http(s), тот же домен, без очевидных файлов и admin/user путей."""
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ['http', 'https']:
      return False
    if parsed_url.netloc and parsed_url.netloc != self.domain:
      return False
    if any(url.endswith(ext) for ext in
           ['.pdf', '.jpg', '.png', '.zip', '.docx', '.xlsx']):
      return False
    if '/admin' in parsed_url.path or '/user' in parsed_url.path:
      return False
    return True

  def crawl(self) -> List[str]:
    """BFS: берём URL из очереди, парсим ссылки, пока не исчерпана глубина."""
    urls_to_visit = [(self.base_url, 0)]

    while urls_to_visit:
      current_url, current_depth = urls_to_visit.pop(0)
      current_url = urljoin(current_url, urlparse(current_url).path)

      if current_url in self.visited_urls or current_depth > self.max_depth:
        continue

      logger.info("Краулер depth=%s url=%s", current_depth, current_url)
      self.visited_urls.add(current_url)

      try:
        response = self._http.get(
            current_url, headers=self.headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        for link in soup.find_all('a', href=True):
          absolute_url = urljoin(self.base_url, link['href'])
          if self._is_valid_url(
              absolute_url) and absolute_url not in self.visited_urls:
            urls_to_visit.append((absolute_url, current_depth + 1))
      except requests.RequestException as e:
        logger.warning("Не удалось загрузить %s: %s", current_url, e)
        continue

    logger.info("Краулер завершён: уникальных URL=%s", len(self.visited_urls))
    return list(self.visited_urls)
