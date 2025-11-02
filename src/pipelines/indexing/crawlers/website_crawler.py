import requests
from typing import List, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup


class WebsiteCrawler:
  """
  Класс для обхода веб-сайта и сбора URL-адресов страниц.
  Его единственная задача — составить список ссылок для дальнейшей обработки.
  Он не занимается парсингом контента.
  """

  def __init__(self, base_url: str, max_depth: int = 2):
    """
    Конструктор краулера.

    Args:
        base_url (str): Начальный URL для обхода.
        max_depth (int): Максимальная глубина рекурсивного обхода.
    """
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
    Внутренний метод для проверки, является ли URL валидным для обхода.
    Игнорирует внешние сайты, ссылки на файлы и административные разделы.
    """
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
    """
    Запускает процесс обхода сайта и возвращает список всех найденных уникальных URL.
    """
    urls_to_visit = [(self.base_url, 0)]

    while urls_to_visit:
      current_url, current_depth = urls_to_visit.pop(0)

      # Нормализуем URL, чтобы убрать параметры и фрагменты, избегая дубликатов.
      current_url = urljoin(current_url, urlparse(current_url).path)

      if current_url in self.visited_urls or current_depth > self.max_depth:
        continue

      print(f"🕸️  [Глубина {current_depth}] Обход страницы: {current_url}")
      self.visited_urls.add(current_url)

      try:
        response = requests.get(current_url, headers=self.headers, verify=False,
                                timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        # Ищем все теги <a> с атрибутом href
        for link in soup.find_all('a', href=True):
          absolute_url = urljoin(self.base_url, link['href'])
          if self._is_valid_url(
              absolute_url) and absolute_url not in self.visited_urls:
            # Добавляем в очередь только новые, валидные URL
            urls_to_visit.append((absolute_url, current_depth + 1))
      except requests.RequestException as e:
        print(f"❌ Не удалось загрузить {current_url} для поиска ссылок: {e}")
        continue

    print(
      f"\n✅ Обход сайта завершен. Найдено {len(self.visited_urls)} уникальных страниц.")
    return list(self.visited_urls)