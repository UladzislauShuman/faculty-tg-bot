"""HTTP-клиент для индексации: Session с ретраями (DNS, таймауты, 5xx)."""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_indexing_session(
    *,
    total_retries: int = 4,
    backoff_factor: float = 0.75,
    verify_ssl: bool = False,
) -> requests.Session:
  """Session с urllib3 Retry на connect/read/status (GET/HEAD).

  ``total_retries`` — число повторных попыток после неудачи (итого до N+1 обращений).
  Поднимает устойчивость к временным сбоям DNS и сети в Docker/CI.
  """
  retry = Retry(
      total=total_retries,
      connect=total_retries,
      read=max(2, total_retries - 1),
      backoff_factor=backoff_factor,
      status_forcelist=(429, 500, 502, 503, 504),
      allowed_methods=frozenset(("GET", "HEAD")),
      raise_on_status=False,
  )
  adapter = HTTPAdapter(max_retries=retry)
  session = requests.Session()
  session.mount("https://", adapter)
  session.mount("http://", adapter)
  session.verify = verify_ssl
  return session
