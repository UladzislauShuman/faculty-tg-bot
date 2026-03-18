import time
import logging
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ProfilingCallbackHandler(BaseCallbackHandler):
  """
  Перехватывает события LangChain для замера времени выполнения каждого этапа RAG.
  """

  def __init__(self):
    self.starts: Dict[str, float] = {}

  def on_retriever_start(self, serialized: Dict[str, Any], query: str,
      **kwargs: Any) -> None:
    logger.info(f"🔍 [RETRIEVER] Начинаем поиск для запроса: '{query}'")
    self.starts['retriever'] = time.perf_counter()

  def on_retriever_end(self, documents: List[Document], **kwargs: Any) -> None:
    elapsed = time.perf_counter() - self.starts.get('retriever',
                                                    time.perf_counter())
    logger.info(
        f"✅ [RETRIEVER] Поиск завершен за {elapsed:.2f} сек. Найдено документов: {len(documents)}")


  def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        logger.info(f"🧠 [LLM] Начинаем генерацию ответа...")
        self.starts['llm'] = time.perf_counter()

  def on_chat_model_start(self, serialized: Dict[str, Any],
      messages: List[List[Any]], **kwargs: Any) -> None:
    logger.info(f"💬[CHAT MODEL] Начинаем генерацию ответа...")
    self.starts['llm'] = time.perf_counter()

  def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
    elapsed = time.perf_counter() - self.starts.get('llm', time.perf_counter())
    logger.info(f"✅ [LLM] Генерация завершена за {elapsed:.2f} сек.")