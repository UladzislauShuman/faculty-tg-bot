"""Оценка только ретривера: вопросы из qa-set, вывод найденных документов в файл paths.retriever."""

import logging
import os
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from src.util.yaml_parser import TestSetLoader

logger = logging.getLogger(__name__)

_PREVIEW_LEN = 120


def _preview(text: str, n: int = _PREVIEW_LEN) -> str:
  if len(text) <= n:
    return text
  return text[: n - 3] + "..."


def _format_retrieval_result(
    index: int,
    question: str,
    relevant_docs: List[Document],
) -> str:
  """Форматирует результат поиска по одному вопросу для файла."""
  header = f"{'=' * 50}\nВОПРОС #{index}: {question}\n{'=' * 50}\n"

  if not relevant_docs:
    return header + ">>> Релевантных документов не найдено.\n\n"

  docs_str_list = []
  for j, doc in enumerate(relevant_docs, 1):
    source = doc.metadata.get("source", "N/A")
    heading = doc.metadata.get("H2", "N/A")

    doc_header = (
        f"--- Документ #{j} (Источник: {source}, Секция: {heading}) ---\n"
    )
    doc_content = doc.page_content
    doc_footer = "\n" + "-" * 40 + "\n"
    docs_str_list.append(doc_header + doc_content + doc_footer)

  return (
      header
      + ">>> Найденные релевантные документы:\n\n"
      + "\n".join(docs_str_list)
  )


def _run_retrieval_loop(
    retriever_chain: Runnable,
    test_set: List[Dict[str, str]],
    file_handle: Any,
) -> None:
  """Итерирует вопросы, вызывает retriever, пишет полный отчёт в file_handle."""
  logger.info("Ретривер готов, начинаем оценку поиска.")
  file_handle.write("Ретривер готов к работе. Начинаем оценку поиска...\n\n")

  for i, qa_pair in enumerate(test_set, 1):
    question = qa_pair.get("question")
    if not question:
      continue

    relevant_docs = retriever_chain.invoke(question)
    logger.info(
        "Вопрос #%s: %s | chunks=%s",
        i,
        _preview(question),
        len(relevant_docs),
    )

    result_str = _format_retrieval_result(i, question, relevant_docs)

    file_handle.write(result_str)
    file_handle.flush()


def run_retrieval_evaluation(
    retriever_chain: Runnable,
    config: Dict[str, Any],
) -> None:
  """Загружает qa_pairs и пишет результаты retrieve в output_dir."""
  output_dir = config["paths"]["output_dir"]
  qa_file_path = config["paths"]["qa_test_set"]
  output_filename = config["paths"]["retriever"]
  output_file_path = os.path.join(output_dir, output_filename)

  os.makedirs(output_dir, exist_ok=True)

  loader = TestSetLoader(qa_file_path)
  test_set = loader.get_qa_pairs()

  if not test_set:
    logger.error("Пустой или не загружен тестовый набор: %s", qa_file_path)
    return

  logger.info("Результаты ретривера: %s", output_file_path)

  with open(output_file_path, "w", encoding="utf-8") as f:
    _run_retrieval_loop(retriever_chain, test_set, f)

  logger.info("Оценка ретривера завершена, файл: %s", output_file_path)
