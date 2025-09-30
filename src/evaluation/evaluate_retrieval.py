
import os
import sys
from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from src.util.yaml_parser import load_qa_test_set


def _format_retrieval_result(index: int, question: str,
    relevant_docs: List[Document]) -> str:
  """
  Форматирует результат поиска по одному вопросу в красивую строку.
  """
  header = f"{'=' * 50}\nВОПРОС #{index}: {question}\n{'=' * 50}\n"

  if not relevant_docs:
    return header + ">>> Релевантных документов не найдено.\n\n"

  docs_str_list = []
  for j, doc in enumerate(relevant_docs, 1):
    source = doc.metadata.get('source', 'N/A')
    heading = doc.metadata.get('heading', 'N/A')

    doc_header = f"--- Документ #{j} (Источник: {source}, Секция: {heading}) ---\n"
    doc_content = doc.page_content
    doc_footer = "\n" + "-" * 40 + "\n"

    docs_str_list.append(doc_header + doc_content + doc_footer)

  return header + ">>> Найденные релевантные документы:\n\n" + "\n".join(
    docs_str_list)


def _run_retrieval_loop(retriever: BaseRetriever,
    test_set: List[Dict[str, str]], file_handle):
  """
  Основной цикл, который итерируется по тестовым вопросам и выполняет поиск документов.
  """
  print("\nРетривер готов к работе. Начинаем оценку поиска...\n",
        file=file_handle)

  for i, qa_pair in enumerate(test_set, 1):
    question = qa_pair.get('question')
    if not question:
      continue

    relevant_docs = retriever.invoke(question)

    result_str = _format_retrieval_result(i, question, relevant_docs)
    file_handle.write(result_str)
    print(f"Обработан вопрос #{i}...")


def run_retrieval_evaluation(retriever: BaseRetriever, config: Dict[str, Any]):
  """
  Главная функция-оркестратор для оценки ретривера.
  """
  output_dir = config['paths']['output_dir']
  qa_file_path = config['paths']['qa_test_set']
  output_filename = config['paths']['retriever']
  output_file_path = os.path.join(output_dir, output_filename)

  os.makedirs(output_dir, exist_ok=True)

  original_stdout = sys.stdout
  print(f"Результаты оценки ретривера будут сохранены в: {output_file_path}")

  try:
    with open(output_file_path, 'w', encoding='utf-8') as f:
      sys.stdout = f

      test_set = load_qa_test_set(qa_file_path)
      if not test_set:
        print(f"Не удалось загрузить тестовый набор из {qa_file_path}")
        return

      _run_retrieval_loop(retriever, test_set, f)

  finally:
    sys.stdout = original_stdout

  print(f"✅ Оценка ретривера завершена. Результаты сохранены.")