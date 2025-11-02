import os
from typing import List, Dict, Any
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from src.util.yaml_parser import load_qa_test_set


def _format_retrieval_result(index: int, question: str,
    relevant_docs: List[Document]) -> str:
  """Форматирует результат поиска по одному вопросу в красивую строку."""
  header = f"{'=' * 50}\nВОПРОС #{index}: {question}\n{'=' * 50}\n"

  if not relevant_docs:
    return header + ">>> Релевантных документов не найдено.\n\n"

  docs_str_list = []
  for j, doc in enumerate(relevant_docs, 1):
    source = doc.metadata.get('source', 'N/A')
    heading = doc.metadata.get('H2', 'N/A')

    doc_header = f"--- Документ #{j} (Источник: {source}, Секция: {heading}) ---\n"
    doc_content = doc.page_content
    doc_footer = "\n" + "-" * 40 + "\n"
    docs_str_list.append(doc_header + doc_content + doc_footer)

  return header + ">>> Найденные релевантные документы:\n\n" + "\n".join(
    docs_str_list)


def _run_retrieval_loop(retriever_chain: Runnable,
    test_set: List[Dict[str, str]], file_handle):
  """
  Основной цикл, который итерируется по тестовым вопросам, выполняет поиск
  и выводит результаты в реальном времени.
  """
  print("\nРетривер готов к работе. Начинаем оценку поиска...\n")
  file_handle.write("Ретривер готов к работе. Начинаем оценку поиска...\n\n")

  for i, qa_pair in enumerate(test_set, 1):
    question = qa_pair.get('question')
    if not question:
      continue

    print(f"--- Обработка вопроса #{i}: '{question}' ---")

    relevant_docs = retriever_chain.invoke(question)

    # Форматируем результат
    result_str = _format_retrieval_result(i, question, relevant_docs)

    # Выводим в консоль и файл
    print(result_str)
    file_handle.write(result_str)

    # Принудительно сбрасываем буфер на диск
    file_handle.flush()


def run_retrieval_evaluation(retriever_chain: Runnable, config: Dict[str, Any]):
  """
  Главная функция-оркестратор для оценки ретривера.
  """
  output_dir = config['paths']['output_dir']
  qa_file_path = config['paths']['qa_test_set']
  output_filename = config['paths']['retriever']
  output_file_path = os.path.join(output_dir, output_filename)

  os.makedirs(output_dir, exist_ok=True)

  test_set = load_qa_test_set(qa_file_path)
  if not test_set:
    print(f"Не удалось загрузить тестовый набор из {qa_file_path}")
    return

  print(f"Результаты оценки ретривера будут сохранены в: {output_file_path}")

  with open(output_file_path, 'w', encoding='utf-8') as f:
    _run_retrieval_loop(retriever_chain, test_set, f)

  print(f"✅ Оценка ретривера завершена. Результаты сохранены.")