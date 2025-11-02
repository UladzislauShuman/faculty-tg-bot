import os
from typing import List, Dict, Any
from langchain_core.runnables import Runnable
from src.util.yaml_parser import load_qa_test_set


def _format_result(index: int, question: str, expected_answer: str,
    bot_answer: str) -> str:
  """Форматирует результат одного тестового прогона в красивую строку."""
  return (
    f"{'=' * 50}\n"
    f"ВОПРОС #{index}: {question}\n"
    f"ОЖИДАЕМЫЙ ОТВЕТ: {expected_answer}\n"
    f"{'-' * 20}\n"
    f"ОТВЕТ БОТА:\n{bot_answer}\n"
    f"{'=' * 50}\n\n"
  )


def _run_test_loop(rag_chain: Runnable, test_set: List[Dict[str, str]],
    file_handle):
  """
  Основной цикл, который итерируется по тестовым вопросам и получает ответы от бота.
  """
  print("\nRAG-цепочка готова к работе. Начинаем оценку...\n")
  file_handle.write("RAG-цепочка готова к работе. Начинаем оценку...\n\n")

  for i, qa_pair in enumerate(test_set, 1):
    question = qa_pair.get('question')
    expected_answer = qa_pair.get('answer')

    if not question:
      continue

    print(f"--- Обработка вопроса #{i}: '{question}' ---")

    # Блокирующий вызов всей RAG-цепочки
    bot_answer = rag_chain.invoke(question)

    # Форматируем результат
    result_str = _format_result(i, question, expected_answer, bot_answer)

    # 1. Выводим в консоль и файл
    print(result_str)
    file_handle.write(result_str)

    # Принудительно сбрасываем буфер на диск, чтобы файл обновлялся в реальном времени
    file_handle.flush()


def run_evaluation_pipeline(rag_chain: Runnable, config: Dict[str, Any]):
  """
  Главная функция-оркестратор для оценки полной RAG-цепочки.
  Отвечает за настройку, управление файлами и вызов основного цикла тестирования.
  """
  output_dir = config['paths']['output_dir']
  qa_file_path = config['paths']['qa_test_set']
  output_filename = config['paths']['run_bot']
  output_file_path = os.path.join(output_dir, output_filename)

  os.makedirs(output_dir, exist_ok=True)

  test_set = load_qa_test_set(qa_file_path)
  if not test_set:
    print(f"Не удалось загрузить тестовый набор из {qa_file_path}")
    return

  print(f"Результаты оценки будут сохранены в: {output_file_path}")

  # Открываем файл и передаем его напрямую в цикл
  with open(output_file_path, 'w', encoding='utf-8') as f:
    _run_test_loop(rag_chain, test_set, f)

  print(f"✅ Оценка завершена. Результаты сохранены.")