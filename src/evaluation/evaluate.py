import os
import sys
from typing import List, Dict, Any
from langchain_core.runnables import Runnable
from src.util.yaml_parser import load_qa_test_set


def _format_result(index: int, question: str, expected_answer: str,
    bot_answer: str) -> str:
  """
  Форматирует результат одного тестового прогона в красивую строку.

  """
  return (
    f"{'=' * 30}\n"
    f"Вопрос #{index}\n"
    f"Формулировка: {question}\n"
    f"Мой ответ: {expected_answer}\n"
    f"Ответ бота: {bot_answer}\n"
    f"{'=' * 30}\n\n"
  )


def _run_test_loop(rag_chain: Runnable, test_set: List[Dict[str, str]],
    file_handle):
  """
  Основной цикл, который итерируется по тестовым вопросам и получает ответы от бота.

  Args:
      rag_chain: Готовая RAG-цепочка, которую мы будем тестировать.
                 С точки зрения этого скрипта, это "черный ящик", который принимает
                 вопрос и возвращает ответ.
      test_set: Список словарей, где каждый словарь содержит 'question' и 'answer'.
      file_handle: Открытый файловый дескриптор, куда будут записываться результаты.
  """
  print("\nRAG-цепочка готова к работе. Начинаем оценку...\n", file=file_handle)

  for i, qa_pair in enumerate(test_set, 1):
    question = qa_pair.get('question')
    expected_answer = qa_pair.get('answer')

    if not question:
      continue

    # запуск всей RAG-цепочки,
    bot_answer = rag_chain.invoke(question)

    # Форматируем и записываем результат в файл
    result_str = _format_result(i, question, expected_answer, bot_answer)
    file_handle.write(result_str)

    print(f"Обработан вопрос #{i}...")


def run_evaluation_pipeline(rag_chain: Runnable, config: Dict[str, Any]):
  """
  Главная функция-оркестратор.

  Отвечает за настройку, управление файлами и вызов основного цикла тестирования.
  """
  # 1. Получаем все необходимые пути из централизованного конфига.
  output_dir = config['paths']['output_dir']
  qa_file_path = config['paths']['qa_test_set']
  output_filename = config['paths']['run_bot']
  output_file_path = os.path.join(output_dir, output_filename)

  # Убеждаемся, что папка для результатов существует.
  os.makedirs(output_dir, exist_ok=True)

  # Сохраняем оригинальный поток вывода (консоль).
  original_stdout = sys.stdout
  print(f"Результаты оценки будут сохранены в: {output_file_path}")

  try:

    with open(output_file_path, 'w', encoding='utf-8') as f:
      # 3. Перенаправляем весь стандартный вывод (все команды print) в этот файл.
      sys.stdout = f

      test_set = load_qa_test_set(qa_file_path)
      if not test_set:
        print(f"Не удалось загрузить тестовый набор из {qa_file_path}")
        return

      # 4. Запускаем основной цикл тестирования
      _run_test_loop(rag_chain, test_set, f)

  finally:
    # 5. Восстанавливаем стандартный вывод обратно в консоль.
    sys.stdout = original_stdout

  print(f"✅ Оценка завершена. Результаты сохранены.")