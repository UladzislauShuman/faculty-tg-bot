import os
import uuid
from typing import List, Dict, Any
from langchain_core.runnables import Runnable
from src.util.yaml_parser import TestSetLoader


def _format_result(index: str, question: str, expected_answer: str,
    bot_answer: str) -> str:
  """Форматирует результат одного тестового прогона в красивую строку."""
  return (
    f"{'=' * 50}\n"
    f"ВОПРОС {index}: {question}\n"
    f"ОЖИДАЕМЫЙ ОТВЕТ: {expected_answer}\n"
    f"{'-' * 20}\n"
    f"ОТВЕТ БОТА:\n{bot_answer}\n"
    f"{'=' * 50}\n\n"
  )


async def run_evaluation_pipeline(rag_chain: Runnable, config: Dict[str, Any]):
  """
  Главная функция-оркестратор для пакетной генерации ответов (Answer).
  """
  output_dir = config['paths']['output_dir']
  qa_file_path = config['paths']['qa_test_set']
  output_filename = config['paths']['run_bot']
  output_file_path = os.path.join(output_dir, output_filename)

  os.makedirs(output_dir, exist_ok=True)

  loader = TestSetLoader(qa_file_path)
  test_mode = config.get('evaluation_settings', {}).get('mode', 'all')

  print(f"Результаты оценки будут сохранены в: {output_file_path}")

  with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(f"RAG-цепочка готова. Режим тестирования: {test_mode}\n\n")

    # --- БЛОК 1: ОДИНОЧНЫЕ ВОПРОСЫ ---
    if test_mode in ['all', 'questions']:
      qa_pairs = loader.get_qa_pairs()
      if qa_pairs:
        print(f"\n--- ОДИНОЧНЫЕ ВОПРОСЫ ({len(qa_pairs)} шт) ---")
        f.write("=== ОДИНОЧНЫЕ ВОПРОСЫ ===\n\n")

        for i, qa_pair in enumerate(qa_pairs, 1):
          question = qa_pair.get('question')
          expected_answer = qa_pair.get('answer')
          if not question: continue

          print(f"Вопрос #{i}: '{question[:50]}...'")

          # Изолированная сессия для каждого одиночного вопроса
          session_id = f"eval_q_{uuid.uuid4().hex[:8]}"
          response = await rag_chain.ainvoke(
              {"input": question},
              config={"configurable": {"session_id": session_id}}
          )

          result_str = _format_result(f"#{i}", question, expected_answer,
                                      response["answer"])
          f.write(result_str)
          f.flush()

    # --- БЛОК 2: МНОГОШАГОВЫЕ СЦЕНАРИИ ---
    if test_mode in ['all', 'scenarios']:
      scenarios = loader.get_test_scenarios()
      if scenarios:
        print(f"\n--- МНОГОШАГОВЫЕ СЦЕНАРИИ ({len(scenarios)} шт) ---")
        f.write("=== МНОГОШАГОВЫЕ СЦЕНАРИИ ===\n\n")

        for sc in scenarios:
          print(f"\nСценарий: {sc['name']}")
          # Единая сессия для всего диалога
          session_id = f"eval_sc_{uuid.uuid4().hex[:8]}"
          f.write(f"--- СЦЕНАРИЙ: {sc['name']} (Session: {session_id}) ---\n\n")

          for j, step in enumerate(sc['steps'], 1):
            question = step['q']
            expected_answer = step['a']

            print(f"  Шаг {j}: '{question[:50]}...'")
            response = await rag_chain.ainvoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
            )

            result_str = _format_result(f"[{sc['name']}] Шаг {j}", question,
                                        expected_answer, response["answer"])
            f.write(result_str)
            f.flush()

  print(f"\n✅ Оценка завершена. Результаты сохранены в {output_file_path}")