"""Пакетный прогон RAG по qa-set: пишет ответы бота в файл paths.run_bot (без judge/trace)."""

import logging
import os
import uuid
from typing import Any, Dict

from langchain_core.runnables import Runnable

from src.util.yaml_parser import TestSetLoader

logger = logging.getLogger(__name__)

_PREVIEW_LEN = 120


def _preview(text: str, n: int = _PREVIEW_LEN) -> str:
  if len(text) <= n:
    return text
  return text[: n - 3] + "..."


def _format_result(
    index: str,
    question: str,
    expected_answer: str,
    bot_answer: str,
) -> str:
  """Форматирует один прогон для записи в файловый отчёт."""
  return (
      f"{'=' * 50}\n"
      f"ВОПРОС {index}: {question}\n"
      f"ОЖИДАЕМЫЙ ОТВЕТ: {expected_answer}\n"
      f"{'-' * 20}\n"
      f"ОТВЕТ БОТА:\n{bot_answer}\n"
      f"{'=' * 50}\n\n"
  )


async def run_evaluation_pipeline(
    rag_chain: Runnable,
    config: Dict[str, Any],
) -> None:
  """Проходит qa_pairs и сценарии из YAML, вызывает rag_chain, сбрасывает вывод в файл."""
  output_dir = config["paths"]["output_dir"]
  qa_file_path = config["paths"]["qa_test_set"]
  output_filename = config["paths"]["run_bot"]
  output_file_path = os.path.join(output_dir, output_filename)

  os.makedirs(output_dir, exist_ok=True)

  loader = TestSetLoader(qa_file_path)
  test_mode = config.get("evaluation_settings", {}).get("mode", "all")

  logger.info("Результаты оценки: %s", output_file_path)

  with open(output_file_path, "w", encoding="utf-8") as f:
    f.write(f"RAG-цепочка готова. Режим тестирования: {test_mode}\n\n")

    if test_mode in ("all", "questions"):
      qa_pairs = loader.get_qa_pairs()
      if qa_pairs:
        logger.info("Одиночные вопросы: %s шт", len(qa_pairs))
        f.write("=== ОДИНОЧНЫЕ ВОПРОСЫ ===\n\n")

        for i, qa_pair in enumerate(qa_pairs, 1):
          question = qa_pair.get("question")
          expected_answer = qa_pair.get("answer")
          if not question:
            continue

          logger.info("Вопрос #%s: %s", i, _preview(question))

          session_id = f"eval_q_{uuid.uuid4().hex[:8]}"
          response = await rag_chain.ainvoke(
              {"input": question},
              config={"configurable": {"session_id": session_id}},
          )

          result_str = _format_result(
              f"#{i}",
              question,
              expected_answer or "",
              response["answer"],
          )
          f.write(result_str)
          f.flush()

    if test_mode in ("all", "scenarios"):
      scenarios = loader.get_test_scenarios()
      if scenarios:
        logger.info("Многошаговые сценарии: %s шт", len(scenarios))
        f.write("=== МНОГОШАГОВЫЕ СЦЕНАРИИ ===\n\n")

        for sc in scenarios:
          logger.info("Сценарий: %s", sc["name"])
          session_id = f"eval_sc_{uuid.uuid4().hex[:8]}"
          f.write(
              f"--- СЦЕНАРИЙ: {sc['name']} (Session: {session_id}) ---\n\n"
          )

          for j, step in enumerate(sc["steps"], 1):
            question = step["q"]
            expected_answer = step["a"]

            logger.info("  Шаг %s: %s", j, _preview(question))
            response = await rag_chain.ainvoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}},
            )

            result_str = _format_result(
                f"[{sc['name']}] Шаг {j}",
                question,
                expected_answer,
                response["answer"],
            )
            f.write(result_str)
            f.flush()

  logger.info("Оценка завершена, файл: %s", output_file_path)
