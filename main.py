import argparse
import yaml
import sys
import os
import shutil
import asyncio
from dotenv import load_dotenv

from src.di_containers import Container
from src.pipelines.indexing.pipeline import run_indexing
from src.evaluation.runner import TestPipelineRunner


# --- УТИЛИТЫ ---

def manage_db_state_for_test(config_data: dict, force_reindex: bool):
  """Управляет путями ТОЛЬКО для ТЕСТОВОГО режима."""
  db_path = config_data['retrievers']['vector_store']['db_path']
  bm25_path = config_data['retrievers']['bm25']['index_path']

  exists = os.path.exists(db_path) and os.path.exists(bm25_path)

  if force_reindex:
    if exists:
      print(f"♻️  [TEST] Удаление старых индексов: {db_path}...")
      try:
        shutil.rmtree(db_path)
        if os.path.exists(bm25_path): os.remove(bm25_path)
      except OSError as e:
        print(f"❌ Ошибка: Файлы БД заняты. {e}")
        sys.exit(1)
    return True

  if not exists:
    print(f"⚠️[TEST] Индексы не найдены. Требуется создание.")
    return True

  print(f"✅  [TEST] Найдены существующие индексы: {db_path}")
  return False


# --- MAIN CLI ---

def main():
  load_dotenv()
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command")

  # TEST
  test_parser = subparsers.add_parser("test")
  test_parser.add_argument("eval_mode", nargs="?",
                           choices=["all", "questions", "scenarios"],
                           help="Режим тестирования")
  test_parser.add_argument("--chunker", type=str, default="markdown",
                           choices=["markdown", "semantic", "unstructured"])
  test_parser.add_argument("--retriever", type=str, default="hybrid")
  test_parser.add_argument("--index-mode", type=str, default="test",
                           choices=["test", "full"])
  test_parser.add_argument("--force-index", action="store_true")

  # INDEX (Production)
  idx_parser = subparsers.add_parser("index")
  idx_parser.add_argument("mode", nargs="?", default="full",
                          choices=["full", "test"])
  idx_parser.add_argument("--chunker", type=str, default="markdown",
                          choices=["markdown", "semantic", "unstructured"])

  # RETRIEVE
  subparsers.add_parser("retrieve").add_argument("-q", "--query")

  # ANSWER
  ans_parser = subparsers.add_parser("answer")
  ans_parser.add_argument("eval_mode", nargs="?",
                          choices=["all", "questions", "scenarios"],
                          help="Режим тестирования")
  ans_parser.add_argument("-q", "--query")

  args = parser.parse_args()

  try:
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
      config = yaml.safe_load(f)
  except Exception as e:
    sys.exit(f"Config missing or invalid: {e}")

  # --- Переопределение режима из CLI ---
  if hasattr(args, 'eval_mode') and args.eval_mode:
    if 'evaluation_settings' not in config:
      config['evaluation_settings'] = {}
    config['evaluation_settings']['mode'] = args.eval_mode
    print(f"🔧 Режим тестирования переопределен через CLI: {args.eval_mode}")

  # Настройка путей для ТЕСТА
  if args.command == "test":
    base_db = config['retrievers']['vector_store'].get('db_path_base',
                                                       'data/chroma_db')
    config['retrievers']['vector_store'][
      'db_path'] = f"{base_db}_{args.chunker}"

    base_bm25 = config['retrievers']['bm25'].get('index_path_base',
                                                 'data/bm25_index')
    base, ext = os.path.splitext(base_bm25)
    config['retrievers']['bm25']['index_path'] = f"{base}_{args.chunker}{ext}"

    args.need_index = manage_db_state_for_test(config, args.force_index)

  container = Container()
  container.config.from_dict(config)

  if args.command == "test":
    runner = TestPipelineRunner(container, config)
    asyncio.run(runner.run(args))

  elif args.command == "index":
    print(f"🚀 Запуск ПРОДАКШН индексации (Chunker: {args.chunker})...")
    prod_db_path = config['retrievers']['vector_store']['db_path']
    prod_bm25_path = config['retrievers']['bm25']['index_path']
    print(f"⚠️  ВНИМАНИЕ: Полная очистка текущей базы: {prod_db_path}")
    try:
      shutil.rmtree(prod_db_path)
      if os.path.exists(prod_bm25_path): os.remove(prod_bm25_path)
    except OSError as e:
      sys.exit(f"❌ Не удалось очистить базу: {e}\nОстановите бота.")

    try:
      processor_provider = getattr(container, f"{args.chunker}_processor")
      container.data_processor.override(processor_provider)
    except AttributeError:
      sys.exit(f"❌ Ошибка: Чанкер {args.chunker} не найден.")

    processor = container.data_processor()
    run_indexing(config, processor, args.mode)

  elif args.command == "retrieve":
    retrieval_step = container.retrieval_chain()
    if args.query:
      print(f"Поиск: {args.query}")
      docs = retrieval_step.invoke(args.query)
      for d in docs:
        print(
            f"\n--- {d.metadata.get('title', 'Doc')} ---\n{d.page_content[:200]}...")


  elif args.command == "answer":
    rag_chain = container.rag_chain()
    if args.query:
      # Одиночный вопрос из консоли
      print(f"Вопрос: {args.query}")
      response = asyncio.run(rag_chain.ainvoke(
          {"input": args.query},
          config={"configurable": {"session_id": "cli_test_session"}}
      ))
      print(response["answer"])
    else:
      # Пакетное тестирование
      from src.evaluation.evaluate import run_evaluation_pipeline
      asyncio.run(run_evaluation_pipeline(rag_chain, config))


if __name__ == "__main__":
  main()