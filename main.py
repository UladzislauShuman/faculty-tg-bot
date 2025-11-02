import argparse
import yaml
import sys
from dotenv import load_dotenv

from src.di_containers import Container
from src.pipelines.indexing.pipeline import run_indexing
from src.evaluation.evaluate import run_evaluation_pipeline
from src.evaluation.evaluate_retrieval import run_retrieval_evaluation


def main():
  load_dotenv()
  parser = argparse.ArgumentParser(
    description="RAG Pipeline for University Bot")
  subparsers = parser.add_subparsers(dest="command", required=True,
                                     help="Available commands")

  parser_index = subparsers.add_parser("index",
                                       help="Run the data indexing pipeline.")
  parser_index.add_argument(
      "mode",
      type=str,
      nargs="?",
      default="full",
      choices=["full", "test"],
      help="Set indexing mode: 'full' (crawl site, default) or 'test' (use test_urls)."
  )

  parser_retrieve = subparsers.add_parser("retrieve",
                                          help="Test the retrieval part of the pipeline.")
  parser_retrieve.add_argument("-q", "--query", type=str,
                               help="A single question to test retrieval against.")

  parser_answer = subparsers.add_parser("answer",
                                        help="Test the full RAG chain.")
  parser_answer.add_argument("-q", "--query", type=str,
                             help="A single question to get an answer for.")

  args = parser.parse_args()

  try:
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
      config_data = yaml.safe_load(f)
  except FileNotFoundError:
    print("❌ Ошибка: Файл config/config.yaml не найден.")
    sys.exit(1)

  container = Container()
  container.config.from_dict(config_data)

  if args.command == 'index':
    print(f"\n🚀 Запуск индексации в режиме '{args.mode}'...")
    processor = container.data_processor()
    run_indexing(config=config_data, processor=processor, mode=args.mode)


  elif args.command == 'retrieve':
    # --- ИСПРАВЛЕННАЯ ЛОГИКА ---
    # Теперь мы просим у контейнера именно цепочку для ретривинга.
    retrieval_step = container.retrieval_chain()
    if args.query:
      print(f"\nПоиск по вашему вопросу: '{args.query}'")
      # Мы используем эту цепочку напрямую. Она уже включает query expansion.
      docs = retrieval_step.invoke(args.query)
      print("\n--- Найденные документы: ---")
      for doc in docs:
        print(f"Источник: {doc.metadata.get('source', 'N/A')}")
        print(f"Заголовок: {doc.metadata.get('title', 'N/A')}")
        if 'H2' in doc.metadata:
          print(f"Секция: {doc.metadata.get('H2', 'N/A')}")
        print(doc.page_content)
        print("-" * 20)
    else:
      # Передаем эту чистую цепочку в функцию оценки.
      run_retrieval_evaluation(retrieval_step, config_data)

  elif args.command == 'answer':
    rag_chain = container.rag_chain()
    if args.query:
      print(f"\nГенерация ответа на ваш вопрос: '{args.query}'")
      print("\n--- Ответ Бота: ---")
      response = rag_chain.invoke(args.query)
      print(response)
    else:
      run_evaluation_pipeline(rag_chain, config_data)


if __name__ == "__main__":
  main()