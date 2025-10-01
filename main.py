# main.py
import argparse
import yaml
import sys
from dotenv import load_dotenv

from src.di_containers import Container
from src.pipelines.indexing import run_indexing
from src.evaluation.evaluate import run_evaluation_pipeline
from src.evaluation.evaluate_retrieval import run_retrieval_evaluation

def main():
    # Загружаем .env, чтобы LLM_PROVIDER был доступен сразу
    load_dotenv()

    # --- Настройка командной строки ---
    parser = argparse.ArgumentParser(description="RAG Pipeline for University Bot")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Команда только для полной индексации
    parser_index = subparsers.add_parser("index", help="Run the full data indexing pipeline.")

    # Команда только для построения карты сайта
    parser_sitemap = subparsers.add_parser("sitemap",
                                           help="Only crawl the site and build the sitemap.txt file.")

    # Команда для тестирования ретривера
    parser_retrieve = subparsers.add_parser("retrieve", help="Test the retrieval part of the pipeline.")
    parser_retrieve.add_argument("-q", "--query", type=str, help="A single question to test retrieval against.")

    # Команда для тестирования полной RAG-цепочки
    parser_answer = subparsers.add_parser("answer", help="Test the full RAG chain (retrieval + generation).")
    parser_answer.add_argument("-q", "--query", type=str, help="A single question to get an answer for.")
    
    args = parser.parse_args()

    try:
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Ошибка: Файл config/config.yaml не найден.")
        sys.exit(1)

    if args.command == 'index':
      config_data['data_source']['sitemap_only'] = False
      run_indexing(config_data)

    elif args.command == 'sitemap':
      config_data['data_source']['sitemap_only'] = True
      run_indexing(config_data)
    
    elif args.command == 'retrieve':
        container = Container()
        container.config.from_dict(config_data)
        retriever = container.final_retriever()
        
        if args.query:
            # Режим одиночного вопроса
            print(f"\nПоиск по вашему вопросу: '{args.query}'")
            docs = retriever.invoke(args.query)
            print("\n--- Найденные документы: ---")
            for doc in docs:
                print(f"Источник: {doc.metadata.get('source', 'N/A')}")
                print(f"Секция: {doc.metadata.get('heading', 'N/A')}")
                print(doc.page_content)
                print("-" * 20)
        else:
            # Режим оценки по qa-test-set.yaml
            run_retrieval_evaluation(retriever, config_data)
            
    elif args.command == 'answer':
        container = Container()
        container.config.from_dict(config_data)
        rag_chain = container.rag_chain()
        
        if args.query:
            # Режим одиночного вопроса
            print(f"\nГенерация ответа на ваш вопрос: '{args.query}'")
            print("\n--- Ответ Бота: ---")
            response = rag_chain.invoke(args.query)
            print(response)
        else:
            # Режим оценки по qa-test-set.yaml
            run_evaluation_pipeline(rag_chain, config_data)

if __name__ == "__main__":
    main()