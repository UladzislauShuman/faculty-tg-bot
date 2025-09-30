import os
import sys
from src.util.yaml_parser import load_qa_test_set

def run_retrieval_evaluation(retriever, config: dict):
    """
    Запускает прогон тестовых вопросов через ретривер и сохраняет результаты.
    """
    output_dir = config['paths']['output_dir']
    qa_file_path = config['paths']['qa_test_set']
    
    output_filename = config['paths']['retriever']
    output_file_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    original_stdout = sys.stdout
    print(f"Результаты оценки ретривера будут сохранены в: {output_file_path}")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        sys.stdout = f

        test_set = load_qa_test_set(qa_file_path)
        if not test_set:
            sys.stdout = original_stdout
            print(f"Не удалось загрузить тестовый набор из {qa_file_path}")
            return

        print("\nРетривер готов к работе. Начинаем оценку поиска...\n")

        for i, qa_pair in enumerate(test_set, 1):
            question = qa_pair.get('question')
            if not question:
                continue

            relevant_docs = retriever.invoke(question)

            print("="*50)
            print(f"ВОПРОС #{i}: {question}")
            print("="*50)
            
            if not relevant_docs:
                print(">>> Релевантных документов не найдено.\n")
                continue

            print(">>> Найденные релевантные документы:\n")
            for j, doc in enumerate(relevant_docs, 1):
                source = doc.metadata.get('source', 'N/A')
                heading = doc.metadata.get('heading', 'N/A')
                print(f"--- Документ #{j} (Источник: {source}, Секция: {heading}) ---")
                print(doc.page_content)
                print("-" * 40 + "\n")
    
    sys.stdout = original_stdout
    print(f"✅ Оценка ретривера завершена. Результаты сохранены.")