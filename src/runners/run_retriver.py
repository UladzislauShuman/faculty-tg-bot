import os
import sys
from dotenv import load_dotenv
from src.util.yaml_parser import load_qa_test_set
from src.components.query_pipeline import load_retriever

load_dotenv()
QA_FILE_PATH = os.getenv("QA_FILE_PATH", "qa-test-set.yaml")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

def run_retrieval_evaluation():
    output_file_path = os.path.join(OUTPUT_DIR, "run_retriver-output.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    original_stdout = sys.stdout
    with open(output_file_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        
        test_set = load_qa_test_set(QA_FILE_PATH)
        if not test_set:
            sys.stdout = original_stdout
            print(f"Не удалось загрузить тестовый набор из {QA_FILE_PATH}")
            return

        print("\nИнициализация ретривера...")
        retriever = load_retriever()
        print("Ретривер готов к работе.\n")

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
                print(f"--- Документ #{j} (Источник: {source}) ---")
                print(doc.page_content)
                print("-" * 40 + "\n")
    
    sys.stdout = original_stdout
    print(f"✅ Результаты поиска по ретриверу сохранены в файл: {output_file_path}")
    
    
if __name__ == "__main__":
    run_retrieval_evaluation()