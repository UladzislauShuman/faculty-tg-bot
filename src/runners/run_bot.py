import os
import sys
from dotenv import load_dotenv
from src.components.rag_bot import get_rag_chain
from src.util.yaml_parser import load_qa_test_set

load_dotenv()
QA_FILE_PATH = os.getenv("QA_FILE_PATH", "qa-test-set.yaml")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

def run_evaluation():
    output_file_path = os.path.join(OUTPUT_DIR, "run_bot-output.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    original_stdout = sys.stdout
    with open(output_file_path, 'w', encoding='utf-8') as f:
        sys.stdout = f

        test_set = load_qa_test_set(QA_FILE_PATH)
        if not test_set:
            sys.stdout = original_stdout
            print(f"Не удалось загрузить тестовый набор из {QA_FILE_PATH}")
            return

        print("\nИнициализация RAG-цепочки... (может занять время)")
        rag_chain = get_rag_chain()
        print("RAG-цепочка готова к работе.\n")

        for i, qa_pair in enumerate(test_set, 1):
            question = qa_pair.get('question')
            expected_answer = qa_pair.get('answer')

            if not question:
                continue

            bot_answer = rag_chain.invoke(question)

            print("="*30)
            print(f"Вопрос #{i}")
            print(f"Формулировка: {question}")
            print(f"Мой ответ: {expected_answer}")
            print(f"Ответ бота: {bot_answer}")
            print("="*30 + "\n")
    
    sys.stdout = original_stdout
    print(f"✅ Результаты ответов бота сохранены в файл: {output_file_path}")


if __name__ == "__main__":
    run_evaluation()