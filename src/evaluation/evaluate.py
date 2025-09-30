import os
import sys
from src.util.yaml_parser import load_qa_test_set

def run_evaluation_pipeline(rag_chain, config: dict):
    """
    Запускает прогон тестового набора вопросов через RAG-цепочку и сохраняет результаты.
    """
    output_dir = config['paths']['output_dir']
    qa_file_path = config['paths']['qa_test_set']
    
    output_filename = config['paths']['run_bot']
    output_file_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    original_stdout = sys.stdout
    print(f"Результаты оценки будут сохранены в: {output_file_path}")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        sys.stdout = f

        test_set = load_qa_test_set(qa_file_path)
        if not test_set:
            sys.stdout = original_stdout
            print(f"Не удалось загрузить тестовый набор из {qa_file_path}")
            return

        print("\nRAG-цепочка готова к работе. Начинаем оценку...\n")

        for i, qa_pair in enumerate(test_set, 1):
            question = qa_pair.get('question')
            expected_answer = qa_pair.get('answer')

            if not question:
                continue

            # Получаем ответ от бота, используя переданную цепочку
            bot_answer = rag_chain.invoke(question)

            print("="*30)
            print(f"Вопрос #{i}")
            print(f"Формулировка: {question}")
            print(f"Мой ответ: {expected_answer}")
            print(f"Ответ бота: {bot_answer}")
            print("="*30 + "\n")
    
    sys.stdout = original_stdout
    print(f"✅ Оценка завершена. Результаты сохранены.")