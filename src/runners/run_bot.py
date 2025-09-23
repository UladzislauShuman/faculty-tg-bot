import os # <-- ДОБАВЛЕНО
from dotenv import load_dotenv # <-- ДОБАВЛЕНО
from src.components.rag_bot import get_rag_chain
from src.util.yaml_parser import load_qa_test_set

QA_FILE_PATH = "qa-test-set.yaml"

def run_evaluation():
    # Загружаем тестовый набор
    test_set = load_qa_test_set(QA_FILE_PATH)
    if not test_set:
        return

    # Создаем нашу RAG-цепочку один раз
    print("\nИнициализация RAG-цепочки... (может занять время)")
    rag_chain = get_rag_chain()
    print("RAG-цепочка готова к работе.\n")

    # Проходим по всем вопросам в цикле
    for i, qa_pair in enumerate(test_set, 1):
        question = qa_pair.get('question')
        expected_answer = qa_pair.get('answer')

        if not question:
            continue

        # Получаем ответ от бота
        bot_answer = rag_chain.invoke(question)

        # Печатаем результат
        print("="*30)
        print(f"Вопрос #{i}")
        print(f"Формулировка: {question}")
        print(f"Мой ответ: {expected_answer}")
        print(f"Ответ бота: {bot_answer}")
        print("="*30 + "\n")


if __name__ == "__main__":
    run_evaluation()