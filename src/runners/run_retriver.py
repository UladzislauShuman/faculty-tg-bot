import os
from dotenv import load_dotenv
from src.util.yaml_parser import load_qa_test_set
from src.components.query_pipeline import load_retriever

load_dotenv()
QA_FILE_PATH = os.getenv("QA_FILE_PATH", "qa-test-set.yaml")

def run_retrieval_evaluation():
    # Загружаем тестовый набор
    test_set = load_qa_test_set(QA_FILE_PATH)
    if not test_set:
        return

    # Инициализируем ретривер один раз
    print("\nИнициализация ретривера...")
    retriever = load_retriever()
    print("Ретривер готов к работе.\n")

    # Проходим по всем вопросам в цикле
    for i, qa_pair in enumerate(test_set, 1):
        question = qa_pair.get('question')
        if not question:
            continue

        # Получаем релевантные документы от ретривера
        # .invoke() - это стандартный способ вызова компонентов LangChain
        relevant_docs = retriever.invoke(question)

        # Печатаем результат в нужном формате
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

if __name__ == "__main__":
    run_retrieval_evaluation()