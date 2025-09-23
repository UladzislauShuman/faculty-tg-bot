import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
DB_DIR = os.getenv("DB_DIR", "chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "cointegrated/rubert-tiny2")

def load_retriever():
    load_dotenv()
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings_model)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    print("✅ Ретривер успешно загружен.")
    return retriever

def search_relevant_documents(query: str):
    print(f"Поиск по запросу: '{query}'")

    retriever = load_retriever()
    relevant_docs = retriever.invoke(query)
    
    print("\n--- Найденные релевантные документы ---")
    if not relevant_docs:
        print("Релевантных документов не найдено.")
        return

    for i, doc in enumerate(relevant_docs):
        print(f"\n--- Документ #{i+1} ---")
        source = doc.metadata.get('source', 'N/A')
        print(f"Источник: {source}")
        print("Содержимое:")
        print(doc.page_content)
        print("-" * 20)
        
    return relevant_docs

if __name__ == "__main__":
    load_dotenv()
    user_query = os.getenv("USER_QUERY")
    if not user_query:
        print("Ошибка: переменная USER_QUERY не найдена в файле .env")
    else:
        search_relevant_documents(user_query)