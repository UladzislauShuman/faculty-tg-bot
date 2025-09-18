# src/query_pipeline.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "chroma_db"
MODEL_NAME = "cointegrated/rubert-tiny2"

def search_relevant_documents(query: str):
    print(f"Поиск по запросу: '{query}'")

    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings_model)
    
    relevant_docs = db.similarity_search(query, k=3)
    
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
    user_query = "декан факультета в настоящее время"
    search_relevant_documents(user_query)