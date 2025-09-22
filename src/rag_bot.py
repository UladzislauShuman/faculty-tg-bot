import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama 
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
DB_DIR = "chroma_db"
MODEL_NAME = "cointegrated/rubert-tiny2"

def load_retriever():
    embeddings_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings_model)
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("Ретривер успешно загружен.")
    return retriever


PROMPT_TEMPLATE = """
Ты — вежливый и полезный ассистент-помощник. Твоя задача — отвечать на вопросы, основываясь ТОЛЬКО на предоставленном ниже контексте.
Если в контексте нет информации для ответа, честно скажи: "К сожалению, я не нашел информации по вашему вопросу в своей базе знаний."
Не придумывай ничего от себя.

Контекст:
{context}

Вопрос:
{question}

Ответ:
"""

def run_rag_chain(query: str):
    retriever = load_retriever()
    ollama_model_name = os.getenv("OLLAMA_MODEL")
    llm = Ollama(model=ollama_model_name)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("\nЗапрос отправлен в RAG-цепочку...")
    response = rag_chain.invoke(query)
    
    print("\n--- Ответ LLM ---")
    print(response)

if __name__ == "__main__":
    user_query = os.getenv("USER_QUERY")
    if not user_query:
        print("Ошибка: переменная USER_QUERY не найдена в файле .env")
    else:
        run_rag_chain(user_query)