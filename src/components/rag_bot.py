import os
from src.util.yaml_parser import load_qa_test_set
from src.components.query_pipeline import load_retriever
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM as Ollama 
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
DB_DIR = os.getenv("DB_DIR", "chroma_db")

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

def get_rag_chain():
    retriever = load_retriever()
    ollama_model_name = os.getenv("OLLAMA_MODEL", "saiga")
    llm = Ollama(model=ollama_model_name)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

if __name__ == "__main__":
    user_query = os.getenv("USER_QUERY")
    if not user_query:
        print("Ошибка: переменная USER_QUERY не найдена в файле .env")
    else:
        print("Ретривер успешно загружен.")
        rag_chain = get_rag_chain()
        print("\nЗапрос отправлен в RAG-цепочку...")
        response = rag_chain.invoke(user_query)
        print("\n--- Ответ LLM ---")
        print(response)