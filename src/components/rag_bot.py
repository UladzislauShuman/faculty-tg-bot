import os
import yaml
from src.util.yaml_parser import load_qa_test_set
from src.components.query_pipeline import load_retriever
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import OllamaLLM as Ollama
from langchain_community.llms import YandexGPT 

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

def get_llm_from_config(provider_config: dict):
    provider_type = provider_config.get("type")

    if provider_type == "ollama":
        print(f"Инициализация LLM от Ollama с моделью: {provider_config.get('model')}")
        return Ollama(model=provider_config.get("model"))
    
    elif provider_type == "yandex_gpt":
        print(f"Инициализация LLM от YandexGPT с моделью: {provider_config.get('model')}")
        
        secret_key = provider_config.get("secret")
        if not secret_key or secret_key == "YOUR_YANDEX_SECRET_KEY_HERE":
             raise NotImplementedError(
                "YandexGPT интеграция настроена, но требует реального API ключа. "
                "Пожалуйста, вставьте ваш IAM-токен или API-ключ в поле 'secret' файла config.yaml."
            )
        
        # Для YandexGPT нужно передавать параметры немного по-другому.
        # Обычно это API-ключ и/или ID каталога.
        # Мы пока просто создаем объект.
        # Чаще всего это выглядит так:
        # return YandexGPT(api_key=secret_key, folder_id="...", ...)
        
        # Заглушка, чтобы показать, что переключение работает
        return YandexGPT(api_key=secret_key)
    else:
        raise ValueError(f"Неизвестный тип провайдера LLM: {provider_type}")

def get_rag_chain():
    # 1. Загружаем общую конфигурацию
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Файл config.yaml не найден в корневой папке.")

    # 2. Определяем, какого провайдера использовать, из .env файла
    provider_name = os.getenv("LLM_PROVIDER")
    if not provider_name:
        raise ValueError("Переменная LLM_PROVIDER не установлена в .env файле.")
    
    # 3. Находим конфигурацию для этого провайдера в config.yaml
    provider_config = config.get('providers', {}).get(provider_name)
    if not provider_config:
        raise ValueError(f"Конфигурация для провайдера '{provider_name}' не найдена в config.yaml.")

    # 4. Создаем экземпляр LLM
    llm = get_llm_from_config(provider_config)
    
    # 5. Собираем RAG-цепочку
    retriever = load_retriever()
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
        try:
            rag_chain = get_rag_chain()
            print("\nЗапрос отправлен в RAG-цепочку...")
            response = rag_chain.invoke(user_query)
            print("\n--- Ответ LLM ---")
            print(response)
        except (ValueError, FileNotFoundError, NotImplementedError) as e:
            print(f"\n❌ Ошибка при инициализации RAG: {e}")