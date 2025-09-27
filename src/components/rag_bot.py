import os
import yaml
from src.util.yaml_parser import load_qa_test_set
from src.components.query_pipeline import load_retriever
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

from langchain_ollama import OllamaLLM as Ollama
from langchain_community.llms import YandexGPT 

load_dotenv()
DB_DIR = os.getenv("DB_DIR", "chroma_db")

PROMPT_TEMPLATE = """
Ты — вежливый и полезный ассистент-помощник. Твоя задача — отвечать на вопросы, основываясь ТОЛЬКО на предоставленном ниже контексте.
Если в контексте нет информации для ответа, честно скажи: "К сожалению, я не нашел информации по вашему вопросу в своей базе знаний."
Не придумывай ничего от себя. Отвечай развернуто, используя всю релевантную информацию из контекста. Указывай полное ФИО и должности, если они есть.

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
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Файл config.yaml не найден в корневой папке.")
    provider_name = os.getenv("LLM_PROVIDER")
    provider_config = config.get('providers', {}).get(provider_name)
    llm = get_llm_from_config(provider_config)
    
    # базовый ретривер, который ищет по векторам
    base_retriever = load_retriever()
    
    # инициализируем модель-кросс-энкодер.
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # создаем "компрессор", который будет использовать модель для переранжировки
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=3)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    print("✅ Ретривер с Re-ranker'ом успешно создан.")
    
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    
    rag_chain = (
        {"context": compression_retriever, "question": RunnablePassthrough()}
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