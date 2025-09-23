import os
import textwrap
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
DB_DIR = os.getenv("DB_DIR", "chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "cointegrated/rubert-tiny2")

def load_data_from_url(url: str):
    print(f"Загрузка данных со страницы: {url}")
    
    # загрузчик для ссылки
    # В headers передаем User-Agent, чтобы сайт не блокировал нас как бота
    loader = WebBaseLoader(
        web_path=url,
        header_template={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }, 
        requests_kwargs={"verify": False} 
    )
    
    # парсим страницу
    # Результат -- список объектов Document
    docs = loader.load()
    print(f"Загружено {len(docs)} документ(ов).")
    
    if docs:
        document = docs[0]
        page_content = document.page_content
        metadata = document.metadata
        
        print("\n--- Метаданные документа ---")
        print(metadata)
        
        print("\n--- Начало содержимого (первые 500 символов) ---")
        print(page_content[:500])
        print("...")
        
        print("\nСохранение содержимого в файл")
        
        # сохранить данные парсинга в файл
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_path = os.path.join(OUTPUT_DIR, "parsed_content.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Источник: {metadata.get('source', 'N/A')}\n")
            f.write(f"Заголовок: {metadata.get('title', 'N/A')}\n")
            f.write("="*30 + "\n\n")
            f.write(page_content)
        print(f"✅ Содержимое успешно сохранено в файл: {file_path}")
        
    return docs

def split_documents(documents: list):
    print("\nРазделение документов на чанки...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ Документы успешно разделены на {len(chunks)} чанков.")
    
    output_file_path = os.path.join(OUTPUT_DIR, "index_pipeline-output.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(f"Исходный документ: {documents[0].metadata.get('source', 'N/A')}\n")
        f.write(f"Всего чанков: {len(chunks)}\n")
        f.write("="*80 + "\n\n")

        for i, chunk in enumerate(chunks):
            f.write(f"--- ЧАНК #{i+1} ---\n")
            f.write(f"Метаданные: {chunk.metadata}\n")
            f.write("-" * 20 + "\n")
            # Используем textwrap для форматирования
            wrapped_text = textwrap.fill(chunk.page_content, width=100)
            f.write(wrapped_text)
            f.write("\n\n" + "="*80 + "\n\n")
            
    print(f"✅ Все чанки сохранены в файл: {output_file_path}")

    return chunks

def create_and_store_embeddings(chunks: list):
    print("\nСоздание и сохранение эмбеддингов...")
    
    model_name = EMBEDDING_MODEL_NAME
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print("Загрузка embedding-модели и создание базы векторов... (может занять время)")
    db = Chroma.from_documents(chunks, embeddings_model, persist_directory=DB_DIR)
    
    print(f"✅ База данных успешно создана и сохранена в папке: {DB_DIR}")
    return db

if __name__ == "__main__":
    TARGET_URL = "https://fpmi.bsu.by/about"
    loaded_documents = load_data_from_url(TARGET_URL)
    if loaded_documents:
        document_chunks = split_documents(loaded_documents)
        vector_database = create_and_store_embeddings(document_chunks)
        print("\n🎉 Все шаги Контура 1 успешно выполнены!")