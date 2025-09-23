import os
import textwrap
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
DB_DIR = os.getenv("DB_DIR", "chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "cointegrated/rubert-tiny2")

def load_and_parse_data(url: str):
    print(f"Загрузка и парсинг данных со страницы: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 1.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Ошибка при загрузке URL: {e}")
        return None

    soup = BeautifulSoup(response.text, 'lxml')
    
    # Блок, где храниться весь текст
    # сейчас для страницы about он выглядит так, но надо тогда 
    # позже далее просмотреть как дела и у других страниц
    content_element = soup.find('div', id='block-famcs-content')
    
    if not content_element:
        content_element = soup.find('article')
        if not content_element:
            print("Не удалось найти основной контент на странице.")
            return None
        
    elements_to_remove = content_element.find_all([
        'script',           
        'style',          
        'nav',             
        'footer',           
        'header',         
        lambda tag: tag.name == 'div' and 'visually-hidden' in tag.get('class', [])
    ])
    
    for element in elements_to_remove:
        element.decompose()
        
    text = content_element.get_text(separator='\n', strip=True)
    
    clean_text = "\n".join([line for line in text.split('\n') if line.strip()])
    
    metadata = {"source": url, "title": soup.title.string if soup.title else "No title"}
    doc = Document(page_content=clean_text, metadata=metadata)
    
    print(f"Загружен и очищен 1 документ. Длина текста: {len(clean_text)} символов.")
    return [doc] 

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
    
    filtered_chunks = [chunk for chunk in chunks if len(chunk.page_content) > 50]
            
    print(f"✅ После фильтрации осталось {len(filtered_chunks)} качественных чанков.")
    
    output_file_path = os.path.join(OUTPUT_DIR, "index_pipeline-output.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(f"Исходный документ: {documents[0].metadata.get('source', 'N/A')}\n")
        f.write(f"Всего чанков: {len(filtered_chunks)}\n")
        f.write("="*80 + "\n\n")

        for i, chunk in enumerate(filtered_chunks):
            f.write(f"--- ЧАНК #{i+1} ---\n")
            f.write(f"Метаданные: {chunk.metadata}\n")
            f.write("-" * 20 + "\n")
            # Используем textwrap для форматирования
            wrapped_text = textwrap.fill(chunk.page_content, width=100)
            f.write(wrapped_text)
            f.write("\n\n" + "="*80 + "\n\n")
            
    print(f"✅ Все чанки сохранены в файл: {output_file_path}")

    return filtered_chunks

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
    loaded_documents = load_and_parse_data(TARGET_URL)
    if loaded_documents:
        document_chunks = split_documents(loaded_documents)
        vector_database = create_and_store_embeddings(document_chunks)
        print("\n🎉 Все шаги Контура 1 успешно выполнены!")