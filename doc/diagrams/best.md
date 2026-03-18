graph TD
    %% --- СТИЛИ ---
    classDef user fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef component fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef storage fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef process fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,stroke-dasharray: 5 5;

    %% --- УЧАСТНИКИ ---
    User((👤 Пользователь)):::user
    WebSite(🌐 Сайт Факультета):::storage

    %% --- КОНТУР 1: ИНДЕКСАЦИЯ ---
    subgraph Indexing_Pipeline ["⚙️ Модуль Индексации"]
        direction TB
        Crawler[Сборщик ссылок]:::component
        Parser[Загрузка и Очистка]:::component
        Chunker[Markdown + Нарезка]:::component
        Embedder_Idx[(Векторизация <br/><i>e5-small</i>)]:::component
    end

    %% --- ХРАНИЛИЩЕ ---
    subgraph Storage ["🗄️ База Знаний"]
        VectorDB[(<b>ChromaDB</b><br/>Векторы)]:::storage
        BM25_Index[(<b>BM25 Index</b><br/>Слова)]:::storage
    end

    %% --- КОНТУР 2: RAG ---
    subgraph RAG_System ["🧠 Система RAG"]
        direction TB
        TG_Bot[(<b>Telegram-бот</b><br/><i>aiogram</i>)]:::component
        RAG_Logic[<b>Модуль RAG</b><br/>Оркестратор]:::component
        
        subgraph Search_Engine ["Поисковый движок"]
            Query_Exp[Расширение запроса]:::component
            Hybrid_Search[Гибридный поиск]:::component
            Reranker[(Переранжирование<br/><i>Cross-Encoder</i>)]:::component
        end
        
        LLM[(<b>LLM</b>)]:::component
    end

    %% --- СВЯЗИ: ИНДЕКСАЦИЯ ---
    WebSite -->|1. HTML-страницы| Crawler
    Crawler --> Parser
    Parser --> Chunker
    Chunker --> Embedder_Idx
    Embedder_Idx -->|Сохранение векторов| VectorDB
    Chunker -->|Сохранение текста| BM25_Index

    %% --- СВЯЗИ: ОБРАБОТКА ЗАПРОСА ---
    User <-->|Сообщения| TG_Bot
    TG_Bot <-->|Текст вопроса| RAG_Logic
    
    RAG_Logic -->|1. Запрос| Query_Exp
    Query_Exp -->|2. Варианты| Hybrid_Search
    
    Hybrid_Search <-->|Векторный поиск| VectorDB
    Hybrid_Search <-->|Лексический поиск| BM25_Index
    
    Hybrid_Search -->|3. Кандидаты| Reranker
    Reranker -->|4. Топ-5 фрагментов| LLM
    LLM -->|5. Финальный ответ| RAG_Logic

    %% --- ГРУППИРОВКА ДЛЯ ПОНИМАНИЯ ---
    linkStyle default stroke-width:2px,fill:none;