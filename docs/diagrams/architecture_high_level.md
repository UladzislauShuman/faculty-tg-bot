graph TD
    classDef user fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef component fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef storage fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef llm fill:#ffcc80,stroke:#e65100,stroke-width:2px;

    User((Пользователь)):::user
    WebSite(Сайт Факультета):::storage
    LLM((LLM)):::llm

    subgraph Indexing [Модуль Индексации]
        direction TB
        Crawler[Сборщик ссылок]:::component
        Parser[Конвертация<br/>HTML в Markdown]:::component
        Chunker[Chunker Parent-Child]:::component
        Embedder[Векторизация]:::component
    end

    subgraph Storage [Хранилище Данных]
        Qdrant[(Qdrant)]:::storage
        Docstore[(Parent Docstore)]:::storage
        Postgres[(PostgreSQL)]:::storage
    end

    subgraph Processing [Обработка Запроса]
        TG_Bot[Telegram-бот]:::component
        Router[Классификатор запросов smalltalk/rag]:::component
        ContextManager[ContextManager]:::component
        
        subgraph RAG [Ветка RAG]
            Contextualizer[Контекстуализация]:::component
            Retriever[Qdrant Retriever]:::component
            Parent_Retriever[Parent Retrieval]:::component
            Reranker[Reranker]:::component
        end
    end

    %% Индексация
    WebSite -->|HTML| Crawler
    Crawler --> Parser
    Parser --> Chunker
    Chunker -->|Дочерние чанки| Embedder
    Chunker -->|Родительские документы| Docstore
    Embedder -->|Векторы| Qdrant

    %% Запрос и Роутинг
    User -->|Вопрос| TG_Bot
    TG_Bot -->|Текст| Router
    Router -->|Определение интента| ContextManager

    %% История
    ContextManager <-->|Чтение / Запись| Postgres
    ContextManager <-->|Суммаризация| LLM

    %% Ветка RAG
    ContextManager -->|rag| Contextualizer
    Contextualizer <-->|Переформулирование| LLM
    Contextualizer -->|Запрос| Retriever
    Retriever <-->|Поиск| Qdrant
    Retriever -->|ID чанков| Parent_Retriever
    Parent_Retriever <-->|Извлечение| Docstore
    Parent_Retriever -->|Документы| Reranker
    Reranker -->|Топ-N| LLM

    %% Ветка Smalltalk
    ContextManager -->|smalltalk / direct_link| LLM

    %% Ответ
    LLM -->|Ответ| TG_Bot
    TG_Bot -->|Ответ| User

    linkStyle default stroke-width:2px,fill:none;
