graph TD
    subgraph "КОНТУР 1: OFLINE (Индексация данных)"
        direction LR
        
        DS1("<b>Источник 1:</b><br>Веб-сайт факультета")
        DS2("<b>Источник 2:</b><br>Excel-файл расписания")
        
        Loader("<b>1. Загрузчики (Loaders)</b><br><i>Функция:</i> Читают данные из разных источников.<br><b>Стек: LangChain (WebBaseLoader, etc.)</b>")
        
        Splitter("<b>2. Разделитель (Splitter)</b><br><i>Функция:</i> Делит тексты на чанки.<br><b>Стек: LangChain</b>")
        
        EmbeddingModel("<b>3. Векторизация</b><br><i>Функция:</i> Превращает чанки в векторы.<br><b>Стек: SentenceTransformers</b>")
        
        VectorDB("<b>4. Векторная База Данных</b><br><i>Функция:</i> Хранит векторы для поиска.<br><b>Стек: ChromaDB</b>")

        DS1 --> Loader
        DS2 --> Loader
        Loader --> Splitter
        Splitter --> EmbeddingModel
        EmbeddingModel --> VectorDB
    end

    subgraph "КОНТУР 2: ONLINE (Обработка запроса)"
        direction TB

        User[("👤 Пользователь")]
        
        Telegram("<b>Интерфейс: Telegram Бот</b><br><i>Стек: python-telegram-bot</i>")

        Orchestrator("<b>Оркестратор (Основная логика)</b><br><i>Функция:</i> Управляет всем процессом ответа.<br><b>Стек: LangChain Expression Language (LCEL)</b>")

        Memory("<b>Память диалога (Conversation Memory)</b><br><i>Функция:</i> Хранит историю переписки<br>для ответов на контекстные вопросы.")

        Reranker("<b>⭐ Переранжировщик (Re-ranker) ⭐</b><br><i>Функция:</i> Точно сортирует найденные документы,<br>выбирая самые релевантные.<br><b>Стек: Cross-Encoder модель</b>")

        LLM("<b>Большая Языковая Модель (LLM)</b><br><i>Функция:</i> Генерирует финальный ответ.<br><b>Стек: Ollama + Llama 3</b>")
        
        User -- "1. Запрос" --> Telegram
        Telegram -- "2. Передает запрос" --> Orchestrator
        Orchestrator -- "3. Обращается к памяти" --> Memory
        Memory -- "4. Возвращает историю + формирует самодостаточный вопрос" --> Orchestrator
        Orchestrator -- "5. Векторизует вопрос и делает широкий поиск (топ-20)" --> VectorDB
        VectorDB -- "6. Возвращает 20 документов-кандидатов" --> Reranker
        Reranker -- "7. Возвращает 5 лучших, самых релевантных документов" --> Orchestrator
        Orchestrator -- "8. Формирует промпт (лучшие документы + вопрос)" --> LLM
        LLM -- "9. Генерирует ответ" --> Orchestrator
        Orchestrator -- "10. Обновляет историю в памяти и отправляет ответ" --> Telegram
        Telegram -- "11. Ответ" --> User
    end

    %% Стилизация
    style VectorDB fill:#d6f5d6,stroke:#333
    style LLM fill:#cde4f7,stroke:#333
    style Reranker fill:#fff8c4,stroke:#333