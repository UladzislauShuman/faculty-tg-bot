graph TD
    subgraph "КОНТУР 1: OFFLINE (Продвинутая индексация)"
        direction LR
        
        DS("<b>Источники данных</b><br>- Веб-сайт<br>- Excel, PDF, Docx")
        
        Chunker("<b>1. Семантическое разделение</b><br><i>Функция:</i> Делит документы на<br>смысловые блоки, а не по размеру.<br><b>Стек: Semantic Chunker</b>")
        
        subgraph "2. Мульти-индексация"
            direction TB
            EmbeddingModel("<b>2а. Векторизация</b><br><b>Стек: SentenceTransformers</b>")
            VectorDB("<b>Векторный индекс (Vector Store)</b><br><i>Для семантического поиска.</i><br><b>Стек: ChromaDB / FAISS</b>")
            KeywordIndex("<b>Индекс по ключевым словам</b><br><i>Для поиска точных терминов.</i><br><b>Стек: BM25</b>")
        end
        
        DS --> Chunker
        Chunker -- "Смысловые чанки" --> EmbeddingModel
        Chunker -- "Текст чанков" --> KeywordIndex
        EmbeddingModel -- "Векторы" --> VectorDB
    end

    subgraph "КОНТУР 2: ONLINE (Интеллектуальный агентский пайплайн)"
        direction TB

        User[("👤 Пользователь")]
        
        Router("<b>A. Маршрутизатор запроса (Query Router)</b><br><i>Функция:</i> 'Мозг' системы. Анализирует запрос<br>и решает, какую стратегию использовать.<br><b>Стек: LLM с Chain of Thought</b>")

        subgraph "B. Гибридный Поиск и Улучшение"
            direction TB
            QueryAugmentation("<b>B.1 Улучшение запроса (Query Augmentation)</b><br><i>Функция:</i> Перефразирование, HyDE.")
            
            subgraph "B.2 Параллельный поиск"
                direction LR
                VectorSearch("<b>Векторный поиск</b>")
                KeywordSearch("<b>Поиск по ключам</b>")
            end
            
            Fusion("<b>B.3 Слияние результатов (RRF)</b><br><i>Функция:</i> Умно объединяет результаты<br>из разных поисковых систем.<br><b>Стек: Reciprocal Rank Fusion</b>")
        end
        
        subgraph "C. Пост-обработка, Генерация и Проверка"
            direction TB
            Reranker("<b>C.1 Переранжирование (Re-ranker)</b><br><i>Функция:</i> Точно сортирует кандидатов.<br><b>Стек: Cross-Encoder</b>")
            
            Compressor("<b>C.2 Сжатие контекста (Context Compressor)</b><br><i>Функция:</i> Оставляет только релевантные предложения.<br><b>Стек: LLMChainExtractor</b>")

            LLM("<b>C.3 Генеративная LLM</b><br><i>Функция:</i> Создает ответ на основе<br>очищенного и сжатого контекста.<br><b>Стек: Ollama + Llama 3</b>")
            
            GroundingCheck("<b>C.4 Проверка ответа (Grounding Check)</b><br><i>Функция:</i> Финальная проверка ответа<br>на предмет галлюцинаций и 'заземленность'.")
        end

        User -- "1. Запрос" --> Router
        Router -- "2. Выбор стратегии + запрос" --> QueryAugmentation
        QueryAugmentation -- "3. Улучшенные запросы" --> VectorSearch & KeywordSearch
        
        VectorSearch -- "Кандидаты 1" --> Fusion
        KeywordSearch -- "Кандидаты 2" --> Fusion
        
        Fusion -- "4. Единый список кандидатов" --> Reranker
        Reranker -- "5. Лучшие N документов" --> Compressor
        Compressor -- "6. Сжатый, кристально чистый контекст" --> LLM
        LLM -- "7. Черновой ответ" --> GroundingCheck
        GroundingCheck -- "8. Финальный, проверенный и надежный ответ" --> User
        
        %% Связи с индексами из Offline-контура
        VectorDB -.-> VectorSearch
        KeywordIndex -.-> KeywordSearch
    end

    %% Стилизация ключевых модулей
    style Router fill:#ffcc99,stroke:#333
    style Fusion fill:#ffcc99,stroke:#333
    style Compressor fill:#ffcc99,stroke:#333
    style Reranker fill:#fff8c4,stroke:#333
    style GroundingCheck fill:#f9caca,stroke:#333
    style LLM fill:#cde4f7,stroke:#333
    style VectorDB fill:#d6f5d6,stroke:#333
    style KeywordIndex fill:#d6f5d6,stroke:#333