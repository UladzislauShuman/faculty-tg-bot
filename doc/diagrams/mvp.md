graph TD
    subgraph "КОНТУР 1: OFFLINE (Индексация данных)"
        direction LR
        
        DS("<b>Источник данных</b><br>- Веб-сайт факультета")
        
        Loader("<b>1. Загрузчик (Loader)</b><br><i>Функция:</i> Читает текст со страниц сайта.")
        
        Splitter("<b>2. Разделитель (Splitter)</b><br><i>Функция:</i> Делит длинный текст на чанки.")
        
        EmbeddingModel("<b>3. Векторизация (Embedding Model)</b><br><i>Функция:</i> Превращает текстовые чанки в векторы.")
        
        VectorDB("<b>4. Векторная База Данных</b><br><i>Функция:</i> Хранит векторы для быстрого поиска.")

        DS --> Loader
        Loader --> Splitter
        Splitter --> EmbeddingModel
        EmbeddingModel --> VectorDB
    end

    subgraph "КОНТУР 2: ONLINE (Обработка запроса)"
        direction TB

        User[("👤 Пользователь")]
        
        Telegram("<b>Интерфейс: Telegram Бот</b><br><i>Функция:</i> Принимает и отправляет сообщения.")

        App("<b>Backend-приложение (Python)</b><br><i>Функция:</i> Управляет RAG-логикой.")

        LLM("<b>Большая Языковая Модель (LLM)</b><br><i>Функция:</i> Генерирует финальный ответ.")
        
        User -- "1. Отправляет вопрос в чат" --> Telegram
        
        Telegram -- "2. Передает текст вопроса" --> App
        
        App -- "3. Векторизует вопрос и ищет в БД" --> VectorDB
        
        VectorDB -- "4. Возвращает похожие чанки (контекст)" --> App
        
        App -- "5. Формирует промпт (контекст + вопрос)" --> LLM
        
        LLM -- "6. Возвращает сгенерированный ответ" --> App
        
        App -- "7. Отправляет готовый ответ боту" --> Telegram

        Telegram -- "8. Показывает ответ пользователю" --> User
    end

    %% Стилизация
    style VectorDB fill:#d6f5d6,stroke:#333
    style LLM fill:#cde4f7,stroke:#333