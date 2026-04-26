FROM python:3.12-slim AS builder

# 1. Копируем сверхбыстрый пакетный менеджер uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Настройки Python и uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

WORKDIR /app

# 2. Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Кеширование слоев: сначала копируем ТОЛЬКО файлы конфигурации зависимостей
COPY pyproject.toml uv.lock ./

# 4. Устанавливаем библиотеки (используем кеш Docker для молниеносной сборки)
# Флаг --no-install-project говорит uv не искать исходный код проекта на этом этапе
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --no-editable

# 5. Добавляем виртуальное окружение uv в PATH
# Теперь команды `python` и `pip` будут автоматически работать внутри окружения
ENV PATH="/app/.venv/bin:$PATH"

# 6. Загружаем NLTK данные (используя python из настроенного окружения)
RUN --mount=type=cache,target=/root/.cache/nltk \
    python -c "import nltk; nltk.download('punkt_tab'); nltk.download('snowball_data'); nltk.download('stopwords')"

# 7. Только теперь копируем остальной код
# Если ты изменишь код в main.py, Docker начнет сборку отсюда.
# Шаг 4 (установка гигабайтов ML библиотек) будет взят из кеша за 0 секунд!
COPY . .

CMD ["python", "main.py", "--help"]