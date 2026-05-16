#!/bin/bash
set -e

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Профили моделей для сравнения A/B — передаются в контейнер как -e RAG_*.
# main.py после загрузки config/config.yaml подменяет ключи см. _apply_rag_stack_env_overrides.
#
# Своя серия экспериментов: измените строки ниже или задайте до запуска:
#   RAG_MODEL_PROFILE=prev ./test.sh          # только предыдущий стек (без вопроса)
#   RAG_MODEL_PROFILE=new ./test.sh          # только новый стек
#   RAG_MODEL_PROFILE=yaml ./test.sh           # только yaml (переменные RAG_* не задаются)
# Либо переопределите по одному прямо в вызове docker (поверх профиля):
#   docker-compose exec -e RAG_OLLAMA_MODEL=mistral rag-cli python main.py test …
#
# Если меняете RAG_EMBEDDING_MODEL относительно уже проиндексированной базы —
# нужна полная переиндексация (в тестах: включите пересборку / --force-index).
# -----------------------------------------------------------------------------
RAG_MODEL_PROFILE="${RAG_MODEL_PROFILE:-}" # prev | new | yaml — если не пустой, без интерактива

# Предыдущий стек (типичный «до смены» в этом репозитории)
RAG_PREV_OLLAMA="${RAG_PREV_OLLAMA:-llama3.1}"
RAG_PREV_EMBEDDING="${RAG_PREV_EMBEDDING:-intfloat/multilingual-e5-large}"
RAG_PREV_RERANKER="${RAG_PREV_RERANKER:-DiTy/cross-encoder-russian-msmarco}"
RAG_PREV_JUDGE="${RAG_PREV_JUDGE:-$RAG_PREV_OLLAMA}"
RAG_PREV_ROUTING="${RAG_PREV_ROUTING:-$RAG_PREV_OLLAMA}"
RAG_PREV_SUMMARY="${RAG_PREV_SUMMARY:-$RAG_PREV_OLLAMA}"
RAG_PREV_HYDE="${RAG_PREV_HYDE:-$RAG_PREV_OLLAMA}"
RAG_PREV_EVAL_EMBEDDING="${RAG_PREV_EVAL_EMBEDDING:-cointegrated/rubert-tiny2}"

# Текущий «новый» стек — совпадайте поля здесь при смене config.yaml
RAG_NEW_OLLAMA="${RAG_NEW_OLLAMA:-qwen2.5:14b-instruct-q4_K_M}"
RAG_NEW_EMBEDDING="${RAG_NEW_EMBEDDING:-BAAI/bge-m3}"
RAG_NEW_RERANKER="${RAG_NEW_RERANKER:-BAAI/bge-reranker-v2-m3}"
RAG_NEW_JUDGE="${RAG_NEW_JUDGE:-$RAG_NEW_OLLAMA}"
RAG_NEW_ROUTING="${RAG_NEW_ROUTING:-$RAG_NEW_OLLAMA}"
RAG_NEW_SUMMARY="${RAG_NEW_SUMMARY:-$RAG_NEW_OLLAMA}"
RAG_NEW_HYDE="${RAG_NEW_HYDE:-$RAG_NEW_OLLAMA}"
RAG_NEW_EVAL_EMBEDDING="${RAG_NEW_EVAL_EMBEDDING:-sentence-transformers/paraphrase-multilingual-mpnet-base-v2}"

DOCKER_RAG_ENV=()

_print_model_profile_summary() {
    echo -e "\n${CYAN}Стеки для сравнения (редактируйте переменные RAG_* в начале test.sh):${NC}"
    echo -e "  ${YELLOW}[prev]${NC} Ollama:${RAG_PREV_OLLAMA} | embed:${RAG_PREV_EMBEDDING} | rerank:${RAG_PREV_RERANKER}"
    echo -e "         judge/routing/summary/hyde: ${RAG_PREV_JUDGE} | eval_emb: ${RAG_PREV_EVAL_EMBEDDING}"
    echo -e "  ${YELLOW}[new] ${NC} Ollama:${RAG_NEW_OLLAMA} | embed:${RAG_NEW_EMBEDDING} | rerank:${RAG_NEW_RERANKER}"
    echo -e "         judge/routing/summary/hyde: ${RAG_NEW_JUDGE} | eval_emb: ${RAG_NEW_EVAL_EMBEDDING}"
}

_fill_docker_rag_env_previous_stack() {
    DOCKER_RAG_ENV=(
        -e "RAG_OLLAMA_MODEL=$RAG_PREV_OLLAMA"
        -e "RAG_EMBEDDING_MODEL=$RAG_PREV_EMBEDDING"
        -e "RAG_RERANKER_MODEL=$RAG_PREV_RERANKER"
        -e "RAG_EVAL_JUDGE_MODEL=$RAG_PREV_JUDGE"
        -e "RAG_ROUTING_MODEL=$RAG_PREV_ROUTING"
        -e "RAG_SUMMARY_MODEL=$RAG_PREV_SUMMARY"
        -e "RAG_HYDE_MODEL=$RAG_PREV_HYDE"
        -e "RAG_EVAL_EMBEDDING_MODEL=$RAG_PREV_EVAL_EMBEDDING"
    )
}

_fill_docker_rag_env_new_stack() {
    DOCKER_RAG_ENV=(
        -e "RAG_OLLAMA_MODEL=$RAG_NEW_OLLAMA"
        -e "RAG_EMBEDDING_MODEL=$RAG_NEW_EMBEDDING"
        -e "RAG_RERANKER_MODEL=$RAG_NEW_RERANKER"
        -e "RAG_EVAL_JUDGE_MODEL=$RAG_NEW_JUDGE"
        -e "RAG_ROUTING_MODEL=$RAG_NEW_ROUTING"
        -e "RAG_SUMMARY_MODEL=$RAG_NEW_SUMMARY"
        -e "RAG_HYDE_MODEL=$RAG_NEW_HYDE"
        -e "RAG_EVAL_EMBEDDING_MODEL=$RAG_NEW_EVAL_EMBEDDING"
    )
}

apply_model_profile_selection() {
    local prof="$1"
    DOCKER_RAG_ENV=()
    prof="$(echo "$prof" | tr '[:upper:]' '[:lower:]')"
    case "$prof" in
        yaml|config| '')
            ;;
        prev|previous|old|p|1)
            _fill_docker_rag_env_previous_stack
            ;;
        new|current|n|2)
            _fill_docker_rag_env_new_stack
            ;;
        *)
            echo -e "${YELLOW}⚠ Неизвестный профиль «$1», использую yaml (без RAG_*).${NC}"
            ;;
    esac
}

_resolve_model_profile() {
    if [ -n "$RAG_MODEL_PROFILE" ]; then
        MODEL_PROFILE_CHOICE=$(echo "$RAG_MODEL_PROFILE" | tr '[:upper:]' '[:lower:]')
        echo -e "${GREEN}Профиль моделей из окружения: ${YELLOW}${MODEL_PROFILE_CHOICE}${NC}"
    else
        _print_model_profile_summary
        echo -e "${CYAN}Какие модели подставить в прогон?${NC}"
        echo -e "  ${GREEN}yaml${NC} — только config.yaml, без переопределения RAG_*"
        echo -e "  ${GREEN}prev${NC} — стек «был раньше» (RAG_PREV_*)"
        echo -e "  ${GREEN}new${NC}  — текущий стек (RAG_NEW_*) — по умолчанию как в этом скрипте"
        read -p "Выбор профиля [yaml/prev/new] (yaml): " MODEL_PROFILE_CHOICE
        MODEL_PROFILE_CHOICE=${MODEL_PROFILE_CHOICE:-yaml}
    fi
    apply_model_profile_selection "$MODEL_PROFILE_CHOICE"
    if [ "${#DOCKER_RAG_ENV[@]}" -gt 0 ]; then
        echo -e "${GREEN}В контейнер передаём переопределения RAG_* (профиль: ${YELLOW}${MODEL_PROFILE_CHOICE}${GREEN}).${NC}"
    else
        echo -e "${GREEN}Модели: только из ${CYAN}config/config.yaml${NC}."
    fi
}

echo -e "${GREEN}🧪 Запуск интерактивного тестирования FPMI RAG Bot...${NC}\n"

# --- Шаг 1: Проверка окружения и запуск контейнеров ---
if [ ! -f .env ]; then
    echo -e "${RED}ОШИБКА: Файл .env не найден. Сначала выполните setup.sh.${NC}"
    exit 1
fi

# Функция для чтения переменных из .env
get_env_var() {
    grep "^$1=" .env | cut -d'=' -f2- | tr -d '"' | tr -d "'"
}

echo -e "${YELLOW}1. Проверка доступности LLM (Ollama)...${NC}"
USE_NATIVE_OLLAMA=false

while true; do
    OLLAMA_URL=$(get_env_var "OLLAMA_HOST")

    if [ -z "$OLLAMA_URL" ]; then
        echo -e "${RED}⚠️  Параметр OLLAMA_HOST в .env пуст.${NC}"
        read -p "Настроить нативную Ollama (Mac GPU)?[y/n]: " choice
        if [[ "$choice" == "y" ]]; then
            echo "OLLAMA_HOST=http://host.docker.internal:11434" >> .env
            continue
        else
            USE_NATIVE_OLLAMA=false
            break
        fi
    fi

    CHECK_URL=${OLLAMA_URL/host.docker.internal/localhost}
    echo -e "🔍 Проверка связи с $CHECK_URL..."
    if curl -s --max-time 2 "$CHECK_URL" > /dev/null; then
        echo -e "      ${GREEN}✓ Ollama доступна!${NC}"
        USE_NATIVE_OLLAMA=true
        break
    else
        echo -e "${RED}❌ Не удается достучаться до $OLLAMA_URL${NC}"
        read -p "Попробовать снова? (y - повторить, n - Docker версия): " choice
        if [[ "$choice" == "y" ]]; then continue; else USE_NATIVE_OLLAMA=false; break; fi
    fi
done

echo -e "\n${YELLOW}2. Проверка состояния контейнеров (без Telegram-бота)...${NC}"
if [ "$USE_NATIVE_OLLAMA" = true ]; then
    # Явно указываем только rag-cli и postgres. Сервис bot не запустится.
    docker-compose up -d rag-cli postgres > /dev/null 2>&1
else
    # Явно указываем rag-cli, postgres и ollama. Сервис bot не запустится.
    docker-compose --profile ollama-docker up -d rag-cli postgres ollama > /dev/null 2>&1
fi
echo -e "      ${GREEN}✓ Контейнеры готовы к работе.${NC}"

echo -e "\n${YELLOW}3. Проверка структуры базы данных (миграции)...${NC}"
# Применяем миграции, чтобы создать таблицу rag_bot_answers, если база пустая
docker-compose exec rag-cli alembic upgrade head > /dev/null 2>&1
echo -e "      ${GREEN}✓ База данных готова.${NC}\n"

# --- Шаг 2: Главное меню ---
echo -e "${CYAN}Выберите способ запуска тестов:${NC}"
echo -e "  (Профиль моделей задаётся сразу после выбора: ${YELLOW}yaml${NC} | ${YELLOW}prev${NC} | ${YELLOW}new${NC} или ${YELLOW}RAG_MODEL_PROFILE=${NC}; см. начало ${CYAN}test.sh${NC}.)"
echo -e "  ${GREEN}0)${NC} Как в ${CYAN}config/config.yaml${NC}: режим ${YELLOW}all${NC} + ${YELLOW}--force-index${NC}"
echo -e "     Chunker — ${CYAN}indexing.chunker${NC}; ${CYAN}retrievers.active_type${NC}, память, HyDE — из yaml (без флагов из скрипта)."
echo -e "     ${YELLOW}--retriever${NC} и ${YELLOW}--index-mode${NC}: дефолты CLI (hybrid, test), см. ${CYAN}main.py test${NC}."
echo -e "  ${GREEN}1)${NC} Быстрый запуск (Дефолтный сценарий Владислава)"
echo -e "     ${YELLOW}[all, markdown, hybrid, test, force-index, chroma_bm25, summary_window]${NC}"
echo -e "     HyDE (опционально): задайте ${CYAN}RAG_HYDE=on${NC} или ${CYAN}RAG_HYDE=off${NC} перед запуском; иначе — как в config.yaml"
echo -e "     Telegram: маршрутизация smalltalk/ссылки — ${CYAN}semantic_routing${NC} в config.yaml (на прогон ${CYAN}main.py test${NC} не влияет)."
echo -e "  ${GREEN}2)${NC} Интерактивная настройка (Выбрать каждый параметр вручную)"
echo -e "  ${GREEN}3)${NC} Матрица сценариев из config.yaml (python main.py test-matrix)"
echo -e "  ${GREEN}4)${NC} Продолжить матрицу после паузы (python main.py test-matrix --resume)"
echo -e "  ${GREEN}5)${NC} Выход"
read -p "Ваш выбор [0-5]: " MENU_CHOICE

if [ "$MENU_CHOICE" == "5" ]; then
    echo "Выход."
    exit 0
fi

_resolve_model_profile

if [ "$MENU_CHOICE" == "0" ]; then
    echo -e "\n${CYAN}=========================================${NC}"
    echo -e "${GREEN}Запуск:${NC} ${YELLOW}main.py test all --force-index${NC} (остальное — из config.yaml, chunker из indexing.chunker)"
    echo -e "${CYAN}=========================================${NC}\n"
    docker-compose exec "${DOCKER_RAG_ENV[@]}" rag-cli python main.py test all --force-index
    echo -e "\n${GREEN}✅ Тестирование завершено!${NC}"
    echo -e "Подробные логи и результаты сохранены в папке ${YELLOW}output/${NC}."
    exit 0
fi

if [ "$MENU_CHOICE" == "3" ]; then
    echo -e "\n${YELLOW}Запуск матрицы evaluation_scenarios (test-matrix)...${NC}"
    docker-compose exec "${DOCKER_RAG_ENV[@]}" rag-cli python main.py test-matrix
    echo -e "\n${GREEN}✅ Готово. Чекпоинт: check_points/default_checkpoint.json${NC}"
    exit 0
fi

if [ "$MENU_CHOICE" == "4" ]; then
    echo -e "\n${YELLOW}Продолжение матрицы (--resume)...${NC}"
    docker-compose exec "${DOCKER_RAG_ENV[@]}" rag-cli python main.py test-matrix --resume
    echo -e "\n${GREEN}✅ Готово.${NC}"
    exit 0
fi

if [ "$MENU_CHOICE" == "1" ]; then
    # Дефолтные настройки
    EVAL_MODE="all"
    CHUNKER="markdown"
    RETRIEVER="hybrid"
    INDEX_MODE="test"
    FORCE_INDEX="--force-index"
    # chroma+bm25 или qdrant; summary_window или чистый window; умная память вкл/выкл
    ACTIVE_TYPE="chroma_bm25"
    MEMORY_TYPE="summary_window"
    MEMORY_OFF_FLAG=""
    # HyDE: RAG_HYDE=on|off — передать в main.py; пусто — не переопределять config.yaml
    HYDE_CLI=""
    case "${RAG_HYDE,,}" in
        on|1|yes|true)  HYDE_CLI="--hyde=on" ;;
        off|0|no|false) HYDE_CLI="--hyde=off" ;;
    esac
else
    echo -e "\n${CYAN}=== ИНТЕРАКТИВНАЯ НАСТРОЙКА ===${NC}"

    # 1. EVAL_MODE
    echo -e "\n${BLUE}1. Режим тестирования (eval_mode)${NC}"
    echo "   Определяет, какие вопросы из qa-test-set.yaml будут запущены."
    echo "   - all       : Запустить всё (одиночные вопросы + диалоги)"
    echo "   - questions : Только одиночные вопросы (без памяти)"
    echo "   - scenarios : Только многошаговые диалоги (проверка контекста)"
    read -p "   Выберите режим [all/questions/scenarios] (По умолчанию: all): " EVAL_MODE
    EVAL_MODE=${EVAL_MODE:-all}

    # 2. CHUNKER
    echo -e "\n${BLUE}2. Алгоритм чанкинга (--chunker)${NC}"
    echo "   Определяет, как HTML-страницы нарезаются на куски текста."
    echo "   - markdown : Конвертация в Markdown и нарезка по заголовкам (Надежно)."
    echo "   - semantic : Умный парсинг таблиц и списков (Экспериментально)."
    echo "   - parent   : Small-to-Big Retrieval (Мелкие чанки для поиска, большие для контекста LLM)."
    read -p "   Выберите чанкер[markdown/semantic/unstructured/parent] (По умолчанию: markdown): " CHUNKER
    CHUNKER=${CHUNKER:-markdown}

    # 3. RETRIEVER
    echo -e "\n${BLUE}3. Стратегия поиска (--retriever)${NC}"
    echo "   Определяет, как система ищет документы в базе."
    echo "   - hybrid : Вектора (смысл) + BM25 (ключевые слова). Лучший результат."
    echo "   - vector : Только векторный поиск (ChromaDB)."
    read -p "   Выберите ретривер [hybrid/vector] (По умолчанию: hybrid): " RETRIEVER
    RETRIEVER=${RETRIEVER:-hybrid}

    # 4. INDEX_MODE
    echo -e "\n${BLUE}4. Объем данных (--index-mode)${NC}"
    echo "   Определяет, откуда брать данные для базы."
    echo "   - test : Индексировать только URL, указанные в qa-test-set.yaml (Быстро)."
    echo "   - full : Запустить краулер по всему сайту (Долго)."
    read -p "   Выберите объем [test/full] (По умолчанию: test): " INDEX_MODE
    INDEX_MODE=${INDEX_MODE:-test}

    # 5. FORCE_INDEX
    echo -e "\n${BLUE}5. Принудительная переиндексация (--force-index)${NC}"
    echo "   Удалить старую базу данных для выбранного чанкера и собрать ее заново?"
    echo "   (Рекомендуется 'y', если вы меняли код парсера или список URL)."
    read -p "   Пересобрать базу? [y/n] (По умолчанию: y): " FORCE_CHOICE
    FORCE_CHOICE=${FORCE_CHOICE:-y}
    if [[ "$FORCE_CHOICE" == "y" || "$FORCE_CHOICE" == "Y" ]]; then
        FORCE_INDEX="--force-index"
    else
        FORCE_INDEX=""
    fi

    # 6. Бэкенд ретривера (см. config retrievers.active_type)
    echo -e "\n${BLUE}6. Тип ретривера в БД (active_type)${NC}"
    echo "   - chroma_bm25 : Chroma (dense) + BM25 (sparse), локальные индексы"
    echo "   - qdrant      : Qdrant hybrid (dense+sparse) в одной коллекции"
    read -p "   Введите [chroma_bm25/qdrant] (По умолчанию: chroma_bm25): " ACTIVE_TYPE
    ACTIVE_TYPE=${ACTIVE_TYPE:-chroma_bm25}

    # 7. Память диалога
    echo -e "\n${BLUE}7. Режим памяти (memory.type)${NC}"
    echo "   - summary_window : после порога — summary + последние N реплик"
    echo "   - window         : только скользящее окно (сырые реплики), без LLM-summary"
    read -p "   Введите [summary_window/window] (По умолчанию: summary_window): " MEMORY_TYPE
    MEMORY_TYPE=${MEMORY_TYPE:-summary_window}

    # 8. Полностью отключить memory.enabled (как в старом боте)
    echo -e "\n${BLUE}8. Отключить умную память (memory.enabled=false)?${NC}"
    read -p "   Полностью выключить memory? [y/N] (по умолчанию: n): " MEMOFF_CH
    MEMOFF_CH=${MEMOFF_CH:-n}
    if [[ "$MEMOFF_CH" == "y" || "$MEMOFF_CH" == "Y" ]]; then
        MEMORY_OFF_FLAG="--memory-off"
    else
        MEMORY_OFF_FLAG=""
    fi

    # 9. HyDE
    echo -e "\n${BLUE}9. HyDE — гипотетический dense-запрос (hyde.enabled)${NC}"
    echo "   - config : не передавать флаг в main.py (брать из config.yaml)"
    echo "   - on     : --hyde=on (модель — hyde.llm в config.yaml; в дефолте та же что providers.ollama)"
    echo "   - off    : --hyde=off (явно выключить, даже если в yaml enabled: true)"
    read -p "   Введите [config/on/off] (по умолчанию: config): " HYDE_IN
    HYDE_IN=${HYDE_IN:-config}
    HYDE_CLI=""
    if [ "$HYDE_IN" == "on" ] || [ "$HYDE_IN" == "ON" ]; then
        HYDE_CLI="--hyde=on"
    elif [ "$HYDE_IN" == "off" ] || [ "$HYDE_IN" == "OFF" ]; then
        HYDE_CLI="--hyde=off"
    fi
fi

# --- Шаг 4: Формирование и запуск команды ---
echo -e "\n${CYAN}=========================================${NC}"
echo -e "${GREEN}🚀 Запускаю тестирование со следующими параметрами:${NC}"
echo -e "   Режим:      ${YELLOW}$EVAL_MODE${NC}"
echo -e "   Чанкер:     ${YELLOW}$CHUNKER${NC}"
echo -e "   Ретривер:   ${YELLOW}$RETRIEVER${NC}"
echo -e "   Данные:     ${YELLOW}$INDEX_MODE${NC}"
echo -e "   Пересборка: ${YELLOW}$([ -n "$FORCE_INDEX" ] && echo "Да" || echo "Нет")${NC}"
echo -e "   active_type: ${YELLOW}${ACTIVE_TYPE}${NC}  (chroma_bm25 | qdrant)"
echo -e "   memory.type: ${YELLOW}${MEMORY_TYPE}${NC}  (summary_window | window)"
if [ -n "$MEMORY_OFF_FLAG" ]; then
    echo -e "   ${YELLOW}memory.enabled: выключена (--memory-off)${NC}"
else
    echo -e "   memory.enabled: ${YELLOW}как в config.yaml${NC}"
fi
if [ -n "$HYDE_CLI" ]; then
    echo -e "   HyDE:         ${YELLOW}${HYDE_CLI}${NC}"
else
    echo -e "   HyDE:         ${YELLOW}как в config.yaml (флаг не передаётся)${NC}"
fi
if [ "${#DOCKER_RAG_ENV[@]}" -gt 0 ]; then
    echo -e "   Модели:       ${YELLOW}RAG_* → контейнер (профиль ${MODEL_PROFILE_CHOICE})${NC}"
else
    echo -e "   Модели:       ${YELLOW}только config.yaml${NC}"
fi
echo -e "${CYAN}=========================================${NC}\n"

docker-compose exec "${DOCKER_RAG_ENV[@]}" rag-cli python main.py test "$EVAL_MODE" \
    --chunker="$CHUNKER" \
    --retriever="$RETRIEVER" \
    --index-mode="$INDEX_MODE" \
    $FORCE_INDEX \
    --active-type="$ACTIVE_TYPE" \
    --memory-type="$MEMORY_TYPE" \
    $MEMORY_OFF_FLAG \
    $HYDE_CLI

echo -e "\n${GREEN}✅ Тестирование завершено!${NC}"
echo -e "Подробные логи и результаты сохранены в папке ${YELLOW}output/${NC}."