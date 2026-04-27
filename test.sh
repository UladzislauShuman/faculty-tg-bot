#!/bin/bash
set -e

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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
echo -e "  ${GREEN}1)${NC} Быстрый запуск (Дефолтный сценарий Владислава)"
echo -e "     ${YELLOW}[all, markdown, hybrid, test, force-index, chroma_bm25, summary_window]${NC}"
echo -e "  ${GREEN}2)${NC} Интерактивная настройка (Выбрать каждый параметр вручную)"
echo -e "  ${GREEN}3)${NC} Выход"
read -p "Ваш выбор [1-3]: " MENU_CHOICE

if [ "$MENU_CHOICE" == "3" ]; then
    echo "Выход."
    exit 0
fi

# --- Шаг 3: Сбор параметров ---
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
    read -p "   Выберите чанкер[markdown/semantic/unstructured] (По умолчанию: markdown): " CHUNKER
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

    # 7. Память диалога (memory.type, Sprint 3)
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
echo -e "${CYAN}=========================================${NC}\n"

# Формируем итоговую команду
CMD="docker-compose exec rag-cli python main.py test $EVAL_MODE --chunker=$CHUNKER --retriever=$RETRIEVER --index-mode=$INDEX_MODE $FORCE_INDEX --active-type=$ACTIVE_TYPE --memory-type=$MEMORY_TYPE $MEMORY_OFF_FLAG"

# Выполняем команду
eval $CMD

echo -e "\n${GREEN}✅ Тестирование завершено!${NC}"
echo -e "Подробные логи и результаты сохранены в папке ${YELLOW}output/${NC}."