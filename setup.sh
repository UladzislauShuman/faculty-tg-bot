#!/bin/bash
set -e

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Запуск настройки окружения для FPMI RAG Bot...${NC}"

# --- Шаг 1: Проверка .env ---
echo -e "\n${YELLOW}1/8. Проверка конфигурации...${NC}"
if [ ! -f .env ]; then
    echo -e "${RED}ОШИБКА: Файл .env не найден.${NC}"
    echo "Пожалуйста, создайте файл .env на основе .env.example и заполните его."
    exit 1
fi

# Функция для чтения переменных из .env
get_env_var() {
    grep "^$1=" .env | cut -d'=' -f2- | tr -d '"' | tr -d "'"
}

OLLAMA_URL=$(get_env_var "OLLAMA_HOST")
OLLAMA_MODEL=$(get_env_var "OLLAMA_MODEL")

# Если модель не указана, ставим дефолт
if [ -z "$OLLAMA_MODEL" ]; then OLLAMA_MODEL="llama3.1"; fi

echo -e "      ${GREEN}✓ Конфигурация загружена: модель $OLLAMA_MODEL${NC}"

# --- Шаг 2: Умная проверка Ollama ---
echo -e "\n${YELLOW}2/8. Проверка доступности LLM (Ollama)...${NC}"

USE_NATIVE_OLLAMA=false

while true; do
    OLLAMA_URL=$(get_env_var "OLLAMA_HOST")

    if [ -z "$OLLAMA_URL" ]; then
        echo -e "${RED}⚠️  Параметр OLLAMA_HOST в .env пуст.${NC}"
        read -p "Настроить нативную Ollama (Mac GPU)? [y/n]: " choice
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

# --- Шаг 3: Запуск Docker ---
echo -e "\n${YELLOW}3/8. Сборка и запуск контейнеров...${NC}"

if [ "$USE_NATIVE_OLLAMA" = true ]; then
    echo -e "${GREEN}💡 РЕЖИМ NATIVE: Запускаю только бота и базу...${NC}"
    # Явно указываем сервисы. Благодаря профилю в docker-compose, ollama не запустится.
    docker-compose up -d --build bot rag-cli postgres qdrant
    # На всякий случай принудительно гасим контейнер, если он висел с прошлого раза
    docker-compose stop ollama || true
else
    echo -e "${YELLOW}📦 РЕЖИМ DOCKER: Запускаю полный стек (включая Ollama)...${NC}"
    # Флаг --profile заставляет Docker запустить "скрытый" сервис ollama
    docker-compose --profile ollama-docker up -d --build
fi
sleep 10

# --- Шаг 4: Загрузка модели ---
while true; do
    # Перечитываем название модели из .env на каждой итерации
    OLLAMA_MODEL=$(get_env_var "OLLAMA_MODEL")
    if [ -z "$OLLAMA_MODEL" ]; then OLLAMA_MODEL="llama3.1"; fi

    echo -e "\n${YELLOW}4/8. Загрузка модели $OLLAMA_MODEL...${NC}"

    # Флаг успеха загрузки
    PULL_SUCCESS=false

    if [ "$USE_NATIVE_OLLAMA" = true ]; then
        echo "Проверка/загрузка в нативной Ollama на хосте..."
        if ollama pull "$OLLAMA_MODEL"; then
            PULL_SUCCESS=true
        fi
    else
        echo "Загрузка модели $OLLAMA_MODEL внутрь Docker..."
        # Здесь мы обращаемся к контейнеру
        if docker-compose --profile ollama-docker exec ollama ollama pull "$OLLAMA_MODEL"; then
            PULL_SUCCESS=true
        fi
    fi

    if [ "$PULL_SUCCESS" = true ]; then
        echo -e "      ${GREEN}✓ Модель $OLLAMA_MODEL готова к работе.${NC}"
        break
    else
        echo -e "${RED}❌ Ошибка: Не удалось загрузить модель '$OLLAMA_MODEL'.${NC}"
        echo "Возможно, название указано неверно или отсутствует интернет-соединение."
        echo "------------------------------------------------------------------"
        echo "1) Я исправлю название в .env и попробую снова"
        echo "2) Использовать проверенную модель по умолчанию (llama3.1)"
        echo "3) Прервать установку"
        read -p "Выберите вариант [1-3]: " model_choice

        case $model_choice in
            1)
                echo -e "${YELLOW}Отредактируйте параметр OLLAMA_MODEL в .env и нажмите Enter...${NC}"
                read
                continue # Возврат в начало цикла
                ;;
            2)
                # Меняем модель в .env на llama3.1 (совместимо с macOS и Linux)
                sed -i.bak "s/^OLLAMA_MODEL=.*/OLLAMA_MODEL=llama3.1/" .env && rm .env.bak || \
                sed -i "s/^OLLAMA_MODEL=.*/OLLAMA_MODEL=llama3.1/" .env
                echo -e "${YELLOW}Модель в .env изменена на llama3.1. Повторяю загрузку...${NC}"
                continue
                ;;
            *)
                echo -e "${RED}Установка прервана пользователем.${NC}"
                exit 1
                ;;
        esac
    fi
done

echo -e "\n${YELLOW}5/8. Применение миграций базы данных...${NC}"
docker-compose exec rag-cli alembic upgrade head
echo -e "      ${GREEN}✓ База данных готова.${NC}"

echo -e "\n${YELLOW}6/8. Индексация данных (RAG)...${NC}"
echo "Запускаем индексацию в тестовом режиме (быстро)..."
sleep 5
docker-compose exec rag-cli python main.py index test
echo -e "      ${GREEN}✓ Данные проиндексированы и векторная база создана.${NC}"

echo -e "\n${YELLOW}7/8. Перезапуск бота...${NC}"
docker-compose restart bot

echo -e "\n${YELLOW}8/8. Готово!${NC}"
echo "------------------------------------------------------------------"
echo -e "РЕЖИМ: $([ "$USE_NATIVE_OLLAMA" = true ] && echo -e "${GREEN}Native GPU" || echo -e "${YELLOW}Docker CPU")${NC}"
echo -e "МОДЕЛЬ: ${GREEN}$OLLAMA_MODEL${NC}"
echo "------------------------------------------------------------------"
echo "Sprint 8 (Telegram): маршрутизация smalltalk / ссылки — config: semantic_routing (enabled, method: regex|llm)."
echo "------------------------------------------------------------------"
echo "ВАЖНО: Для работы с Telegram нужен HTTPS туннель (По умолчанию используется localtunnel и запуститься далее именно он)"
echo "1. Убедитесь, что у вас установлен Node.js."
echo "2. Откройте НОВОЕ окно терминала и запустите:"
echo -e "${YELLOW}   npx localtunnel --port 8080 --subdomain famcsanswerbot${NC}"
echo "3. Скопируйте полученную ссылку (https://....loca.lt) в ваш .env файл"
echo "   в переменную TGSERVER__WEBHOOK_URL (не забудьте добавить /webhook в конце!)."
echo "Чтобы перезапустить бота: docker-compose restart bot"
echo "Чтобы остановить бота: docker-compose down"
echo "------------------------------------------------------------------"

echo -e "\n${YELLOW}Выполняю команду npx localtunnel --port 8080 --subdomain famcsanswerbot${NC}"
npx localtunnel --port 8080 --subdomain famcsanswerbot