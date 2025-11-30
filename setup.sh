#!/bin/bash
set -e

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Запуск настройки окружения для FPMI RAG Bot...${NC}"

# --- Шаг 1: Проверка .env ---
echo -e "\n${YELLOW}1/7. Проверка конфигурации...${NC}"
if [ ! -f .env ]; then
    echo -e "${RED}ОШИБКА: Файл .env не найден.${NC}"
    echo "Пожалуйста, создайте файл .env на основе .env.example и заполните его."
    exit 1
fi
echo -e "      ${GREEN}✓ Файл .env найден.${NC}"

# --- Шаг 2: Docker Compose ---
echo -e "\n${YELLOW}2/7. Сборка и запуск контейнеров...${NC}"
docker-compose up -d --build

echo -e "      ${GREEN}✓ Контейнеры запущены. Ожидание готовности сервисов (10 сек)...${NC}"
sleep 10

# --- Шаг 3: Скачивание модели Ollama ---
echo -e "\n${YELLOW}3/7. Загрузка LLM модели (llama3.1) внутрь Docker...${NC}"
echo "Это может занять время в зависимости от скорости интернета..."
docker-compose exec ollama ollama pull llama3.1
echo -e "      ${GREEN}✓ Модель успешно загружена.${NC}"

# --- Шаг 4: Миграции БД ---
echo -e "\n${YELLOW}4/7. Применение миграций базы данных...${NC}"
docker-compose exec rag-cli alembic upgrade head
echo -e "      ${GREEN}✓ База данных готова.${NC}"

# --- Шаг 5: Индексация данных ---
echo -e "\n${YELLOW}5/7. Индексация данных (RAG)...${NC}"
echo "Запускаем индексацию в тестовом режиме (быстро)..."
docker-compose exec rag-cli python main.py index test
echo -e "      ${GREEN}✓ Данные проиндексированы и векторная база создана.${NC}"

echo ""
echo "6/7. Перезапуск сервиса бота... (чтобы он подхватил изменения в data/)"
# Это нужно, чтобы бот подхватил новую базу данных, которую мы только что создали
docker-compose restart bot

# --- Шаг 7: Финал ---
echo -e "\n${YELLOW}7/7. Настройка завершена!${NC}"
echo -e "${GREEN}Бот запущен и готов к работе.${NC}"
echo "------------------------------------------------------------------"
echo "ВАЖНО: Для работы с Telegram нужен HTTPS туннель (По умолчанию используется localtunnel и запуститься именно он)"
echo "1. Убедитесь, что у вас установлен Node.js."
echo "2. Откройте НОВОЕ окно терминала и запустите:"
echo -e "${YELLOW}   npx localtunnel --port 8080 --subdomain famcsanswerbot${NC}"
echo "3. Скопируйте полученную ссылку (https://....loca.lt) в ваш .env файл"
echo "   в переменную TGSERVER__WEBHOOK_URL (не забудьте добавить /webhook в конце!)."
echo "Чтобы перезапустить бота: docker-compose restart bot"
echo "Чтобы остановить бота: docker-compose down"
echo "------------------------------------------------------------------"
npx localtunnel --port 8080 --subdomain famcsanswerbot