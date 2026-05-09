#!/bin/bash
set -e

# Продолжить прогон матрицы (check_points/default_checkpoint.json + --resume)
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

if [ ! -f .env ]; then
    echo -e "${RED}ОШИБКА: Файл .env не найден. Сначала выполните setup.sh или test.sh.${NC}"
    exit 1
fi

echo -e "${GREEN}▶ Продолжение test-matrix (--resume) в контейнере rag-cli...${NC}"
docker-compose exec rag-cli python main.py test-matrix --resume

echo -e "${GREEN}✅ Готово.${NC}"
