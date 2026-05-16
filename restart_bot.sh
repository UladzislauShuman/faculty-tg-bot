#!/usr/bin/env bash
# Перезапуск только Telegram-бота (сервис bot в docker-compose).
#
# Достаточно после правок в src/ и config/ — они смонтированы в контейнер.
# Полный ./setup.sh нужен, если менялись зависимости (pyproject.toml), Dockerfile,
# или поднимаете стек с нуля.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  docker compose restart bot
elif command -v docker-compose >/dev/null 2>&1; then
  docker-compose restart bot
else
  echo "Не найдены docker compose или docker-compose." >&2
  exit 1
fi

echo "Сервис bot перезапущен."
