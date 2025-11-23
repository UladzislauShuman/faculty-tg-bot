import logging
import os
import sys
import yaml
from aiohttp import web
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from src.di_containers import Container
from src.tg_bot.handlers import main_router
from src.tg_bot.services.interfaces import IUserService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def on_startup(bot: Bot):
    logger.info("Веб-сервер запускается, устанавливаем вебхук...")
    webhook_url = os.getenv("TGSERVER__WEBHOOK_URL")
    if not webhook_url:
        logger.error("Переменная окружения TGSERVER__WEBHOOK_URL не установлена!")
        return
    await bot.set_webhook(webhook_url)
    logger.info(f"Вебхук успешно установлен на {webhook_url}")

async def on_shutdown(bot: Bot):
    logger.info("Веб-сервер останавливается, удаляем вебхук...")
    await bot.delete_webhook()

def main():
    try:
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        logger.info("✅ Конфигурация config.yaml успешно загружена.")
    except FileNotFoundError:
        logger.error("❌ Ошибка: Файл config/config.yaml не найден.")
        sys.exit(1)

    container = Container()
    container.config.from_dict(config_data)

    bot = Bot(
        token=os.getenv("TGSERVER__TOKEN"),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()


    logger.info("Инициализация RAG-компонентов...")
    dp["user_service"] = container.bot_user_service()
    dp["answer_service"] = container.bot_answer_service()
    dp["rag_chain"] = container.rag_chain()
    logger.info("RAG-компоненты готовы.")

    dp.include_router(main_router)
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    app = web.Application()
    webhook_requests_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
    )

    webhook_path = "/webhook"
    webhook_requests_handler.register(app, path=webhook_path)
    setup_application(app, dp, bot=bot)

    web_app_host = "0.0.0.0"
    web_app_port = 8080
    logger.info(f"Запуск веб-сервера на {web_app_host}:{web_app_port}")

    web.run_app(app, host=web_app_host, port=web_app_port)

if __name__ == '__main__':
    main()