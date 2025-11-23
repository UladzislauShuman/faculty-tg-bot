from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Импортируем наш новый, простой объект настроек
from src.tg_bot.db.settings import settings

# Основа для всех наших моделей
metadata = MetaData()
Base = declarative_base(metadata=metadata)

# Асинхронный движок для работы с БД
aengine = create_async_engine(
    settings.DB.adsn,
    echo=False, # Включать только для отладки SQL-запросов
)

# Фабрика для создания асинхронных сессий
asession = sessionmaker(
    aengine,
    expire_on_commit=False,
    class_=AsyncSession,
)