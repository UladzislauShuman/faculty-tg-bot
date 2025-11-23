# src/tg_bot/db/settings.py (ПРОСТАЯ И НАДЕЖНАЯ ВЕРСИЯ)

from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseSettings(BaseSettings):
    """Настройки для подключения к базе данных. Читает переменные с префиксом DB__"""
    HOST: str = "localhost"
    PORT: int = 5432
    NAME: str
    USER: str
    PASS: str

    class Config:
        env_prefix = 'DB__' # Указываем префикс для этой модели

    @property
    def adsn(self) -> str:
        return f"postgresql+asyncpg://{self.USER}:{self.PASS}@{self.HOST}:{self.PORT}/{self.NAME}"

class BotSettings(BaseSettings):
    """Настройки для Telegram-бота. Читает переменные с префиксом TGSERVER__"""
    TOKEN: str
    WEBHOOK_URL: str | None = None

    class Config:
        env_prefix = 'TGSERVER__' # Указываем префикс для этой модели

class Settings(BaseSettings):
    """Главный класс настроек."""
    DB: DatabaseSettings = DatabaseSettings()
    TGSERVER: BotSettings = BotSettings() # Имя поля должно совпадать с префиксом

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings = Settings()