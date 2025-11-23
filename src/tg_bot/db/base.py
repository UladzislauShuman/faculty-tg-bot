# src/common/db/base.py

from typing import Generic, TypeVar, Optional, List
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
import sqlalchemy as sa

from src.tg_bot.db.orm import Base, asession
from src.tg_bot.db.session_manager import get_session

SchemaType = TypeVar('SchemaType', bound=BaseModel)

class BaseStorage(Generic[SchemaType]):
    """
    Упрощенная базовая реализация репозитория.
    Предоставляет основные CRUD-операции.
    """
    model_cls: type[Base] = None
    schema_cls: type[SchemaType] = None # Pydantic-схема для валидации

    async def get_or_create(
        self,
        session: Optional[AsyncSession] = None,
        defaults: Optional[dict] = None,
        **kwargs,
    ) -> SchemaType:
        """
        Находит объект по **kwargs или создает новый с `defaults`.
        """
        async with get_session(session) as s:
            stmt = sa.select(self.model_cls).filter_by(**kwargs)
            result = await s.execute(stmt)
            instance = result.scalar_one_or_none()

            if instance:
                return self.schema_cls.model_validate(instance)
            else:
                create_kwargs = {**(defaults or {}), **kwargs}
                instance = self.model_cls(**create_kwargs)
                s.add(instance)
                await s.flush() # Получаем ID и другие значения от БД
                return self.schema_cls.model_validate(instance)

    async def create(
        self,
        session: Optional[AsyncSession] = None,
        **kwargs
    ) -> SchemaType:
        """Просто создает новый объект."""
        return await self.get_or_create(session, defaults=kwargs, **kwargs)

    async def list_all(
        self,
        session: Optional[AsyncSession] = None,
        **kwargs
    ) -> List[SchemaType]:
        """Возвращает список всех объектов, подходящих под фильтр."""
        async with get_session(session) as s:
            stmt = sa.select(self.model_cls).filter_by(**kwargs)
            result = await s.execute(stmt)
            instances = result.scalars().all()
            return [self.schema_cls.model_validate(inst) for inst in instances]