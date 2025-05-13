from abc import ABC, abstractmethod
from typing import Any


class FactoryInterface(ABC):
    """
    Интерфейс для Factory-классов (PolicyFactory, QNetFactory, ReplayBufferFactory и т.д.).
    """
    @classmethod
    @abstractmethod
    def create(cls, *args: Any, **kwargs: Any) -> Any:
        """Создает объект по заданным параметрам"""
        pass 