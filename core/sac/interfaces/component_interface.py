from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


class ComponentInterface(ABC):
    """
    Интерфейс для SAC-компонентов (policy, critic, alpha, нормализатор).
    """
    @abstractmethod
    def to_device(self, device: Union[str, Any]) -> Any:
        """Перенос компонента на устройство"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Сохранение состояния компонента"""
        pass

    @abstractmethod
    def load(self, path: str, map_location: Optional[Union[str, Any]] = None) -> None:
        """Загрузка состояния компонента"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Сброс внутреннего состояния"""
        pass 