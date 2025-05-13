from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class PluginInterface(ABC):
    """
    Интерфейс плагина агента SAC.
    """
    @abstractmethod
    def on_init(self) -> None:
        """Вызывается при инициализации агента."""
        pass

    @abstractmethod
    def before_update(self, batch: Dict[str, Any]) -> None:
        """Хук перед обновлением агента."""
        pass

    @abstractmethod
    def after_update(
        self, metrics: Dict[str, Any], batch: Optional[Dict[str, Any]] = None
    ) -> None:
        """Хук после обновления агента."""
        pass

    @abstractmethod
    def before_action(
        self, obs: Any, deterministic: bool = False
    ) -> Optional[Any]:
        """Хук перед выбором действия."""
        pass

    @abstractmethod
    def after_action(self, action: Any, obs: Any) -> Any:
        """Хук после выбора действия."""
        pass 