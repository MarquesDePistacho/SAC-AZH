from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from core.sac.interfaces.plugin_interface import PluginInterface
import inspect

# Реестр плагинов агента
PLUGIN_REGISTRY: Dict[str, Any] = {}

def register_plugin(name: str):
    """Декоратор для регистрации плагинов агента."""
    def decorator(cls):
        PLUGIN_REGISTRY[name] = cls
        return cls
    return decorator

class AgentPlugin(PluginInterface, ABC):
    """
    Базовый класс для плагинов SACAgent. Плагины могут реагировать на события агента.
    """
    def __init__(self, agent: Any, **config: Any):
        self.agent = agent
        self.config = config
        # Автоматическая регистрация хуков
        self._implemented_hooks = self._find_hooks()

    def _find_hooks(self):
        hooks = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith(('on_', 'before_', 'after_')) and method.__func__ is not getattr(AgentPlugin, name, None):
                hooks.append(name)
        return hooks

    def on_init(self) -> None:
        """Вызывается при инициализации агента после создания плагинов."""
        pass

    def before_update(self, batch: Dict[str, Any]) -> None:
        """Вызывается перед обновлением компонентов (update step)."""
        pass

    def after_update(
        self,
        metrics: Dict[str, Any],
        batch: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Хук после выполнения update() агента.

        Args:
            metrics (Dict[str, Any]): Метрики обновления агента.
            batch (Optional[Dict[str, Any]]): Последний обновлённый батч данных.
        """
        pass

    def before_action(
        self,
        obs: Any,
        deterministic: bool = False,
    ) -> Optional[Any]:
        """
        Хук перед выбором действия. Может вернуть действие (torch.Tensor или np.ndarray) для перехвата.
        Если возвращает None, будет вызван стандартный select_action агента.
        """
        return None

    def after_action(
        self,
        action: Any,
        obs: Any,
    ) -> Any:
        """
        Хук после выбора действия. Может изменить или отформатировать действие перед возвращением.
        Возвращает новое действие.
        """
        return action 