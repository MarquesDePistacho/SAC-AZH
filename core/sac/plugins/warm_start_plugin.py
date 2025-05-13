from typing import Any, Dict, Optional
import numpy as np
import torch

from core.sac.plugins.plugin_base import AgentPlugin, register_plugin
from core.sac.components.warm_start_component import WarmStartComponent

@register_plugin("warm_start")
class WarmStartPlugin(AgentPlugin):
    """
    Плагин теплого старта: генерирует случайные действия на начальных шагах,
    предсказывает действия через WarmStartComponent и обновляет её буферы.
    """
    def __init__(self, agent: Any, **config: Any):
        super().__init__(agent, **config)
        # Инициализируем внутренний компонент для хранения данных
        self.component = WarmStartComponent(
            action_dim=agent._action_dim,
            obs_dim=agent._obs_dim,
            device=agent.device,
            **config
        )

    def on_init(self) -> None:
        """Хук после инициализации агента."""
        pass

    def before_action(self, obs: Any, deterministic: bool = False) -> Optional[Any]:
        """
        Вызывается перед select_action.
        Возвращает действие, если нужно пропустить выбор политикой.
        """
        # Случайные действия в первые random_steps
        if self.component.should_use_random_action():
            action_np = np.random.uniform(-1.0, 1.0, size=self.component.action_dim)
            return torch.as_tensor(action_np, dtype=torch.float32, device=self.agent.device)
        # Предсказанное действие регрессором
        predicted = self.component.predict_action(obs)
        if predicted is not None:
            return predicted
        return None

    def after_action(self, action: Any, obs: Any) -> Any:
        """
        Вызывается после получения действия.
        Добавляет данные в буфер и обновляет состояние компонента.
        """
        self.component.add_sample(obs, action)
        self.component.step()
        return action

    def after_update(self, metrics: Dict[str, Any], batch: Optional[Dict[str, Any]] = None) -> None:
        """
        Вызывается после update: можно обучить регрессор.
        """
        self.component.train_model() 