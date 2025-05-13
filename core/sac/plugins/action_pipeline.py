from typing import Any, Optional
import numpy as np
import torch

from core.sac.plugins.plugin_base import AgentPlugin, register_plugin

@register_plugin("action_pipeline")
class ActionPipelinePlugin(AgentPlugin):
    """
    Плагин для управления конвейером действий агента:
      - Вызывает перед выбором действий (before_action)
      - Преобразует тип действия после выборки (after_action)
    """
    def before_action(self, obs: Any, deterministic: bool = False) -> Optional[Any]:
        """
        Хук перед выбором действия: возвращает None, если агент должен выбрать действие.
        """
        return None

    def after_action(self, action: Any, obs: Any) -> Any:
        """
        Хук после выбора действия: преобразует torch.Tensor в np.ndarray, если входное obs было np.ndarray.
        """
        # Преобразуем тензор на CPU в numpy, если вход был numpy
        if isinstance(obs, np.ndarray):
            if isinstance(action, torch.Tensor):
                return action.detach().cpu().numpy()
            return action
        return action 