from core.sac.plugins.plugin_base import AgentPlugin, register_plugin
from typing import Any, Dict
import math
import numpy as np

@register_plugin("adaptive_explorer")
class AdaptiveExplorerPlugin(AgentPlugin):
    """
    Плагин для расчёта адаптивного log_std на основе скользящего среднего reward.
    """
    def __init__(
        self,
        agent: Any,
        min_log_std: float = -2.0,
        max_log_std: float = 2.0,
        ma_decay: float = 0.99,
        **config: Any
    ):
        super().__init__(agent, min_log_std=min_log_std, max_log_std=max_log_std, ma_decay=ma_decay)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.ma_decay = ma_decay
        self.reward_ma = None
        self.prev_reward_ma = None

    def after_update(self, metrics: Dict[str, float], batch: Dict[str, Any]) -> None:
        """
        Вызывается после обновления компонентов агента, чтобы скорректировать log_std и метрики исследования.
        """
        # Получаем награды из батча и приводим к numpy
        rewards = batch.get('rewards')
        if rewards is None:
            return
        rewards_np = rewards.float().cpu().numpy()
        # Обновляем скользящее среднее и рассчитываем новый log_std
        new_log_std, explorer_metrics = self._compute_exploration(rewards_np)
        # Обновляем параметры политики, если есть log_std_head
        policy_net = getattr(self.agent.policy, 'policy_net', None)
        if policy_net and hasattr(policy_net, 'log_std_head'):
            policy_net.log_std_head.bias.data.fill_(new_log_std)
        # Обновляем метрики агента
        metrics.update(explorer_metrics)
        metrics['adaptive_log_std'] = new_log_std

    def _compute_exploration(self, rewards: np.ndarray) -> (float, Dict[str, float]):
        """
        Расчет нового значения log_std и метрик исследования.
        """
        r_mean = float(np.mean(rewards))
        # Инициализация
        if self.reward_ma is None:
            self.reward_ma = r_mean
            self.prev_reward_ma = r_mean
        else:
            self.prev_reward_ma = self.reward_ma
            self.reward_ma = self.ma_decay * self.reward_ma + (1 - self.ma_decay) * r_mean
        reward_abs = abs(self.reward_ma) + 1e-3
        # Гауссова функция для базового исследования
        base_exploration = math.exp(-(reward_abs**2) / (2 * (self.max_log_std**2)))
        # Динамический фактор исследования
        improvement = self.reward_ma - self.prev_reward_ma
        dyn_factor = 1.0 + max(-improvement / 10.0, 0)
        # Итоговый уровень исследования
        exploration_level = base_exploration * dyn_factor
        # Вычисление нового log_std
        log_range = self.max_log_std - self.min_log_std
        new_log_std = float(np.clip(
            self.min_log_std + exploration_level * log_range,
            self.min_log_std,
            self.max_log_std
        ))
        metrics = {
            'reward_abs': reward_abs,
            'base_exploration': base_exploration,
            'dynamic_factor': dyn_factor,
            'exploration_level': exploration_level
        }
        return new_log_std, metrics 