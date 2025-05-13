from core.sac.plugins.plugin_base import AgentPlugin, register_plugin
from typing import Any
import torch
import math

from core.logging.logger import get_logger

logger = get_logger("jit_compiler")

@register_plugin("jit_compiler")
class JITCompiler(AgentPlugin):
    """
    Плагин для JIT-компиляции критических функций агента на GPU.
    """
    def __init__(self, agent: Any, **config: Any):
        super().__init__(agent, **config)

    def on_init(self) -> None:
        """
        Выполняет JIT-компиляцию методов policy.sample, critic.forward, policy.evaluate, select_action, act при инициализации агента.
        """
        device = self.agent.device
        if not torch.cuda.is_available() or device.type != 'cuda':
            return
        try:
            # Компилируем sample
            if hasattr(self.agent.policy, 'sample'):
                test_obs = torch.zeros((1, self.agent._obs_dim), device=device)
                _ = self.agent.policy.sample(test_obs)
            # Компилируем critic.forward
            if hasattr(self.agent.critic, 'forward'):
                test_obs = torch.zeros((1, self.agent._obs_dim), device=device)
                test_action = torch.zeros((1, self.agent._action_dim), device=device)
                _ = self.agent.critic.forward(test_obs, test_action)
            # Компилируем policy.evaluate
            if hasattr(self.agent.policy, 'evaluate'):
                test_obs = torch.zeros((1, self.agent._obs_dim), device=device)
                _ = self.agent.policy.evaluate(test_obs)
            # Компилируем select_action
            if hasattr(self.agent, 'select_action'):
                test_obs = torch.zeros((1, self.agent._obs_dim), device=device)
                _ = self.agent.select_action(test_obs)
            # Компилируем act
            if hasattr(self.agent, 'act'):
                test_obs = torch.zeros((1, self.agent._obs_dim), device=device)
                _ = self.agent.act(test_obs)
        except Exception as e:
            logger.error(f"Ошибка при JIT компиляции: {e}") 

