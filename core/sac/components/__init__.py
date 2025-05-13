from core.sac.components.base_component import SACComponent
from core.sac.components.policy_component import PolicyComponent
from core.sac.components.critic_component import CriticComponent
from core.sac.components.alpha_component import AlphaComponent
from core.sac.components.normalizer_component import NormalizerComponent

# Список экспортируемых компонентов
__all__ = [
    'SACComponent',         # Базовый компонент
    'PolicyComponent',      # Компонент политики (актор)
    'CriticComponent',      # Компонент критика (Q-сети)
    'AlphaComponent',       # Компонент для alpha (коэффициент энтропии)
    'NormalizerComponent',  # Компонент для нормализации наблюдений
] 