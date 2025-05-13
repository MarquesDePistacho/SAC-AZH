import torch
from torch import Tensor
from typing import Tuple


def compute_critic_target(
    rewards: Tensor,
    next_q_values: Tensor,
    next_log_probs: Tensor,
    dones: Tensor,
    gamma: float,
    alpha: Tensor,
) -> Tensor:
    """
    Вычисляет целевые значения Q-функции для критика SAC.
    Формула: reward + gamma * (1 - done) * (next_q_value - alpha * next_log_prob)
    """
    return rewards + gamma * (1.0 - dones) * (next_q_values - alpha * next_log_probs)


def compute_critic_loss(
    q1: Tensor, q2: Tensor, q_target: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Вычисляет потерю критика и TD-ошибки.
    Возвращает (loss, td_errors).
    """
    td_error1 = q_target - q1
    td_error2 = q_target - q2
    # Абсолютные TD-ошибки как максимум двух
    td_errors = torch.maximum(td_error1.abs(), td_error2.abs())
    # MSE loss: 0.5*(e1^2 + e2^2)
    critic_loss = 0.5 * (td_error1.pow(2) + td_error2.pow(2))
    return critic_loss, td_errors


def compute_actor_loss(
    log_probs: Tensor, q_values: Tensor, alpha: Tensor
) -> Tensor:
    """
    Вычисляет функцию потерь актора: alpha * log_prob - Q(s, a).
    """
    return alpha * log_probs - q_values 