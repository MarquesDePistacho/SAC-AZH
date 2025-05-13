import torch
from core.sac.losses import compute_critic_target, compute_critic_loss, compute_actor_loss


def test_compute_critic_target_and_loss():
    # Простые входы для проверки формул
    rewards = torch.tensor([1.0, 2.0])
    next_q = torch.tensor([0.5, 1.5])
    next_log = torch.tensor([0.1, 0.2])
    dones = torch.tensor([0.0, 1.0])
    gamma = 0.9
    alpha = torch.tensor(0.2)

    target = compute_critic_target(rewards, next_q, next_log, dones, gamma, alpha)
    expected_target = rewards + gamma * (1 - dones) * (next_q - alpha * next_log)
    assert torch.allclose(target, expected_target)

    # Проверяем compute_critic_loss
    q1 = torch.tensor([1.0, 2.0])
    q2 = torch.tensor([0.5, 1.5])
    loss, td_errors = compute_critic_loss(q1, q2, target)
    # Форма выходов
    assert loss.shape == (2,)
    assert td_errors.shape == (2,)
    # Проверяем td_errors = max(|target-q1|, |target-q2|)
    td1 = (target - q1).abs()
    td2 = (target - q2).abs()
    assert torch.allclose(td_errors, torch.maximum(td1, td2))

def test_compute_actor_loss_function():
    # Проверяем compute_actor_loss
    log_probs = torch.tensor([-0.5, -1.0])
    q_values = torch.tensor([0.1, 0.2])
    alpha = torch.tensor(0.2)

    loss = compute_actor_loss(log_probs, q_values, alpha)
    expected = alpha * log_probs - q_values
    assert torch.allclose(loss, expected) 