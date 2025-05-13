import torch
from core.sac.components.alpha_component import AlphaComponent
import math

def test_alpha_update_and_set():
    comp = AlphaComponent(action_dim=2, initial_alpha=0.2, learn_alpha=True)
    log_probs = torch.randn(8)
    loss = comp.update(log_probs)
    assert isinstance(loss, float)
    old_alpha = comp.alpha.item()
    comp.set_alpha(0.5)
    assert abs(comp.alpha.item() - 0.5) < 1e-5
    comp.set_alpha(-1.0)  # Должно быть >= 1e-8
    assert math.isclose(comp.alpha.item(), 1e-8, rel_tol=1e-6)