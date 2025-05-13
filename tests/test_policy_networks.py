import torch
import pytest
from core.sac.networks import MLPPolicy, LSTMPolicy

@pytest.mark.parametrize("batch,obs_dim,action_dim", [(4, 8, 3), (1, 5, 2)])
def test_mlp_policy_forward(batch, obs_dim, action_dim):
    net = MLPPolicy(input_dim=obs_dim, action_dim=action_dim, hidden_dims=[16, 16])
    obs = torch.randn(batch, obs_dim)
    mean, log_std = net(obs)
    assert mean.shape == (batch, action_dim)
    assert log_std.shape == (batch, action_dim)
    assert mean.dtype == torch.float32

def test_lstm_policy_forward():
    obs_dim, action_dim, batch, seq = 6, 2, 3, 5
    net = LSTMPolicy(input_dim=obs_dim, action_dim=action_dim, hidden_dim=8, num_layers=1)
    obs = torch.randn(batch, seq, obs_dim)
    h0 = (torch.zeros(1, batch, 8), torch.zeros(1, batch, 8))
    mean, log_std, new_hidden = net(obs, h0)
    assert mean.shape == (batch, seq, action_dim)
    assert log_std.shape == (batch, seq, action_dim)
    assert isinstance(new_hidden, tuple)