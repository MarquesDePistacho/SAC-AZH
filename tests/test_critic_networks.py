import torch
import pytest
from core.sac.networks import MLPQNet, LSTMQNet, DualMLPQNet, DualLSTMQNet

def test_mlpqnet_forward():
    obs_dim, action_dim, batch = 5, 2, 4
    net = MLPQNet(obs_dim, action_dim, [16, 16])
    obs = torch.randn(batch, obs_dim)
    act = torch.randn(batch, action_dim)
    q = net(obs, act)
    assert q.shape == (batch, 1)

def test_dualmlpqnet_forward():
    obs_dim, action_dim, batch = 5, 2, 4
    net = DualMLPQNet(obs_dim, action_dim, [16, 16])
    obs = torch.randn(batch, obs_dim)
    act = torch.randn(batch, action_dim)
    q1, q2 = net(obs, act)
    assert q1.shape == (batch, 1)
    assert q2.shape == (batch, 1)

def test_lstmqnet_forward():
    obs_dim, action_dim, batch, seq = 4, 2, 3, 6
    net = LSTMQNet(obs_dim, action_dim, hidden_dim=8, num_layers=1)
    obs = torch.randn(batch, seq, obs_dim)
    act = torch.randn(batch, seq, action_dim)
    h0 = (torch.zeros(1, batch, 8), torch.zeros(1, batch, 8))
    q, h1 = net(obs, act, h0)
    assert q.shape == (batch, seq, 1)
    assert isinstance(h1, tuple)

def test_duallstmqnet_forward():
    obs_dim, action_dim, batch, seq = 4, 2, 3, 6
    net = DualLSTMQNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=8,
        num_layers=1
    )
    obs = torch.randn(batch, seq, obs_dim)
    act = torch.randn(batch, seq, action_dim)
    h0 = (torch.zeros(1, batch, 8), torch.zeros(1, batch, 8))
    q1, q2, h1, h2 = net(obs, act, h0, h0)
    assert q1.shape == (batch, seq, 1)
    assert q2.shape == (batch, seq, 1)
    assert isinstance(h1, tuple) and isinstance(h2, tuple)
