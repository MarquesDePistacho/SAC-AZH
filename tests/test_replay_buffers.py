import torch
import numpy as np
import pytest
from core.sac.buffers.replay import ReplayBuffer, PrioritizedReplayBuffer, SequenceReplayBuffer, PrioritizedSequenceReplayBuffer

@pytest.mark.parametrize("BufferClass", [ReplayBuffer, PrioritizedReplayBuffer])
def test_replay_buffer_add_and_sample(BufferClass):
    buf = BufferClass(capacity=10, obs_dim=3, action_dim=2)
    for _ in range(8):
        buf.add(np.ones(3), np.ones(2), 1.0, np.ones(3), False)
    batch = buf.sample(4)
    assert batch["obs"].shape == (4, 3)
    assert batch["actions"].shape == (4, 2)
    assert batch["rewards"].shape == (4, 1)
    assert buf.can_sample(4)
    assert isinstance(batch["obs"], torch.Tensor)
    assert isinstance(batch["actions"], torch.Tensor)
    # Проверка clear и повторного заполнения
    buf.clear()
    assert buf.is_empty() or len(buf) == 0
    for _ in range(5):
        buf.add(np.ones(3), np.ones(2), 1.0, np.ones(3), False)
    assert buf.can_sample(2)

def test_sequence_replay_buffer_add_and_sample():
    buf = SequenceReplayBuffer(capacity=10, obs_dim=2, action_dim=1, sequence_length=3)
    for _ in range(10):
        buf.add(np.ones(2), np.ones(1), 1.0, np.ones(2), False)
    batch = buf.sample(2)
    assert batch["obs"].shape == (2, 3, 2)
    assert batch["actions"].shape == (2, 3, 1)
    assert isinstance(batch["obs"], torch.Tensor)
    assert isinstance(batch["actions"], torch.Tensor)
    # Проверка can_sample и clear
    assert buf.can_sample(2)
    buf.clear()
    assert buf.is_empty() or len(buf) == 0

def test_prioritized_sequence_replay_buffer_add_and_sample():
    buf = PrioritizedSequenceReplayBuffer(capacity=10, obs_dim=2, action_dim=1, sequence_length=3)
    for _ in range(10):
        buf.add(np.ones(2), np.ones(1), 1.0, np.ones(2), False)
    batch = buf.sample(2)
    assert batch["obs"].shape[1:] == (3, 2)
    assert isinstance(batch["obs"], torch.Tensor)
    # Проверка can_sample и clear
    assert buf.can_sample(2)
    buf.clear()
    assert buf.is_empty() or len(buf) == 0

def test_empty_buffer_sample():
    buf = ReplayBuffer(capacity=5, obs_dim=2, action_dim=1)
    with pytest.raises(AssertionError):
        buf.sample(1)