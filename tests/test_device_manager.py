import torch
import pytest
from core.sac.device import device_manager


def test_get_default_device():
    dev = device_manager.get_device(None)
    assert isinstance(dev, torch.device)


def test_to_device_tensor_and_model():
    t = torch.arange(4)
    moved = device_manager.to_device(t, 'cpu')
    assert moved.device.type == 'cpu'

    import torch.nn as nn
    model = nn.Linear(2, 2)
    target_dev = device_manager.get_device('cpu')
    moved_model = device_manager.to_device(model, target_dev)
    for p in moved_model.parameters():
        assert p.device == target_dev


def test_to_device_nested_structures():
    t = torch.zeros(3)
    data = {'x': t, 'lst': [t, {'inner': t}]}
    moved_data = device_manager.to_device(data, 'cpu')
    assert isinstance(moved_data, dict)
    assert moved_data['x'].device.type == 'cpu'
    assert isinstance(moved_data['lst'], list)
    assert moved_data['lst'][0].device.type == 'cpu'
    assert moved_data['lst'][1]['inner'].device.type == 'cpu'


def test_async_data_transfer_nested():
    t = torch.zeros(3)
    data = {'x': t, 'lst': [t, {'inner': t}]}
    moved = device_manager.async_data_transfer(data, source_device=torch.device('cpu'), target_device=torch.device('cpu'))
    assert isinstance(moved, dict)
    assert moved['x'].device.type == 'cpu'
    assert moved['lst'][1]['inner'].device.type == 'cpu' 