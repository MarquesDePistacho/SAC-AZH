import torch
import numpy as np
from core.sac.normalizers import WelfordObservationNormalizer, BatchMeanStdNormalizer

def test_welford_normalizer_update_and_normalize():
    norm = WelfordObservationNormalizer(obs_dim=3)
    data = torch.randn(10, 3)
    norm.update(data)
    normed = norm.normalize(data)
    assert normed.shape == data.shape
    assert torch.isfinite(normed).all()

def test_batch_mean_std_normalizer():
    norm = BatchMeanStdNormalizer(shape=(3,))
    data = np.random.randn(20, 3).astype(np.float32)
    norm.update(torch.from_numpy(data))
    normed = norm.normalize(torch.from_numpy(data))
    assert normed.shape == data.shape
    assert torch.isfinite(normed).all()