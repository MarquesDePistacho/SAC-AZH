import torch
from core.sac.batching import SampleDataset, BatchFetcher


def test_sample_dataset():
    class DummyBuffer:
        def __init__(self):
            self.counter = 0
        def sample(self, batch_size):
            self.counter += 1
            return {'value': torch.tensor([self.counter] * batch_size)}

    buf = DummyBuffer()
    ds = SampleDataset(buf, batch_size=3)
    it = iter(ds)
    batch1 = next(it)
    batch2 = next(it)
    assert batch1['value'].tolist() == [1, 1, 1]
    assert batch2['value'].tolist() == [2, 2, 2]


def test_batch_fetcher_no_workers():
    class DummyBuffer:
        def __init__(self):
            self.calls = 0
        def sample(self, batch_size):
            self.calls += 1
            return {'x': torch.arange(batch_size) + self.calls}

    buf = DummyBuffer()
    fetcher = BatchFetcher(buf, num_workers=0)
    batch = fetcher.next_batch(4)
    assert 'x' in batch
    assert batch['x'].shape[0] == 4
    assert buf.calls == 1


def test_batch_fetcher_with_workers(tmp_path):
    # Проверим, что DataLoader и IterableDataset работают (num_workers=0 ещё имитируем)
    class DummyBuffer2:
        def sample(self, batch_size):
            return {'x': torch.zeros(batch_size)}

    buf2 = DummyBuffer2()
    fetcher2 = BatchFetcher(buf2, num_workers=0)
    batch = fetcher2.next_batch(2)
    assert torch.all(batch['x'] == 0)


def test_batch_fetcher_next_batch():
    class DummyBuffer:
        def __init__(self):
            self.calls = 0
        def sample(self, batch_size):
            self.calls += batch_size
            return {'x': torch.arange(batch_size) + self.calls}

    buf = DummyBuffer()
    fetcher = BatchFetcher(buf, num_workers=0)
    batch = fetcher.next_batch(4)
    assert 'x' in batch
    assert batch['x'].tolist() == [4, 5, 6, 7] 