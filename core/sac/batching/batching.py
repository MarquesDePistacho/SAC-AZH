from torch.utils.data import IterableDataset, DataLoader
from core.utils.device_utils import device_manager


class SampleDataset(IterableDataset):
    """
    IterableDataset, возвращающий батчи из replay_buffer.sample().
    """
    def __init__(self, buffer, batch_size: int):
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield self.buffer.sample(self.batch_size)


class BatchFetcher:
    """
    Сервис для получения батчей из буфера с помощью DataLoader.
    """
    def __init__(self, buffer, num_workers: int = 0):
        self.buffer = buffer
        self.num_workers = num_workers
        self._dataloader = None
        self._dataloader_iter = None
        self._batch_size = None

    def next_batch(self, batch_size: int):
        if self.num_workers > 0:
            if self._dataloader is None or self._batch_size != batch_size:
                dataset = SampleDataset(self.buffer, batch_size)
                self._dataloader = DataLoader(
                    dataset,
                    batch_size=None,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    prefetch_factor=2,
                )
                self._dataloader_iter = iter(self._dataloader)
                self._batch_size = batch_size
            batch = next(self._dataloader_iter)
        else:
            batch = self.buffer.sample(batch_size)
        return batch 