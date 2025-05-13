import time
from time import perf_counter
import torch
import numpy as np
from typing import List, Optional, Any

from core.sac.buffers import ReplayBuffer, PrioritizedReplayBuffer, SequenceReplayBuffer, PrioritizedSequenceReplayBuffer
from core.sac.batching import BatchFetcher, SampleDataset
from core.sac.losses import compute_actor_loss
from core.sac.components import PolicyComponent, CriticComponent, AlphaComponent, NormalizerComponent
from core.sac.agent import SACAgent
from core.utils.device_utils import device_manager
from core.training.mlflow import benchmark_run, log_metrics, log_params, BENCHMARK_EXPERIMENT


@benchmark_run(experiment_name=BENCHMARK_EXPERIMENT)
def benchmark_buffers(
    capacities: List[int] = [1000000],
    batch_sizes: List[int] = [32, 64, 128, 256, 512],
    runs: int = 5,
    obs_dim: int = 64,
    action_dim: int = 2,
    device: str = "cpu",
) -> None:
    # Логирование параметров происходит автоматически в декораторе benchmark_run
    # Цикл бенчмарков
    for BufClass in [ReplayBuffer, PrioritizedReplayBuffer, SequenceReplayBuffer, PrioritizedSequenceReplayBuffer]:
        for capacity in capacities:
            buf = BufClass(capacity, obs_dim, action_dim, device=device,
                            storage_dtype=torch.float32, use_pinned_memory=False)
            # замер fill
            t0 = perf_counter()
            for _ in range(capacity):
                buf.add(torch.randn(obs_dim), torch.randn(action_dim), 0.0, torch.randn(obs_dim), False)
            fill_time = perf_counter() - t0
            log_metrics({f"{BufClass.__name__}_fill_time": fill_time}, 0)
            # выборка и add/update
            for bs in batch_sizes:
                # sample
                sample_times = []
                for _ in range(runs):
                    t0 = perf_counter()
                    batch = buf.sample(bs)
                    if buf.target_device.type=='cuda': torch.cuda.synchronize()
                    sample_times.append(perf_counter()-t0)
                log_metrics({
                    f"{BufClass.__name__}_sample_mean": np.mean(sample_times),
                    f"{BufClass.__name__}_sample_std": np.std(sample_times)
                }, bs)
                # add
                add_times = []
                for _ in range(runs):
                    t0 = perf_counter()
                    buf.add(torch.randn(obs_dim), torch.randn(action_dim),0.0, torch.randn(obs_dim), False)
                    if buf.target_device.type=='cuda': torch.cuda.synchronize()
                    add_times.append(perf_counter()-t0)
                log_metrics({f"{BufClass.__name__}_add_mean": np.mean(add_times)}, bs)
                # update_priorities для PER
                if hasattr(buf, 'update_priorities'):
                    td_times = []
                    for _ in range(runs):
                        t0 = perf_counter()
                        idxs = batch.get('tree_indices')
                        if idxs is not None:
                            td = torch.rand_like(batch['rewards']).view(-1)
                            buf.update_priorities(idxs, td)
                            if buf.target_device.type=='cuda': torch.cuda.synchronize()
                        td_times.append(perf_counter()-t0)
                    log_metrics({f"{BufClass.__name__}_update_prio_mean": np.mean(td_times)}, bs)


@benchmark_run(experiment_name=BENCHMARK_EXPERIMENT)
def benchmark_batching(
    obs_dim: int = 64,
    action_dim: int = 2,
    batch_sizes: List[int] = [32, 64, 128, 256, 512],
    runs: int = 5,
    device: str = "cpu",
) -> None:
    dummy = ReplayBuffer(1000, obs_dim, action_dim, device=device)
    fetcher = BatchFetcher(dummy, num_workers=0)
    for bs in batch_sizes:
        times=[]
        for _ in range(runs):
            t0=perf_counter()
            _=fetcher.next_batch(bs)
            times.append(perf_counter()-t0)
        log_metrics({
            "batch_fetch_time_mean": np.mean(times),
            "batch_fetch_time_std": np.std(times)
        }, bs)


@benchmark_run(experiment_name=BENCHMARK_EXPERIMENT)
def benchmark_device(
    obs_dim: int = 64,
    runs: int = 5,
    device: str = "cpu",
) -> None:
    # to_device time
    times=[]
    for _ in range(runs):
        t=torch.randn(obs_dim)
        t0=perf_counter(); device_manager.to_device(t,device); times.append(perf_counter()-t0)
    log_metrics({
        "to_device_time_mean": np.mean(times),
        "to_device_time_std": np.std(times)
    }, 0)
    # async_transfer time
    times=[]
    for _ in range(runs):
        batch={'x':torch.randn(obs_dim)}
        t0=perf_counter(); device_manager.async_data_transfer(batch,torch.device(device),torch.device(device)); times.append(perf_counter()-t0)
    log_metrics({
        "async_transfer_time_mean": np.mean(times),
        "async_transfer_time_std": np.std(times)
    }, 0)


@benchmark_run(experiment_name=BENCHMARK_EXPERIMENT)
def benchmark_components(
    obs_dim: int = 64,
    action_dim: int = 2,
    batch_size: int = 128,
    runs: int = 5,
    device: str = "cpu",
) -> None:
    # Policy timing
    import torch.nn as nn
    net = nn.Linear(obs_dim, action_dim).to(device)
    comp = PolicyComponent(net, torch.optim.Adam(net.parameters()), device=device)
    obs = torch.randn(batch_size, obs_dim, device=device)
    # sample timing
    times = []
    for _ in range(runs):
        t0 = perf_counter()
        comp.sample(obs)
        times.append(perf_counter() - t0)
    log_metrics({
        "sample_time_mean": np.mean(times),
        "sample_time_std": np.std(times)
    }, 0)
    # update timing
    times = []
    for _ in range(runs):
        t0 = perf_counter()
        comp.update(torch.randn(batch_size, device=device))
        times.append(perf_counter() - t0)
    log_metrics({
        "update_time_mean": np.mean(times),
        "update_time_std": np.std(times)
    }, 0)
    # Critic timing
    qnet = nn.Linear(obs_dim + action_dim, 1).to(device)
    comp = CriticComponent(qnet, torch.optim.Adam(qnet.parameters()), device=device)
    obs = torch.randn(batch_size, obs_dim, device=device)
    act = torch.randn(batch_size, action_dim, device=device)
    # forward timing
    times = []
    for _ in range(runs):
        t0 = perf_counter()
        comp.forward(obs, act)
        times.append(perf_counter() - t0)
    log_metrics({
        "forward_time_mean": np.mean(times),
        "forward_time_std": np.std(times)
    }, 0)
    # soft_update timing
    times = []
    for _ in range(runs):
        t0 = perf_counter()
        comp.soft_update()
        times.append(perf_counter() - t0)
    log_metrics({
        "soft_update_time_mean": np.mean(times),
        "soft_update_time_std": np.std(times)
    }, 0)


@benchmark_run(experiment_name=BENCHMARK_EXPERIMENT)
def benchmark_agent(
    obs_dim: int = 64,
    action_dim: int = 2,
    batch_size: int = 128,
    runs: int = 5,
    device: str = "cpu",
) -> None:
    # dummy buffer
    dummy = type('B', (), {
        'can_sample': lambda s: True,
        'sample': lambda bs: {
            'obs': torch.randn(bs, obs_dim, device=device),
            'actions': torch.randn(bs, action_dim, device=device),
            'rewards': torch.zeros(bs,1, device=device),
            'next_obs': torch.randn(bs, obs_dim, device=device),
            'dones': torch.zeros(bs,1, device=device),
            'tree_indices': None
        }
    })()
    agent = SACAgent(
        policy_net=torch.nn.Linear(obs_dim, action_dim).to(device),
        q_net=torch.nn.Linear(obs_dim+action_dim,1).to(device),
        replay_buffer=dummy
    )
    # act timing
    times = []
    obs = torch.randn(obs_dim, device=device)
    for _ in range(runs):
        t0 = perf_counter()
        agent.act(obs)
        times.append(perf_counter() - t0)
    log_metrics({
        "act_time_mean": np.mean(times),
        "act_time_std": np.std(times)
    }, 0)
    # update timing
    times = []
    for _ in range(runs):
        t0 = perf_counter()
        agent.update(batch_size)
        times.append(perf_counter() - t0)
    log_metrics({
        "update_time_mean": np.mean(times),
        "update_time_std": np.std(times)
    }, 0)


@benchmark_run(experiment_name=BENCHMARK_EXPERIMENT)
def benchmark_alpha(
    batch_sizes: List[int] = [32,64,128],
    runs: int = 5,
    device: str = "cpu",
) -> None:
    """Бенчмарк обновления AlphaComponent"""
    for bs in batch_sizes:
        comp = AlphaComponent(bs, initial_alpha=0.2, learn_alpha=True, device=device)
        log_probs = torch.randn(bs, device=device)
        times=[]
        for _ in range(runs):
            t0=perf_counter(); comp.update(log_probs); times.append(perf_counter()-t0)
        log_metrics({
            "alpha_update_mean": np.mean(times),
            "alpha_update_std": np.std(times)
        }, bs)


@benchmark_run(experiment_name=BENCHMARK_EXPERIMENT)
def benchmark_normalizer(
    obs_dim: int = 64,
    batch_sizes: List[int] = [32,64,128],
    runs: int = 5,
    device: str = "cpu",
) -> None:
    """Бенчмарк NormalizerComponent.normalize"""
    norm = NormalizerComponent(normalizer=None, clip_obs=10.0, device=device, obs_dim=obs_dim)
    for bs in batch_sizes:
        data = torch.randn(bs, obs_dim, device=device)
        times=[]
        for _ in range(runs):
            t0=perf_counter(); norm.normalize(data); times.append(perf_counter()-t0)
        log_metrics({
            "normalize_mean": np.mean(times),
            "normalize_std": np.std(times)
        }, bs)
