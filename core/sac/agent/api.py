from typing import Any, Dict, Optional, Union, List
import numpy as np
import torch
from torch.amp import autocast
from torch import Tensor

from core.sac.agent.core import CoreSACAgent
from core.logging.logger import get_logger, log_method_call, DEBUG

logger = get_logger("sac_agent")

class SACAgent(CoreSACAgent):
    """
    Публичный API SACAgent с логгированием и делегацией CoreSACAgent.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("SACAgent инициализирован")

    @log_method_call()
    def train(self, mode: bool = True) -> 'SACAgent':
        super().train(mode)
        return self

    @log_method_call(log_level=DEBUG, log_args=False, log_return=False)
    def select_action(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
        evaluate: bool = False,
    ) -> torch.Tensor:
        return super().select_action(obs, deterministic, evaluate)

    @log_method_call(log_level=DEBUG)
    def act(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        return super().act(obs, deterministic)

    @log_method_call(log_level=DEBUG)
    def update(self, batch_size: int) -> Optional[Dict[str, float]]:
        return super().update(batch_size)

    @log_method_call()
    def perform_updates(
        self, num_updates: int, batch_size: int
    ) -> Optional[Dict[str, float]]:
        return super().perform_updates(num_updates, batch_size)

    @log_method_call()
    def to_device(self, device: Union[str, torch.device]) -> 'SACAgent':
        super().to_device(device)
        return self

    @log_method_call()
    def export_to_onnx(
        self,
        obs_shape: Union[tuple, List[tuple], Tensor],
        export_dir: str,
        filename: str = "policy.onnx",
    ) -> Optional[str]:
        return super().export_to_onnx(obs_shape, export_dir, filename)

    @log_method_call()
    def save(self, path: str) -> None:
        super().save(path)

    @log_method_call()
    def load(self, path: str, map_location: Optional[str] = None) -> None:
        super().load(path, map_location)

    @log_method_call()
    def eval(self) -> 'SACAgent':
        super().eval()
        return self

    @log_method_call()
    def reset_hidden(self) -> None:
        super().reset_hidden()

    def start_parallel(
        self, env: Any, batch_size: int, max_update_steps: Optional[int] = None
    ) -> None:
        super().start_parallel(env, batch_size, max_update_steps)

    def stop_parallel(self) -> None:
        super().stop_parallel()

__all__ = ['SACAgent'] 