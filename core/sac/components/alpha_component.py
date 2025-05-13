import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union

from core.sac.components.base_component import PlugAndPlayComponent, register_component
from core.logging.logger import get_logger, log_method_call, log_tensor_info, DEBUG
from core.utils.device_utils import device_manager

logger = get_logger("alpha_component")


@register_component("alpha")
class AlphaComponent(PlugAndPlayComponent):
    """
    Компонент для управления alpha (энтропия) в SAC.

    Attributes:
        action_dim (int): Размерность действия.
        learn_alpha (bool): Флаг, указывающий, нужно ли обучать alpha.
        alpha_value (float): Текущее значение alpha.
        clip_grad_norm (float): Максимальная норма градиента для clipping'a.
        target_entropy (Optional[float]): Целевая энтропия для действий.
        log_alpha (nn.Parameter или torch.Tensor): Логарифм значения alpha.
        optimizer (Optional[torch.optim.Optimizer]): Оптимизатор для обучения alpha.
        adaptive_target_entropy (bool): Использовать ли адаптивную целевую энтропию.
        entropy_ma_decay (float): Коэффициент затухания для скользящего среднего энтропии.
        _entropy_ma (Optional[float]): Скользящее среднее энтропии (для адаптации).
    """

    def __init__(
        self,
        action_dim: int,
        initial_alpha: float = 0.2,
        learn_alpha: bool = True,
        target_entropy: Optional[float] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 3e-4,
        clip_grad_norm: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        name: str = "alpha",
        adaptive_target_entropy: bool = False,
        entropy_ma_decay: float = 0.99,
    ):
        """
        Инициализация alpha-компонента.\n
        1. Настраивает параметры и целевую энтропию.\n
        2. Создаёт параметр log_alpha.\n
        3. Создаёт оптимизатор.\n

        Args:
            action_dim (int): Размерность пространства действий.
            initial_alpha (float): Начальное значение alpha.
            learn_alpha (bool): Обучать ли alpha во время работы.
            target_entropy (Optional[float]): Целевое значение энтропии.
            optimizer (Optional[torch.optim.Optimizer]): Внешний оптимизатор для alpha.
            lr (float): Скорость обучения, если оптимизатор не задан.
            clip_grad_norm (float): Порог для clipping'а градиентов.
            device (Optional[Union[str, torch.device]]): Устройство для размещения параметров.
            name (str): Имя компонента (для логирования).
            adaptive_target_entropy (bool): Автоматически обновлять целевую энтропию.
            entropy_ma_decay (float): Декаи для усреднения энтропии.
        """
        super().__init__(name=name, device=device)

        self.action_dim = action_dim
        self.learn_alpha = learn_alpha
        self.alpha_value = initial_alpha
        self.clip_grad_norm = clip_grad_norm

        if target_entropy is None and learn_alpha:
            self.target_entropy = -float(action_dim) * 0.98
            # logger.debug(f"Целевая энтропия не указана, установлена эвристически: {self.target_entropy:.4f}")
        else:
            self.target_entropy = target_entropy
            # logger.debug(f"Использована указанная целевая энтропия: {self.target_entropy}")

        self._init_parameters(initial_alpha)

        self._init_optimizer(optimizer, lr)

        logger.info(
            f"Инициализирован компонент Alpha: learn_alpha={learn_alpha}, "
            f"initial_alpha={initial_alpha:.4f}, target_entropy={self.target_entropy:.4f}, "
            f"clip_grad_norm={self.clip_grad_norm}"
        )

        self.adaptive_target_entropy = adaptive_target_entropy
        self.entropy_ma_decay = entropy_ma_decay
        self._entropy_ma = None

    def _init_parameters(self, initial_alpha: float) -> None:
        """
        Инициализирует параметры alpha.

        Args:
            initial_alpha (float): Начальное значение alpha.
        """
        log_alpha_value = torch.log(
            torch.tensor(max(initial_alpha, 1e-8), device=self.device)
        )

        if self.learn_alpha:
            self.log_alpha = nn.Parameter(
                log_alpha_value.clone().detach().to(torch.float32), requires_grad=True
            )
            # logger.debug(f"Создан обучаемый параметр log_alpha: {self.log_alpha.item():.4f}")
        else:
            self.register_buffer(
                "log_alpha", log_alpha_value.clone().detach().to(torch.float32)
            )
            # logger.debug(f"Создан необучаемый буфер log_alpha: {self.log_alpha.item():.4f}")

    def _init_optimizer(
        self, optimizer: Optional[torch.optim.Optimizer], lr: float
    ) -> None:
        """
        Инициализирует оптимизатор для alpha.

        Args:
            optimizer (Optional[torch.optim.Optimizer]): Оптимизатор, переданный извне.
            lr (float): Скорость обучения для внутреннего Adam.
        """
        if not self.learn_alpha:
            self.optimizer = None
            logger.debug("Оптимизатор не требуется (learn_alpha=False)")
            return

        if optimizer is not None:
            self.optimizer = optimizer
            logger.debug(f"Использован внешний оптимизатор: {type(optimizer).__name__}")
        else:
            self.optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            logger.debug(f"Создан внутренний оптимизатор Adam с lr={lr}")

    @property
    def alpha(self) -> torch.Tensor:
        """
        Возвращает текущее значение alpha как тензор.
        """
        return self.log_alpha.exp()

    @property
    def alpha_numpy(self) -> float:
        """
        Возвращает текущее значение alpha как число (float).
        """
        return self.alpha_value

    @property
    def log_alpha_param(self) -> list:
        """
        Возвращает список с параметром log_alpha, если alpha обучается.
        """
        return [self.log_alpha] if self.learn_alpha else []

    def _should_update_alpha(self) -> bool:
        return self.learn_alpha and self.optimizer is not None

    def _sanitize_alpha_value(self, value: float) -> float:
        if not isinstance(value, float):
            try:
                value = float(value)
            except Exception:
                value = 1e-8
        if not (value > 0) or value != value or value == float('inf') or value == float('-inf'):
            return 1e-8
        return value

    @log_method_call()
    def update(self, log_probs: torch.Tensor) -> float:
        if not self._should_update_alpha():
            return 0.0
        if log_probs is None:
            logger.warning("Пропуск обновления alpha (log_probs=None)")
            return 0.0
        log_probs = device_manager.ensure_tensor_on_device(log_probs, self.device, self.log_alpha.dtype, name="log_probs")
        log_probs = device_manager.clean_tensor(log_probs)
        if self.adaptive_target_entropy:
            batch_entropy = -log_probs.mean().item()
            if self._entropy_ma is None:
                self._entropy_ma = batch_entropy
            else:
                self._entropy_ma = (
                    self.entropy_ma_decay * self._entropy_ma
                    + (1 - self.entropy_ma_decay) * batch_entropy
                )
            self.target_entropy = self._entropy_ma
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()
        log_tensor_info(logger, "alpha_loss", alpha_loss, level=DEBUG)
        logger.debug(
            f"Перед обновлением: alpha={self.alpha_numpy:.4f}, target_entropy={self.target_entropy:.4f}"
        )
        self.optimizer.zero_grad()
        alpha_loss.backward()
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.log_alpha_param, self.clip_grad_norm
            )
        self.optimizer.step()
        with torch.no_grad():
            self.alpha_value = self._sanitize_alpha_value(self.alpha.item())
            if self.alpha_value <= 1e-8:
                logger.warning(
                    f"Значение alpha стало слишком маленьким ({self.alpha_value:.2e}), устанавливаем в 1e-8"
                )
                self.set_alpha(1e-8)
        logger.debug(f"После обновления: alpha={self.alpha_numpy:.4f}")
        return alpha_loss.item()

    @log_method_call()
    def set_alpha(self, alpha_value: float) -> None:
        """
        Устанавливает значение alpha вручную.

        Args:
            alpha_value (float): Новое значение alpha.
        """
        with torch.no_grad():
            if alpha_value <= 0:
                logger.warning(
                    f"Попытка установить неположительное alpha: {alpha_value}, используем 1e-8"
                )
                alpha_value = 1e-8

            log_alpha_value = torch.log(
                torch.tensor(alpha_value, dtype=torch.float32, device=self.device)
            )
            if self.learn_alpha:
                self.log_alpha.data.copy_(log_alpha_value)
            else:
                self.log_alpha.copy_(log_alpha_value)

            self.alpha_value = alpha_value

        # logger.debug(f"Alpha установлено вручную: {alpha_value:.4f}")

    @log_method_call()
    def set_target_entropy(self, target_entropy: float) -> None:
        """
        Устанавливает новое значение целевой энтропии.

        Args:
            target_entropy (float): Новая целевая энтропия.
        """
        if self.learn_alpha:
            self.target_entropy = target_entropy
            # logger.debug(f"Целевая энтропия установлена: {target_entropy:.4f}")
        else:
            logger.warning(
                "Попытка установить целевую энтропию при отключенной оптимизации alpha"
            )

    def get_config(self) -> Dict[str, Any]:
        """
        Возвращает конфигурацию компонента в виде словаря.
        """
        config = super().get_config()
        config.update(
            {
                "action_dim": self.action_dim,
                "learn_alpha": self.learn_alpha,
                "alpha": self.alpha_numpy,
                "target_entropy": self.target_entropy,
                "clip_grad_norm": self.clip_grad_norm,
            }
        )
        return config

    def _compute_alpha_loss(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет функцию потерь для alpha.

        Args:
            log_probs (torch.Tensor): Логарифмы вероятностей действий.

        Returns:
            torch.Tensor: Значение функции потерь.
        """
        if not self.learn_alpha:
            return torch.tensor(0.0, device=self.device)

        if log_probs.dim() > 1:
            log_probs = log_probs.mean(dim=0)

        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        return alpha_loss
