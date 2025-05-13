import torch
import torch.nn as nn
from torch.amp import autocast
import functools
from typing import Tuple, Dict, Any, Optional, Union, List, Type, Callable

from core.sac.networks import MLPPolicy, LSTMPolicy, BasePolicy
from core.sac.components.base_component import PlugAndPlayComponent, register_component
from core.logging.logger import get_logger, log_method_call, log_tensor_info
from core.utils.device_utils import device_manager

logger = get_logger("policy_component")


@register_component("policy")
class PolicyComponent(PlugAndPlayComponent):
    """
    Компонент политики SAC. Управляет сетью, оптимизатором и выбором действий.
    """

    def __init__(
        self,
        policy_net: Union[MLPPolicy, LSTMPolicy, BasePolicy],
        optimizer: Optional[torch.optim.Optimizer] = None,
        clip_grad_norm: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        name: str = "policy",
    ):
        """
        Инициализация компонента политики.\n
        1. Перемещает сеть на устройство.\n
        2. Определяет размерность действий.\n
        3. Сохраняет оптимизатор.\n

        Args:
            policy_net (Union[MLPPolicy, LSTMPolicy, BasePolicy]): Нейросеть для политики.
            optimizer (Optional[torch.optim.Optimizer]): Внешний оптимизатор политики.
            clip_grad_norm (float): Порог для clipping'а градиентов.
            device (Optional[Union[str, torch.device]]): Целевое устройство (CPU/GPU).
            name (str): Имя компонента (для логирования).
        """
        super().__init__(name=name, device=device)

        # Перемещаем сеть на устройство компонента
        self.policy_net = policy_net.to(self.device)

        # Определяем, использует ли политика LSTM
        self.use_lstm = isinstance(policy_net, LSTMPolicy)
        self.clip_grad_norm = clip_grad_norm

        # Сохраняем оптимизатор
        self.optimizer = optimizer

        # Получаем размерность действий
        if hasattr(policy_net, "action_dim"):
            self.action_dim = policy_net.action_dim
        elif hasattr(policy_net, "mean_head") and isinstance(
            policy_net.mean_head, nn.Linear
        ):
            self.action_dim = policy_net.mean_head.out_features
        elif hasattr(policy_net, "combined_head") and isinstance(
            policy_net.combined_head, nn.Linear
        ):
            self.action_dim = policy_net.combined_head.out_features // 2
        else:
            # Пытаемся получить из последнего слоя, если это Linear
            last_linear = None
            for module in reversed(list(policy_net.modules())):
                if isinstance(module, nn.Linear):
                    last_linear = module
                    break
            if last_linear:
                self.action_dim = last_linear.out_features // 2 if 'combined' in last_linear.__class__.__name__.lower() else last_linear.out_features
                logger.warning(
                    f"Не удалось определить action_dim напрямую, предполагаем {self.action_dim} из последнего Linear слоя."
                )
            else:
                self.action_dim = 1  # Значение по умолчанию, если не удалось определить
                logger.error("Не удалось определить action_dim для политики!")

        # Кэшируем dtype параметров сети для быстрого доступа
        try:
            self.param_dtype = next(self.policy_net.parameters()).dtype
        except StopIteration:
            self.param_dtype = torch.float32
            logger.warning(f"Не удалось определить dtype параметров сети, используем {self.param_dtype}")

        logger.info(
            f"Инициализирован компонент Policy '{self.name}': "
            f"Тип={type(policy_net).__name__}, use_lstm={self.use_lstm}, "
            f"action_dim={self.action_dim}, clip_grad_norm={self.clip_grad_norm}"
        )

    def _prepare_obs_and_hidden(self, obs, hidden):
        """
        Подготавливает наблюдения и скрытое состояние для LSTM:
        - Приводит форму наблюдений к [batch, seq, obs_dim].
        - Сбрасывает или инициализирует скрытое состояние при необходимости. 
        """
        obs = device_manager.prepare_input_tensor(obs, self.device, self.param_dtype, min_dim=3, name="obs")
        batch_size = obs.shape[0]
        if hidden is None or hidden[0].shape[1] != batch_size:
            self.policy_net.reset_hidden(batch_size=batch_size, device=self.device)
            hidden = self.policy_net.get_hidden()
        return obs, hidden

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Прямой проход через политику.

        Args:
            obs (torch.Tensor): Тензор наблюдений.
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]]): Скрытое состояние LSTM (если используется).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:\n
                - mean: Среднее значение действия.\n
                - log_std: Логарифм стандартного отклонения.\n
                - new_hidden: Новое скрытое состояние LSTM (если используется).\n
        """
        if self.use_lstm:
            obs, hidden = self._prepare_obs_and_hidden(obs, hidden)
        else:
            obs = device_manager.prepare_input_tensor(obs, self.device, self.param_dtype, min_dim=2, name="obs")
        device_type = self.device.type
        with autocast(
            device_type=device_type, enabled=(self.enable_amp and device_type == "cuda")
        ):
            if self.use_lstm:
                mean, log_std, new_hidden = self.policy_net(obs, hidden)
                return mean, log_std, new_hidden
            else:
                mean, log_std = self.policy_net(obs)
                return mean, log_std, None

    def sample(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,
    ]:
        """
        Выборка действия из политики на основе наблюдения.

        Args:
            obs (torch.Tensor): Тензор наблюдений.
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]]): Скрытое состояние LSTM (если используется).
            deterministic (bool): Флаг детерминированного выбора действий.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Tuple], torch.Tensor]:\n
                - action: Выбранное действие.\n
                - log_prob: Логарифм вероятности действия.\n
                - new_hidden: Новое скрытое состояние LSTM (если используется).\n
                - mean: Среднее значение действия.\n
        """
        mean, log_std, new_hidden = self.forward(obs, hidden)
        
        with torch.no_grad(), autocast(
                device_type=self.device.type,
                enabled=(self.enable_amp and self.device.type == "cuda"),
            ):
            if deterministic:
                action = torch.tanh(mean)
                log_prob = torch.zeros_like(mean)
            else:
                log_std = device_manager.clean_tensor(torch.clamp(log_std, -20, 2))
                std = device_manager.clean_tensor(log_std.exp())
                mean = device_manager.clean_tensor(mean)
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.rsample()
                action = torch.tanh(x_t)
                log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
                if log_prob.dim() > 1:
                    log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, new_hidden, mean

    @log_method_call(log_args=False, log_return=False)
    def evaluate(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        epsilon: float = 1e-6,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,
    ]:
        """
        Оценка действия и вычисление логарифма вероятности для обучения.

        Args:
            obs (torch.Tensor): Тензор наблюдений.
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]]): Скрытое состояние LSTM.
            epsilon (float): Малое число для предотвращения деления на ноль.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Tuple], torch.Tensor]:\n
                - action: Выбранное действие.\n
                - log_prob: Логарифм вероятности действия.\n
                - new_hidden: Новое скрытое состояние LSTM.\n
                - mean: Среднее значение действия.\n
        """
        if self.use_lstm:
            obs, hidden = self._prepare_obs_and_hidden(obs, hidden)
        device_type = self.device.type
        with autocast(
            device_type=device_type, enabled=(self.enable_amp and device_type == "cuda")
        ):
            mean, log_std, new_hidden = self.forward(obs, hidden)
            log_std = torch.clamp(log_std, -20, 2)
            log_std = torch.where(
                torch.isfinite(log_std), log_std, torch.zeros_like(log_std)
            )
            std = log_std.exp()
            std = torch.where(
                torch.isfinite(std) & (std > 0), std, torch.ones_like(std) * 1e-3
            )
            mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + epsilon)
            if log_prob.dim() > 1:
                log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, new_hidden, mean

    def _compute_log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисляет логарифм вероятности заданного действия для заданного распределения.

        Args:
            mean (torch.Tensor): Среднее значение действия.
            log_std (torch.Tensor): Логарифм стандартного отклонения.
            action (torch.Tensor): Действие, для которого вычисляется вероятность.

        Returns:
            torch.Tensor: Логарифм вероятности действия.
        """
        log_std = device_manager.clean_tensor(torch.clamp(log_std, -20, 2))
        std = device_manager.clean_tensor(log_std.exp())
        mean = device_manager.clean_tensor(mean)
        normal = torch.distributions.Normal(mean, std)
        action_safe = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
        x_t = torch.atanh(action_safe)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action_safe.pow(2) + 1e-6)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = device_manager.clean_tensor(log_prob)
        return log_prob

    @log_method_call()
    def reset_hidden(self, batch_size: int = 1) -> None:
        """
        Сбрасывает скрытое состояние LSTM (если используется).

        Args:
            batch_size (int): Размер батча для инициализации начального скрытого состояния.
        """
        if self.use_lstm and hasattr(self.policy_net, "reset_hidden"):
            self.policy_net.reset_hidden(batch_size=batch_size, device=self.device)
            logger.debug(
                f"Сброшено скрытое состояние LSTM для политики '{self.name}', batch_size={batch_size}"
            )

    @log_method_call()
    def update(self, loss: torch.Tensor) -> float:
        """
        Обновляет веса сети политики с поддержкой AMP и GradScaler.

        Args:
            loss (torch.Tensor): Значение функции потерь для обратного распространения ошибки.

        Returns:
            float: Значение потерь как число.
        """
        if self.optimizer is None:
            logger.warning(
                f"Оптимизатор для политики '{self.name}' не установлен, обновление пропущено."
            )
            return loss.item() if isinstance(loss, torch.Tensor) else float(loss)

        self.optimizer.zero_grad()
        
        if self.enable_amp:
            # Масштабируем loss и делаем backward
            self.scaler.scale(loss).backward()
            # Всегда выполняем unscale для регистрации inf checks
            try:
                self.scaler.unscale_(self.optimizer)
            except RuntimeError:
                pass
            # Обрезка градиентов при необходимости
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), self.clip_grad_norm
                )
            # Шаг оптимизатора с проверкой на Inf (fallback на обычный шаг при ошибке)
            try:
                self.scaler.step(self.optimizer)
            except Exception:
                self.optimizer.step()
            # Обновление scale-фактора, игнорировать отсутствие inf checks
            try:
                self.scaler.update()
            except AssertionError:
                pass
        else:
            # Стандартное backward и шаг оптимизатора без AMP
            loss.backward()
        return loss.item()

    def get_config(self) -> Dict[str, Any]:
        """
        Возвращает конфигурацию компонента политики.

        Returns:
            Dict[str, Any]: Конфигурационные данные политики.
        """
        config = super().get_config()
        config.update(
            {
                "policy_type": type(self.policy_net).__name__,
                "use_lstm": self.use_lstm,
                "action_dim": self.action_dim,
                "clip_grad_norm": self.clip_grad_norm,
            }
        )
        return config

    def train(self, mode: bool = True):
        """
        Переключает внутреннюю сеть политики в режим обучения.

        Args:
            mode (bool): True - режим тренировки, False - режим оценки.

        Returns:
            PolicyComponent: Объект в нужном режиме.
        """
        logger.debug(
            f"PolicyComponent '{self.name}' setting mode to {'train' if mode else 'eval'}"
        )
        self.policy_net.train(mode)
        return self

    def eval(self):
        """
        Переключает внутреннюю сеть политики в режим оценки.

        Returns:
            PolicyComponent: Объект в режиме оценки.
        """
        return self.train(False)
