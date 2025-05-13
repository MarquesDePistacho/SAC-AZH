import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
import functools
from typing import Tuple, Dict, Any, Optional, Union
from copy import deepcopy
from core.logging.logger import get_logger, log_method_call, log_tensor_info
from core.utils.device_utils import device_manager

from core.sac.networks import DualMLPQNet, DualLSTMQNet
from core.sac.components.base_component import PlugAndPlayComponent, register_component

logger = get_logger("critic_component")


@register_component("critic")
class CriticComponent(PlugAndPlayComponent):
    """
    Критик SAC: две Q-сети, целевые сети, оптимизатор, обновление весов.
    """

    def __init__(
        self,
        q_net: Union[DualMLPQNet, DualLSTMQNet],
        optimizer: Optional[torch.optim.Optimizer] = None,
        tau: float = 0.005,
        clip_grad_norm: float = 0.0,
        target_update_interval: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        name: str = "critic",
        adaptive_tau: bool = False,
        tau_min: float = 0.001,
        tau_max: float = 0.05,
        tau_ma_decay: float = 0.99,
    ):
        """
        Инициализация критика.\n
        1. Перемещает сети на устройство.\n
        2. Создаёт целевую копию.\n
        3. Настраивает параметры Polyak.\n

        Args:
            q_net (Union[DualMLPQNet, DualLSTMQNet]): Сеть Q-функции.
            optimizer (Optional[torch.optim.Optimizer]): Внешний оптимизатор для обучения.
            tau (float): Коэффициент Polyak для мягкого обновления целевых сетей.
            clip_grad_norm (float): Порог clipping'а градиентов.
            target_update_interval (int): Интервал обновления целевых сетей.
            device (Optional[Union[str, torch.device]]): Устройство вычислений (CPU/GPU).
            name (str): Имя компонента (для логирования).
            adaptive_tau (bool): Флаг использования адаптивного tau.
            tau_min (float): Минимальное значение tau при адаптации.
            tau_max (float): Максимальное значение tau при адаптации.
            tau_ma_decay (float): Декаи скользящего среднего для tau.
        """
        super().__init__(name=name, device=device)

        # Перемещаем основную сеть на устройство компонента
        self.q_net = q_net.to(self.device)

        # Создаем глубокую копию Q-сети для целевой сети и перемещаем на устройство
        self.target_q_net = deepcopy(q_net).to(self.device)

        # Отключаем вычисление градиентов для целевой сети
        for param in self.target_q_net.parameters():
            param.requires_grad = False

        # Определяем, использует ли критик LSTM
        self.use_lstm = isinstance(q_net, DualLSTMQNet)
        self.clip_grad_norm = clip_grad_norm

        # Сохраняем оптимизатор и параметры обновления
        self.optimizer = optimizer
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.update_counter = 0

        # Adaptive tau (Polyak averaging)
        self.adaptive_tau = adaptive_tau
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_ma_decay = tau_ma_decay
        self._loss_ma = None
        self._loss_var = None
        
        # Кэшируем dtype параметров сети для быстрого доступа
        try:
            self.param_dtype = next(self.q_net.parameters()).dtype
        except StopIteration:
            self.param_dtype = torch.float32
            logger.warning(f"Не удалось определить dtype параметров сети, используем {self.param_dtype}")

        logger.info(
            f"Инициализирован компонент Critic '{self.name}': "
            f"Тип={type(q_net).__name__}, use_lstm={self.use_lstm}, tau={tau}, "
            f"clip_grad_norm={self.clip_grad_norm}, target_update_interval={target_update_interval}"
        )

    def _ensure_tensor_on_device(self, tensor, tensor_name="input"):
        return device_manager.ensure_tensor_on_device(tensor, self.device, self.param_dtype, name=tensor_name)

    @torch.jit.export
    @log_method_call(log_args=False, log_return=False)
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden1: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden2: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Прямой проход через обе Q-сети.\n
        1. Приводит входы к нужному типу и устройству.\n
        2. Применяет autocast для AMP.\n
        3. Возвращает Q1, Q2 (и скрытые состояния для LSTM).\n

        Args:
            obs (torch.Tensor): Тензор наблюдений.
            action (torch.Tensor): Тензор действий.
            hidden1 (Optional[Tuple[torch.Tensor, torch.Tensor]]): Скрытое состояние первой LSTM.
            hidden2 (Optional[Tuple[torch.Tensor, torch.Tensor]]): Скрытое состояние второй LSTM.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Tuple], Optional[Tuple]]:\n
                - Q1: Выход первой Q-сети.\n
                - Q2: Выход второй Q-сети.\n
                - Новое скрытое состояние первой LSTM (если используется).\n
                - Новое скрытое состояние второй LSTM (если используется).\n
        """
        # Обеспечиваем правильные типы и устройства тензоров
        obs = self._ensure_tensor_on_device(obs, "obs")
        action = self._ensure_tensor_on_device(action, "action")

        # Выполняем прямой проход с AMP
        device_type = self.device.type
        with autocast(
            device_type=device_type, enabled=(self.enable_amp and device_type == "cuda")
        ):
            if self.use_lstm:
                # Для LSTM Q-сетей
                q1, q2, hidden1_new, hidden2_new = self.q_net(
                    obs, action, hidden1, hidden2
                )
                return q1, q2, hidden1_new, hidden2_new
            else:
                # Для MLP Q-сетей
                q1, q2 = self.q_net(obs, action)
                return q1, q2, None, None

    @torch.jit.export
    @log_method_call(log_args=False, log_return=False)
    def target_forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden1: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden2: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Прямой проход через целевые Q-сети (без градиентов).

        Args:
            obs (torch.Tensor): Тензор наблюдений.
            action (torch.Tensor): Тензор действий.
            hidden1 (Optional[Tuple[torch.Tensor, torch.Tensor]]): Скрытое состояние первой LSTM.
            hidden2 (Optional[Tuple[torch.Tensor, torch.Tensor]]): Скрытое состояние второй LSTM.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Tuple], Optional[Tuple]]:\n
                - Q1_target: Выход первой целевой Q-сети.\n
                - Q2_target: Выход второй целевой Q-сети.\n
                - Новое скрытое состояние первой LSTM (если используется).\n
                - Новое скрытое состояние второй LSTM (если используется).\n
        """
        # Обеспечиваем правильные типы и устройства тензоров
        obs = self._ensure_tensor_on_device(obs, "obs")
        action = self._ensure_tensor_on_device(action, "action")

        # Выполняем без градиентов и с AMP
        device_type = self.device.type
        with torch.no_grad(), autocast(
                device_type=device_type,
                enabled=(self.enable_amp and device_type == "cuda"),
            ):
            if self.use_lstm:
                # Для LSTM Q-сетей
                q1, q2, hidden1_new, hidden2_new = self.target_q_net(
                    obs, action, hidden1, hidden2
                )
                return q1, q2, hidden1_new, hidden2_new
            else:
                # Для MLP Q-сетей
                q1, q2 = self.target_q_net(obs, action)
                return q1, q2, None, None

    @log_method_call(log_args=False, log_return=False)
    def q1_forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Прямой проход только через первую Q-сеть (для актора).

        Args:
            obs (torch.Tensor): Тензор наблюдений.
            action (torch.Tensor): Тензор действий.
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]]): Скрытое состояние LSTM.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple]]:\n
                - Q1: Значение первой Q-сети.\n
                - Новое скрытое состояние LSTM (если используется).\n
        """
        # Обеспечиваем правильные типы и устройства тензоров
        obs = self._ensure_tensor_on_device(obs, "obs")
        action = self._ensure_tensor_on_device(action, "action")

        # Выполняем с AMP
        device_type = self.device.type
        with autocast(
            device_type=device_type, enabled=(self.enable_amp and device_type == "cuda")
        ):
            if self.use_lstm:
                # Для LSTM Q-сетей
                q1, hidden_new = self.q_net.q1_value(obs, action, hidden)
                return q1, hidden_new
            else:
                # Для MLP Q-сетей
                q1 = self.q_net.q1_value(obs, action)
                return q1, None

    @log_method_call()
    def soft_update(self) -> None:
        """
        Мягкое обновление целевых Q-сетей с использованием коэффициента tau.
        """
        with torch.no_grad():
            for param, target_param in zip(
                self.q_net.parameters(), self.target_q_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

        logger.debug(f"Целевые Q-сети '{self.name}' обновлены с tau={self.tau:.4f}")

    @log_method_call()
    def compute_critic_loss(
        self,
        batch: Dict[str, torch.Tensor],
        gamma: float,
        next_q_values: torch.Tensor,
        next_action_log_probs: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Считает MSE-лосс критика и TD-ошибку.\n
        1. Вычисляет целевые значения.\n
        2. Считает текущие Q.\n
        3. Считает TD-ошибки и лосс.\n
        4. Возвращает лосс и TD-ошибку для PER.\n

        Args:
            batch (Dict[str, torch.Tensor]): Батч данных в формате словаря.
            gamma (float): Коэффициент дисконтирования.
            next_q_values (torch.Tensor): Следующие значения Q-функции.
            next_action_log_probs (torch.Tensor): Логарифмы вероятностей действий.
            alpha (torch.Tensor): Коэффициент энтропии.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:\n
                - critic_loss: Общая потеря критика.\n
                - td_error: TD-ошибка для PER.\n
        """
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        # next_obs уже нормализован в SACAgent._update_critic
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        # Веса для PER (если используются, иначе единицы)
        weights = batch.get("weights", torch.ones_like(rewards))

        # Вычисление целевого Q-значения с AMP
        device_type = self.device.type
        with autocast(
            device_type=device_type, enabled=(self.enable_amp and device_type == "cuda")
        ):
            with torch.no_grad():
                # Цель = r + gamma * (1 - done) * (min_Q_target(s', a') - alpha * log_pi(a'|s'))
                next_q_target = next_q_values - alpha * next_action_log_probs
                target_q = rewards + gamma * (1.0 - dones) * next_q_target
                log_tensor_info(logger, "target_q", target_q)

            # Получаем текущие Q-значения от основных сетей
            q1, q2, _, _ = self.forward(obs, actions)
            log_tensor_info(logger, "q1_current", q1)
            log_tensor_info(logger, "q2_current", q2)

            # Вычисляем TD-ошибки
            td_error1 = q1 - target_q
            td_error2 = q2 - target_q

            # Вычисляем взвешенные потери MSE для каждой Q-сети
            # loss = E[weights * (Q(s,a) - Target)^2]
            q1_loss = (weights * td_error1.pow(2)).mean()
            q2_loss = (weights * td_error2.pow(2)).mean()

            # Общая потеря критика
            critic_loss = q1_loss + q2_loss

            # TD-ошибка для обновления приоритетов в PER
            # Используем векторизованный минимум для лучшей производительности
            td_error = torch.min(
                torch.stack([td_error1.abs(), td_error2.abs()]), dim=0
            )[0]

        log_tensor_info(logger, "critic_loss", critic_loss)
        log_tensor_info(logger, "td_error", td_error)
        logger.debug(
            f"Потери критика: q1_loss={q1_loss.item():.6f}, q2_loss={q2_loss.item():.6f}"
        )
        return critic_loss, td_error

    @log_method_call()
    def update(self, loss: torch.Tensor) -> float:
        """
        Обновляет веса критика (AMP/GradScaler поддерживаются).\n
        1. Обнуляет градиенты.\n
        2. Делает backward (AMP если включён).\n
        3. Клипует градиенты.\n
        4. Шаг оптимизатора.\n
        5. Адаптирует tau если нужно.\n
        6. Обновляет целевые сети.\n
        
        Args:
            loss (torch.Tensor): Значение функции потерь для обратного распространения ошибки.

        Returns:
            float: Значение потерь как число.
        """
        if self.optimizer is None:
            logger.warning(
                f"Оптимизатор для критика '{self.name}' не установлен, обновление пропущено."
            )
            return loss.item() if isinstance(loss, torch.Tensor) else float(loss)

        self.optimizer.zero_grad()
        # AMP everywhere: autocast + GradScaler
        if self.enable_amp:
            from torch.amp import autocast

            with autocast(
                device_type=self.device.type,
                enabled=True,
                dtype=getattr(self, "amp_dtype", torch.float16),
            ):
                self.scaler.scale(loss).backward()
            if self.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.q_net.parameters(), self.clip_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.q_net.parameters(), self.clip_grad_norm
                )
            self.optimizer.step()
        # Изменяем tau на основе скользящего среднего и дисперсии loss
        if self.adaptive_tau:
            loss_val = loss.item()
            if self._loss_ma is None:
                self._loss_ma = loss_val
                self._loss_var = 0.0
            else:
                prev_ma = self._loss_ma
                self._loss_ma = (
                    self.tau_ma_decay * self._loss_ma
                    + (1 - self.tau_ma_decay) * loss_val
                )
                self._loss_var = (
                    self.tau_ma_decay * self._loss_var
                    + (1 - self.tau_ma_decay) * (loss_val - prev_ma) ** 2
                )
            # Чем стабильнее loss (меньше var), тем больше tau
            tau = float(
                np.clip(self.tau_max - self._loss_var, self.tau_min, self.tau_max)
            )
            self.tau = tau
            logger.debug(
                f"[adaptive_tau] loss_ma={self._loss_ma:.6f}, loss_var={self._loss_var:.6f}, tau={self.tau:.6f}"
            )
        # Обновляем целевые сети (мягкое обновление), если нужно
        self.update_counter += 1
        self.update_targets()
        return loss.item()

    @log_method_call()
    def reset_hidden(self, batch_size: int = 1) -> None:
        """
        Сброс скрытых состояний LSTM (если есть).

        Args:
            batch_size (int): Размер батча для инициализации начального скрытого состояния.
        """
        if self.use_lstm and hasattr(self.q_net, "reset_hidden"):
            # Сбрасываем скрытые состояния для основных и целевых сетей
            self.q_net.reset_hidden(batch_size=batch_size, device=self.device)
            self.target_q_net.reset_hidden(batch_size=batch_size, device=self.device)
            logger.debug(
                f"Сброшены скрытые состояния LSTM в Q-сетях '{self.name}', batch_size={batch_size}"
            )

    def get_config(self) -> Dict[str, Any]:
        """
        Возвращает конфигурацию критика в виде словаря.

        Returns:
            Dict[str, Any]: Конфигурационные данные критика.
        """
        config = super().get_config()
        config.update(
            {
                "q_net_type": type(self.q_net).__name__,
                "use_lstm": self.use_lstm,
                "tau": self.tau,
                "clip_grad_norm": self.clip_grad_norm,
                "target_update_interval": self.target_update_interval,
            }
        )
        return config

    def update_targets(self) -> None:
        """
        Обновляет целевые сети, если пришло время (на основе update_counter).
        """
        # Выполняем мягкое обновление только если счетчик достиг интервала
        if self.update_counter % self.target_update_interval == 0:
            self.soft_update()
