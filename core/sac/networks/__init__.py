import torch
import torch.nn as nn
from torch.amp import autocast
import numpy as np
from typing import Tuple, List, Optional, Union, Any
from abc import ABC, abstractmethod
import math

from core.sac.activations import (
    ReLUActivation,
    LeakyReLUActivation,
    ELUActivation,
    SELUActivation,
    TanhActivation,
    SigmoidActivation,
    SwishActivation,
    MishActivation,
    GeluActivation,
)
from core.logging.logger import get_logger, log_method_call, log_tensor_info
from core.utils.device_utils import device_manager

logger = get_logger("networks")

# --- Словарь доступных активационных функций ---
_activations = {
    "relu": ReLUActivation,
    "leaky_relu": LeakyReLUActivation,
    "elu": ELUActivation,
    "selu": SELUActivation,
    "tanh": TanhActivation,
    "sigmoid": SigmoidActivation,
    "swish": SwishActivation,
    "mish": MishActivation,
    "gelu": GeluActivation,
}


# --- Фабрика для создания активационных функций ---
def _create_activation(activation_type: str, **kwargs) -> nn.Module:
    """
    Создает экземпляр модуля активации по его строковому имени.
    """
    activation_type = activation_type.lower()
    if activation_type not in _activations:
        available = ", ".join(_activations.keys())
        logger.error(
            f"Запрошена неподдерживаемая активация: {activation_type}. Доступные: {available}"
        )
        raise ValueError(
            f"Неподдерживаемая активация: {activation_type}. Доступные: {available}"
        )

    activation_class = _activations[activation_type]
    try:
        return activation_class(**kwargs)
    except Exception as e:
        logger.error(
            f"Ошибка при создании активации '{activation_type}' с параметрами {kwargs}: {e}",
            exc_info=True,
        )
        raise


# --- Базовый класс для всех политик ---
class BasePolicy(nn.Module, ABC):
    """
    Абстрактный базовый класс для всех политик. Определяет интерфейс инициализации, сброса состояния и прямого прохода.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, obs: torch.Tensor, hidden: Optional[Any] = None
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Any]
    ]:
        """
        Выполняет прямой проход через сеть политики для получения параметров распределения действий.
        Реализован в наследниках класса.
        """
        pass

    @abstractmethod
    def sample(
        self,
        obs: torch.Tensor,
        hidden: Optional[Any] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[Any]]:
        """
        Сэмплирует действие из политики на основе наблюдения.
        Реализован в наследниках класса.
        """
        pass

    @abstractmethod
    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_random_init: bool = False,
        random_scale: float = 0.1,
    ) -> None:
        """
        Сбрасывает скрытое состояние (если применимо).
        Реализован в наследниках класса.
        """
        pass

    @abstractmethod
    def get_hidden(self) -> Optional[Any]:
        """
        Возвращает текущее скрытое состояние (если применимо).
        Реализован в наследниках класса.
        """
        pass


# --- MLP политика для SAC ---
class MLPPolicy(BasePolicy):
    """
    MLP-политика для SAC с поддержкой layer norm, dropout, AMP, кастомной инициализации и автоматическим выбором устройства/типа.
    """

    @log_method_call()
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = False,
        use_amp: bool = False,
        custom_init=None,
    ):
        """
        Инициализирует MLP политику.

        Args:
            input_dim (int): Размерность входных данных.
            action_dim (int): Размерность действий.
            hidden_dims (List[int]): Список размерностей скрытых слоев.
            activation (str, optional): Тип активационной функции. По умолчанию "relu".
            log_std_min (float, optional): Минимальное значение логарифма стандартного отклонения. По умолчанию -20.0.
            log_std_max (float, optional): Максимальное значение логарифма стандартного отклонения. По умолчанию 2.0.
            dropout_rate (float, optional): Вероятность dropout. По умолчанию 0.0.
            use_layer_norm (bool, optional): Флаг использования нормализации слоя. По умолчанию False.
            use_amp (bool, optional): Флаг использования автоматического смешанного прецизионного обучения. По умолчанию False.
            custom_init (Optional[Callable], optional): Кастомная функция инициализации весов. По умолчанию None.
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim
        self.enable_amp = use_amp

        layers = []
        prev_dim = input_dim
        logger.debug(
            f"Создание MLPPolicy: input={input_dim}, hidden={hidden_dims}, action={action_dim}, activation={activation}, ln={use_layer_norm}, dropout={dropout_rate}, amp={use_amp}"
        )
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
                logger.debug(f"  Добавлен LayerNorm после слоя {i}")
            layers.append(_create_activation(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
                logger.debug(f"  Добавлен Dropout (p={dropout_rate}) после слоя {i}")
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        logger.debug(f"Создан feature_extractor: {self.feature_extractor}")

        self.combined_head = nn.Linear(prev_dim, action_dim * 2)

        self._init_weights(custom_init)

        logger.info(f"MLPPolicy успешно инициализирована.")

    def _init_weights(self, custom_init=None):
        """
        Инициализирует веса сети с использованием Xavier/Orthogonal инициализации.

        Args:
            custom_init (Optional[Callable], optional): Кастомная функция инициализации весов.
                Если передана, используется вместо стандартной. По умолчанию None.

        Raises:
            Может выбросить исключение, если кастомная инициализация некорректна.
        """
        logger.debug("Инициализация весов MLPPolicy...")
        if custom_init:
            custom_init(self)
            return
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # Используем меньший gain для объединенной головы
        nn.init.xavier_uniform_(self.combined_head.weight, gain=0.01)
        if self.combined_head.bias is not None:
            nn.init.zeros_(self.combined_head.bias)

        logger.debug("Веса MLPPolicy инициализированы.")

    def forward(
        self, obs: torch.Tensor, hidden: Optional[Any] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Выполняет прямой проход для получения среднего и логарифма стандартного отклонения.

        Args:
            obs (torch.Tensor): Наблюдение.
            hidden (Optional[Any], optional): Скрытое состояние (если применимо). По умолчанию None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Кортеж из mean и log_std.
        """
        # Убираем лишнюю размерность, если есть (например, [1, seq_len, dim])
        if obs.dim() == 3 and obs.shape[0] == 1:
            obs = obs.squeeze(0)
        elif obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Добавляем batch_size=1

        device_type = "cuda" if obs.device.type == "cuda" else "cpu"
        amp_enabled = (
            self.enable_amp and device_type == "cuda" and torch.cuda.is_available()
        )

        with autocast(device_type=device_type, enabled=amp_enabled):
            features = self.feature_extractor(obs)
            # Используем единый проход через объединенную голову
            combined_output = self.combined_head(features)
            # Разделяем результат на mean и log_std
            mean, log_std = torch.chunk(combined_output, 2, dim=-1)
        
        # Ограничиваем log_std для стабильности
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        hidden: Optional[Any] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[Any]]:
        """
        Сэмплирует действие из Гауссова распределения с параметрами, предсказанными сетью.

        Args:
            obs (torch.Tensor): Наблюдение.
            hidden (Optional[Any], optional): Скрытое состояние (если применимо). По умолчанию None.
            deterministic (bool, optional): Флаг детерминированного выбора действия. По умолчанию False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[Any]]:
                Кортеж из действия, логарифма вероятности, среднего значения и скрытого состояния.
        """
        mean, log_std = self.forward(obs)
        
        if deterministic:
            action = torch.tanh(mean)
            return action, None, mean, None  

        # Репараметризационный трюк с использованием шума
        std = log_std.exp()
        noise = torch.randn_like(mean)
        x_t = mean + std * noise
        action = torch.tanh(x_t)
        # Вычисляем log_prob для гауссиана и поправку якобиана для tanh
        log_prob = -0.5 * (noise.pow(2) + 2 * log_std + math.log(2 * math.pi))
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean, None  

    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_random_init: bool = False,
        random_scale: float = 0.1,
    ) -> None:
        """
        Сбрасывает скрытое состояние (для совместимости, но не используется в MLPPolicy).

        Args:
            batch_size (int, optional): Размер батча. По умолчанию 1.
            device (Optional[torch.device], optional): Устройство хранения тензора. По умолчанию None.
            dtype (Optional[torch.dtype], optional): Тип данных тензора. По умолчанию None.
            use_random_init (bool, optional): Флаг случайной инициализации. По умолчанию False.
            random_scale (float, optional): Масштаб случайной инициализации. По умолчанию 0.1.
        """
        pass

    def get_hidden(self) -> Optional[Any]:
        """
        Возвращает текущее скрытое состояние (для совместимости, всегда None в MLPPolicy).

        Returns:
            Optional[Any]: Текущее скрытое состояние. Для MLPPolicy всегда None.
        """
        return None


# --- LSTM политика для SAC ---
class LSTMPolicy(BasePolicy):
    """
    LSTM-политика для SAC с поддержкой layer norm, dropout, AMP, random init hidden, автоматическим выбором устройства/типа.
    """

    @log_method_call()
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        activation: str = "tanh",  
        dropout_rate: float = 0.0,
        bidirectional: bool = False,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        use_layer_norm: bool = False,
        use_amp: bool = False,
        custom_init=None,
    ):
        """
        Инициализирует LSTM политику.

        Args:
            input_dim (int): Размерность входных данных.
            action_dim (int): Размерность действий.
            hidden_dim (int): Размерность скрытого слоя LSTM.
            num_layers (int, optional): Число слоёв LSTM. По умолчанию 1.
            activation (str, optional): Тип активационной функции. По умолчанию "tanh".
            dropout_rate (float, optional): Вероятность dropout. По умолчанию 0.0.
            bidirectional (bool, optional): Двунаправленный LSTM? По умолчанию False.
            log_std_min (float, optional): Минимальное значение логарифма стандартного отклонения. По умолчанию -20.0.
            log_std_max (float, optional): Максимальное значение логарифма стандартного отклонения. По умолчанию 2.0.
            use_layer_norm (bool, optional): Флаг использования нормализации слоя. По умолчанию False.
            use_amp (bool, optional): Флаг использования автоматического смешанного прецизионного обучения. По умолчанию False.
            custom_init (Optional[Callable], optional): Кастомная инициализация. По умолчанию None.
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.action_dim = action_dim
        self.enable_amp = use_amp
        self.num_directions = 2 if bidirectional else 1
        lstm_out_dim = hidden_dim * self.num_directions

        logger.debug(
            f"Создание LSTMPolicy: input={input_dim}, action={action_dim}, hidden={hidden_dim}, layers={num_layers}, activation={activation}, bidir={bidirectional}, ln={use_layer_norm}, dropout={dropout_rate}, amp={use_amp}"
        )

        # Слой LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  
            dropout=dropout_rate if num_layers > 1 else 0, 
            bidirectional=bidirectional,
        )
        logger.debug(f"Создан LSTM: {self.lstm}")

        # Дополнительные слои после LSTM 
        post_lstm_layers = []
        if use_layer_norm:
            post_lstm_layers.append(nn.LayerNorm(lstm_out_dim))
            logger.debug("Добавлен LayerNorm после LSTM")
        if activation:  # Активация может быть не нужна
            try:
                post_lstm_layers.append(_create_activation(activation))
            except ValueError:
                logger.warning(
                    f"Не удалось создать активацию '{activation}' после LSTM, пропускаем."
                )
        if dropout_rate > 0:
            post_lstm_layers.append(nn.Dropout(p=dropout_rate))
            logger.debug(f"  Добавлен Dropout (p={dropout_rate}) после LSTM")

        # Если есть доп. слои, создаем Sequential
        self.post_lstm = (
            nn.Sequential(*post_lstm_layers) if post_lstm_layers else nn.Identity()
        )
        logger.debug(f"Создан post_lstm модуль: {self.post_lstm}")

        # Головы для предсказания параметров распределения
        self.mean_head = nn.Linear(lstm_out_dim, action_dim)
        self.log_std_head = nn.Linear(lstm_out_dim, action_dim)

        self._init_weights(custom_init)

        # Инициализация скрытого состояния (будет создано при первом вызове или сбросе)
        self.hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        logger.info(f"LSTMPolicy успешно инициализирована.")

    def _init_weights(self, custom_init=None):
        """
        Инициализирует веса сети с использованием Xavier/Orthogonal инициализации.

        Args:
            custom_init (Optional[Callable], optional): Кастомная функция инициализации весов.
                Если передана, используется вместо стандартной. По умолчанию None.

        Raises:
            Может выбросить исключение, если кастомная инициализация некорректна.
        """
        logger.debug("Инициализация весов LSTMPolicy...")
        if custom_init:
            custom_init(self)
            return
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:  # Входные веса
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:  # Скрытые веса
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Инициализация forget gate bias в 1.0 (или другое значение) может улучшить обучение
                # bias состоит из [bias_i, bias_f, bias_g, bias_o]
                n = param.size(0)
                start, end = n // 4, n // 2
                nn.init.constant_(param[start:end], 1.0)
                logger.debug(f"Forget gate bias LSTM инициализирован значением 1.0")

        for m in self.post_lstm.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        if self.mean_head.bias is not None:
            nn.init.zeros_(self.mean_head.bias)

        nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.01)
        if self.log_std_head.bias is not None:
            nn.init.zeros_(self.log_std_head.bias)

        logger.debug("Веса LSTMPolicy инициализированы.")

    @log_method_call()
    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_random_init: bool = False,
        random_scale: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Создает новое скрытое состояние LSTM.

        Args:
            batch_size (int, optional): Размер батча. По умолчанию 1.
            device (Optional[torch.device], optional): Устройство хранения тензора. По умолчанию None.
            dtype (Optional[torch.dtype], optional): Тип данных тензора. По умолчанию None.
            use_random_init (bool, optional): Флаг случайной инициализации. По умолчанию False.
            random_scale (float, optional): Масштаб случайной инициализации. По умолчанию 0.1.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Новое скрытое состояние (h, c).
        """
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
                logger.warning("Сеть без параметров, hidden на CPU.")

        if use_random_init:
            # Случайная инициализация для потенциально лучшей сходимости
            h_0 = (
                torch.randn(
                    self.num_layers * self.num_directions,
                    batch_size,
                    self.hidden_dim,
                    device=device,
                )
                * random_scale
            )
            c_0 = (
                torch.randn(
                    self.num_layers * self.num_directions,
                    batch_size,
                    self.hidden_dim,
                    device=device,
                )
                * random_scale
            )
            logger.debug(
                f"Скрытое состояние LSTM инициализировано случайными значениями (scale={random_scale})"
            )
        else:
            # Стандартная инициализация нулями
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_dim,
                device=device,
            )
            c_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_dim,
                device=device,
            )

        log_tensor_info(
            logger, f"LSTMPolicy скрытое состояние h сброшено (batch={batch_size})", h_0
        )
        log_tensor_info(
            logger, f"LSTMPolicy скрытое состояние c сброшено (batch={batch_size})", c_0
        )
        
        return (h_0, c_0)

    def _prepare_obs(self, obs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Подготавливает входные наблюдения для работы с LSTM.

        Args:
            obs (Union[torch.Tensor, np.ndarray]): Входное наблюдение.

        Returns:
            torch.Tensor: Преобразованное наблюдение, готовое к использованию в модели.
        """
        if not isinstance(obs, torch.Tensor):
            target_device = next(self.parameters()).device
            obs = torch.as_tensor(obs, dtype=torch.float32, device=target_device)
            logger.warning(
                f"Вход obs был преобразован в тензор на устройстве {target_device}"
            )

        # LSTM ожидает [batch_size, seq_len, input_dim]
        if obs.dim() == 1:  # [input_dim] -> [1, 1, input_dim]
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif (
            obs.dim() == 2
        ):  # [batch_size, input_dim] -> [batch_size, 1, input_dim] (считаем seq_len=1)
            obs = obs.unsqueeze(1)
        elif obs.dim() != 3:
            raise ValueError(
                f"Неожиданная размерность obs: {obs.shape}. Ожидалась 1, 2 или 3."
            )

        return obs

    def _prepare_hidden(
        self,
        obs_batch_size: int,
        device: torch.device,
        hidden_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Подготавливает скрытое состояние LSTM.
        
        Args:
            obs_batch_size (int): Размер батча входных данных
            device (torch.device): Устройство для размещения тензоров
            hidden_input (Optional[Tuple[torch.Tensor, torch.Tensor]]): Внешнее скрытое состояние
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Подготовленное скрытое состояние (h, c)
        """
        # Если передано явное скрытое состояние, используем его
        if hidden_input is not None:
            h, c = hidden_input
            
            # Проверяем устройство
            if h.device != device:
                return (
                    h.to(device=device, non_blocking=True),
                    c.to(device=device, non_blocking=True)
                )
            
            # Проверяем размер батча
            current_batch_size = h.size(1)
            if current_batch_size == obs_batch_size:
                return hidden_input  
                
            # Создаем новое скрытое состояние с нужным размером батча
            h_new = torch.zeros(
                self.num_layers * self.num_directions,
                obs_batch_size, 
                self.hidden_dim, 
                device=device,
                dtype=h.dtype
            )
            c_new = torch.zeros(
                self.num_layers * self.num_directions,
                obs_batch_size, 
                self.hidden_dim, 
                device=device,
                dtype=c.dtype
            )
            
            # Инициализируем новое состояние
            return (h_new, c_new)
            
        # Если скрытое состояние не передано, создаем новое
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            obs_batch_size,
            self.hidden_dim,
            device=device,
        )
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            obs_batch_size,
            self.hidden_dim,
            device=device,
        )
        
        return (h_0, c_0)

    # Не используем @torch.jit.export здесь, т.к. он может конфликтовать с управлением hidden состоянием
    def forward(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Выполняет прямой проход через LSTM и головы для получения параметров распределения.

        Args:
            obs (Union[torch.Tensor, np.ndarray]): Наблюдение. Может быть тензором или массивом NumPy.
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Скрытое состояние LSTM.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:\n
                - mean: Предсказанное среднее значение действий.\n
                - log_std: Предсказанный логарифм стандартного отклонения.\n
                - next_hidden_state: Обновлённое скрытое состояние LSTM (h_n, c_n).\n
        """
        log_tensor_info(logger, "LSTMPolicy.forward вход obs", obs)
        if hidden:
            log_tensor_info(logger, "LSTMPolicy.forward вход hidden[0]", hidden[0])

        x = self._prepare_obs(obs)
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device

        # Подготавливаем скрытое состояние для текущего вызова
        hidden_state = self._prepare_hidden(batch_size, device, hidden)

        device_type = "cuda" if device.type == "cuda" else "cpu"
        amp_enabled = (
            self.enable_amp and device_type == "cuda" and torch.cuda.is_available()
        )

        with autocast(device_type=device_type, enabled=amp_enabled):
            # Прямой проход через LSTM
            # lstm_out: [batch_size, seq_len, num_directions * hidden_dim]
            # next_hidden: Tuple([num_layers * num_dir, batch, hidden], [num_layers * num_dir, batch, hidden])
            lstm_out, next_hidden_state = self.lstm(x, hidden_state)
            log_tensor_info(logger, "LSTMPolicy.forward выход LSTM lstm_out", lstm_out)

            # Если работаем с последовательностью (seq_len > 1), берем выход каждого шага.
            # Если работаем с батчем одиночных шагов (seq_len == 1), берем только его.
            # LSTM возвращает все выходы, post_lstm ожидает [batch * seq, features] или [batch, features]

            if seq_len > 1:
                # Обработка последовательности: применяем post_lstm к каждому шагу
                lstm_out_reshaped = lstm_out.contiguous().view(batch_size * seq_len, -1)
                features = self.post_lstm(lstm_out_reshaped)
                features = features.view(
                    batch_size, seq_len, -1
                )  # Возвращаем форму [batch, seq, feature]
            else:
                # Обработка одиночного шага (или батча одиночных шагов)
                features = self.post_lstm(
                    lstm_out.squeeze(1)
                )  # Убираем seq_len=1 -> [batch, feature]

            log_tensor_info(
                logger, "LSTMPolicy.forward выход post_lstm features", features
            )

            # Получаем параметры распределения
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        log_tensor_info(logger, "LSTMPolicy.forward выход mean", mean)
        log_tensor_info(logger, "LSTMPolicy.forward выход log_std", log_std)
        return mean, log_std, next_hidden_state

    @log_method_call()
    def sample(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Сэмплирует действие из политики LSTM.

        Args:
            obs (Union[torch.Tensor, np.ndarray]): Наблюдение.
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Скрытое состояние.
            deterministic (bool, optional): Флаг детерминированного выбора действия. По умолчанию False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        # Получаем параметры распределения и следующее скрытое состояние
        mean, log_std, next_hidden = self.forward(obs, hidden)

        if deterministic:
            action = torch.tanh(mean)
            log_tensor_info(logger, "LSTMPolicy.sample (deterministic) action", action)
            return action, None, mean, next_hidden

        std = log_std.exp()
        normal_dist = torch.distributions.Normal(mean, std)

        x_t = normal_dist.rsample()
        action = torch.tanh(x_t)

        log_prob = normal_dist.log_prob(x_t)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        log_tensor_info(logger, "LSTMPolicy.sample (stochastic) action", action)
        log_tensor_info(logger, "LSTMPolicy.sample log_prob", log_prob)

        return action, log_prob, mean, next_hidden

    def get_hidden(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Метод присутствует для совместимости с интерфейсом BasePolicy.

        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor]]: Всегда None в этой реализации.
        """
        return None


# --- Базовый класс для Q-сетей ---
class BaseQNet(nn.Module, ABC):
    """
    Абстрактный базовый класс для Q-сетей (оценщиков ценности действия).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, hidden: Optional[Any] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Выполняет прямой проход для оценки Q-значения пары (наблюдение, действие).
        Реализован в наследниках класса.
        """
        pass

    @abstractmethod
    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_random_init: bool = False,
        random_scale: float = 0.1,
    ) -> None:
        """
        Сбрасывает скрытое состояние (если применимо).
        Реализован в наследниках класса.
        """
        pass

    @abstractmethod
    def get_hidden(self) -> Optional[Any]:
        """
        Возвращает текущее скрытое состояние (если применимо).
        Реализован в наследниках класса.
        """
        pass


# --- MLP Q-сеть ---
class MLPQNet(BaseQNet):
    """
    MLP-модель Q-функции для SAC с опциональными layer norm, dropout, AMP, кастомной инициализацией.
    """

    @log_method_call()
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_layer_norm: bool = False,
        use_amp: bool = False,
        concat_layer: int = 0,
        custom_init=None,
    ):
        """
        Инициализирует MLP модель Q-функции.

        Args:
            obs_dim (int): Размерность наблюдения.
            action_dim (int): Размерность действия.
            hidden_dims (List[int]): Список размерностей скрытых слоев.
            activation (str, optional): Тип активационной функции. По умолчанию "relu".
            dropout_rate (float, optional): Вероятность dropout. По умолчанию 0.0.
            use_layer_norm (bool, optional): Флаг использования нормализации слоя. По умолчанию False.
            use_amp (bool, optional): Флаг использования автоматического смешанного прецизионного обучения. По умолчанию False.
            concat_layer (int, optional): Индекс слоя, после которого происходит конкатенация с действием. По умолчанию 0.
            custom_init (Optional[Callable], optional): Кастомная функция инициализации весов. По умолчанию None.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.concat_layer = min(concat_layer, len(hidden_dims))
        self.enable_amp = use_amp

        # Вычисляем размерности для слоёв до и после слоя конкатенации с action
        pre_layers = []
        post_layers = []

        # Строим слои до конкатенации с действием
        prev_dim = obs_dim
        for i in range(self.concat_layer):
            h_dim = hidden_dims[i]
            pre_layers.append(nn.Linear(prev_dim, h_dim))
            if use_layer_norm:
                pre_layers.append(nn.LayerNorm(h_dim))
            pre_layers.append(_create_activation(activation))
            if dropout_rate > 0:
                pre_layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim

        # Размерность входа после конкатенации
        if self.concat_layer == 0:
            post_input_dim = obs_dim + action_dim
        else:
            post_input_dim = hidden_dims[self.concat_layer - 1] + action_dim

        # Строим слои после конкатенации
        prev_dim = post_input_dim
        for i in range(self.concat_layer, len(hidden_dims)):
            h_dim = hidden_dims[i]
            post_layers.append(nn.Linear(prev_dim, h_dim))
            if use_layer_norm:
                post_layers.append(nn.LayerNorm(h_dim))
            post_layers.append(_create_activation(activation))
            if dropout_rate > 0:
                post_layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim

        # Последний слой для получения одного значения Q
        post_layers.append(nn.Linear(prev_dim, 1))

        # Объединяем все в Sequential для удобства
        self.pre_layers = nn.Sequential(*pre_layers) if pre_layers else None
        self.post_layers = nn.Sequential(*post_layers)

        self._init_weights(custom_init)

        logger.info(f"MLPQNet успешно инициализирована. Слой конкатенации: {concat_layer}")

    def _init_weights(self, custom_init=None):
        """
        Инициализирует веса сети с использованием Xavier/Orthogonal инициализации.

        Args:
            custom_init (Optional[Callable], optional): Кастомная функция инициализации весов.
                Если передана, используется вместо стандартной. По умолчанию None.

        Raises:
            Может выбросить исключение, если кастомная инициализация некорректна.
        """
        logger.debug("Инициализация весов MLPQNet...")
        if custom_init:
            custom_init(self)
            return

        # Применяем Xavier к pre_layers
        if self.pre_layers:
            for m in self.pre_layers.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

        # Применяем Xavier к post_layers
        for i, m in enumerate(self.post_layers.modules()):
            if isinstance(m, nn.Linear):
                # Для последнего слоя используем меньший gain
                if i == len(list(self.post_layers.modules())) - 1:
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        logger.debug("Веса MLPQNet инициализированы.")

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход для вычисления значения Q(obs, action).

        Args:
            obs (torch.Tensor): Тензор наблюдения формы [batch_size, obs_dim].
            action (torch.Tensor): Тензор действия формы [batch_size, action_dim].

        Returns:
            torch.Tensor: Значение Q(obs, action) формы [batch_size, 1].
        """
        # Убираем лишнюю размерность, если есть
        if obs.dim() == 3 and obs.shape[0] == 1:
            obs = obs.squeeze(0)
        if action.dim() == 3 and action.shape[0] == 1:
            action = action.squeeze(0)
            
        # Добавляем batch dimension, если его нет
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Определяем тип устройства для autocast
        device_type = "cuda" if obs.device.type == "cuda" else "cpu"
        amp_enabled = (
            self.enable_amp and device_type == "cuda" and torch.cuda.is_available()
        )

        with autocast(device_type=device_type, enabled=amp_enabled):
            # Обработка наблюдения через pre_layers, если они существуют
            if self.pre_layers is not None:
                obs_features = self.pre_layers(obs)
                # Конкатенация с действием
                combined = torch.cat([obs_features, action], dim=-1)
            else:
                # Конкатенация напрямую, если нет pre_layers
                combined = torch.cat([obs, action], dim=-1)
                
            # Получение значения Q через post_layers
            q_value = self.post_layers(combined)

        return q_value

    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_random_init: bool = False,
        random_scale: float = 0.1,
    ) -> None:
        pass

    def get_hidden(self) -> Optional[Any]:
        return None


# --- LSTM Q-сеть ---
class LSTMQNet(BaseQNet):
    """
    Реализует LSTM Q-сеть, используемую в SAC для оценки Q-значений при наличии истории наблюдений и действий.
    """

    @log_method_call()
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        activation: str = "tanh",
        dropout_rate: float = 0.0,
        bidirectional: bool = False,
        use_layer_norm: bool = False,
        use_amp: bool = False,
        custom_init=None,
    ):
        """
        Конструктор LSTM Q-сети.

        Args:
            obs_dim (int): Размерность наблюдений.
            action_dim (int): Размерность действий.
            hidden_dim (int): Размерность скрытого слоя LSTM.
            num_layers (int, optional): Число слоёв LSTM. По умолчанию 1.
            activation (str, optional): Активационная функция. По умолчанию 'tanh'.
            dropout_rate (float, optional): Вероятность dropout. По умолчанию 0.0.
            bidirectional (bool, optional): Двунаправленный LSTM? По умолчанию False.
            use_layer_norm (bool, optional): Использовать layer norm? По умолчанию False.
            use_amp (bool, optional): Использовать AMP? По умолчанию False.
            custom_init (Optional[Callable], optional): Кастомная инициализация. По умолчанию None.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.enable_amp = use_amp
        self.num_directions = 2 if bidirectional else 1
        lstm_out_dim = hidden_dim * self.num_directions

        logger.debug(
            f"Создание LSTMQNet: obs={obs_dim}, action={action_dim}, hidden={hidden_dim}, layers={num_layers}, activation={activation}, bidir={bidirectional}, ln={use_layer_norm}, dropout={dropout_rate}, amp={use_amp}"
        )

        # LSTM обрабатывает только наблюдения
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        logger.debug(f"Создан LSTM: {self.lstm}")

        # MLP после LSTM для обработки выхода LSTM + действия
        post_lstm_input_dim = lstm_out_dim + action_dim
        post_layers = []
        if use_layer_norm:
            post_layers.append(nn.LayerNorm(post_lstm_input_dim))
            logger.debug(" Добавлен LayerNorm после LSTM+Action")
        if activation:
            try:
                post_layers.append(_create_activation(activation))
            except ValueError:
                logger.warning(
                    f"Не удалось создать активацию '{activation}' после LSTM+Action, пропускаем."
                )
        if dropout_rate > 0:
            post_layers.append(nn.Dropout(p=dropout_rate))
            logger.debug(f"  Добавлен Dropout (p={dropout_rate}) после LSTM+Action")

        # Если есть доп. слои, создаем Sequential
        self.post_lstm_processor = (
            nn.Sequential(*post_layers) if post_layers else nn.Identity()
        )
        logger.debug(f"Создан post_lstm_processor: {self.post_lstm_processor}")

        self.q_head = nn.Linear(post_lstm_input_dim, 1)
        self._init_weights(custom_init)
        self.hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        logger.info("LSTMQNet успешно инициализирована.")

    def _init_weights(self, custom_init=None):
        """
        Инициализирует веса сети с использованием Xavier/Orthogonal инициализации.

        Args:
            custom_init (Optional[Callable], optional): Кастомная функция инициализации весов.
                Если передана, используется вместо стандартной. По умолчанию None.

        Raises:
            Может выбросить исключение, если кастомная инициализация некорректна.
        """
        logger.debug("Инициализация весов LSTMQNet...")
        if custom_init:
            custom_init(self)
            return
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                start, end = n // 4, n // 2
                nn.init.constant_(param[start:end], 1.0)
                logger.debug("Forget gate bias LSTM инициализирован значением 1.0")

        for m in self.post_lstm_processor.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        nn.init.xavier_uniform_(self.q_head.weight, gain=1.0)
        if self.q_head.bias is not None:
            nn.init.zeros_(self.q_head.bias)
        logger.debug("Веса LSTMQNet инициализированы.")

    @log_method_call()
    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        use_random_init: bool = False,
        random_scale: float = 0.1,
    ) -> None:
        """
        Сбрасывает скрытое состояние LSTM.
        """
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
                logger.warning("Сеть без параметров, hidden на CPU.")

        if use_random_init:
            # Случайная инициализация для потенциально лучшей сходимости
            h_0 = (
                torch.randn(
                    self.num_layers * self.num_directions,
                    batch_size,
                    self.hidden_dim,
                    device=device,
                )
                * random_scale
            )
            c_0 = (
                torch.randn(
                    self.num_layers * self.num_directions,
                    batch_size,
                    self.hidden_dim,
                    device=device,
                )
                * random_scale
            )
            logger.debug(
                f"Скрытое состояние LSTMQNet инициализировано случайными значениями (scale={random_scale})"
            )
        else:
            # Стандартная инициализация нулями
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_dim,
                device=device,
            )
            c_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_dim,
                device=device,
            )

        self.hidden = (h_0, c_0)
        log_tensor_info(
            logger, f"LSTMQNet скрытое состояние h сброшено (batch={batch_size})", h_0
        )
        log_tensor_info(
            logger, f"LSTMQNet скрытое состояние c сброшено (batch={batch_size})", c_0
        )

    # Копируем вспомогательные методы из LSTMPolicy для DRY
    _prepare_obs = LSTMPolicy._prepare_obs
    _prepare_hidden = LSTMPolicy._prepare_hidden

    def _init_hidden(self, batch_size, device, dtype):
        """
        Вспомогательный метод для инициализации начального скрытого состояния LSTM.

        Args:
            batch_size (int): Размер батча.
            device (torch.device): Устройство хранения тензора (CPU или GPU).
            dtype (torch.dtype): Тип данных тензора.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Начальные значения h_0 и c_0 (состояние LSTM).
        """
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        return (h_0, c_0)

    def forward(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Выполняет прямой проход через LSTM и головы для получения Q-значения.

        Args:
            obs (Union[torch.Tensor, np.ndarray]): Наблюдение.
            action (Union[torch.Tensor, np.ndarray]): Действие.
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Скрытое состояние. По умолчанию None.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:\n
                - q_value: Предсказанное значение Q-функции.\n
                - next_hidden_state: Обновлённое скрытое состояние LSTM.\n
        """
        log_tensor_info(logger, "LSTMQNet.forward вход obs", obs)
        log_tensor_info(logger, "LSTMQNet.forward вход action", action)
        
        # Подготавливаем наблюдение в формат для LSTM
        obs_tensor = self._prepare_obs(obs)
        batch_size = obs_tensor.size(0)
        seq_len = obs_tensor.size(1)
        device = obs_tensor.device
        
        # Преобразуем действие в тензор, если оно не является тензором
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=device)
        elif action.device != device:
            action = action.to(device)
            
        # Если actions имеет dim=1, добавляем batch dimension
        if action.dim() == 1:
            action = action.unsqueeze(0)  

        # Подготавливаем скрытое состояние
        hidden_state = self._prepare_hidden(batch_size, device, hidden)

        device_type = device.type
        amp_enabled = (
            self.enable_amp and device_type == "cuda" and torch.cuda.is_available()
        )

        with autocast(device_type=device_type, enabled=amp_enabled):
            # Прямой проход через LSTM (только для наблюдений)
            lstm_out, next_hidden_state = self.lstm(obs_tensor, hidden_state)
            log_tensor_info(logger, "LSTMQNet.forward выход LSTM lstm_out", lstm_out)

            if seq_len > 1:
                # Обрабатываем последовательность
                lstm_out = lstm_out.contiguous().view(batch_size * seq_len, -1)
                # Приводим action к [batch*seq, action_dim]
                action_expanded = action.view(batch_size * seq_len, -1)
                # Конкатенируем выход LSTM и действие
                combined = torch.cat([lstm_out, action_expanded], dim=-1)
                # Применяем пост-обработку
                features = self.post_lstm_processor(combined)
                # Получаем Q-значение
                q_value = self.q_head(features)
                # Восстанавливаем форму [batch, seq, 1]
                q_value = q_value.view(batch_size, seq_len, 1)
            else:
                # Обрабатываем одиночный шаг
                lstm_out = lstm_out.squeeze(1)  # [batch, 1, hidden] -> [batch, hidden]
                # Конкатенируем выход LSTM и действие
                combined = torch.cat([lstm_out, action], dim=-1)
                # Применяем пост-обработку
                features = self.post_lstm_processor(combined)
                # Получаем Q-значение
                q_value = self.q_head(features)

            log_tensor_info(logger, "LSTMQNet.forward выход q_value", q_value)

        # Обновляем внутреннее состояние только если hidden не был передан явно
        if hidden is None:
            self.hidden = next_hidden_state

        return q_value, next_hidden_state

    def get_hidden(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Возвращает текущее внутреннее скрытое состояние."""
        return self.hidden

    def q1_value(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Возвращает значение и следующее скрытое состояние только первой Q-сети Q1.

        Args:
            obs (Union[torch.Tensor, np.ndarray]): Наблюдение.
            action (Union[torch.Tensor, np.ndarray]): Действие.
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Скрытое состояние. По умолчанию None.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - q1_value: Значение Q1.
                - new_hidden: Обновлённое скрытое состояние первой сети.
        """
        # AMP обрабатывается внутри self.q1.forward
        return self.q1(obs, action, hidden)


# --- Двойные Q-сети для SAC (устраняют переоценку) ---
class DualMLPQNet(nn.Module):
    """
    Двойная MLP Q-сеть для SAC. Содержит две отдельные Q-функции для уменьшения переоценивания.
    """

    @log_method_call()
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_layer_norm: bool = False,
        use_amp: bool = False,
    ):
        """
        Инициализирует двойную MLP Q-сеть.

        Args:
            obs_dim (int): Размерность наблюдения.
            action_dim (int): Размерность действия.
            hidden_dims (List[int]): Список размерностей скрытых слоев.
            activation (str, optional): Тип активационной функции. По умолчанию "relu".
            dropout_rate (float, optional): Вероятность dropout. По умолчанию 0.0.
            use_layer_norm (bool, optional): Флаг использования нормализации слоя. По умолчанию False.
            use_amp (bool, optional): Флаг использования автоматического смешанного прецизионного обучения. По умолчанию False.
        """
        super().__init__()
        logger.debug("Создание DualMLPQNet...")
        self.enable_amp = use_amp
        self.q1 = MLPQNet(
            obs_dim,
            action_dim,
            hidden_dims,
            activation,
            dropout_rate,
            use_layer_norm,
            use_amp,
        )
        self.q2 = MLPQNet(
            obs_dim,
            action_dim,
            hidden_dims,
            activation,
            dropout_rate,
            use_layer_norm,
            use_amp,
        )
        logger.info("DualMLPQNet успешно инициализирована.")

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисляет Q-значения для обеих сетей: Q1(s, a) и Q2(s, a).

        Args:
            obs (torch.Tensor): Наблюдение.
            action (torch.Tensor): Действие.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Значения Q1 и Q2.
        """
        # Используем отдельные Q-сети, каждая со своим управлением AMP
        q1_value = self.q1(obs, action)
        q2_value = self.q2(obs, action)
        return q1_value, q2_value

    def q1_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Возвращает значение только первой Q-сети Q1(s, a).
        Удобно для вычисления потерь политики в SAC.

        Args:
            obs (torch.Tensor): Наблюдение.
            action (torch.Tensor): Действие.

        Returns:
            torch.Tensor: Значение Q1.
        """
        # AMP будет обработан внутри self.q1.forward
        return self.q1(obs, action)


class DualLSTMQNet(nn.Module):
    """
    Обертка для двух независимых LSTM Q-сетей (Q1 и Q2).
    Управляет скрытыми состояниями обеих сетей.
    """

    @log_method_call()
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        activation: str = "tanh",
        dropout_rate: float = 0.0,
        bidirectional: bool = False,
        use_layer_norm: bool = False,
        use_amp: bool = False,
    ):
        """
        Инициализирует двойную LSTM Q-сеть.

        Args:
            obs_dim (int): Размерность наблюдения.
            action_dim (int): Размерность действия.
            hidden_dim (int): Размерность скрытого слоя LSTM.
            num_layers (int, optional): Число слоёв LSTM. По умолчанию 1.
            activation (str, optional): Тип активационной функции. По умолчанию "tanh".
            dropout_rate (float, optional): Вероятность dropout. По умолчанию 0.0.
            bidirectional (bool, optional): Флаг двунаправленного LSTM. По умолчанию False.
            use_layer_norm (bool, optional): Флаг использования нормализации слоя. По умолчанию False.
            use_amp (bool, optional): Флаг использования автоматического смешанного прецизионного обучения. По умолчанию False.
        """
        super().__init__()
        logger.debug("Создание DualLSTMQNet...")
        self.enable_amp = use_amp
        self.q1 = LSTMQNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            use_layer_norm=use_layer_norm,
            use_amp=use_amp,
        )
        self.q2 = LSTMQNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            use_layer_norm=use_layer_norm,
            use_amp=use_amp,
        )
        logger.info("DualLSTMQNet успешно инициализирована.")

    @log_method_call()
    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        use_random_init: bool = False,
        random_scale: float = 0.1,
    ) -> None:
        """
        Сбрасывает скрытые состояния обеих LSTM сетей.

        Args:
            batch_size (int, optional): Размер батча. По умолчанию 1.
            device (Optional[torch.device], optional): Устройство хранения тензора. По умолчанию None.
            use_random_init (bool, optional): Флаг случайной инициализации. По умолчанию False.
            random_scale (float, optional): Масштаб случайной инициализации. По умолчанию 0.1.
        """
        # Сбрасываем скрытые состояния обеих сетей
        self.q1.reset_hidden(batch_size, device, use_random_init, random_scale)
        self.q2.reset_hidden(batch_size, device, use_random_init, random_scale)
        logger.debug(
            f"Сброшены скрытые состояния обеих Q-сетей, batch_size={batch_size}, use_random_init={use_random_init}"
        )

    def forward(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        hidden1: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden2: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Выполняет прямой проход для обеих LSTM Q-сетей.

        Если hidden1/hidden2 не переданы или их размер не совпадает с batch_size, инициализируются автоматически.

        Args:
            obs (Union[torch.Tensor, np.ndarray]): Наблюдение.
            action (Union[torch.Tensor, np.ndarray]): Действие.
            hidden1 (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Скрытое состояние первой сети. По умолчанию None.
            hidden2 (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Скрытое состояние второй сети. По умолчанию None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]: \n
                - q1_value: Значение Q1.\n
                - q2_value: Значение Q2.\n
                - new_hidden1: Обновлённое скрытое состояние первой сети.\n
                - new_hidden2: Обновлённое скрытое состояние второй сети.\n
        """
        # Подготавливаем тензоры для каждой сети
        x = self.q1._prepare_obs(obs)
        batch_size = x.shape[0]
        device = x.device

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=device)
        elif action.device != device:
            action = action.to(device)

        # Подготавливаем скрытые состояния
        if hidden1 is None:
            hidden1 = self.q1._prepare_hidden(batch_size, device, self.q1.hidden)
        if hidden2 is None:
            hidden2 = self.q2._prepare_hidden(batch_size, device, self.q2.hidden)

        # Определяем тип устройства для AMP
        device_type = device.type
        amp_enabled = (
            self.enable_amp and device_type == "cuda" and torch.cuda.is_available()
        )

        with autocast(device_type=device_type, enabled=amp_enabled):
            # Вычисляем Q1 и Q2 значения и обновленные скрытые состояния
            q1_value, new_hidden1 = self.q1(obs, action, hidden1)
            q2_value, new_hidden2 = self.q2(obs, action, hidden2)

        # Логируем результаты
        log_tensor_info(logger, "DualLSTMQNet.forward выход q1", q1_value)
        log_tensor_info(logger, "DualLSTMQNet.forward выход q2", q2_value)
        
        return q1_value, q2_value, new_hidden1, new_hidden2

    def q1_value(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Возвращает значение и следующее скрытое состояние только первой Q-сети Q1.

        Args:
            obs (Union[torch.Tensor, np.ndarray]): Наблюдение.
            action (Union[torch.Tensor, np.ndarray]): Действие.
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Скрытое состояние. По умолчанию None.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - q1_value: Значение Q1.
                - new_hidden: Обновлённое скрытое состояние первой сети.
        """
        # AMP обрабатывается внутри self.q1.forward
        return self.q1(obs, action, hidden)

    def get_hidden(
        self,
    ) -> Tuple[
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Возвращает текущие внутренние скрытые состояния обеих Q-сетей.

        Returns:
            Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                - Скрытое состояние первой Q-сети.
                - Скрытое состояние второй Q-сети.
        """
        return self.q1.get_hidden(), self.q2.get_hidden()
