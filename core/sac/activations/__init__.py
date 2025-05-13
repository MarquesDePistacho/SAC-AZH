import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from typing import Optional, Union
import weakref
from torch.amp import autocast

from core.logging.logger import get_logger
from core.utils.device_utils import device_manager

logger = get_logger("activations")

class ActivationCache:
    """
    Кэш для хранения выходных значений функций активации.
    Использует weakref.WeakKeyDictionary для автоматического удаления записей,
    когда входной тензор больше не используется.
    """
    def __init__(self, activation_name: str):
        """
        Конструктор класса ActivationCache.

        Args:
            activation_name (str): Имя активации, для которой создаётся кэш.
        """
        self.activation_name = activation_name
        self.cache = weakref.WeakKeyDictionary()
        self.enabled = True
        logger.debug(f"Инициализирован кэш для активации '{activation_name}'")
    
    def get_cached_output(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Пытается получить кэшированный результат для входного тензора.

        Args:
            x (torch.Tensor): Входной тензор.

        Returns:
            Optional[torch.Tensor]: Кэшированное значение, если оно существует и соответствует условиям.
        """
        if not self.enabled:
            return None
            
        if x in self.cache:
            cached_output = self.cache[x]
            if cached_output.device == x.device:
                logger.debug(f"Результат для активации '{self.activation_name}' найден в кэше.")
                return cached_output
            else:
                logger.debug(f"Запись в кэше для '{self.activation_name}' на другом устройстве ({cached_output.device}), игнорируем.")
                del self.cache[x]

        return None
    
    def update_cache(self, x: torch.Tensor, output: torch.Tensor) -> None:
        """
        Добавляет или обновляет запись в кэше.

        Args:
            x (torch.Tensor): Входной тензор.
            output (torch.Tensor): Выходной тензор для сохранения в кэше.
        """
        if not self.enabled:
            return
            
        self.cache[x] = output
        logger.debug(f"Кэш активации '{self.activation_name}' обновлен.")
    
    def clear(self) -> None:
        """
        Очищает кэш.
        """
        self.cache.clear()
        logger.debug(f"Кэш активации '{self.activation_name}' очищен.")
    
    def to_device(self, device: Union[str, torch.device]) -> None:
        """
        Перемещает все кэшированные тензоры на указанное устройство.
        Предполагается, что этот метод очищает кэш, так как перемещение устройства может повлиять на актуальность данных.

        Note:
            Внимание: Это может быть неэффективно, если кэш большой.
        
        Args:
            device (Union[str, torch.device]): Целевое устройство для перемещения.
        """
        target_device = device_manager.get_device(device)
        logger.debug(f"Попытка перемещения кэша '{self.activation_name}' на устройство {target_device} (текущий размер: {len(self.cache)})...")

        self.clear()
        logger.debug(f"Кэш '{self.activation_name}' очищен при перемещении на устройство {target_device}.")

def _add_cache_methods(cls):
    """
    Декоратор: Добавляет методы управления кэшем (`clear_cache`, `enable_cache`, `disable_cache`) в класс активации.
    
    Args:
        cls: Декорируемый класс.
        
    Returns:
        cls: Класс с добавленными методами.
    """
    def clear_cache(self):
        """Очистка кэша активации."""
        if hasattr(self, '_cache') and isinstance(self._cache, ActivationCache):
            self._cache.clear()
        return self
        
    def enable_cache(self):
        """Включение кэширования для этой активации."""
        if hasattr(self, '_cache') and isinstance(self._cache, ActivationCache):
            self._cache.enabled = True
            logger.debug(f"Кэширование включено для '{self._cache.activation_name}'")
        return self

    def disable_cache(self):
        """Выключение кэширования для этой активации."""
        if hasattr(self, '_cache') and isinstance(self._cache, ActivationCache):
            self._cache.enabled = False
            logger.debug(f"Кэширование выключено для '{self._cache.activation_name}'")
        return self

    cls.clear_cache = clear_cache
    cls.enable_cache = enable_cache
    cls.disable_cache = disable_cache
    return cls

def _modify_to_method(cls):
    """
    Декоратор: Модифицирует метод `to()` модуля nn.Module, чтобы он очищал кэш при перемещении.

    Args:
        cls: Декорируемый класс.

    Returns:
        cls: Класс с изменённым методом `to()`.
    """
    if not hasattr(cls, 'to'):
        return cls

    orig_to = cls.to
    
    @functools.wraps(orig_to)
    def to_with_cache_clear(self, *args, **kwargs):
        """
        Перемещает модуль и очищает кэш активации.
        """
        result = orig_to(self, *args, **kwargs)

        if hasattr(self, '_cache') and isinstance(self._cache, ActivationCache):
            logger.debug(f"Очистка кэша '{self._cache.activation_name}' после вызова to().")
            self._cache.clear()

        return result
        
    cls.to = to_with_cache_clear
    return cls

def _modify_train_method(cls):
    """
    Декоратор: Модифицирует метод `train()` модуля nn.Module, чтобы он очищал кэш при переходе в режим обучения.

    Args:
        cls: Декорируемый класс.

    Returns:
        cls: Класс с изменённым методом `train()`.
    """
    if not hasattr(cls, 'train'):
        return cls

    orig_train = cls.train
    
    @functools.wraps(orig_train)
    def train_with_cache_clear(self, mode: bool = True):
        """
        Устанавливает режим обучения/оценки и очищает кэш при включении режима обучения.
        """
        result = orig_train(self, mode=mode)

        if mode and hasattr(self, '_cache') and isinstance(self._cache, ActivationCache):
            logger.debug(f"Очистка кэша '{self._cache.activation_name}' при переходе в режим обучения (train).")
            self._cache.clear()

        return result
        
    cls.train = train_with_cache_clear
    return cls

def _modify_forward_method(cls):
    """
    Декоратор: Модифицирует метод `forward()` для использования кэша в режиме оценки.

    Args:
        cls: Декорируемый класс.

    Returns:
        cls: Класс с изменённым методом `forward()`.
    """
    if not hasattr(cls, 'forward'):
        raise TypeError(f"Класс {cls.__name__} должен иметь метод forward для кэширования.")

    orig_forward = cls.forward
    
    @functools.wraps(orig_forward)
    def cached_forward(self, x: torch.Tensor):
        """
        Версия `forward`, использующая кэш в режиме оценки (`self.training == False`).
        """
        use_cache = hasattr(self, '_cache') and isinstance(self._cache, ActivationCache) and not self.training

        if use_cache:
            cached_result = self._cache.get_cached_output(x)
            if cached_result is not None:
                return cached_result

            device_type = x.device.type
            amp_enabled = torch.cuda.is_available() and device_type == "cuda"
            with autocast(device_type=device_type, enabled=amp_enabled):
                result = orig_forward(self, x)

            self._cache.update_cache(x, result)
            return result
        else:
            device_type = x.device.type
            amp_enabled = torch.cuda.is_available() and device_type == "cuda"
            with autocast(device_type=device_type, enabled=amp_enabled):
                return orig_forward(self, x)
        
    cls.forward = cached_forward
    return cls

def cached_activation(cls):
    """
    Декоратор класса для добавления функциональности кэширования к функции активации nn.Module.
    Модифицирует __init__, forward, to, train и добавляет методы управления кэшем.

    Args:
        cls: Декорируемый класс.

    Returns:
        cls: Класс с поддержкой кэширования.
    """
    logger.debug(f"Применение декоратора @cached_activation к классу {cls.__name__}")
    if not issubclass(cls, nn.Module):
        raise TypeError("Декоратор @cached_activation может применяться только к подклассам nn.Module.")

    orig_init = cls.__init__
    
    @functools.wraps(orig_init)
    def init_with_cache(self, *args, **kwargs):
        """
        Модифицированный __init__, который создает экземпляр ActivationCache.
        """
        orig_init(self, *args, **kwargs)
        self._cache = ActivationCache(cls.__name__)

    cls.__init__ = init_with_cache
    cls = _modify_forward_method(cls)
    cls = _modify_to_method(cls)
    cls = _modify_train_method(cls)
    cls = _add_cache_methods(cls)
    
    return cls

class ActivationFunction(nn.Module):
    """
    Базовый класс для модулей функций активации.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Метод forward должен быть реализован в подклассе.")

@cached_activation
class ReLUActivation(ActivationFunction):
    """
    Реализация активации ReLU (Rectified Linear Unit) с кэшированием.

    Args:
        inplace (bool): Если True, выполняет операцию на месте.
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        logger.debug(f"Инициализирована ReLUActivation(inplace={inplace})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x, inplace=self.inplace)

@cached_activation
class LeakyReLUActivation(ActivationFunction):
    """
    Реализация активации Leaky ReLU с кэшированием.

    Args:
        negative_slope (float): Угловой коэффициент для отрицательной части функции.
        inplace (bool): Если True, выполняет операцию на месте.
    """
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
        logger.debug(f"Инициализирована LeakyReLUActivation(negative_slope={negative_slope}, inplace={inplace})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=self.inplace)

@cached_activation
class ELUActivation(ActivationFunction):
    """
    Реализация активации ELU (Exponential Linear Unit) с кэшированием.

    Args:
        alpha (float): Коэффициент для отрицательной части функции.
        inplace (bool): Если True, выполняет операцию на месте.
    """
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace
        logger.debug(f"Инициализирована ELUActivation(alpha={alpha}, inplace={inplace})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha, inplace=self.inplace)

@cached_activation
class SELUActivation(ActivationFunction):
    """
    Реализация активации SELU (Scaled Exponential Linear Unit) с кэшированием.

    Args:
        inplace (bool): Если True, выполняет операцию на месте.
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        logger.debug(f"Инициализирована SELUActivation(inplace={inplace})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.selu(x, inplace=self.inplace)

@cached_activation
class TanhActivation(ActivationFunction):
    """Активация Tanh (гиперболический тангенс) с кэшированием."""
    def __init__(self):
        super().__init__()
        logger.debug("Инициализирована TanhActivation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

@cached_activation
class SigmoidActivation(ActivationFunction):
    """Активация Sigmoid с кэшированием."""
    def __init__(self):
        super().__init__()
        logger.debug("Инициализирована SigmoidActivation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

@cached_activation
class GeluActivation(ActivationFunction):
    """Активация GELU (Gaussian Error Linear Unit) с кэшированием."""
    def __init__(self):
        super().__init__()
        logger.debug("Инициализирована GeluActivation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)

@cached_activation
class SwishActivation(ActivationFunction):
    """Активация Swish (x * sigmoid(x)) с кэшированием (здесь beta=1)."""
    def __init__(self):
        super().__init__()
        logger.debug("Инициализирована SwishActivation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

@cached_activation
class MishActivation(ActivationFunction):
    """Активация Mish (x * tanh(softplus(x))) с кэшированием."""
    def __init__(self):
        super().__init__()
        logger.debug("Инициализирована MishActivation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))

@cached_activation
class FusedSiLUActivation(ActivationFunction):
    """
    Реализация активации SiLU (Swish) с использованием fused CUDA kernel при доступности.
    """
    def __init__(self):
        super().__init__()
        logger.debug("Инициализирована FusedSiLUActivation")
        self._has_fused_kernel = hasattr(torch.nn.functional, 'silu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._has_fused_kernel:
            return F.silu(x)  # Использует оптимизированный CUDA kernel в PyTorch >= 1.7
        else:
            return x * torch.sigmoid(x)  # Fallback для старых версий

@cached_activation
class FastGELUActivation(ActivationFunction):
    """
    Быстрая аппроксимация GELU активации.
    """
    def __init__(self):
        super().__init__()
        logger.debug("Инициализирована FastGELUActivation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Быстрая аппроксимация GELU (используется во многих современных моделях)
        return x * torch.sigmoid(1.702 * x)

@cached_activation
class HardSwishActivation(ActivationFunction):
    """
    Hard-Swish активация, более быстрая аппроксимация Swish.
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        logger.debug(f"Инициализирована HardSwishActivation(inplace={inplace})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, 'hardswish'):
            return F.hardswish(x, inplace=self.inplace)  # PyTorch >= 1.6
        else:
            # Ручная реализация для обратной совместимости
            if self.inplace:
                return x.mul_(F.relu6(x + 3, inplace=True).div_(6))
            else:
                return x * F.relu6(x + 3, inplace=False) / 6

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
    "fused_silu": FusedSiLUActivation,
    "fast_gelu": FastGELUActivation,
    "hard_swish": HardSwishActivation,
}

def create_activation(activation_type: str, **kwargs) -> nn.Module:
    """
    Создает экземпляр функции активации по ее имени.
    """
    activation_type = activation_type.lower()
    activation_class = _activations.get(activation_type)

    if activation_class:
        try:
            return activation_class(**kwargs)
        except TypeError as e:
            logger.error(f"Ошибка при создании активации '{activation_type}' с параметрами {kwargs}: {e}")
            try:
                logger.warning(f"Попытка создать '{activation_type}' без параметров.")
                return activation_class()
            except TypeError:
                logger.error(f"Не удалось создать '{activation_type}' даже без параметров.")
                raise ValueError(f"Не удалось создать активацию '{activation_type}'. Проверьте параметры: {kwargs}")
    else:
        available = ", ".join(_activations.keys())
        logger.error(f"Неизвестный тип функции активации: '{activation_type}'. Доступные типы: {available}")
        raise ValueError(f"Неизвестный тип функции активации: '{activation_type}'. Доступные: {available}") 