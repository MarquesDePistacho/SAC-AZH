import torch
import torch.nn as nn
from torch.amp import GradScaler
from typing import Dict, Any, Optional, Union
import os
from functools import lru_cache

from core.logging.logger import get_logger, log_method_call
from core.utils.device_utils import device_manager
from core.sac.interfaces.component_interface import ComponentInterface

logger = get_logger("components")

# --- Plug-and-Play Registry для компонентов ---
COMPONENT_REGISTRY = {}


def register_component(name):
    def decorator(cls):
        COMPONENT_REGISTRY[name] = cls
        return cls

    return decorator


class SACComponent(ComponentInterface, nn.Module):
    """
    Базовый компонент SAC: устройство, AMP, сохранение, загрузка, параметры.

    Attributes:
        name (str): Имя компонента.
        _device (torch.device): Устройство, на котором работает компонент.
        enable_amp (bool): Включен ли автоматическое смешанное предCISION (AMP).
        scaler (GradScaler): Шкалер для AMP, используется при обучении на GPU.
    """

    def __init__(self, name: str, device: Optional[Union[str, torch.device]] = None):
        """
        Инициализация базового компонента.
        1. Настраивает устройство.\n
        2. Создаёт GradScaler для AMP.\n

        Args:
            name (str): Имя компонента.
            device (Optional[Union[str, torch.device]]): Целевое устройство (CPU или GPU).
        """
        super(SACComponent, self).__init__()
        self.name = name

        # Инициализация устройства
        self._device = device_manager.get_device(
            device
        )  

        # Инициализация GradScaler для AMP (только для CUDA)
        self.enable_amp = self._device.type == "cuda"
        # Создаем GradScaler всегда, но включаем его только для CUDA
        self.scaler = GradScaler(enabled=self.enable_amp)

        logger.debug(
            f"Инициализирован компонент '{name}' на устройстве {self._device}. AMP {'включен' if self.enable_amp else 'выключен'}."
        )

    @property
    @lru_cache(maxsize=1)
    def device(self) -> torch.device:
        """
        Возвращает текущее устройство компонента (torch.device).
        Кэшируется для ускорения.
        """
        return self._device

    def _is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def _is_cpu(self) -> bool:
        return self.device.type == "cpu"

    @log_method_call()
    def to_device(self, device: Union[str, torch.device]) -> "SACComponent":
        """
        Перемещает компонент на указанное устройство и обновляет GradScaler.

        Args:
            device (Union[str, torch.device]): Целевое устройство.

        Returns:
            SACComponent: Объект, перемещённый на новое устройство.
        """
        # Получаем объект torch.device
        target_device = device_manager.get_device(device)

        # Пропускаем перенос, если устройство не изменилось
        if self._device == target_device:
            logger.debug(
                f"Компонент '{self.name}' уже находится на устройстве {target_device}"
            )
            return self

        # Запоминаем предыдущее устройство для логирования
        prev_device = self._device

        # Обновляем устройство компонента перед переносом
        self._device = target_device

        # Переносим основной модуль (параметры и буферы) с non_blocking=True
        # для асинхронной передачи данных на CUDA
        result = super().to(target_device, non_blocking=True)

        # Обновляем состояние GradScaler в зависимости от устройства
        self.enable_amp = self._device.type == "cuda"
        self.scaler = GradScaler(enabled=self.enable_amp)

        logger.debug(
            f"Компонент '{self.name}' перемещен с {prev_device} на {target_device}. AMP {'включен' if self.enable_amp else 'выключен'}."
        )
        return result

    @log_method_call()
    def save(self, path: str) -> None:
        """
        Сохраняет состояние компонента в файл.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            state_dict = self.state_dict()

            cpu_state_dict = {}
            for k, v in state_dict.items():
                v_clean = device_manager.clean_tensor(v)
                cpu_state_dict[k] = device_manager.safe_to_cpu(v_clean)

            torch.save(cpu_state_dict, path)
            logger.info(f"Компонент '{self.name}' сохранен в {path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения компонента '{self.name}': {e}")
            minimal_state = {'__config__': self.get_config()}
            torch.save(minimal_state, path)
            logger.warning(f"Сохранен только конфиг компонента '{self.name}'")

    @log_method_call()
    def load(
        self, path: str, map_location: Optional[Union[str, torch.device]] = None
    ) -> None:
        """
        Загружает состояние компонента из файла.

        Args:
            path (str): Путь к файлу с состоянием.
            map_location (Optional[Union[str, torch.device]]): Устройство, на которое будет загружено состояние.
        """
        if not os.path.exists(path):
            logger.warning(
                f"Файл {path} не найден, загрузка компонента '{self.name}' пропущена."
            )
            return

        try:
            # Определяем устройство для загрузки
            target_device = map_location or self.device
            # Получаем объект torch.device
            load_device = device_manager.get_device(target_device)

            # Загружаем состояние
            state_dict = torch.load(path, map_location=load_device)
            self.load_state_dict(state_dict)
            logger.info(
                f"Компонент '{self.name}' успешно загружен из {path} на устройство {load_device}"
            )

            # Если загрузка была на другое устройство, перемещаем компонент
            if load_device != self.device:
                self.to_device(load_device)

        except Exception as e:
            logger.error(f"Ошибка при загрузке компонента '{self.name}' из {path}: {e}")
            raise

    def get_config(self) -> Dict[str, Any]:
        """
        Возвращает базовый конфиг компонента в виде словаря.

        Note:
            Используется в наследниках класса.

        Returns:
            Dict[str, Any]: Конфигурационные данные компонента.
        """
        return {
            "name": self.name,
            "component_type": self.__class__.__name__,
            "device": str(self.device),
        }

    @log_method_call()
    def reset(self) -> None:
        """
        Сбрасывает внутреннее состояние компонента (если есть).
        Реализуется в подклассах по необходимости.
        """
        # Базовый компонент не имеет сбрасываемого состояния
        pass

    def get_parameter_count(self) -> Dict[str, int]:
        """
        Возвращает количество параметров (всего, обучаемых, замороженных).

        Returns:
            Dict[str, int]: Словарь с количеством параметров.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        counts = {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
        }
        logger.debug(f"Параметры компонента '{self.name}': {counts}")
        return counts

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Оценивает использование памяти моделью (в мегабайтах).

        """
        try:
            from torchinfo import summary
            info = summary(self, verbose=0, depth=1)
            total_memory_mb = info.total_params * 4 / (1024 * 1024)  # float32
            usage = {
                "total_mb": round(total_memory_mb, 2),
                "parameters_mb": round(total_memory_mb, 2),
            }
            return usage
        except Exception:
            # Старый способ
            total_memory = 0
            for param in self.parameters():
                param_size = param.numel() * param.element_size()
                if param.requires_grad:
                    param_size *= 2
                total_memory += param_size
            total_memory_mb = total_memory / (1024 * 1024)
            usage = {
                "total_mb": round(total_memory_mb, 2),
                "parameters_mb": round(total_memory_mb, 2),
            }
            return usage

    @log_method_call()
    def summary(self) -> str:
        """
        Возвращает краткую информацию о компоненте в виде строки.

        Returns:
            str: Описание компонента с ключевыми характеристиками.
        """
        param_counts = self.get_parameter_count()
        memory_usage = self.get_memory_usage()

        summary_str = f"Компонент: {self.name}\n"
        summary_str += f"Устройство: {self.device}\n"
        summary_str += f"AMP: {'Включен' if self.enable_amp else 'Выключен'}\n"
        summary_str += f"Параметры: {param_counts['total']:,} ({memory_usage['parameters']:.2f} МБ)\n"
        summary_str += f"Буферы: {memory_usage['buffers']:.2f} МБ\n"
        summary_str += f"Всего память: {memory_usage['total']:.2f} МБ\n"

        return summary_str

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Возвращает state_dict компонента для сохранения.

        Returns:
            Dict[str, Any]: Состояние модели в виде словаря.
        """
        return self.state_dict()


# Базовый интерфейс для plug-and-play компонентов
class PlugAndPlayComponent(SACComponent):
    """
    Plug-and-play компонент SAC. Наследники должны регистрироваться через декоратор.

    Attributes:
        COMPONENT_REGISTRY (Dict[str, type]): Регистр всех зарегистрированных компонентов.
    """

    @classmethod
    def from_config(cls, config: dict):
        """
        Создаёт экземпляр компонента из конфигурации.

        Args:
            config (Dict[str, Any]): Конфигурация компонента.

        Returns:
            PlugAndPlayComponent: Экземпляр компонента.
        """
        return cls(**config)
