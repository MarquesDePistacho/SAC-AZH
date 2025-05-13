import torch
import os
from typing import Union, Optional, Any, Dict, List

from core.logging.logger import get_logger

logger = get_logger("device_utils")


class DeviceManager:
    """
    Менеджер устройств для централизованного управления GPU/CPU ресурсами
    """

    _instance = None

    def __new__(cls) -> "DeviceManager":
        """
        Реализует паттерн Singleton. Создает единственный экземпляр класса, если он еще не существует.

        Returns:
            DeviceManager: Единственный экземпляр класса DeviceManager.
        """
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Инициализирует внутренние параметры менеджера устройств: проверяет доступность CUDA и сканирует GPU.
        """
        self._default_device = None
        self._available_gpus = []
        self._scan_available_devices()
        logger.debug(f"DeviceManager инициализирован: {self.device_summary()}")

        if self._cuda_available:
            self._cuda_stream = torch.cuda.Stream()
        else:
            self._cuda_stream = None

    def _scan_available_devices(self):
        """
        Сканирует и обновляет список доступных GPU-устройств и их характеристики.
        """
        self._cuda_available = torch.cuda.is_available()

        if self._cuda_available:
            self._cuda_device_count = torch.cuda.device_count()
            self._available_gpus = list(range(self._cuda_device_count))

            # Получаем информацию о GPU
            self._gpu_info = []
            for i in range(self._cuda_device_count):
                name = torch.cuda.get_device_name(i)
                mem_info = self.get_cuda_memory_info(i)
                self._gpu_info.append({"index": i, "name": name, "memory": mem_info})
        else:
            self._cuda_device_count = 0
            self._available_gpus = []
            self._gpu_info = []

    def get_device(self, device: Union[str, torch.device, None] = None) -> torch.device:
        """
        Возвращает объект устройства (CPU или GPU), основываясь на входном значении.

        Args:
            device (Union[str, torch.device, None]): Описание устройства ('cuda', 'cpu', 'cuda:0') или объект torch.device.
                Если None — выбирается доступное по умолчанию устройство.

        Returns:
            torch.device: Объект устройства (например, cuda:0 или cpu).
        """
        if device is None:
            # Автоматический выбор: CUDA если доступно, иначе CPU
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Если уже получили объект устройства - возвращаем как есть
        if isinstance(device, torch.device):
            return device

        # Приводим строку к нижнему регистру
        device_str = device.lower()

        # Проверяем различные варианты названия CUDA
        if device_str in ["cuda", "gpu"]:
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning("CUDA недоступно, используется CPU")
                return torch.device("cpu")

        # Проверка для конкретного GPU
        elif device_str.startswith("cuda:") or device_str.startswith("gpu:"):
            gpu_id = int(device_str.split(":")[1])
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                return torch.device(f"cuda:{gpu_id}")
            else:
                logger.warning(f"GPU {gpu_id} недоступен, используется CPU")
                return torch.device("cpu")

        # По умолчанию используем CPU
        else:
            return torch.device("cpu")

    def set_default_device(self, device: Union[str, torch.device]) -> torch.device:
        """
        Устанавливает новое устройство по умолчанию.

        Args:
            device (Union[str, torch.device]): Новое устройство.

        Returns:
            torch.device: Установленное устройство по умолчанию.
        """
        device_obj = self.get_device(device)
        self._default_device = device_obj
        logger.info(f"Изменено устройство по умолчанию: {device_obj}")
        return device_obj

    def get_default_device(self) -> torch.device:
        """
        Возвращает текущее устройство по умолчанию (CUDA, если доступно).

        Returns:
            torch.device: Устройство по умолчанию.
        """
        if self._default_device is None:
            self._default_device = self.get_device()
        return self._default_device

    def to_device(
        self,
        obj: Any,
        device: Union[str, torch.device, None] = None,
        non_blocking: bool = False,
    ) -> Any:
        """
        Перемещает объект (тензор, модель, словарь, список и т.д.) на указанное устройство.

        Args:
            obj (Any): Объект для перемещения.
            device (Union[str, torch.device, None]): Целевое устройство. Если None — используется устройство по умолчанию.
            non_blocking (bool): Если True, копирование может быть асинхронным.

        Returns:
            Any: Объект, перемещенный на целевое устройство.
        """
        target_device = self.get_device(device)

        if obj is None:
            return None

        # Если объект имеет метод .to() (тензоры, модули)
        if hasattr(obj, "to") and callable(getattr(obj, "to")):
            # Для тензоров можем использовать non_blocking
            if isinstance(obj, torch.Tensor):
                return obj.to(target_device, non_blocking=non_blocking)
            return obj.to(target_device)

        # Списки и кортежи обрабатываем рекурсивно (с сохранением типа)
        elif isinstance(obj, (list, tuple)):
            container_type = type(obj)
            return container_type(
                self.to_device(x, target_device, non_blocking) for x in obj
            )

        # Словари обрабатываем рекурсивно (только значения)
        elif isinstance(obj, dict):
            return {
                k: self.to_device(v, target_device, non_blocking)
                for k, v in obj.items()
            }

        # Другие типы данных оставляем без изменений
        else:
            return obj

    def async_data_transfer(
        self,
        data: Union[Dict[str, torch.Tensor], torch.Tensor],
        source_device: Union[str, torch.device, None] = "cpu",
        target_device: Union[str, torch.device, None] = None,
        use_pinned_memory: bool = True,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Асинхронная передача данных между устройствами с использованием pinned memory.
        
        Args:
            data (Union[Dict[str, torch.Tensor], torch.Tensor]): Данные для передачи 
            source_device (Union[str, torch.device, None]): Исходное устройство 
            target_device (Union[str, torch.device, None]): Целевое устройство 
            use_pinned_memory (bool): Использовать pinned memory для промежуточного хранения при передаче с CPU на GPU
            stream (Optional[torch.cuda.Stream]): CUDA поток для асинхронной передачи
        
        Returns:
            Union[Dict[str, torch.Tensor], torch.Tensor]: Переданные данные на целевом устройстве
        """
        source_device = self.get_device(source_device)
        target_device = self.get_device(target_device or self.get_default_device())
        
        # Если устройства совпадают, не нужна передача
        if source_device == target_device:
            return data
        
        # Если передаем с CPU на CUDA с использованием pinned memory
        if source_device.type == "cpu" and target_device.type == "cuda" and use_pinned_memory:
            # Используем переданный поток или создаем новый
            cuda_stream = stream or self._cuda_stream or torch.cuda.Stream(device=target_device)
            
            # Обрабатываем словарь тензоров
            if isinstance(data, dict):
                result = {}
                with torch.cuda.stream(cuda_stream):
                    for key, tensor in data.items():
                        if isinstance(tensor, torch.Tensor):
                            # Только CPU тензоры пиним и отправляем асинхронно
                            if tensor.device.type == 'cpu':
                                pinned_tensor = tensor.pin_memory() if not tensor.is_pinned() else tensor
                                result[key] = pinned_tensor.to(target_device, non_blocking=True)
                            else:
                                # Если тензор уже на GPU, просто используем его или копируем при необходимости
                                result[key] = tensor if tensor.device == target_device else tensor.to(target_device)
                        else:
                            # Не тензоры копируем как обычно
                            result[key] = self.to_device(tensor, target_device)
                
                # Синхронизируем текущий поток с cuda_stream
                torch.cuda.current_stream().wait_stream(cuda_stream)
                return result
            
            # Обрабатываем единичный тензор
            elif isinstance(data, torch.Tensor):
                # Пиним и передаем асинхронно только CPU тензоры
                if data.device.type == 'cpu':
                    with torch.cuda.stream(cuda_stream):
                        pinned_data = data.pin_memory() if not data.is_pinned() else data
                        result = pinned_data.to(target_device, non_blocking=True)
                    torch.cuda.current_stream().wait_stream(cuda_stream)
                    return result
                else:
                    # Если тензор уже на GPU, возвращаем или перемещаем при необходимости
                    return data if data.device == target_device else data.to(target_device)
            
            # Другие типы данных
            else:
                return self.to_device(data, target_device)
                
        # Для других комбинаций устройств или без pinned memory
        return self.to_device(data, target_device, non_blocking=(target_device.type == "cuda"))

    def transfer(
        self,
        data: Union[Dict[str, torch.Tensor], torch.Tensor, Any],
        target_device: Union[str, torch.device, None] = None,
        async_transfer: bool = True,
        **kwargs
    ) -> Any:
        """
        Универсальный метод для передачи объектов между устройствами.
        Если async_transfer=True, используется pinned memory и асинхронная передача.
        Иначе вызывается to_device для синхронного переноса.
        """
        # Определяем целевое устройство
        tgt = self.get_device(target_device)
        if async_transfer:
            # Пытаемся асинхронно через pinned memory
            return self.async_data_transfer(
                data, source_device=self.get_default_device(), target_device=tgt, **kwargs
            )
        else:
            # Синхронная передача через .to()
            return self.to_device(data, tgt, non_blocking=False)

    def set_cuda_visible_devices(self, devices: Union[int, List[int], str]) -> None:
        """
        Устанавливает видимые CUDA-устройства через переменную окружения CUDA_VISIBLE_DEVICES.
        """
        # Преобразуем вход в строку
        if isinstance(devices, int):
            devices_str = str(devices)
        elif isinstance(devices, list):
            devices_str = ",".join(map(str, devices))
        else:
            devices_str = devices

        # Устанавливаем переменную окружения
        os.environ["CUDA_VISIBLE_DEVICES"] = devices_str
        logger.info(f"Установлены видимые CUDA устройства: {devices_str}")

        # Пересканируем доступные устройства
        self._scan_available_devices()

    def get_cuda_memory_info(
        self, device: Optional[Union[int, torch.device]] = None
    ) -> Dict[str, float]:
        """
        Возвращает информацию о памяти указанного CUDA-устройства.

        Args:
            device (Optional[Union[int, torch.device]]): Устройство, для которого запрашивается информация.

        Returns:
            Dict[str, float]: Словарь с информацией о свободной, занятой и общей памяти (в ГБ и процентах).
        """
        if not self._cuda_available:
            logger.warning("CUDA недоступна. Информация о памяти недоступна.")
            return {"free": 0, "total": 0, "used": 0, "percent_used": 0}

        # Определяем индекс устройства
        device_idx = 0
        if device is not None:
            if isinstance(device, torch.device) and device.type == "cuda":
                device_idx = device.index if device.index is not None else 0
            elif isinstance(device, int):
                device_idx = device

        # Проверяем валидность индекса
        if device_idx >= self._cuda_device_count:
            logger.warning(
                f"Индекс устройства {device_idx} вне диапазона. Доступно устройств: {self._cuda_device_count}"
            )
            device_idx = 0

        # Получаем информацию о памяти
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            used_gb = total_gb - free_gb

            return {
                "free": free_gb,
                "total": total_gb,
                "used": used_gb,
                "percent_used": (used_gb / total_gb) * 100 if total_gb > 0 else 0,
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о памяти: {str(e)}")
            return {"free": 0, "total": 0, "used": 0, "percent_used": 0}

    def clear_cuda_cache(self) -> None:
        """
        Очищает кэш CUDA для освобождения выделенной, но неиспользуемой памяти.
        """
        if self._cuda_available:
            torch.cuda.empty_cache()
            logger.debug("Кэш CUDA очищен")

    def device_summary(self) -> str:
        """
        Возвращает текстовую сводку о доступных устройствах и текущих настройках.

        Returns:
            str: Текстовое описание состояния устройств (CUDA/GPU/память).
        """
        summary = f"CUDA доступна: {self._cuda_available}\n"

        if self._cuda_available:
            summary += f"Количество GPU: {self._cuda_device_count}\n"
            for idx, info in enumerate(self._gpu_info):
                summary += f"GPU {idx}: {info['name']}, Память: {info['memory']['used']:.2f}/{info['memory']['total']:.2f} GB ({info['memory']['percent_used']:.1f}%)\n"

        if self._default_device:
            summary += f"Устройство по умолчанию: {self._default_device}"

        return summary

    def find_optimal_device(self) -> torch.device:
        """
        Находит оптимальное устройство (GPU с наибольшим количеством свободной памяти).

        Returns:
            torch.device: Лучшее доступное устройство (например, cuda:0).
        """
        if not self._cuda_available:
            logger.info("CUDA недоступна, используется CPU")
            return torch.device("cpu")

        if not self._gpu_info:
            logger.info("Нет информации о GPU, используется CPU")
            return torch.device("cpu")

        # Ищем GPU с наибольшим количеством свободной памяти
        best_gpu = max(self._gpu_info, key=lambda x: x["memory"]["free"])
        device_str = f"cuda:{best_gpu['index']}"
        logger.info(
            f"Выбрано оптимальное устройство: {device_str} ({best_gpu['name']}), "
            f"Свободно: {best_gpu['memory']['free']:.2f} GB"
        )
        return torch.device(device_str)

    def get_all_devices_info(self) -> List[Dict[str, Any]]:
        """
        Возвращает информацию о всех доступных GPU-устройствах.

        Returns:
            List[Dict[str, Any]]: Список словарей с информацией о каждом GPU.
        """
        return self._gpu_info.copy() if self._cuda_available else []

    def clean_tensor(self, tensor: Any) -> Any:
        """
        Заменяет NaN/Inf в тензоре на 0, без изменения остальных значений.
        """
        if not isinstance(tensor, torch.Tensor):
            return tensor
        try:
            return _torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
        except Exception:
            return tensor

    def safe_to_cpu(self, tensor: Any) -> Any:
        """
        Переносит тензор на CPU; при ошибке создаёт новый нулевой тензор на CPU того же размера.
        """

        if not isinstance(tensor, torch.Tensor):
            return tensor
        try:
            return tensor.detach().cpu()
        except Exception:
            try:
                return torch.zeros_like(tensor, device='cpu')
            except Exception:
                return tensor

    def ensure_tensor_on_device(
        self,
        tensor: Any,
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
        name: str = "tensor"
    ) -> torch.Tensor:
        """
        Приводит вход к torch.Tensor, переносит на нужное устройство и тип.
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if device is not None and tensor.device != device:
            tensor = tensor.to(device=device)
        return tensor

    def prepare_input_tensor(
        self,
        x: Any,
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
        min_dim: int = 2,
        name: str = "input"
    ) -> torch.Tensor:
        """
        Приводит x к torch.Tensor, переносит на нужное устройство и тип, добавляет batch dim если нужно.
        """
        tensor = self.ensure_tensor_on_device(x, device, dtype, name)
        while tensor.dim() < min_dim:
            tensor = tensor.unsqueeze(0)
        return tensor

# Создаем глобальный экземпляр менеджера устройств
device_manager = DeviceManager()
