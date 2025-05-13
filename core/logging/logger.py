import logging
import os
import sys
import functools
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Union, Callable

# Уровни логирования для удобного доступа
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
TRACE = 5  # Уровень ниже DEBUG для очень подробных логов
logging.addLevelName(TRACE, "TRACE")

# Форматтер по умолчанию
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class LoggerManager:
    """
    Синглтон-менеджер для централизованной настройки и управления логгерами в приложении.
    """
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _root_logger: Optional[logging.Logger] = None
    _file_handler: Optional[logging.FileHandler] = None
    _console_handler: Optional[logging.StreamHandler] = None
    _log_dir: str = "logs"
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    def setup(
        self,
        log_dir: str = "logs",
        console_level: int = INFO,
        file_level: int = INFO,
        root_name: str = "sac_app",
        log_format: str = DEFAULT_LOG_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
        use_file_handler: bool = True,
        use_console_handler: bool = True,
    ) -> logging.Logger:
        """
        Настраивает корневой логгер и обработчики логирования (консоль и файл).

        Args:
            log_dir (str): Путь к директории логов.
            console_level (int): Уровень логов для консоли.
            file_level (int): Уровень логов для файла.
            root_name (str): Имя корневого логгера.
            log_format (str): Формат строк лога.
            date_format (str): Формат даты в логах.
            use_file_handler (bool): Включить запись в файл.
            use_console_handler (bool): Включить вывод в консоль.

        Returns:
            logging.Logger: Настроенный корневой логгер.
        """
        if self._initialized:
            return self._root_logger

        self._log_dir = log_dir

        # Создаем корневой логгер
        self._root_logger = logging.getLogger(root_name)
        
        minimum_level = min(console_level if use_console_handler else INFO, 
                            file_level if use_file_handler else INFO)
        self._root_logger.setLevel(minimum_level)
        
        # Очищаем обработчики
        self._root_logger.handlers = []

        # Создаем форматтер
        try:
            formatter = logging.Formatter(log_format, datefmt=date_format)
        except Exception:
            formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

        # Добавляем консольный обработчик
        if use_console_handler:
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.setLevel(console_level)
            self._console_handler.setFormatter(formatter)
            self._root_logger.addHandler(self._console_handler)

        # Добавляем файловый обработчик
        if use_file_handler:
            try:
                os.makedirs(log_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(log_dir, f"{root_name}_{timestamp}.log")
                
                self._file_handler = logging.FileHandler(log_file, encoding="utf-8")
                self._file_handler.setLevel(file_level)
                self._file_handler.setFormatter(formatter)
                self._root_logger.addHandler(self._file_handler)
            except Exception as e:
                print(f"Ошибка создания файлового обработчика: {e}", file=sys.stderr)
                self._file_handler = None

        self._initialized = True
        
        if self._root_logger:
            console_status = f"{logging.getLevelName(console_level)}" if self._console_handler else "Отключена"
            file_status = f"{logging.getLevelName(file_level)}" if self._file_handler else "Отключен"
            self._root_logger.info(f"Логгер настроен. Консоль: {console_status}, Файл: {file_status}")
        
        return self._root_logger

    def get_logger(self, name: str) -> logging.Logger:
        """Возвращает логгер с указанным именем"""
        if name not in self._loggers:
            if self._root_logger:
                logger_name = f"{self._root_logger.name}.{name}"
            else:
                logger_name = name
                if not logging.root.handlers:
                    logging.basicConfig(level=INFO)

            self._loggers[name] = logging.getLogger(logger_name)

        return self._loggers[name]

    def set_level(self, level: Union[int, str], logger_name: Optional[str] = None) -> None:
        """Устанавливает уровень логирования для указанного или всех логгеров"""
        if isinstance(level, str):
            level_int = logging.getLevelName(level.upper())
            if not isinstance(level_int, int):
                return
            level = level_int

        if logger_name:
            if logger_name in self._loggers:
                self._loggers[logger_name].setLevel(level)
            elif self._root_logger and logger_name == self._root_logger.name:
                self._root_logger.setLevel(level)
        else:
            # Устанавливаем для всех логгеров
            for logger_instance in self._loggers.values():
                logger_instance.setLevel(level)
            if self._root_logger:
                self._root_logger.setLevel(level)
            if self._console_handler:
                self._console_handler.setLevel(level)
            if self._file_handler:
                self._file_handler.setLevel(level)


# Глобальный экземпляр и функции доступа
logger_manager = LoggerManager()

def setup_logging(*args, **kwargs) -> logging.Logger:
    """Настраивает систему логирования"""
    return logger_manager.setup(*args, **kwargs)

def get_logger(name: str = "default") -> logging.Logger:
    """Возвращает логгер для компонента"""
    if not logger_manager._initialized:
        logger_manager.setup()
    return logger_manager.get_logger(name)

def set_log_level(level: Union[int, str], logger_name: Optional[str] = None) -> None:
    """Устанавливает уровень логирования"""
    logger_manager.set_level(level, logger_name)

def log_device_info(logger_instance: Optional[logging.Logger] = None) -> None:
    """Логирует информацию о вычислительных устройствах"""
    target_logger = logger_instance or logger_manager._root_logger
    if not target_logger:
        return

    target_logger.info("Получение информации об устройствах...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        target_logger.info(f"torch.cuda.is_available(): {cuda_available}")
        if cuda_available:
            count = torch.cuda.device_count()
            target_logger.info(f"torch.cuda.device_count(): {count}")
            for i in range(count):
                try:
                    name = torch.cuda.get_device_name(i)
                    free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                    free_gb = free_bytes / (1024**3)
                    total_gb = total_bytes / (1024**3)
                    target_logger.info(f"  GPU {i}: {name}, Память: {free_gb:.2f}/{total_gb:.2f} GB свободно")
                except Exception as e:
                    target_logger.warning(f"  Не удалось получить информацию для GPU {i}: {e}")
        else:
            target_logger.info("Используется CPU.")
    except ImportError:
        target_logger.warning("PyTorch не найден. Информация об устройствах недоступна.")
    except Exception as e:
        target_logger.error(f"Ошибка при получении информации о CUDA: {e}", exc_info=True)


# Декораторы для логирования
def log_method_call(log_level: int = DEBUG, log_args: bool = True, log_return: bool = False):
    """Декоратор для логирования вызовов методов и функций"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Получаем логгер
            logger_instance = None
            func_name = func.__qualname__

            if args:
                first_arg = args[0]
                if hasattr(first_arg, "logger") and isinstance(getattr(first_arg, "logger"), logging.Logger):
                    logger_instance = getattr(first_arg, "logger")
                elif isinstance(first_arg, type) and hasattr(first_arg, "_logger"):
                    logger_instance = getattr(first_arg, "_logger")
                elif hasattr(first_arg, "__class__"):
                    try:
                        logger_instance = get_logger(first_arg.__class__.__name__.lower())
                    except Exception:
                        pass

            if logger_instance is None:
                module_name = func.__module__
                logger_instance = get_logger(module_name)

            # Логируем вызов
            if log_args:
                args_repr = [repr(a) for a in args[1:]]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                logger_instance.log(log_level, f"Вызов {func_name}({signature})")
            else:
                logger_instance.log(log_level, f"Вызов {func_name}(...)")

            # Выполняем метод и логируем результат
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = end_time - start_time

                log_msg = f"Завершение {func_name}. Длительность: {duration}"
                if log_return:
                    try:
                        result_repr = repr(result)
                        if len(result_repr) > 100:
                            result_repr = result_repr[:100] + "..."
                        log_msg += f", Результат: {result_repr}"
                    except Exception:
                        log_msg += ", Результат: <не удалось представить>"

                logger_instance.log(log_level, log_msg)
                return result
            except Exception as e:
                end_time = datetime.now()
                duration = end_time - start_time
                logger_instance.error(f"Ошибка в {func_name} после {duration}: {e}", exc_info=True)
                raise

        return wrapper
    return decorator


def log_tensor_info(logger: logging.Logger, name: str, tensor: Any, 
                    level: int = TRACE, log_stats: bool = False) -> None:
    """Логирует информацию о тензоре"""
    if not logger.isEnabledFor(level):
        return

    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            info_str = f"Тензор '{name}': shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"

            if log_stats and tensor.numel() > 0 and tensor.is_floating_point():
                try:
                    tensor_cpu = tensor.detach().float().cpu()
                    info_str += f", stats=[min={tensor_cpu.min().item():.4g}, max={tensor_cpu.max().item():.4g}, mean={tensor_cpu.mean().item():.4g}, std={tensor_cpu.std().item():.4g}]"
                except Exception as e:
                    info_str += f", stats=<ошибка: {e}>"
            
            logger.log(level, info_str)
        elif isinstance(tensor, (np.ndarray, np.generic)):
            info_str = f"Numpy массив '{name}': shape={tensor.shape}, dtype={tensor.dtype}"
            
            if log_stats and tensor.size > 0 and np.issubdtype(tensor.dtype, np.number):
                try:
                    info_str += f", stats=[min={np.min(tensor):.4g}, max={np.max(tensor):.4g}, mean={np.mean(tensor):.4g}, std={np.std(tensor):.4g}]"
                except Exception as e:
                    info_str += f", stats=<ошибка: {e}>"
            
            logger.log(level, info_str)
        else:
            logger.log(level, f"Переменная '{name}': type={type(tensor).__name__}")
    except ImportError:
        logger.log(level, f"Переменная '{name}': type={type(tensor).__name__} (PyTorch/Numpy не импортированы)")
    except Exception as e:
        logger.log(level, f"Не удалось залогировать информацию о '{name}': {e}")
