import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
import os
import numpy as np

from core.logging.logger import get_logger
from core.utils.device_utils import device_manager

logger = get_logger("normalizers")

# --- Нормализатор на основе алгоритма Велфорда ---
class WelfordObservationNormalizer(nn.Module):
    """
    Нормализатор наблюдений, использующий алгоритм Велфорда для онлайн-вычисления 
    среднего и стандартного отклонения с поэлементным обновлением.

    Attributes:
        running_mean (torch.Tensor): Текущее среднее значение наблюдений.
        running_var (torch.Tensor): Текущая дисперсия наблюдений.
        count (torch.Tensor): Общее количество прошедших через нормализатор наблюдений.
        obs_dim (int): Размерность входных наблюдений.
        epsilon (float): Малое значение для предотвращения деления на ноль.
        clip_range (float): Диапазон ограничения нормализованных значений.
    """
    def __init__(self, 
                 obs_dim: int, 
                 epsilon: float = 1e-8, 
                 clip_range: float = 5.0):
        """
        Инициализирует экземпляр WelfordObservationNormalizer.

        Args:
            obs_dim (int): Размерность наблюдений.
            epsilon (float): Малое значение для предотвращения деления на ноль.
            clip_range (float): Диапазон ограничения нормализованных данных.
        """
        super(WelfordObservationNormalizer, self).__init__()
        
        # Register buffers for statistics
        self.register_buffer("running_mean", torch.zeros(obs_dim))
        self.register_buffer("running_var", torch.ones(obs_dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.float32))
        
        self.obs_dim = obs_dim
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        logger.debug(f"Инициализирован WelfordObservationNormalizer с размерностью {obs_dim} и clip_range={clip_range}")
    
    def update(self, obs: torch.Tensor) -> None:
        """
        Обновляет статистику (среднее и дисперсию) на основе новых наблюдений.

        """
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.running_mean.device)
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            batch_count = obs.shape[0]
            # Векторное обновление по алгоритму Велфорда
            total_count = self.count + batch_count
            delta = obs - self.running_mean  # [batch, obs_dim]
            mean_batch = delta.sum(dim=0) / total_count
            new_mean = self.running_mean + mean_batch
            delta2 = obs - new_mean
            m_a = self.running_var * self.count
            m_b = ((delta * delta2).sum(dim=0))
            new_var = (m_a + m_b) / total_count
            self.running_mean = new_mean
            self.running_var = new_var
            self.count = total_count
    
    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Нормализует наблюдения на основе текущей статистики.

        Args:
            obs (torch.Tensor): Входной тензор наблюдений.

        Returns:
            torch.Tensor: Нормализованный тензор наблюдений.
        """
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.running_mean.device)
            
            if self.count <= 1:
                return obs
            
            std = torch.sqrt(self.running_var + self.epsilon)
            normalized_obs = (obs - self.running_mean) / std
            normalized_obs = torch.clamp(normalized_obs, -self.clip_range, self.clip_range)
            
            return normalized_obs
    
    def denormalize(self, normalized_obs: torch.Tensor) -> torch.Tensor:
        """
        Денормализует наблюдения, возвращая их к исходному масштабу
        """
        with torch.no_grad():
            if not isinstance(normalized_obs, torch.Tensor):
                normalized_obs = torch.as_tensor(normalized_obs, dtype=torch.float32, device=self.running_mean.device)
            
            if self.count <= 1:
                return normalized_obs
            
            std = torch.sqrt(self.running_var + self.epsilon)
            denormalized_obs = normalized_obs * std + self.running_mean
            
            return denormalized_obs
    
    def save(self, path: str) -> None:
        """
        Сохраняет текущую статистику нормализатора в файл.

        Args:
            path (str): Путь к файлу, в который будет сохранена статистика.
        """
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'count': self.count,
            'obs_dim': self.obs_dim,
            'epsilon': self.epsilon,
            'clip_range': self.clip_range
        }, path)
        logger.debug(f"Статистика WelfordObservationNormalizer сохранена в {path}")
    
    def load(self, path: str, map_location: Optional[str] = None) -> None:
        """
        Загружает статистику из файла.

        Args:
            path (str): Путь к файлу со статистикой.
            map_location (Optional[str]): Устройство, на которое будет загружен тензор.
        """
        try:
            state = torch.load(path, map_location=map_location)
            self.running_mean = state['running_mean']
            self.running_var = state['running_var']
            self.count = state['count']
            self.obs_dim = state['obs_dim']
            self.epsilon = state['epsilon']
            self.clip_range = state['clip_range']
            logger.debug(f"Статистика WelfordObservationNormalizer загружена из {path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке статистики WelfordObservationNormalizer: {str(e)}")
            raise
            
    def to(self, device: Union[str, torch.device]) -> 'WelfordObservationNormalizer':
        """
        Перемещает нормализатор на указанное устройство (CPU / GPU).

        Args:
            device (Union[str, torch.device]): Целевое устройство.

        Returns:
            WelfordObservationNormalizer: Экземпляр класса на новом устройстве.
        """
        device = device_manager.get_device(device) if isinstance(device, str) else device
        return super().to(device)
        
    def is_dummy(self) -> bool:
        """
        Проверяет, является ли нормализатор фиктивным.

        Returns:
            bool: Всегда возвращает False.
        """
        return False


# --- Нормализатор на основе скользящего среднего и стандартного отклонения ---
class BatchMeanStdNormalizer(nn.Module):
    """
    Нормализатор, вычисляющий скользящие среднее и стандартное отклонение
    с пакетным обновлением для эффективного использования на GPU.

    Attributes:
        mean (torch.Tensor): Текущее среднее значение наблюдений.
        var (torch.Tensor): Текущая дисперсия наблюдений.
        count (torch.Tensor): Общее количество обработанных наблюдений.
        shape (Tuple[int, ...]): Форма входного тензора.
        epsilon (float): Малое значение для предотвращения деления на ноль.
        clip_range (float): Диапазон ограничения нормализованных значений.
    """
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4, clip_range: float = 5.0):
        """
        Инициализирует экземпляр BatchMeanStdNormalizer.

        Args:
            shape (Tuple[int, ...]): Форма входного тензора.
            epsilon (float): Малое значение для предотвращения деления на ноль.
            clip_range (float): Диапазон ограничения нормализованных данных.
        """
        super(BatchMeanStdNormalizer, self).__init__()
        
        # Регистрируем буферы для средних и стандартных отклонений
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(0, dtype=torch.float32))
        
        self.shape = shape
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        logger.debug(f"Инициализирован BatchMeanStdNormalizer с формой {shape} и clip_range={clip_range}")
    
    def update(self, x: Union[torch.Tensor, torch.Tensor]) -> None:
        """
        Обновляет статистику (среднее и дисперсию) на основе нового батча данных.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Батч новых наблюдений.
        """
        with torch.no_grad():
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
            batch_mean_tensor = torch.tensor(batch_mean, device=self.mean.device)
            batch_var_tensor = torch.tensor(batch_var, device=self.var.device)
            new_count = self.count + batch_count
            delta = batch_mean_tensor - self.mean
            new_mean = self.mean + delta * batch_count / new_count
            m_a = self.var * self.count
            m_b = batch_var_tensor * batch_count
            M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / new_count
            new_var = M2 / new_count
            self.mean = new_mean
            self.var = new_var
            self.count = new_count
    
    def normalize(self, x: Union[torch.Tensor, torch.Tensor]) -> Union[torch.Tensor, torch.Tensor]:
        """
        Нормализует данные на основе текущей статистики.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Входной тензор или массив.

        Returns:
            Union[torch.Tensor, np.ndarray]: Нормализованный тензор или массив.
        """
        with torch.no_grad():
            is_tensor = isinstance(x, torch.Tensor)
            
            if is_tensor:
                mean = self.mean
                std = torch.sqrt(self.var + self.epsilon)
                
                # Нормализуем данные
                normalized_x = (x - mean) / std
                
                # Обрезаем значения, выходящие за пределы диапазона
                normalized_x = torch.clamp(normalized_x, -self.clip_range, self.clip_range)
            else:
                # Переводим параметры в numpy
                mean = self.mean.detach().cpu().numpy()
                std = torch.sqrt(self.var.detach().cpu().numpy() + self.epsilon)
                
                # Нормализуем данные
                normalized_x = (x - mean) / std
                
                # Обрезаем значения, выходящие за пределы диапазона
                normalized_x = torch.clip(normalized_x, -self.clip_range, self.clip_range)
        
        return normalized_x
    
    def denormalize(self, normalized_x: Union[torch.Tensor, torch.Tensor]) -> Union[torch.Tensor, torch.Tensor]:
        """
        Денормализует данные, возвращая их к исходному масштабу.

        Args:
            normalized_x (Union[torch.Tensor, np.ndarray]): Нормализованный тензор или массив.

        Returns:
            Union[torch.Tensor, np.ndarray]: Оригинальный масштабированный тензор или массив.
        """
        with torch.no_grad():
            is_tensor = isinstance(normalized_x, torch.Tensor)
            
            if is_tensor:
                mean = self.mean
                std = torch.sqrt(self.var + self.epsilon)
                
                # Денормализуем данные
                denormalized_x = normalized_x * std + mean
            else:
                # Переводим параметры в numpy
                mean = self.mean.detach().cpu().numpy()
                std = torch.sqrt(self.var.detach().cpu().numpy() + self.epsilon)
                
                # Денормализуем данные
                denormalized_x = normalized_x * std + mean
        
        return denormalized_x
    
    def save(self, path: str) -> None:
        """
        Сохраняет текущую статистику нормализатора в файл.

        Args:
            path (str): Путь к файлу, в который будет сохранена статистика.
        """
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'mean': self.mean,
            'var': self.var,
            'count': self.count,
            'shape': self.shape,
            'epsilon': self.epsilon,
            'clip_range': self.clip_range
        }, path)
        logger.debug(f"Статистика BatchMeanStdNormalizer сохранена в {path}")
    
    def load(self, path: str, map_location: Optional[str] = None) -> None:
        """
        Загружает статистику из файла.

        Args:
            path (str): Путь к файлу со статистикой.
            map_location (Optional[str]): Устройство, на которое будет загружен тензор.
        """
        try:
            state = torch.load(path, map_location=map_location)
            self.mean = state['mean']
            self.var = state['var']
            self.count = state['count']
            self.shape = state['shape']
            self.epsilon = state['epsilon']
            self.clip_range = state['clip_range']
            logger.debug(f"Статистика BatchMeanStdNormalizer загружена из {path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке статистики BatchMeanStdNormalizer: {str(e)}")
            raise
            
    def to(self, device: Union[str, torch.device]) -> 'BatchMeanStdNormalizer':
        """
        Перемещает нормализатор на указанное устройство (CPU / GPU).

        Args:
            device (Union[str, torch.device]): Целевое устройство.

        Returns:
            BatchMeanStdNormalizer: Экземпляр класса на новом устройстве.
        """
        device = device_manager.get_device(device) if isinstance(device, str) else device
        return super().to(device)
        
    def is_dummy(self) -> bool:
        """
        Проверяет, является ли нормализатор фиктивным.

        Returns:
            bool: Всегда возвращает False.
        """
        return False


# --- Фиктивный нормализатор (для случаев, когда нормализация не нужна) ---
class DummyNormalizer(nn.Module):
    """
    Фиктивный нормализатор, не выполняющий фактической нормализации.

    Attributes:
        obs_dim (Optional[int]): Размерность наблюдений (необязательная).
    """
    def __init__(self, obs_dim=None):
        """
        Инициализирует фиктивный нормализатор.

        Args:
            obs_dim (Optional[int]): Размерность наблюдений.
        """
        super(DummyNormalizer, self).__init__()
        self.obs_dim = obs_dim
        logger.debug(f"Инициализирован DummyNormalizer (obs_dim={obs_dim})")
    
    def update(self, x: Union[torch.Tensor, torch.Tensor]) -> None:
        """
        Фиктивное обновление. Не изменяет состояние нормализатора.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Входные данные (игнорируются).
        """
        pass
    
    def normalize(self, x: Union[torch.Tensor, torch.Tensor]) -> Union[torch.Tensor, torch.Tensor]:
        """
        Возвращает входные данные без изменений.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Входные данные.

        Returns:
            Union[torch.Tensor, np.ndarray]: То же самое, что и на входе.
        """
        return x
    
    def denormalize(self, x: Union[torch.Tensor, torch.Tensor]) -> Union[torch.Tensor, torch.Tensor]:
        """
        Возвращает входные данные без изменений.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Входные данные.

        Returns:
            Union[torch.Tensor, np.ndarray]: То же самое, что и на входе.
        """
        return x
    
    def save(self, path: str) -> None:
        """
        Не сохраняет ничего. Только логирует.

        Args:
            path (str): Путь к файлу (игнорируется).
        """
        logger.debug(f"DummyNormalizer: сохранение в {path} не требуется")
    
    def load(self, path: str, map_location: Optional[str] = None) -> None:
        """
        Не загружает ничего. Только логирует.

        Args:
            path (str): Путь к файлу (игнорируется).
            map_location (Optional[str]): Устройство (игнорируется).
        """
        logger.debug(f"DummyNormalizer: загрузка из {path} не требуется")
        
    def to(self, device: Union[str, torch.device]) -> 'DummyNormalizer':
        """
        Перемещает нормализатор на устройство (ничего не меняет).

        Args:
            device (Union[str, torch.device]): Целевое устройство.

        Returns:
            DummyNormalizer: Сам объект.
        """
        device = device_manager.get_device(device) if isinstance(device, str) else device
        return super().to(device)
        
    def is_dummy(self) -> bool:
        """
        Проверяет, является ли нормализатор фиктивным.

        Returns:
            bool: Всегда возвращает True.
        """
        return True