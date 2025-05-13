import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass

from core.sac.buffers.segment_tree import SumTree, check_bfloat16_support
from core.logging.logger import get_logger
from core.utils.device_utils import device_manager

logger = get_logger("buffers")


# Отдельные JIT функции для критических операций
@torch.jit.script
def compute_is_weights(
    priorities: torch.Tensor,
    total_sum: float,
    beta: float,
    batch_size: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Вычисляет веса Importance Sampling для коррекции смещения приоритетного семплирования.
    Оптимизировано для векторного выполнения и численной стабильности.
    """
    with torch.no_grad():
        # Рассчитываем вероятности P(i) = priority / total_sum
        probabilities = priorities / (total_sum + epsilon)

        # Вычисляем веса IS: w_i = (N * P(i)) ^ (-beta)
        # N - размер буфера (или батча, если нормализуем по батчу)
        # Здесь N предполагается равным batch_size для нормализации по батчу
        weights = (batch_size * probabilities).pow(-beta)

        # Нормализуем веса по максимальному весу в батче: w_i / max(w_j)
        max_weight = torch.max(weights)
        normalized_weights = weights / (max_weight + epsilon)

        # Убедимся, что веса не содержат NaN/Inf
        normalized_weights = torch.nan_to_num(
            normalized_weights, nan=0.0, posinf=1.0, neginf=0.0
        )

    return normalized_weights


@torch.jit.script
def update_priorities_batch(
    tree: torch.Tensor, tree_indices: torch.Tensor, td_abs: torch.Tensor, alpha: float
) -> None:
    """
    Обновляет приоритеты в дереве сумм и распространяет изменения вверх.

    Note:
        - Использует степенное преобразование TD-ошибок с параметром alpha. \n
        - Все значения приводятся к типу дерева (tree.dtype) для поддержки bfloat16/float16/float32. \n

    Args:
        tree (torch.Tensor): Дерево SumTree.
        tree_indices (torch.Tensor): Индексы листьев дерева.
        td_abs (torch.Tensor): По модулю TD-ошибки.
        alpha (float): Параметр степенного преобразования приоритетов.
    """
    with torch.no_grad():
        # Переносим индексы и TD-ошибки на устройство дерева
        idxs = tree_indices.to(tree.device) if tree_indices.device != tree.device else tree_indices
        errors = td_abs.to(tree.device) if td_abs.device != tree.device else td_abs
        if idxs.numel() == 0:
            return  # Нечего обновлять, если нет индексов
        # Проверяем и корректируем индексы
        valid_mask = (idxs >= 0) & (idxs < tree.size(0))
        if not torch.all(valid_mask):
            idxs = idxs[valid_mask]
            errors = errors[valid_mask]
            if idxs.numel() == 0:
                return  # После фильтрации нет валидных индексов
        # Обрабатываем NaN/Inf в ошибках
        if torch.isnan(errors).any() or torch.isinf(errors).any():
            errors = torch.nan_to_num(errors, nan=1e-2, posinf=1.0, neginf=1e-2)
        # Ограничиваем TD-ошибки и приводим к типу дерева
        errors_clamped = torch.clamp(errors, min=1e-8, max=1e2).to(dtype=tree.dtype)
        # Вычисляем приоритеты
        priorities = (errors_clamped.pow(alpha) + 1e-6).to(dtype=tree.dtype)
        # Обновляем листья дерева
        tree[idxs] = priorities
        # Подготавливаем индексы для обновления родителей
        current_indices = idxs.clone()
        while current_indices.numel() > 0:
            parent_indices = (current_indices - 1) // 2
            parent_indices = torch.unique(parent_indices)
            parent_indices = parent_indices[parent_indices >= 0]
            if parent_indices.numel() == 0:
                break  # Все элементы достигли корня
            left_indices = 2 * parent_indices + 1
            right_indices = 2 * parent_indices + 2
            valid_left = (left_indices >= 0) & (left_indices < tree.size(0))
            valid_right = (right_indices >= 0) & (right_indices < tree.size(0))
            left_values = torch.zeros(
                parent_indices.size(0), dtype=tree.dtype, device=tree.device
            )
            right_values = torch.zeros(
                parent_indices.size(0), dtype=tree.dtype, device=tree.device
            )
            if valid_left.any():
                left_values[valid_left] = tree[left_indices[valid_left]].to(
                    dtype=tree.dtype
                )
            if valid_right.any():
                right_values[valid_right] = tree[right_indices[valid_right]].to(
                    dtype=tree.dtype
                )
            parent_values = (left_values + right_values).to(dtype=tree.dtype)
            tree[parent_indices] = parent_values
            current_indices = parent_indices


@torch.jit.script
def compute_sequence_weights(
    seq_td_errors: torch.Tensor, alpha: float, epsilon: float = 1e-5
) -> torch.Tensor:
    """
    Вычисляет приоритеты для последовательностей TD-ошибок.
    Оптимизировано для минимизации операций.

    Args:
        seq_td_errors (torch.Tensor): TD-ошибки для последовательности.
        alpha (float): Коэффициент приоритета.
        epsilon (float): Малое значение для стабильности.

    Returns:
        torch.Tensor: Взвешенные приоритеты последовательностей.
    """
    with torch.no_grad():
        # Вариант 1: Максимальная TD-ошибка в последовательности
        max_td = torch.max(seq_td_errors, dim=1)[0]

        # Вариант 2: Среднее значение TD-ошибок
        # mean_td = torch.mean(seq_td_errors, dim=1)

        # Применяем степенное преобразование и добавляем эпсилон
        return max_td.pow(alpha) + epsilon


class ReplayBufferDataset(Dataset):
    """
    Адаптер для использования буферов воспроизведения с DataLoader.
    """

    def __init__(self, buffer: "BaseReplayBuffer"):
        self.buffer = buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Используем семплирование по индексу при наличии метода
        if hasattr(self.buffer, "get_item_by_idx"):
            return self.buffer.get_item_by_idx(idx)
        # Запасной вариант - простое семплирование 1 элемента
        return self.buffer.sample(1)


@dataclass
class BufferConfig:
    capacity: int
    obs_dim: Union[int, Tuple]
    action_dim: int
    device: str = "cpu"
    storage_dtype: torch.dtype = torch.float32
    use_pinned_memory: bool = True
    # Параметры приоритезации (используются в приоритетных буферах)
    alpha: Optional[float] = None
    beta: Optional[float] = None
    beta_annealing_steps: Optional[int] = None
    epsilon: float = 1e-5
    tree_dtype: Optional[torch.dtype] = torch.float32


class StorageMixin:
    """
    Миксин для общей инициализации тензорного хранилища в буферах.
    """
    def init_storage(self):
        cfg = self.config
        # Определяем форму наблюдений
        if isinstance(cfg.obs_dim, tuple):
            obs_shape = (cfg.capacity,) + cfg.obs_dim
        else:
            obs_shape = (cfg.capacity, cfg.obs_dim)
        # Общие параметры для тензоров
        tensor_kwargs = {"dtype": cfg.storage_dtype, "device": torch.device("cpu")}
        # Выбираем фабрику с поддержкой pinned memory
        target_dev = device_manager.get_device(cfg.device)
        if cfg.use_pinned_memory and target_dev.type == "cuda":
            def factory(*size, **kwargs):
                t = torch.zeros(*size, **kwargs)
                return t.pin_memory()
        else:
            factory = torch.zeros
        # Создаем хранилище
        self.obs = factory(obs_shape, **tensor_kwargs)
        self.next_obs = factory(obs_shape, **tensor_kwargs)
        self.actions = factory((cfg.capacity, cfg.action_dim), **tensor_kwargs)
        self.rewards = factory((cfg.capacity, 1), **tensor_kwargs)
        self.dones = factory((cfg.capacity, 1), **tensor_kwargs)
        # Счетчики
        self.position = 0
        self._size = 0
        # Устройства
        self.storage_device = torch.device("cpu")
        self.target_device = target_dev


class PriorityMixin:
    """
    Миксин для инициализации SumTree и параметров приоритетного семплирования.
    """
    def init_priority(self) -> None:
        """
        Настраивает SumTree и параметры приоритетного семплирования из config.
        """
        cfg = self.config
        # Извлекаем параметры приоритезации
        alpha = cfg.alpha or 0.0
        beta = cfg.beta or 0.0
        self.alpha = alpha
        self.beta = beta
        self._initial_beta = beta
        self.epsilon = cfg.epsilon
        self.beta_annealing_steps = cfg.beta_annealing_steps
        self.beta_increment = (
            (1.0 - beta) / cfg.beta_annealing_steps
            if cfg.beta_annealing_steps else 0
        )
        self.annealing_step = 0
        # Определяем тип дерева с учётом поддержки bfloat16
        self.supports_bfloat16 = check_bfloat16_support(cfg.device)
        dtype = (
            torch.bfloat16
            if cfg.tree_dtype == torch.float16 and self.supports_bfloat16
            else cfg.tree_dtype
        )
        self.tree_dtype = dtype
        # Создаём SumTree
        self.sum_tree = SumTree(
            cfg.capacity, device=cfg.device, dtype=self.tree_dtype
        )


class BaseReplayBuffer:
    """
    Абстрактный базовый класс для буферов воспроизведения.
    Определяет общий интерфейс для всех типов буферов.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: Union[int, Tuple],
        action_dim: int,
        device: str = "cpu",
        storage_dtype=torch.float32,
    ):
        """
        Инициализация базового буфера.
        """
        # Храним данные на CPU pinned, целевое устройство для выборки сохраняем отдельно
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        target_dev = device_manager.get_device(device)
        self.target_device = target_dev
        self.storage_device = torch.device("cpu")
        self.device = self.storage_device
        
        # Автоматически используем bfloat16 вместо float16, если устройство поддерживает
        if storage_dtype == torch.float16 and check_bfloat16_support(self.device):
            self.storage_dtype = torch.bfloat16
        else:
            self.storage_dtype = storage_dtype
        
        self._size = 0

    def to_storage_tensor(self, x, dtype, device):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        return torch.as_tensor(x, dtype=dtype, device=device)

    def add(
        self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool
    ) -> None:
        """
        Добавляет новый переход в буфер: конвертация данных, запись в storage, обновление счетчиков.
        """
        idx = getattr(self, 'position', 0) % self.capacity
        obs = self.to_storage_tensor(obs, self.storage_dtype, self.storage_device)
        action = self.to_storage_tensor(action, self.storage_dtype, self.storage_device)
        next_obs = self.to_storage_tensor(next_obs, self.storage_dtype, self.storage_device)
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)
        self.position = (idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Сэмплирует один батч и переводит его на целевое устройство.
        """
        assert self._size > 0, "Буфер пуст!"
        # Индексы на CPU
        indices = torch.randint(0, self._size, (batch_size,), device=self.storage_device)
        # Срезы из хранения (CPU pinned)
        obs_batch = self.obs[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        next_obs_batch = self.next_obs[indices]
        dones_batch = self.dones[indices]
        # Собираем словарь
        batch = {
            "obs": obs_batch,
            "actions": actions_batch,
            "rewards": rewards_batch,
            "next_obs": next_obs_batch,
            "dones": dones_batch,
        }
        # Асинхронная передача на устройство через менеджер устройств
        if self.target_device != self.storage_device:
            batch = device_manager.async_data_transfer(
                batch,
                source_device=self.storage_device,
                target_device=self.target_device
            )
        return batch

    def can_sample(self, batch_size: int) -> bool:
        """
        Проверяет, достаточно ли данных для семплирования.
        """
        return self._size >= batch_size

    def __len__(self) -> int:
        """
        Возвращает текущее количество переходов в буфере.
        """
        return self._size

    def to_device(self, device: Union[str, torch.device]) -> "BaseReplayBuffer":
        """
        Обновляет целевое устройство для выборки. Хранилище остаётся на CPU.
        """
        target_dev = device_manager.get_device(device)
        self.target_device = target_dev
        return self

    def is_empty(self) -> bool:
        """
        Проверяет, пуст ли буфер.
        """
        return self._size == 0

    def clear(self) -> None:
        """
        Очищает буфер, сбрасывая счетчики и пересоздавая хранилище.
        """
        # Переинициализация storage через StorageMixin, если есть
        if hasattr(self, 'init_storage'):
            self.init_storage()
        else:
            # сбрасываем счетчики
            self._size = 0
            self.position = 0


# --- Стандартный буфер воспроизведения (FIFO) --- #
class ReplayBuffer(BaseReplayBuffer, StorageMixin):
    """
    Реализация стандартного циклического буфера FIFO (First-In-First-Out) для воспроизведения.
    Хранит данные в тензорах и позволяет быстро выполнять семплирование.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: Union[int, Tuple],
        action_dim: int,
        device: str = "cpu",
        storage_dtype=torch.float32,
        use_pinned_memory: bool = True
    ):
        """
        Инициализирует буфер воспроизведения с заданной вместимостью.

        Args:
            capacity (int): Максимальное количество хранимых переходов.
            obs_dim (Union[int, Tuple]): Размерность наблюдений.
            action_dim (int): Размерность действий.
            device (str): Устройство ('cpu' или 'cuda').
            storage_dtype (torch.dtype): Тип данных для хранения. 
            use_pinned_memory (bool): Использовать pinned memory.
        """
        # Формируем конфиг и инициализируем хранилище через миксин
        cfg = BufferConfig(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            storage_dtype=storage_dtype,
            use_pinned_memory=use_pinned_memory
        )
        self.config = cfg
        super().__init__(capacity, obs_dim, action_dim, device, storage_dtype)
        # Создаем тензорное хранилище
        self.init_storage()
        logger.info(
            f"Создан ReplayBuffer: capacity={cfg.capacity}, dtype={cfg.storage_dtype}, "
            f"storage_device={self.storage_device}, target_device={self.target_device}, pinned_memory={cfg.use_pinned_memory}"
        )

    # Использует базовый add() и sample()
    def clear(self) -> None:
        super().clear()


# --- Приоритезированный буфер воспроизведения (PER) --- #
class PrioritizedReplayBuffer(ReplayBuffer, PriorityMixin):
    """
    Буфер с приоритетным семплированием, использующий SumTree.
    Поддерживает векторизованные операции и коррекцию смещения IS.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: Union[int, Tuple],
        action_dim: int,
        alpha: float = 0.6,  # Коэффициент приоритезации (0: равномерно, 1: полностью по приоритету)
        beta: float = 0.4,  # Начальный коэффициент коррекции IS (0 -> 1)
        beta_annealing_steps: Optional[
            int
        ] = None,  # Количество шагов для отжига beta до 1.0
        epsilon: float = 1e-5,  # Малая добавка к приоритету для ненулевой вероятности выбора
        device: str = "cpu",
        storage_dtype=torch.float32,
        tree_dtype: torch.dtype = torch.float32,
        use_pinned_memory: bool = True
    ):
        """
        Инициализирует буфер с приоритетным семплированием.
        
        Args:
            capacity (int): Максимальное количество хранимых переходов.
            obs_dim (Union[int, Tuple]): Размерность наблюдений.
            action_dim (int): Размерность действий.
            alpha (float): Коэффициент приоритезации (0: равномерно, 1: полностью по приоритету).
            beta (float): Начальный коэффициент коррекции IS (0 -> 1).
            beta_annealing_steps (Optional[int]): Количество шагов для отжига beta до 1.0.
            epsilon (float): Малая добавка к приоритету для ненулевой вероятности выбора.
            device (str): Устройство ('cpu' или 'cuda').
            storage_dtype (torch.dtype): Тип данных для хранения основных тензоров буфера.
            tree_dtype (torch.dtype): Тип данных для дерева приоритетов. 
            use_pinned_memory (bool): Использовать pinned memory для оптимизации передачи данных между CPU и GPU.
        """
        # Конфигурация и инициализация хранилища
        cfg = BufferConfig(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            storage_dtype=storage_dtype,
            use_pinned_memory=use_pinned_memory,
            tree_dtype=tree_dtype,
            alpha=alpha,
            beta=beta,
            beta_annealing_steps=beta_annealing_steps,
            epsilon=epsilon,
        )
        self.config = cfg
        super().__init__(capacity, obs_dim, action_dim, device, storage_dtype, use_pinned_memory)
        # Инициализируем приоритетное дерево и параметры
        self.init_priority()
        logger.info(
            f"Создан PrioritizedReplayBuffer: capacity={cfg.capacity}, storage_dtype={cfg.storage_dtype}, "
            f"tree_dtype={self.tree_dtype}, device={self.target_device}, pinned_memory={cfg.use_pinned_memory}, "
            f"alpha={alpha}, beta={beta}"
        )

    def add(
        self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool
    ) -> None:
        """
        Добавляет переход и обновляет приоритет в дереве.
        """
        # Основное сохранение данных
        super().add(obs, action, reward, next_obs, done)
        # Индекс в дереве: leaf offset + position-1
        leaf_idx = (self.position - 1) % self.capacity + self.sum_tree.capacity - 1
        # Обновляем приоритет
        self.sum_tree.update(leaf_idx, self.sum_tree.get_max_priority())

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Сэмплирует батч с приоритетами и переводит его на целевое устройство.
        """
        assert self._size > 0, "Буфер пуст!"
        # Получаем индексы и приоритеты из дерева
        tree_indices, priorities, indices = self.sum_tree.get_batch(batch_size)
        # Переносим индексы на устройство хранения (CPU) перед индексированием тензоров
        indices = indices.to(self.storage_device)
        # Срезы из хранения (CPU pinned)
        obs_batch = self.obs[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        next_obs_batch = self.next_obs[indices]
        dones_batch = self.dones[indices]
        # Вычисляем IS-веса
        total_sum = self.sum_tree.sum()
        with torch.no_grad():
            is_weights = compute_is_weights(priorities, total_sum, self.beta, batch_size).unsqueeze(1)
        # Составляем словарь батча
        batch = {
            "obs": obs_batch,
            "actions": actions_batch,
            "rewards": rewards_batch,
            "next_obs": next_obs_batch,
            "dones": dones_batch,
            "tree_indices": tree_indices,
            "is_weights": is_weights,
        }
        # Асинхронная передача на устройство через менеджер устройств
        if self.target_device != self.storage_device:
            batch = device_manager.async_data_transfer(
                batch,
                source_device=self.storage_device,
                target_device=self.target_device
            )
        return batch

    def update_priorities(
        self,
        tree_indices: torch.Tensor,
        td_errors: torch.Tensor,
        min_priority: float = 1e-6,
        max_priority: float = 1e2,
        dynamic_alpha_fn=None,
    ) -> None:
        """
        Обновляет приоритеты на основе TD-ошибок.
        """
        # Преобразуем TD-ошибки к плоскому тензору при необходимости
        if td_errors.dim() > 1:
            td_errors = td_errors.view(-1)

        # Используем асинхронные операции на GPU
        if (
            self.device.type == "cuda"
            and hasattr(self.sum_tree, "_cuda_stream")
            and self.sum_tree._cuda_stream is not None
        ):
            with torch.cuda.stream(self.sum_tree._cuda_stream):
                # Используем dynamic_alpha_fn, если передана, иначе self.alpha
                alpha = (
                    dynamic_alpha_fn(td_errors)
                    if dynamic_alpha_fn is not None
                    else self.alpha
                )

                # Оптимизированное обновление приоритетов
                update_priorities_batch(
                    self.sum_tree.tree, tree_indices, td_errors, alpha
                )

            # Нет необходимости в синхронизации здесь, продолжаем выполнение
        else:
            # Для CPU используем стандартное обновление
            alpha = (
                dynamic_alpha_fn(td_errors)
                if dynamic_alpha_fn is not None
                else self.alpha
            )
            update_priorities_batch(self.sum_tree.tree, tree_indices, td_errors, alpha)

    def _update_beta(self) -> None:
        """
        Обновляет параметр beta согласно схеме отжига.
        """
        if (
            self.beta_annealing_steps
            and self.annealing_step < self.beta_annealing_steps
        ):
            self.beta = min(
                1.0, self._initial_beta + self.beta_increment * self.annealing_step
            )
            self.annealing_step += 1

    def clear(self) -> None:
        """
        Очищает буфер и сбрасывает приоритеты.
        """
        super().clear()
        self.sum_tree.clear()
        self.annealing_step = 0
        self.beta = self._initial_beta


# --- Буфер воспроизведения для последовательностей (для RNN) --- #
class SequenceReplayBuffer(BaseReplayBuffer, StorageMixin):
    """
    Буфер воспроизведения, предназначенный для хранения и семплирования последовательностей.
    Особенно полезен при обучении рекуррентными нейросетями (RNN, LSTM), где требуется контекст предыдущих наблюдений.

    Attributes:
        sequence_length (int): Длина последовательности, которую нужно возвращать при семплировании.
        current_episode (Dict[str, List[torch.Tensor]]): Текущий эпизод, который ещё не завершён.
        episodes (List[Dict[str, torch.Tensor]]): Список завершённых эпизодов.
        episode_lengths (List[int]): Длины сохранённых эпизодов.
        _current_episode_len (int): Текущая длина текущего эпизода.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: Union[int, Tuple],
        action_dim: int,
        sequence_length: int = 10,
        device: str = "cpu",
        storage_dtype: torch.dtype = torch.float32,
        use_pinned_memory: bool = True,
        tree_dtype: torch.dtype = torch.float32,
    ):
        """
        Инициализирует буфер последовательностей (хранение на CPU pinned) с заданной длиной sequence_length.
        """
        # Конфигурация и инициализация хранилища
        cfg = BufferConfig(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            storage_dtype=storage_dtype,
            use_pinned_memory=use_pinned_memory,
            tree_dtype=tree_dtype,
        )
        self.config = cfg
        # Инициализация sequence-хранилища
        super().__init__(capacity, obs_dim, action_dim, device, storage_dtype)
        self.init_storage()
        logger.info(
            f"Создан SequenceReplayBuffer: capacity={cfg.capacity}, sequence_length={sequence_length}, "
            f"storage_dtype={cfg.storage_dtype}, tree_dtype={tree_dtype}, "
            f"device={self.target_device}, pinned_memory={cfg.use_pinned_memory}"
        )
        # Параметры последовательностей
        self.sequence_length = sequence_length
        self._indices_buffer = None

    def add(
        self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool
    ) -> None:
        """
        Добавляет новую транзицию в текущий эпизод. Если эпизод завершён, добавляет его в список завершённых.

        Args:
            obs (Any): Текущее наблюдение.
            action (Any): Выполненное действие.
            reward (float): Полученная награда.
            next_obs (Any): Новое наблюдение после действия.
            done (bool): Флаг завершения эпизода.
        """
        idx = self.position

        # Эффективное преобразование к тензорам
        obs = self.to_storage_tensor(obs, self.storage_dtype, self.device)
        action = self.to_storage_tensor(action, self.storage_dtype, self.device)
        next_obs = self.to_storage_tensor(next_obs, self.storage_dtype, self.device)

        # Запись данных
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)

        # Обновление позиции и размера
        self.position = (self.position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

        # Если это конец эпизода, сбрасываем позицию для обеспечения непрерывности последовательностей
        if done:
            self.position = 0

    def can_sample(self, batch_size: int) -> bool:
        """
        Проверяет, достаточно ли данных для семплирования.
        """
        return self._size >= self.sequence_length and self._size >= batch_size

    def __len__(self) -> int:
        """
        Возвращает текущее количество переходов в буфере.
        """
        return self._size

    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Сэмплирует batch_size последовательностей длины sequence_length и переводит на target_device.
        Теперь реализовано батчево без цикла for.
        """
        if not self.can_sample(batch_size):
            return None
        max_start = self._size - self.sequence_length
        if self._indices_buffer is None or self._indices_buffer.size(0) < batch_size:
            self._indices_buffer = torch.zeros(batch_size, dtype=torch.int64, device=self.storage_device)
        torch.randint(0, max_start + 1, (batch_size,), device=self.storage_device, out=self._indices_buffer[:batch_size])
        starts = self._indices_buffer[:batch_size]
        # Индексы для батчевого копирования
        idx = starts.unsqueeze(1) + torch.arange(self.sequence_length, device=self.storage_device).unsqueeze(0)  # [batch, seq]
        # Копируем батчом
        seq_obs = self.obs[idx]
        seq_next = self.next_obs[idx]
        seq_act = self.actions[idx]
        seq_rew = self.rewards[idx]
        seq_dne = self.dones[idx]
        batch = {
            "obs": seq_obs,
            "actions": seq_act,
            "rewards": seq_rew,
            "next_obs": seq_next,
            "dones": seq_dne,
        }
        if self.target_device != self.storage_device:
            batch = device_manager.async_data_transfer(
                batch,
                source_device=self.storage_device,
                target_device=self.target_device
            )
        return batch

    def update_priorities(
        self,
        tree_indices: torch.Tensor,
        sequence_td_errors: torch.Tensor,
        min_priority: float = 1e-6,
        max_priority: float = 1e2,
        use_softmax_priority: bool = True,
        dynamic_alpha_fn=None,
    ) -> None:
        """
        Обновляет приоритеты в дереве для заданных индексов на основе ошибки TD или другого критерия.

        Args:
            tree_indices (torch.Tensor): Индексы в дереве, которым соответствуют обновляемые последовательности.
            priorities (torch.Tensor): Новые значения приоритетов.
        """
        batch_size = tree_indices.size(0)
        if not isinstance(sequence_td_errors, torch.Tensor):
            sequence_td_errors = torch.tensor(
                sequence_td_errors, device=self.device, dtype=self.tree_dtype
            )
        elif sequence_td_errors.device != self.device:
            sequence_td_errors = sequence_td_errors.to(
                device=self.device, dtype=self.tree_dtype
            )
        alpha = (
            dynamic_alpha_fn(sequence_td_errors)
            if dynamic_alpha_fn is not None
            else getattr(self, "alpha", 0.6)
        )
        if len(sequence_td_errors.shape) > 1:
            if use_softmax_priority:
                weights = torch.softmax(sequence_td_errors, dim=1)
                td_errors = (weights * sequence_td_errors).sum(dim=1)
            else:
                td_errors = compute_sequence_weights(
                    sequence_td_errors, alpha, self.epsilon
                )
        else:
            td_errors = sequence_td_errors
        td_errors = torch.clamp(td_errors.abs(), min=min_priority, max=max_priority)

        # Векторизованное обновление приоритетов
        self.sum_tree.update_batch(tree_indices, td_errors)

    def _update_beta(self) -> None:
        """
        Обновляет параметр beta согласно схеме отжига.
        """
        if (
            self.beta_annealing_steps
            and self.annealing_step < self.beta_annealing_steps
        ):
            self.beta = min(
                1.0, self._initial_beta + self.beta_increment * self.annealing_step
            )
            self.annealing_step += 1

    def clear(self) -> None:
        """
        Очищает буфер последовательностей, сбрасывая storage и индексы.
        """
        super().clear()
        # Сбрасываем буфер случайных стартов
        self._indices_buffer = None


# --- Приоритезированный буфер для последовательностей --- #
class PrioritizedSequenceReplayBuffer(SequenceReplayBuffer, PriorityMixin):
    """
    Приоритезированный буфер последовательностей.

    Attributes:
        alpha (float): Параметр регулирования силой приоритезации (0 — равномерное семплирование, 1 — максимум приоритета).
        beta (float): Начальное значение коэффициента Importance Sampling для компенсации смещения.
        _initial_beta (float): Сохранённое начальное значение beta для отжига.
        epsilon (float): Малое значение, добавляемое к приоритетам, чтобы избежать нулевой вероятности выборки.
        beta_annealing_steps (Optional[int]): Число шагов для постепенного роста beta до 1.0.
        beta_increment (float): Шаг увеличения beta при каждом обновлении.
        annealing_step (int): Счётчик шагов для отжига beta.
        tree_dtype (torch.dtype): Тип данных дерева приоритетов (например, torch.float32 или torch.bfloat16).
        sum_tree (SumTree): Дерево сумм для хранения приоритетов.
        _mapping_tensor (torch.Tensor): Тензор для отображения (ep_idx, start_pos) -> tree_idx.
        _active_indices_mask (torch.Tensor): Маска активных индексов в mapping_tensor.
        _reverse_mapping (torch.Tensor): Обратное отображение tree_idx -> индекс в mapping_tensor.
        _is_weights_buffer (torch.Tensor): Предварительно выделённый буфер для весов IS.
        _cached_batch (Dict[str, Optional[torch.Tensor]]): Кэшированный батч для повышения производительности.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: Union[int, Tuple],
        action_dim: int,
        sequence_length: int = 10,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing_steps: Optional[int] = None,
        epsilon: float = 1e-5,
        device: str = "cuda",
        storage_dtype: torch.dtype = torch.float32,
        use_amp: Optional[bool] = None,
        tree_dtype: torch.dtype = torch.float32,
        use_pinned_memory: bool = True,
    ):
        """
        Инициализирует приоритезированный буфер последовательностей.

        Args:
            capacity (int): Максимальное количество хранимых переходов.
            obs_dim (Union[int, Tuple]): Размерность наблюдений (скаляр или кортеж).
            action_dim (int): Размерность действия.
            sequence_length (int): Длина последовательностей, используемых для обучения.
            alpha (float): Параметр приоритезации: 0 — равномерное семплирование, 1 — максимальный приоритет.
            beta (float): Начальное значение коэффициента importance sampling.
            beta_annealing_steps (Optional[int]): Количество шагов, за которое beta возрастает до 1.0.
            epsilon (float): Малое число, добавляемое к приоритетам для устойчивости.
            device (str): Устройство, на котором хранятся данные ("cpu" или "cuda").
            storage_dtype (torch.dtype): Тип хранения тензоров (по умолчанию float32).
            use_amp (Optional[bool]): Включает автоматическое смешивание точности (AMP), если доступно.
            tree_dtype (torch.dtype): Тип данных, используемый в дереве приоритетов.
            use_pinned_memory (bool): Использовать pinned memory.
        """
        # Конфигурация и инициализация хранилища
        cfg = BufferConfig(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            storage_dtype=storage_dtype,
            use_pinned_memory=use_pinned_memory,
            tree_dtype=tree_dtype,
            alpha=alpha,
            beta=beta,
            beta_annealing_steps=beta_annealing_steps,
            epsilon=epsilon,
        )
        self.config = cfg
        # Инициализация sequence-хранилища
        super().__init__(capacity, obs_dim, action_dim, sequence_length, device, storage_dtype, use_pinned_memory)
        # Инициализируем приоритетное дерево и параметры
        self.init_priority()
        logger.info(
            f"Создан PrioritizedSequenceReplayBuffer: capacity={cfg.capacity}, sequence_length={sequence_length}, "
            f"storage_dtype={cfg.storage_dtype}, tree_dtype={self.tree_dtype}, "
            f"device={self.target_device}, pinned_memory={cfg.use_pinned_memory}, "
            f"alpha={alpha}, beta={beta}"
        )
        
        self._mapping_device = torch.device('cpu')
        
        self._max_valid_starts = capacity
        self._valid_start_count = 0
        
        # Матрица для отображения (ep_idx, start_pos) -> tree_idx
        self._mapping_tensor = torch.full(
            (self._max_valid_starts, 3), -1, 
            dtype=torch.int64, device=self._mapping_device
        )
        
        # Обратное отображение tree_idx -> индекс в _mapping_tensor
        self._reverse_mapping = torch.full(
            (self.capacity,), -1, 
            dtype=torch.int64, device=self._mapping_device
        )
        
        # Маска активных (валидных) индексов
        self._active_indices_mask = torch.zeros(
            self._max_valid_starts, 
            dtype=torch.bool, device=self._mapping_device
        )
        
        self._next_tree_idx = 0

    def _add_valid_start_to_tree(self, ep_idx: int, start_pos: int) -> None:
        """
        Добавляет допустимую стартовую позицию последовательности в дерево приоритетов
        с максимальным приоритетом.

        Args:
            ep_idx (int): Индекс эпизода (внутренний счётчик).
            start_pos (int): Позиция начала последовательности в эпизоде.
        """
        # Получаем максимальный приоритет из дерева (или 1.0 если пусто)
        max_priority = self.sum_tree.get_max_priority()

        # Находим свободный индекс в дереве
        data_idx = self._next_tree_idx
        self._next_tree_idx = (self._next_tree_idx + 1) % self.sum_tree.capacity

        # Если перезаписываем старый индекс, очищаем старые отображения
        if self._reverse_mapping[data_idx] >= 0:
            mapping_idx = self._reverse_mapping[data_idx].item()
            # Сбрасываем значения
            self._mapping_tensor[mapping_idx, 0] = -1
            self._mapping_tensor[mapping_idx, 1] = -1
            self._mapping_tensor[mapping_idx, 2] = -1
            self._active_indices_mask[mapping_idx] = False
            self._reverse_mapping[data_idx] = -1
        
        # Ищем первый неактивный индекс
        free_indices = torch.nonzero(~self._active_indices_mask, as_tuple=True)[0]
        if free_indices.numel() > 0:
            # Используем первый свободный индекс
            mapping_idx = free_indices[0].item()
        else:
            # Если нет свободных индексов, увеличиваем счетчик
            if self._valid_start_count < self._max_valid_starts:
                mapping_idx = self._valid_start_count
                self._valid_start_count += 1
            else:
                mapping_idx = 0
                # Сбрасываем связанные данные
                old_tree_idx = self._mapping_tensor[mapping_idx, 2].item()
                if old_tree_idx >= 0:
                    self._reverse_mapping[old_tree_idx] = -1
        
        # Устанавливаем новые отображения
        self._mapping_tensor[mapping_idx, 0] = ep_idx
        self._mapping_tensor[mapping_idx, 1] = start_pos
        self._mapping_tensor[mapping_idx, 2] = data_idx
        self._reverse_mapping[data_idx] = mapping_idx
        self._active_indices_mask[mapping_idx] = True

        # Добавляем новую точку с максимальным приоритетом
        self.sum_tree.update(data_idx, max_priority)

    def _remove_valid_starts_from_tree(self, removed_episode_idx: int) -> None:
        """
        Удаляет из дерева приоритетов все стартовые позиции, связанные с указанным эпизодом.

        Используется при переполнении буфера или переиспользовании места в циклической памяти.

        Args:
            removed_episode_idx (int): Индекс эпизода, чьи последовательности нужно удалить.
        """
        # Находим все записи для указанного эпизода
        episode_mask = (self._mapping_tensor[:, 0] == removed_episode_idx) & self._active_indices_mask
        if not episode_mask.any():
            return  
        
        # Получаем индексы в дереве для записей этого эпизода
        tree_indices = self._mapping_tensor[episode_mask, 2]
        
        # Обнуляем приоритеты в дереве векторизованно
        if tree_indices.numel() > 0:
            for tree_idx in tree_indices:
                if tree_idx >= 0:
                    self.sum_tree.update(tree_idx.item(), 0.0)
                    # Очищаем обратное отображение
                    self._reverse_mapping[tree_idx] = -1
        
        # Очищаем данные в тензоре отображений
        self._mapping_tensor[episode_mask, 0] = -1
        self._mapping_tensor[episode_mask, 1] = -1
        self._mapping_tensor[episode_mask, 2] = -1
        self._active_indices_mask[episode_mask] = False

    def can_sample(self, batch_size: int) -> bool:
        """
        Проверяет, достаточно ли валидных стартовых точек в дереве для семплирования.
        """
        if batch_size <= 0:
            return False

        # Проверяем, что есть достаточно данных для семплирования
        active_count = self._active_indices_mask.sum().item()
        return active_count >= batch_size and self.sum_tree.sum() > 0

    def __len__(self) -> int:
        """
        Возвращает количество валидных стартовых точек.
        """
        return int(self._active_indices_mask.sum().item())

    def clear(self) -> None:
        """
        Очищает буфер, дерево сумм и все отображения.
        """
        super().clear()
        self.sum_tree.clear()
        self._mapping_tensor.fill_(-1)
        self._reverse_mapping.fill_(-1)
        self._active_indices_mask.fill_(False)
        self._valid_start_count = 0
        self._next_tree_idx = 0

    def add(self, obs, action, reward, next_obs, done):
        super().add(obs, action, reward, next_obs, done)
        # Автоматически добавляем валидный старт, если достаточно данных для последовательности
        if self._size >= self.sequence_length:
            start_pos = (self.position - self.sequence_length) % self.capacity
            self._add_valid_start_to_tree(0, start_pos)
