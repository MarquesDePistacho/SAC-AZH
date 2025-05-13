import torch
import operator
from typing import Callable, Union, Tuple

from core.utils.device_utils import device_manager
from core.logging.logger import get_logger

logger = get_logger("segment_tree")


# --- JIT-функции для работы с деревом --- #
@torch.jit.script
def update_tree_vectorized(
    tree: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    min_priority: float = 1e-6,
    max_priority: float = 1e2,
) -> None:
    """
    Векторизованное обновление нескольких узлов дерева сумм с клиповкой приоритетов.
    Значения клиппируются в диапазоне [min_priority, max_priority] и приводятся к типу дерева.

    Args:
        tree (torch.Tensor): Тензор дерева, содержащий значения приоритетов.
        indices (torch.Tensor): Индексы листьев, которые нужно обновить.
        values (torch.Tensor): Новые значения для индексов.
        min_priority (float): Минимально допустимое значение приоритета.
        max_priority (float): Максимально допустимое значение приоритета.
    """
    if tree.size(0) == 0 or indices.size(0) == 0:
        return

    # Проверяем валидность индексов
    valid_mask = (indices >= 0) & (indices < tree.size(0))
    if not valid_mask.all():
        indices = indices[valid_mask]
        values = values[valid_mask]
        if indices.size(0) == 0:
            return

    # Клипуем значения и приводим к типу дерева
    values = torch.clamp(values, min=min_priority, max=max_priority).to(
        dtype=tree.dtype
    )

    # Эффективное обновление с использованием index_copy_ вместо индексирования по одному элементу
    tree.index_copy_(0, indices, values)

    # Вычисляем высоту дерева один раз
    LOG2 = torch.log(torch.tensor(2.0, dtype=torch.float32, device=tree.device))
    height = int(
        torch.log(
            torch.tensor(tree.size(0) + 1, dtype=torch.float32, device=tree.device)
        )
        / LOG2
    )

    # Создаем словарь для хранения индексов по уровням
    level_indices = {height - 1: indices}

    for level in range(height - 1, 0, -1):
        start_idx = int(2**level - 1)
        end_idx = int(2 ** (level + 1) - 1)

        if start_idx >= tree.size(0) or start_idx < 0:
            continue

        # Получаем индексы текущего уровня
        if level not in level_indices:
            continue

        current_indices = level_indices[level]
        if current_indices.size(0) == 0:
            continue

        # Вычисляем индексы родителей текущего уровня
        parent_indices = (current_indices - 1) // 2

        # Удаляем дубликаты для предотвращения конфликтов записи и сортируем для оптимизации доступа к памяти
        parent_indices = torch.unique(parent_indices, sorted=True)
        level_indices[level - 1] = parent_indices

        # Обновляем значения родителей
        for parent_idx in parent_indices:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1

            # Проверяем валидность индексов
            if left_idx < tree.size(0):
                left_val = tree[left_idx]
                if right_idx < tree.size(0):
                    # Если оба потомка существуют
                    right_val = tree[right_idx]
                    tree[parent_idx] = (left_val + right_val).to(dtype=tree.dtype)
                else:
                    # Если только левый потомок существует
                    tree[parent_idx] = left_val.to(dtype=tree.dtype)


@torch.jit.script
def find_prefixsum_idx_vectorized(
    tree: torch.Tensor, prefixsums: torch.Tensor, capacity: int
) -> torch.Tensor:
    """
    Векторизованный поиск индексов по префиксным суммам.

    Args:
        tree (torch.Tensor): Дерево приоритетов.
        prefixsums (torch.Tensor): Префиксные суммы для поиска в дереве.
        capacity (int): Емкость дерева (число листьев).

    Returns:
        torch.Tensor: Индексы в массиве данных, соответствующие префиксным суммам.
    """
    batch_size = prefixsums.size(0)
    indices = torch.zeros(batch_size, dtype=torch.int64, device=tree.device)

    # Эффективное вычисление высоты дерева один раз
    LOG2 = torch.log(torch.tensor(2.0, dtype=torch.float32, device=tree.device))
    height = int(
        torch.log(
            torch.tensor(float(capacity), dtype=torch.float32, device=tree.device)
        )
        / LOG2
    )
    height = max(height, 0)

    # Предварительно выделяем буферы для промежуточных результатов
    left_children = torch.zeros_like(indices)
    left_values = torch.zeros(batch_size, dtype=tree.dtype, device=tree.device)
    valid_indices = torch.zeros(batch_size, dtype=torch.bool, device=tree.device)
    go_right = torch.zeros(batch_size, dtype=torch.bool, device=tree.device)

    for level in range(height):
        # Вычисляем индексы левых потомков (2*idx + 1)
        left_children = 2 * indices + 1

        # Находим валидные индексы (в пределах дерева)
        valid_indices = left_children < tree.size(0)

        # Сбрасываем значения в ноль
        left_values.zero_()

        # Получаем значения левых потомков для валидных индексов
        if valid_indices.any():
            valid_left = left_children[valid_indices]
            left_values[valid_indices] = tree[valid_left].to(dtype=tree.dtype)

        # Определяем, нужно ли идти вправо
        go_right = prefixsums > left_values.to(dtype=prefixsums.dtype)

        # Обновляем префиксные суммы и индексы
        prefixsums = torch.where(
            go_right, prefixsums - left_values.to(dtype=prefixsums.dtype), prefixsums
        )
        indices = torch.where(go_right, left_children + 1, left_children)

    # Преобразуем индексы дерева в индексы данных
    result = indices - (capacity - 1)

    # Безопасно клиппуем результат
    result = torch.clamp(result, 0, capacity - 1)

    return result


class SegmentTree:
    """
    Реализация дерева отрезков для операций редукции.
    Использует эффективную векторную реализацию на основе тензоров PyTorch.

    Attributes:
        capacity (int): Число элементов данных (листовых узлов).
        operation (Callable): Бинарная операция для агрегации (например, сложение или минимум).
        neutral_element (float): Нейтральный элемент для операции.
        device (torch.device): Устройство, на котором хранится дерево.
        tree (torch.Tensor): Тензор, представляющий структуру дерева.
    """

    def __init__(
        self,
        capacity: int,
        operation: Union[Callable, str],
        neutral_element: float,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Инициализирует дерево отрезков с оптимизацией памяти и вычислений.

        Args:
            capacity (int): Максимальное количество хранимых элементов.
            operation (Union[Callable, str]): Операция над элементами ('sum', 'min' или пользовательская функция).
            neutral_element (float): Нейтральное значение для операции.
            device (Union[str, torch.device]): Устройство для хранения дерева.
            dtype (torch.dtype): Тип данных для значений дерева.
        """
        self.capacity = capacity
        if isinstance(operation, str):
            if operation == "sum":
                self.operation = operator.add
            elif operation == "min":
                self.operation = min
            else:
                raise ValueError(f"Неподдерживаемая строковая операция: {operation}")
        else:
            self.operation = operation
        self.neutral_element = neutral_element
        self.device = device_manager.get_device(device)
        # Дерево минимального размера: 2*capacity-1
        self.tree = torch.full(
            (2 * self.capacity - 1,), neutral_element, dtype=dtype, device=self.device
        )
        logger.info(
            f"Создано {self.__class__.__name__}: capacity={self.capacity}, operation={self.operation.__name__}, device={self.device}, dtype={dtype}"
        )

    def update(self, idx: int, value: float) -> None:
        """Обновляет значение в заданной позиции.

        Args:
            idx (int): Позиция
            value (float): Значение
        """
        if not (0 <= idx < self.capacity):
            return
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            left = 2 * tree_idx + 1
            right = left + 1
            left_val = (
                self.tree[left] if left < self.tree.size(0) else self.neutral_element
            )
            right_val = (
                self.tree[right] if right < self.tree.size(0) else self.neutral_element
            )
            self.tree[tree_idx] = self.operation(left_val, right_val)

    def get(self, idx: int) -> float:
        """Получает значение элемента по индексу.

        Args:
            idx (int): Индекс элемента

        Returns:
            float: Значение элемента
        """
        tree_idx = idx + self.capacity - 1
        return self.tree[tree_idx].item()

    def get_all_values(self) -> torch.Tensor:
        """Возвращает все значения из листьев дерева.

        Returns:
            torch.Tensor: Значения листов
        """
        return self.tree[self.capacity - 1 : 2 * self.capacity - 1]

    def total(self) -> float:
        """Возвращает общую сумму/значение корня дерева.

        Returns:
            float: Общая сумму/значение корня дерева.
        """
        if self.tree.size(0) == 0 or not torch.isfinite(self.tree[0]):
            logger.warning(
                "Дерево содержит невалидные значения, возвращаем нейтральный элемент"
            )
            return float(self.neutral_element)
        try:
            return self.tree[0].item()
        except RuntimeError as e:
            logger.error(f"Ошибка при получении total(): {e}")
            return float(self.neutral_element)

    def to(self, device: Union[str, torch.device]) -> "SegmentTree":
        """Перемещает дерево на другое устройство.

        Args:
            device (Union[str, torch.device]): Устройство, на которое нужно переместить дерево

        Returns:
            SegmentTree: Сегментное дерево
        """
        target_device = device_manager.get_device(device)
        if target_device == self.device:
            return self
        self.tree = self.tree.to(target_device)
        self.device = target_device
        return self

    def clear(self) -> None:
        """Очищает дерево, заполняя его нейтральными элементами."""
        self.tree.fill_(self.neutral_element)


class SumTree(SegmentTree):
    """
    Оптимизированное дерево сумм для приоритетного семплирования.
    Использует векторизованные и параллельные операции для ускорения работы на GPU.

    Attributes:
        _cached_batch_size (Optional[int]): Размер последнего батча для кэширования.
        _cached_random_offsets (Optional[torch.Tensor]): Кэшированные случайные смещения для семплирования.
        _cached_prefixsums (Optional[torch.Tensor]): Кэшированные префиксные суммы.
        _cuda_stream (Optional[torch.cuda.Stream]): CUDA-поток для асинхронных операций.
        _max_update_batch_size (int): Максимальный размер батча для обновления.
        _update_parent_indices_buffer (torch.Tensor): Буфер для индексов родителей при обновлении.
    """

    def __init__(
        self,
        capacity: int,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Инициализирует дерево сумм и заполняет его начальными значениями.

        Args:
            capacity (int): Емкость дерева (максимальное число элементов).
            device (Union[str, torch.device]): Устройство, на котором будет работать дерево.
            dtype (torch.dtype): Тип данных для значений дерева.
        """
        super().__init__(capacity, operator.add, 0.0, device, dtype)
        self.initialize()

        # Кэшируем буферы для повторного использования в get_batch
        self._cached_batch_size = None
        self._cached_random_offsets = None
        self._cached_prefixsums = None

        # Создаем CUDA-поток для асинхронных операций (если на GPU)
        self._cuda_stream = None
        if self.device.type == "cuda":
            try:
                self._cuda_stream = torch.cuda.Stream(device=self.device)
                logger.info(
                    f"Создан CUDA-поток для асинхронных операций на {self.device}"
                )
            except Exception as e:
                logger.warning(f"Не удалось создать CUDA-поток: {e}")

        # Буферы для обновления приоритетов (предварительно выделяем)
        self._max_update_batch_size = 1024
        self._update_parent_indices_buffer = torch.zeros(
            self._max_update_batch_size, dtype=torch.int64, device=self.device
        )

    def initialize(self, value: float = 1.0) -> None:
        """
        Инициализирует листья одинаковым значением для равномерного семплирования.

        Args:
            value (float): Значение, которым будут заполнены листья дерева.
        """
        if self.capacity <= 0:
            logger.warning("Попытка инициализировать дерево с capacity <= 0")
            return
        # Проверяем размер дерева
        expected_size = 2 * self.capacity - 1
        if self.tree.size(0) != expected_size:
            logger.warning(
                f"Размер дерева ({self.tree.size(0)}) не соответствует ожидаемому ({expected_size}), переинициализация"
            )
            self.tree = torch.full(
                (expected_size,),
                self.neutral_element,
                dtype=self.tree.dtype,
                device=self.device,
            )

        # Инициализация листьев с использованием векторизованных операций
        leaf_start = self.capacity - 1
        leaf_end = 2 * self.capacity - 1

        # Заполняем листья значением
        self.tree[leaf_start:leaf_end] = value

        # Обновляем дерево снизу вверх
        for idx in range(self.capacity - 2, -1, -1):
            left = 2 * idx + 1
            right = left + 1
            if left < self.tree.size(0) and right < self.tree.size(0):
                self.tree[idx] = self.tree[left] + self.tree[right]
            elif left < self.tree.size(0):
                self.tree[idx] = self.tree[left]
            else:
                self.tree[idx] = self.neutral_element

    def total(self) -> float:
        """
        Возвращает сумму всех элементов с защитой от ошибок.

        Returns:
            float: Сумма всех элементов дерева.
        """
        if self.tree.size(0) == 0:
            logger.warning("Дерево пустое, возвращаем нейтральный элемент")
            return float(self.neutral_element)

        try:
            root_value = self.tree[0]
            # Проверяем валидность корневого значения
            if not torch.isfinite(root_value):
                logger.warning(
                    "Корень дерева содержит невалидное значение, возвращаем нейтральный элемент"
                )
                return float(self.neutral_element)
            return root_value.item()
        except RuntimeError as e:
            logger.error(f"Ошибка при получении total(): {e}")
            return float(self.neutral_element)

    def sum(self) -> float:
        """
        Алиас для метода total().

        Returns:
            float: Сумма всех элементов дерева.
        """
        return self.total()

    def update_batch(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        min_priority: float = 1e-6,
        max_priority: float = 1e2,
    ) -> None:
        """
        Векторизованное обновление приоритетов с клиппингом.

        Args:
            indices (torch.Tensor): Индексы листьев, которые нужно обновить.
            values (torch.Tensor): Новые значения для индексов.
            min_priority (float): Минимально допустимое значение приоритета.
            max_priority (float): Максимально допустимое значение приоритета.
        """
        # Проверяем размер батча для расширения буферов при необходимости
        batch_size = indices.size(0)
        if batch_size > self._max_update_batch_size:
            self._max_update_batch_size = batch_size * 2
            self._update_parent_indices_buffer = torch.zeros(
                self._max_update_batch_size, dtype=torch.int64, device=self.device
            )

        # Используем асинхронные операции на GPU, если возможно
        if self.device.type == "cuda" and self._cuda_stream is not None:
            with torch.cuda.stream(self._cuda_stream):
                # Клиппируем значения и приводим к типу дерева
                values = torch.clamp(values, min=min_priority, max=max_priority).to(
                    dtype=self.tree.dtype
                )
                update_tree_vectorized(
                    self.tree, indices, values, min_priority, max_priority
                )
            # Можно выполнять другие операции параллельно
        else:
            # Обычное обновление на CPU или без потока
            values = torch.clamp(values, min=min_priority, max=max_priority)
            update_tree_vectorized(
                self.tree, indices, values, min_priority, max_priority
            )

    def get_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Оптимизированное получение батча индексов по приоритетам.

        Args:
            batch_size (int): Количество индексов, которые нужно выбрать.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Индексы в дереве (tree_indices)
                - Приоритеты для выбранных индексов
                - Индексы в данных (data_indices)
        """
        if batch_size <= 0:
            logger.warning(f"get_batch вызван с недопустимым batch_size: {batch_size}")
            empty = torch.tensor([], device=self.device, dtype=torch.int64)
            empty_priorities = torch.tensor(
                [], device=self.device, dtype=self.tree.dtype
            )
            return empty, empty_priorities, empty

        # Проверяем валидность суммы дерева
        total_sum = self.total()
        if (
            not torch.isfinite(torch.tensor(total_sum, device=self.device))
            or total_sum <= 0
        ):
            logger.warning(f"Сумма дерева невалидна ({total_sum}), переинициализация")
            self.initialize()
            total_sum = self.total()

            # Запасной вариант - равномерное семплирование
            if total_sum <= 0:
                idx = torch.randint(0, self.capacity, (batch_size,), device=self.device)
                tree_idx = idx + self.capacity - 1
                priorities = torch.ones(
                    batch_size, device=self.device, dtype=self.tree.dtype
                )
                return tree_idx, priorities, idx

        # Размер сегмента для равномерного разделения суммы
        segment = total_sum / batch_size

        # Переиспользуем буферы, если batch_size не изменился
        use_cached_buffers = (
            self._cached_batch_size == batch_size
            and self._cached_random_offsets is not None
            and self._cached_prefixsums is not None
        )

        # Асинхронная подготовка буферов, если на GPU
        if self.device.type == "cuda" and self._cuda_stream is not None:
            with torch.cuda.stream(self._cuda_stream):
                if not use_cached_buffers:
                    # Создаем новые буферы с правильным размером
                    self._cached_random_offsets = torch.zeros(
                        batch_size, device=self.device
                    )
                    self._cached_prefixsums = torch.zeros(
                        batch_size, device=self.device
                    )
                    self._cached_batch_size = batch_size

                # Генерируем случайные смещения (0-1)
                self._cached_random_offsets.uniform_(0, 1)

                # Вычисляем префиксные суммы: [0, 1, 2, ...] * segment + random_offset * segment
                torch.arange(
                    0, batch_size, device=self.device, out=self._cached_prefixsums
                )
                self._cached_prefixsums.mul_(segment)
                self._cached_prefixsums.add_(self._cached_random_offsets.mul_(segment))

                # Находим индексы по префиксным суммам
                indices = find_prefixsum_idx_vectorized(
                    self.tree, self._cached_prefixsums, self.capacity
                )
                tree_indices = indices + self.capacity - 1

                # Получаем приоритеты
                priorities = torch.index_select(self.tree, 0, tree_indices)

            # Синхронизируем для получения результатов
            torch.cuda.current_stream().wait_stream(self._cuda_stream)
        else:
            # Для CPU используем прямой подход без асинхронности
            if not use_cached_buffers:
                self._cached_random_offsets = torch.rand(batch_size, device=self.device)
                self._cached_prefixsums = (
                    torch.arange(batch_size, device=self.device) * segment
                    + self._cached_random_offsets * segment
                )
                self._cached_batch_size = batch_size
            else:
                # Обновляем только случайности
                self._cached_random_offsets.uniform_(0, 1)
                torch.arange(
                    0, batch_size, device=self.device, out=self._cached_prefixsums
                )
                self._cached_prefixsums.mul_(segment)
                self._cached_prefixsums.add_(self._cached_random_offsets.mul_(segment))

            indices = find_prefixsum_idx_vectorized(
                self.tree, self._cached_prefixsums, self.capacity
            )
            tree_indices = indices + self.capacity - 1
            priorities = self.tree[tree_indices]

        return tree_indices, priorities, indices

    def get_max_priority(self) -> float:
        """
        Возвращает максимальный приоритет среди конечных значений.

        Returns:
            float: Максимальный приоритет среди листьев дерева.
        """
        try:
            # Оптимизированный доступ только к листьям дерева
            leaf_start = self.capacity - 1
            leaf_end = 2 * self.capacity - 1

            if leaf_end > self.tree.size(0):
                leaf_end = self.tree.size(0)

            all_values = self.tree[leaf_start:leaf_end]
            finite_values = all_values[torch.isfinite(all_values)]

            if finite_values.numel() > 0:
                max_prio = finite_values.max().item()
            else:
                max_prio = 0.0

        except Exception as e:
            logger.error(f"Ошибка при вычислении максимального приоритета: {e}")
            max_prio = 0.0

        # Возвращаем максимум из найденного или 1.0
        return max(max_prio, 1.0)

    def to(self, device: Union[str, torch.device]) -> "SumTree":
        """
        Перемещает дерево на указанное устройство.

        Args:
            device (Union[str, torch.device]): Целевое устройство.

        Note:
            Обновляет CUDA поток, если перемещаем на/с GPU.

        Returns:
            SumTree: Обновлённый экземпляр дерева на целевом устройстве.
        """
        target_device = device_manager.get_device(device)

        # Если устройство не меняется, ничего не делаем
        if target_device == self.device:
            return self

        # Перемещаем дерево
        super().to(target_device)

        # Обновляем кэшированные буферы
        self._cached_random_offsets = None
        self._cached_prefixsums = None
        self._cached_batch_size = None

        # Обновляем CUDA поток при перемещении на/с GPU
        if target_device.type == "cuda" and self._cuda_stream is None:
            try:
                self._cuda_stream = torch.cuda.Stream(device=target_device)
            except Exception as e:
                logger.warning(f"Не удалось создать CUDA-поток: {e}")
        elif target_device.type != "cuda":
            self._cuda_stream = None

        # Перемещаем буферы
        self._update_parent_indices_buffer = torch.zeros(
            self._max_update_batch_size, dtype=torch.int64, device=target_device
        )

        return self

    def clear(self) -> None:
        """
        Очищает дерево, сбрасывает кэшированные данные.
        """
        super().clear()
        self._cached_random_offsets = None
        self._cached_prefixsums = None
        self._cached_batch_size = None


# Проверка поддержки bfloat16 для устройства
def check_bfloat16_support(device: Union[str, torch.device]) -> bool:
    """
    Проверяет поддержку bfloat16 на указанном устройстве.

    Args:
        device (Union[str, torch.device]): Устройство для проверки.

    Returns:
        bool: True, если bfloat16 поддерживается, иначе False.
    """
    device = device_manager.get_device(device)

    # На CPU поддержка bfloat16 зависит от версии PyTorch
    if device.type == "cpu":
        return hasattr(torch, "bfloat16")

    # На CUDA проверяем возможность создания тензора
    elif device.type == "cuda":
        try:
            # Проверка через создание тестового тензора
            if torch.cuda.is_available():
                test_tensor = torch.zeros(1, dtype=torch.bfloat16, device=device)
                del test_tensor
                return True
        except RuntimeError:
            return False

    return False


# Функция для активации режима отладки CUDA
def enable_cuda_debugging() -> bool:
    """
    Рекомендации по активации режима отладки CUDA.
    Выводит инструкции для настройки переменных окружения.

    Returns:
        bool: True, если всё выполнено успешно.
    """
    import os

    # Проверяем, установлена ли уже переменная CUDA_LAUNCH_BLOCKING
    if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        logger.info(
            "Установлена переменная CUDA_LAUNCH_BLOCKING=1 для улучшения отладки CUDA ошибок"
        )

    # Выводим рекомендации по компиляции PyTorch с DSA
    logger.info(
        "Для включения device-side assertions при компиляции PyTorch из исходников:"
    )
    logger.info(
        "1. Клонируйте репозиторий PyTorch: git clone --recursive https://github.com/pytorch/pytorch"
    )
    logger.info("2. Установите переменную окружения: export TORCH_USE_CUDA_DSA=1")
    logger.info("3. Соберите PyTorch: python setup.py install")

    return True
