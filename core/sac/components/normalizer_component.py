import torch
from typing import Dict, Optional, Any, Union
import os
from functools import lru_cache

from core.sac.components.base_component import SACComponent
from core.sac.normalizers import (
    WelfordObservationNormalizer,
    BatchMeanStdNormalizer,
    DummyNormalizer,
)
from core.logging.logger import get_logger, log_method_call, log_tensor_info

logger = get_logger("normalizer_component")


class NormalizerComponent(SACComponent):
    """
    Компонент нормализации наблюдений. Делегирует операции базовому нормализатору.
    """

    def __init__(
        self,
        normalizer: Optional[Any] = None,
        normalize_obs: bool = True,
        clip_obs: Optional[float] = 10.0,
        epsilon: float = 1e-8,
        device: Optional[Union[str, torch.device]] = None,
        name: str = "normalizer",
        obs_dim: Optional[int] = None,
    ):
        """
        Инициализация компонента нормализации.\n
        1. Определяет размерность.\n
        2. Создаёт или использует нормализатор.\n
        3. Перемещает на нужное устройство.\n
        
        Args:
            normalizer (Optional[Any]): Внешний нормализатор (если не задан — создается автоматически).
            normalize_obs (bool): Флаг включения нормализации.
            clip_obs (Optional[float]): Диапазон ограничения нормализованных значений.
            epsilon (float): Малое значение для предотвращения деления на ноль.
            device (Optional[Union[str, torch.device]]): Устройство для вычислений (CPU/GPU).
            name (str): Имя компонента (для логирования).
            obs_dim (Optional[int]): Размерность входных наблюдений.
        """
        super(NormalizerComponent, self).__init__(name=name, device=device)

        # Сохраняем настройки
        self.normalize_obs = normalize_obs
        self.clip_obs = clip_obs

        # Определяем размерность наблюдений
        if normalizer is not None and hasattr(normalizer, "obs_dim"):
            self.obs_dim = normalizer.obs_dim
        elif obs_dim is not None:
            self.obs_dim = obs_dim
        else:
            raise ValueError(
                "Необходимо указать obs_dim явно или передать normalizer с атрибутом obs_dim"
            )

        # Определяем, нужно ли создавать нормализатор
        should_create_normalizer = normalizer is None and self.normalize_obs

        if normalizer is not None:
            # Используем переданный нормализатор
            self.normalizer = normalizer
            logger.info(
                f"Использован внешний нормализатор типа {type(normalizer).__name__}"
            )
            # Перемещаем внешний нормализатор на нужное устройство, если у него есть метод to()
            if hasattr(self.normalizer, "to") and callable(
                getattr(self.normalizer, "to")
            ):
                try:
                    self.normalizer = self.normalizer.to(self.device)
                    logger.debug(f"Внешний нормализатор перемещен на {self.device}")
                except Exception as e:
                    logger.warning(
                        f"Не удалось переместить внешний нормализатор на устройство {self.device}: {e}"
                    )
            # Убеждаемся, что normalize_obs соответствует типу нормализатора
            if isinstance(self.normalizer, DummyNormalizer):
                self.normalize_obs = False
            else:
                self.normalize_obs = True  # Если нормализатор не фиктивный, считаем, что нормализация включена

        elif should_create_normalizer:
            # Создаем BatchMeanStdNormalizer по умолчанию, если нормализация включена
            # Welford менее эффективен для пакетного обучения
            self.normalizer = BatchMeanStdNormalizer(
                shape=(self.obs_dim,), clip_range=clip_obs, epsilon=epsilon
            ).to(self.device)
            logger.info(
                f"Создан внутренний нормализатор BatchMeanStdNormalizer: obs_dim={self.obs_dim}, clip={clip_obs}, eps={epsilon}"
            )
        else:
            # Создаем DummyNormalizer, если нормализация выключена или нормализатор не передан
            self.normalizer = DummyNormalizer(obs_dim=self.obs_dim)
            self.normalize_obs = False  # Явно выключаем флаг
            logger.info(
                "Нормализация отключена, создан фиктивный нормализатор (DummyNormalizer)"
            )

        logger.info(
            f"Инициализирован компонент '{self.name}': normalize_obs={self.normalize_obs}, clip_obs={self.clip_obs}, тип={type(self.normalizer).__name__}"
        )

    def is_dummy(self) -> bool:
        """
        Проверяет, является ли используемый нормализатор фиктивным.

        Returns:
            bool: True, если используется DummyNormalizer или нормализация отключена.
        """
        # Сверяемся с типом и флагом normalize_obs
        return isinstance(self.normalizer, DummyNormalizer) or not self.normalize_obs

    def normalize(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Нормализует наблюдения (или возвращает как есть, если нормализация отключена).

        Args:
            observations (torch.Tensor): Входной тензор наблюдений.

        Returns:
            torch.Tensor: Нормализованный тензор наблюдений.
        """
        if not self.normalize_obs or self.is_dummy():
            return observations

        # Убедимся, что входной тензор находится на нужном устройстве, сохраняя dtype
        if not isinstance(observations, torch.Tensor):
            observations = torch.as_tensor(observations, device=self.device)
        elif observations.device != self.device:
            observations = observations.to(self.device)

        # Выполняем без градиентов
        with torch.no_grad():
            # Делегируем нормализацию базовому нормализатору
            # Убедимся, что нормализатор реализует метод normalize
            if hasattr(self.normalizer, "normalize") and callable(
                getattr(self.normalizer, "normalize")
            ):
                normalized = self.normalizer.normalize(observations)
                if normalized.dtype != observations.dtype:
                    logger.debug(
                        f"Приводим нормализованные данные обратно к {observations.dtype}"
                    )
                    normalized = normalized.to(dtype=observations.dtype)
                return normalized
            else:
                logger.warning(
                    f"Нормализатор типа {type(self.normalizer).__name__} не имеет метода normalize(). Нормализация пропущена."
                )
                return observations

    def update(self, observations: torch.Tensor) -> None:
        """
        Обновляет статистику нормализатора на основе новых наблюдений.

        Args:
            observations (torch.Tensor): Тензор новых наблюдений.
        """
        if not self.normalize_obs or self.is_dummy():
            return

        # Убедимся, что входной тензор находится на нужном устройстве
        # Большинство нормализаторов работают с float32, но мы сохраняем исходный dtype
        # для последующего преобразования обратно
        original_dtype = None
        if not isinstance(observations, torch.Tensor):
            observations = torch.as_tensor(observations, device=self.device)
        elif observations.device != self.device:
            observations = observations.to(self.device)

        # Сохраняем исходный dtype
        original_dtype = observations.dtype

        # Для внутренних расчетов статистики обычно требуется float32
        if observations.dtype != torch.float32:
            # Конвертируем только для обновления статистики, но не для будущих выходных данных
            logger.debug(
                f"Преобразование наблюдений из {observations.dtype} в float32 для обновления статистики"
            )
            observations = observations.to(dtype=torch.float32)

        # Делегируем обновление статистики базовому нормализатору
        # Убедимся, что нормализатор реализует метод update
        if hasattr(self.normalizer, "update") and callable(
            getattr(self.normalizer, "update")
        ):
            self.normalizer.update(observations)
        else:
            logger.debug(
                f"Нормализатор типа {type(self.normalizer).__name__} не имеет метода update(). Обновление статистики пропущено."
            )

    @log_method_call()
    def save(self, path: str) -> None:
        """
        Сохраняет состояние нормализатора в файл.

        Args:
            path (str): Путь к файлу для сохранения.
        """
        # Создаем директорию, если ее нет
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Делегируем сохранение, если возможно
        if hasattr(self.normalizer, "save") and callable(
            getattr(self.normalizer, "save")
        ):
            try:
                self.normalizer.save(path)
                logger.info(
                    f"Состояние нормализатора '{self.name}' ({type(self.normalizer).__name__}) сохранено в {path}"
                )
            except Exception as e:
                logger.error(
                    f"Ошибка при сохранении состояния нормализатора '{self.name}': {e}"
                )
        else:
            logger.warning(
                f"Нормализатор типа {type(self.normalizer).__name__} не поддерживает сохранение. Сохранены только метаданные компонента."
            )
            # Сохраняем базовую конфигурацию компонента, чтобы знать настройки при загрузке
            config = self.get_config()
            state_dict = {
                "config": config
                # Не сохраняем сам нормализатор, т.к. он не поддерживает save
            }
            torch.save(state_dict, path)

    @log_method_call()
    def load(
        self, path: str, map_location: Optional[Union[str, torch.device]] = None
    ) -> None:
        """
        Загружает состояние нормализатора из файла.

        Args:
            path (str): Путь к файлу со состоянием.
            map_location (Optional[Union[str, torch.device]]): Устройство, на которое будет загружено состояние.
        """
        if not os.path.exists(path):
            logger.warning(
                f"Файл состояния нормализатора {path} не найден. Загрузка пропущена."
            )
            return

        # Делегируем загрузку, если возможно
        if hasattr(self.normalizer, "load") and callable(
            getattr(self.normalizer, "load")
        ):
            try:
                # Передаем текущее устройство компонента как map_location
                self.normalizer.load(path, map_location=self.device)
                logger.info(
                    f"Состояние нормализатора '{self.name}' ({type(self.normalizer).__name__}) загружено из {path} на {self.device}"
                )
            except Exception as e:
                logger.error(
                    f"Ошибка при загрузке состояния нормализатора '{self.name}': {e}"
                )
        else:
            logger.warning(
                f"Нормализатор типа {type(self.normalizer).__name__} не поддерживает загрузку. Загружены только метаданные компонента."
            )
            # Загружаем базовую конфигурацию, чтобы проверить настройки
            try:
                state_dict = torch.load(path, map_location=self.device)
                loaded_config = state_dict.get("config", {})
                logger.info(
                    f"Загружены метаданные компонента нормализатора из {path}: {loaded_config}"
                )
                # Можно добавить проверки на совпадение конфигураций, если нужно
            except Exception as e:
                logger.error(
                    f"Ошибка при загрузке метаданных компонента нормализатора '{self.name}': {e}"
                )

    def get_config(self) -> Dict[str, Any]:
        """
        Возвращает конфигурацию компонента нормализации в виде словаря.

        Returns:
            Dict[str, Any]: Конфигурационные данные нормализатора.
        """
        config = super().get_config()
        config.update(
            {
                "normalize_obs": self.normalize_obs,
                "clip_obs": self.clip_obs,
                "obs_dim": self.obs_dim,
                "normalizer_type": type(self.normalizer).__name__,
            }
        )
        # Добавляем специфичные параметры нормализатора, если они есть
        if hasattr(self.normalizer, "epsilon"):
            config["epsilon"] = self.normalizer.epsilon
        return config

    # Добавляем метод save_state для совместимости с agent.py
    @log_method_call()
    def save_state(self, path: str) -> None:
        """
        Псевдоним для метода save(). Используется для совместимости с agent.py.

        Args:
            path (str): Путь к файлу для сохранения.
        """
        self.save(path)

    # Добавляем метод load_state для совместимости с agent.py
    @log_method_call()
    def load_state(
        self, path: str, map_location: Optional[Union[str, torch.device]] = None
    ) -> None:
        """
        Псевдоним для метода load(). Используется для совместимости с agent.py.

        Args:
            path (str): Путь к файлу со состоянием.
            map_location (Optional[Union[str, torch.device]]): Устройство, на которое будет загружено состояние.
        """
        self.load(path, map_location=map_location)

    @property
    @lru_cache(maxsize=1)
    def cached_obs_dim(self):
        return self.obs_dim
