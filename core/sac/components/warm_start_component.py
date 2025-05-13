import torch
import numpy as np
from typing import Dict, Any, Optional, Union
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
import os
import joblib

from core.sac.components.base_component import SACComponent
from core.logging.logger import get_logger, log_method_call, log_tensor_info

logger = get_logger("warm_start_component")


class WarmStartComponent(SACComponent):
    """
    Компонент "теплого старта" для SAC. Использует регрессор (DecisionTree, RandomForest)
    для предсказания действий на ранних этапах обучения.

    Attributes:
        enabled (bool): Включён ли теплый старт.
        obs_dim (int): Размерность наблюдений.
        action_dim (int): Размерность действий.
        regressor_type (str): Тип регрессора ('decision_tree' или 'random_forest').
        active_steps (int): Сколько шагов компонент будет активен.
        random_steps (int): Сколько первых шагей использовать случайные действия.
        exploration_noise (float): Шум исследования при использовании регрессора.
        initialized (bool): Обучена ли модель.
        step_count (int): Счётчик пройденных шагов.
        model (Optional): Модель регрессора (DecisionTreeRegressor или RandomForestRegressor).
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        enabled: bool = False,
        regressor_type: str = "decision_tree",
        steps: int = 10000,
        random_steps: int = 1000,
        action_space: Optional[Any] = None,
        max_depth: int = 10,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
        exploration_noise: float = 0.1,
        regressor: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        name: str = "warm_start",
    ):
        """
        Инициализация компонента теплого старта.
        1. Создаёт регрессор.
        2. Настраивает параметры и буферы.

        Args:
            action_dim (int): Размерность действий.
            obs_dim (int): Размерность наблюдений.
            enabled (bool): Флаг включения компонента.
            regressor_type (str): Тип регрессора ('decision_tree', 'random_forest').
            steps (int): Количество шагов, в течение которых компонент будет активным.
            random_steps (int): Первые N шагов — использовать полностью случайные действия.
            action_space (Optional[Any]): Пространство действий среды.
            max_depth (int): Максимальная глубина дерева решений.
            min_samples_split (int): Минимальное количество сэмплов для разделения.
            random_state (Optional[int]): Зерно для генератора случайных чисел.
            exploration_noise (float): Добавляемый шум к предсказанным действиям.
            regressor (Optional[Any]): Внешний регрессор.
            device (Optional[Union[str, torch.device]]): Устройство вычислений (CPU/GPU).
            name (str): Имя компонента (для логирования).
        """
        super(WarmStartComponent, self).__init__(name=name, device=device)

        self.enabled = enabled
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.regressor_type = (
            regressor_type.lower()
            if isinstance(regressor_type, str)
            else "decision_tree"
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.active_steps = steps
        self.random_steps = min(random_steps, steps)
        self.exploration_noise = exploration_noise

        self.initialized = False
        self.step_count = 0
        self.model = None

        # Если передан внешний регрессор, используем его
        if regressor is not None:
            self.model = regressor
            self.initialized = True
            self.reset_buffers()
            logger.info(f"WarmStartComponent '{self.name}' получил внешний регрессор.")
        # Иначе создаем модель на основе типа и настроек
        elif self.enabled:
            try:
                if self.regressor_type == "random_forest":
                    self.model = RandomForestRegressor(
                        n_estimators=10,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                    logger.debug(
                        f"Создан RandomForestRegressor(max_depth={max_depth}, min_samples={min_samples_split})"
                    )
                elif self.regressor_type == "decision_tree":
                    self.model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state,
                    )
                    logger.debug(
                        f"Создан DecisionTreeRegressor(max_depth={max_depth}, min_samples={min_samples_split})"
                    )
                else:
                    logger.warning(
                        f"Неизвестный тип регрессора: '{self.regressor_type}'. WarmStart отключен."
                    )
                    self.enabled = False
                if self.enabled:
                    self.reset_buffers()
            except ImportError:
                logger.error(
                    "scikit-learn не найден. Установите `scikit-learn` для WarmStartComponent."
                )
                self.enabled = False
            except Exception as e:
                logger.error(f"Ошибка создания регрессора: {e}")
                self.enabled = False

        if self.enabled:
            logger.info(
                f"Инициализирован компонент WarmStart '{self.name}': "
                f"enabled={enabled}, regressor_type={self.regressor_type}, steps={steps}, "
                f"random_steps={self.random_steps}, action_dim={action_dim}, obs_dim={obs_dim}"
            )
        else:
            logger.info(f"Компонент WarmStart '{self.name}' отключен.")

    def reset_buffers(self) -> None:
        """
        Сбрасывает буферы наблюдений и действий для сбора данных.
        Вызывается при инициализации и после обучения модели.
        """
        if not self.enabled:
            return
        self.obs_buffer = np.empty((0, self.obs_dim), dtype=np.float32)
        self.action_buffer = np.empty((0, self.action_dim), dtype=np.float32)
        logger.debug(f"Буферы компонента '{self.name}' сброшены.")

    def to_numpy_flat(self, x, dtype=np.float32):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().flatten().astype(dtype)
        return np.array(x, dtype=dtype).flatten()

    def add_sample(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """
        Добавляет пару (наблюдение, действие) в буферы для последующего обучения регрессора.

        Args:
            obs (Union[np.ndarray, torch.Tensor]): Тензор или массив наблюдений.
            action (Union[np.ndarray, torch.Tensor]): Тензор или массив действий.
        """
        if not self.enabled or self.step_count >= self.active_steps:
            return
        obs_np = self.to_numpy_flat(obs, np.float32)
        action_np = self.to_numpy_flat(action, np.float32)
        if obs_np.shape[0] != self.obs_dim or action_np.shape[0] != self.action_dim:
            logger.warning(
                f"Неверная размерность в add_sample: obs {obs_np.shape}, action {action_np.shape}. Пропуск."
            )
            return
        self.obs_buffer = np.vstack([self.obs_buffer, obs_np])
        self.action_buffer = np.vstack([self.action_buffer, action_np])
        buffer_len = self.obs_buffer.shape[0]
        if not self.initialized and buffer_len > self.min_samples_split:
            if buffer_len % 100 == 0 or buffer_len == 500:
                logger.debug(
                    f"Накоплено {buffer_len} сэмплов, попытка обучения регрессора..."
                )
                self.train_model()

    def train_model(self) -> None:
        """
        Обучает регрессор на собранных данных.
        Если данных недостаточно, обучение пропускается.
        """
        if not self.enabled or self.obs_buffer.shape[0] < self.min_samples_split:
            logger.debug(
                "Обучение регрессора пропущено (компонент выключен или буферы пусты)."
            )
            return
        logger.info(
            f"Начало обучения регрессора '{self.name}' ({self.regressor_type}) на {self.obs_buffer.shape[0]} сэмплах..."
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = self.obs_buffer
            y = self.action_buffer
            try:
                self.model.fit(X, y)
                self.initialized = True
                self.reset_buffers()
                logger.info(f"Регрессор '{self.name}' успешно обучен. Буферы очищены.")
            except Exception as e:
                logger.error(f"Ошибка при обучении регрессора '{self.name}': {e}")
                self.initialized = False

    def predict_action(
        self, obs: Union[np.ndarray, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Предсказывает действие на основе наблюдения с помощью обученного регрессора.

        Args:
            obs (Union[np.ndarray, torch.Tensor]): Входные наблюдения.

        Returns:
            Optional[torch.Tensor]: Предсказанное действие или None, если компонент неактивен.
        """
        if (
            not self.should_use_warm_start()
            or not self.initialized
            or self.model is None
        ):
            return None

        try:
            obs_np = self.to_numpy_flat(obs, np.float32)

            if obs_np.shape[0] != self.obs_dim:
                logger.warning(
                    f"Неверная размерность наблюдения в predict_action: ожидалось {self.obs_dim}, получено {obs_np.shape[0]}. Возвращаем None."
                )
                return None

            action_np = self.model.predict(obs_np.reshape(1, -1))[0]

            if self.exploration_noise > 0:
                noise = np.random.normal(
                    0, self.exploration_noise, size=self.action_dim
                )
                action_np = action_np + noise
                logger.debug(
                    f"Добавлен шум исследования (std={self.exploration_noise:.3f}) к действию регрессора."
                )

            if (
                self.action_space is not None
                and hasattr(self.action_space, "low")
                and hasattr(self.action_space, "high")
            ):
                action_np = np.clip(
                    action_np, self.action_space.low, self.action_space.high
                )
                logger.debug(
                    f"Действие регрессора клиппировано в диапазон [{self.action_space.low}, {self.action_space.high}]"
                )
            else:
                action_np = np.clip(action_np, -1.0, 1.0)

            action = torch.tensor(action_np, dtype=torch.float32, device=self.device)
            log_tensor_info(logger, f"{self.name}_predicted_action", action)

            return action

        except Exception as e:
            logger.error(
                f"Ошибка при предсказании действия регрессором '{self.name}': {e}. Возвращаем None."
            )
            return None

    def step(self) -> None:
        """
        Увеличивает внутренний счётчик шагов.
        Используется для управления активностью компонента по времени.
        """
        if not self.enabled:
            return
        self.step_count += 1

    def is_active(self) -> bool:
        """
        Проверяет, находится ли компонент в активном состоянии.

        Returns:
            bool: True, если компонент активен.
        """
        return (
            self.enabled and self.step_count <= self.active_steps and self.initialized
        )

    def should_use_warm_start(self) -> bool:
        """
        Проверяет, нужно ли использовать теплый старт сейчас.

        Returns:
            bool: True, если компонент активен и прошёл инициализацию.
        """
        return self.enabled and self.step_count < self.active_steps and self.initialized

    def should_use_random_action(self) -> bool:
        """
        Проверяет, нужно ли использовать случайные действия на ранних этапах.

        Returns:
            bool: True, если счётчик шагов ещё меньше random_steps.
        """
        return self.enabled and self.step_count < self.random_steps

    @log_method_call()
    def save(self, path: str) -> None:
        """
        Сохраняет модель регрессора и текущее состояние компонента в файл.

        Args:
            path (str): Путь к файлу, в который будет сохранено состояние.
        """
        if not self.enabled or self.model is None:
            logger.debug(
                f"Сохранение компонента '{self.name}' пропущено (отключен или нет модели)."
            )
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)

        state_dict = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "regressor_type": self.regressor_type,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
            "active_steps": self.active_steps,
            "random_steps": self.random_steps,
            "exploration_noise": self.exploration_noise,
            "initialized": self.initialized,
            "step_count": self.step_count,
        }

        try:
            joblib.dump({"model": self.model, "state": state_dict}, path, compress=3)
            logger.info(f"Компонент WarmStart '{self.name}' сохранен в {path}")
        except Exception as e:
            logger.error(
                f"Ошибка при сохранении компонента WarmStart '{self.name}' в {path}: {e}"
            )

    @log_method_call()
    def load(self, path: str) -> None:
        """
        Загружает модель регрессора и состояние компонента из файла.

        Args:
            path (str): Путь к файлу с состоянием компонента.
        """
        if not self.enabled:
            logger.debug(
                f"Загрузка компонента '{self.name}' пропущена (отключен при инициализации)."
            )
            return

        if not os.path.exists(path):
            logger.warning(
                f"Файл для загрузки компонента '{self.name}' не найден: {path}"
            )
            return

        try:
            data = joblib.load(path)

            state_dict = data.get("state")
            if state_dict:
                self.obs_dim = state_dict.get("obs_dim", self.obs_dim)
                self.action_dim = state_dict.get("action_dim", self.action_dim)
                self.regressor_type = state_dict.get(
                    "regressor_type", self.regressor_type
                )
                self.max_depth = state_dict.get("max_depth", self.max_depth)
                self.min_samples_split = state_dict.get(
                    "min_samples_split", self.min_samples_split
                )
                self.random_state = state_dict.get("random_state", self.random_state)
                self.active_steps = state_dict.get("active_steps", self.active_steps)
                self.random_steps = state_dict.get("random_steps", self.random_steps)
                self.exploration_noise = state_dict.get(
                    "exploration_noise", self.exploration_noise
                )
                self.initialized = state_dict.get("initialized", self.initialized)
                self.step_count = state_dict.get("step_count", self.step_count)
            else:
                logger.warning(f"Словарь состояния 'state' не найден в файле {path}")

            loaded_model = data.get("model")
            if loaded_model is not None:
                self.model = loaded_model
                if not self.initialized:
                    self.initialized = True
                    logger.info(
                        f"Загруженная модель регрессора '{self.name}' помечена как инициализированная."
                    )
                logger.info(
                    f"Компонент WarmStart '{self.name}' успешно загружен из {path}"
                )
            else:
                logger.warning(
                    f"Модель 'model' не найдена в файле {path}. Компонент может быть неинициализирован."
                )
                if self.initialized:
                    logger.warning(
                        f"Сбрасываем флаг initialized для '{self.name}', так как модель не загружена."
                    )
                    self.initialized = False

        except Exception as e:
            logger.error(
                f"Ошибка при загрузке компонента WarmStart '{self.name}' из {path}: {e}"
            )
            self.model = None
            self.initialized = False

    def get_config(self) -> Dict[str, Any]:
        """
        Возвращает конфигурацию компонента теплого старта.

        Returns:
            Dict[str, Any]: Словарь с параметрами и настройками компонента.
        """
        config = super().get_config()
        config.update(
            {
                "enabled": self.enabled,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "regressor_type": self.regressor_type,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "random_state": self.random_state,
                "active_steps": self.active_steps,
                "random_steps": self.random_steps,
                "exploration_noise": self.exploration_noise,
                "initialized": self.initialized,
                "step_count": self.step_count,
            }
        )
        return config

    @log_method_call()
    def set_regressor(self, regressor: Any) -> None:
        """
        Устанавливает внешнюю модель регрессора в компонент.

        Args:
            regressor (Any): Объект регрессора с методом predict().
        """
        if not self.enabled:
            logger.warning(
                f"Попытка установить регрессор для отключенного компонента '{self.name}'."
            )
            return

        if not hasattr(regressor, "predict") or not callable(
            getattr(regressor, "predict")
        ):
            logger.error(
                f"Переданный объект типа {type(regressor).__name__} не является валидным регрессором (отсутствует метод predict)."
            )
            return

        self.model = regressor
        self.initialized = True
        logger.info(
            f"Для компонента '{self.name}' установлен внешний регрессор типа {type(regressor).__name__}. Компонент инициализирован."
        )
