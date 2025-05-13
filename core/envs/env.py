import gym
from gym import spaces
import numpy as np
import torch
import torch.cuda.amp
import os
import json
import yaml
import time
from typing import Dict, List, Tuple, Optional, Union, Any

try:
    from mlagents_envs.environment import UnityEnvironment, ActionTuple
    from mlagents_envs.side_channel.engine_configuration_channel import (
        EngineConfigurationChannel,
    )
    from mlagents_envs.side_channel.float_properties_channel import (
        FloatPropertiesChannel,
    )
    from mlagents_envs.exception import (
        UnityWorkerInUseException,
        UnityEnvironmentException,
    )
    from mlagents_envs.base_env import DecisionSteps, TerminalSteps

    unity_available = True
except ImportError:
    unity_available = False

from core.logging.logger import get_logger, log_method_call, DEBUG
from core.utils.device_utils import device_manager

logger = get_logger("unity_env")


# --- Конфигурация среды Unity ---
class UnityEnvConfig:
    """
    Класс для конфигурации среды Unity.
    """

    @log_method_call(log_level=DEBUG, log_args=True)
    def __init__(
        self,
        config_file: Optional[str] = None,
        env_name: str = "default",
        file_name: Optional[str] = None,
        worker_id: int = 0,
        time_scale: float = 20.0,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 300,
        base_port: int = 5005,
        memory_buffer_size: int = 4096,
        flatten_obs: bool = True,
        normalize_actions: bool = True,
        width: int = 1280,
        height: int = 720,
        quality_level: int = 5,
        target_frame_rate: int = -1,
        capture_frame_rate: int = 60,
        multi_agent: bool = False,
        device: str = "cuda",
    ) -> None:
        """
        Конструктор класса UnityEnvConfig.

        Args:
            config_file (Optional[str]): Путь к файлу конфигурации.
            env_name (str): Имя среды.
            file_name (Optional[str]): Имя файла среды.
            worker_id (int): Идентификатор рабочего процесса.
            time_scale (float): Масштаб времени.
            seed (int): Сид для случайных чисел.
            no_graphics (bool): Флаг для отключения графики.
            timeout_wait (int): Время ожидания при подключении.
            base_port (int): Базовый порт для соединения.
            memory_buffer_size (int): Размер буфера памяти.
            flatten_obs (bool): Флаг для уплощения наблюдений.
            normalize_actions (bool): Флаг для нормализации действий.
            width (int): Ширина экрана.
            height (int): Высота экрана.
            quality_level (int): Уровень качества графики.
            target_frame_rate (int): Целевая частота кадров.
            capture_frame_rate (int): Частота захвата кадров.
            multi_agent (bool): Флаг для мультиагентных сред.
            device (str, optinal): Устройство, на котором обучается модель.
        """
        self.env_name = env_name
        self.file_name = file_name
        self.worker_id = worker_id
        self.time_scale = time_scale
        self.seed = seed
        self.no_graphics = no_graphics
        self.timeout_wait = timeout_wait
        self.base_port = base_port
        self.memory_buffer_size = memory_buffer_size
        self.flatten_obs = flatten_obs
        self.normalize_actions = normalize_actions
        self.width = width
        self.height = height
        self.quality_level = quality_level
        self.target_frame_rate = target_frame_rate
        self.capture_frame_rate = capture_frame_rate
        self.multi_agent = multi_agent
        self.device = device_manager.get_device(device)  # Для обработки тензоров

        # Дополнительные свойства
        self.properties = {}

        # Загрузка конфигурации из файла, если указан
        if config_file is not None:
            self.load_config(config_file)

        logger.info(
            f"Инициализирована конфигурация UnityEnv: env_name={env_name}, worker_id={worker_id}, device={self.device}"
        )

    @log_method_call(log_level=DEBUG, log_args=True)
    def load_config(self, config_file: str) -> None:
        """
        Загружает конфигурацию из файла (JSON или YAML).

        Args:
            config_file (str): Путь к файлу конфигурации.
        """
        if not os.path.exists(config_file):
            logger.warning(f"Файл конфигурации не найден: {config_file}")
            return

        try:
            ext = os.path.splitext(config_file)[1].lower()

            if ext == ".json":
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            elif ext in [".yaml", ".yml"]:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
            else:
                logger.error(f"Неподдерживаемый формат файла конфигурации: {ext}")
                return

            # Обновляем конфигурацию
            self._update_from_dict(config)

            logger.info(f"Загружена конфигурация из {config_file}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")

    @log_method_call(log_level=DEBUG, log_args=True)
    def save_config(self, config_file: str) -> None:
        """
        Сохраняет текущую конфигурацию в файл (JSON или YAML).

        Args:
            config_file (str): Путь к файлу, в который будет сохранена конфигурация.
        """
        try:
            config = self.to_dict()

            # Создаем директорию, если не существует
            directory = os.path.dirname(config_file)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            ext = os.path.splitext(config_file)[1].lower()

            if ext == ".json":
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
            elif ext in [".yaml", ".yml"]:
                with open(config_file, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            else:
                logger.error(f"Неподдерживаемый формат файла конфигурации: {ext}")
                return

            logger.info(f"Конфигурация сохранена в {config_file}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении конфигурации: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует текущую конфигурацию в словарь для последующей сериализации.

        Returns:
            Dict[str, Any]: Словарь с параметрами конфигурации.
        """
        return {
            "env_name": self.env_name,
            "file_name": self.file_name,
            "worker_id": self.worker_id,
            "time_scale": self.time_scale,
            "seed": self.seed,
            "no_graphics": self.no_graphics,
            "timeout_wait": self.timeout_wait,
            "base_port": self.base_port,
            "memory_buffer_size": self.memory_buffer_size,
            "flatten_obs": self.flatten_obs,
            "normalize_actions": self.normalize_actions,
            "width": self.width,
            "height": self.height,
            "quality_level": self.quality_level,
            "target_frame_rate": self.target_frame_rate,
            "capture_frame_rate": self.capture_frame_rate,
            "multi_agent": self.multi_agent,
            "properties": self.properties,
        }

    def _update_from_dict(self, config: Dict[str, Any]) -> None:
        """
        Обновляет атрибуты объекта на основе словаря конфигурации.

        Args:
            config (Dict[str, Any]): Словарь с новыми значениями параметров.
        """
        # Обновляем основные параметры
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key == "properties":
                # Обрабатываем дополнительные свойства
                if isinstance(value, dict):
                    self.properties.update(value)

        logger.debug(f"Конфигурация обновлена из словаря: {list(config.keys())}")

    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Получает значение дополнительного свойства по ключу.

        Args:
            key (str): Ключ свойства.
            default (Any): Значение по умолчанию, если свойство не найдено.

        Returns:
            Any: Значение свойства или значение по умолчанию.
        """
        return self.properties.get(key, default)

    def set_property(self, key: str, value: Any) -> None:
        """
        Устанавливает значение дополнительного свойства.

        Args:
            key (str): Ключ свойства.
            value (Any): Новое значение свойства.
        """
        self.properties[key] = value
        logger.debug(f"Установлено свойство {key}={value}")

    def get_env_kwargs(self) -> Dict[str, Any]:
        """
        Возвращает параметры среды Unity для передачи в её инициализатор.

        Returns:
            Dict[str, Any]: Словарь с параметрами среды.
        """
        return {
            "file_name": self.file_name,
            "worker_id": self.worker_id,
            "time_scale": self.time_scale,
            "seed": self.seed,
            "no_graphics": self.no_graphics,
            "timeout_wait": self.timeout_wait,
            "base_port": self.base_port,
            "memory_buffer_size": self.memory_buffer_size,
        }

    def get_gym_kwargs(self) -> Dict[str, Any]:
        """
        Возвращает параметры для создания Gym-обертки вокруг среды Unity.

        Returns:
            Dict[str, Any]: Словарь с параметрами Gym-среды.
        """
        return {
            "file_name": self.file_name,
            "worker_id": self.worker_id,
            "time_scale": self.time_scale,
            "seed": self.seed,
            "no_graphics": self.no_graphics,
            "flatten_obs": self.flatten_obs,
            "normalize_actions": self.normalize_actions,
            "timeout_wait": self.timeout_wait,
            "base_port": self.base_port,
            "memory_buffer_size": self.memory_buffer_size,
        }

    def get_engine_config(self) -> Dict[str, Any]:
        """
        Возвращает параметры конфигурации движка Unity.

        Returns:
            Dict[str, Any]: Словарь с параметрами движка Unity.
        """
        return {
            "width": self.width,
            "height": self.height,
            "quality_level": self.quality_level,
            "time_scale": self.time_scale,
            "target_frame_rate": self.target_frame_rate,
            "capture_frame_rate": self.capture_frame_rate,
        }

    def __str__(self) -> str:
        """
        Возвращает строковое представление объекта конфигурации.

        Returns:
            str: Строка с информацией о конфигурации.
        """
        return f"UnityEnvConfig(env_name={self.env_name}, file_name={self.file_name}, worker_id={self.worker_id}, multi_agent={self.multi_agent})"

    def __repr__(self) -> str:
        """
        Возвращает строковое представление объекта конфигурации.

        Returns:
            str: Строка с информацией о конфигурации.
        """
        return f"UnityEnvConfig(env_name={self.env_name}, file_name={self.file_name}, worker_id={self.worker_id}, multi_agent={self.multi_agent})"


# --- Класс обертки среды Unity для Gym ---
class UnityToGymWrapper(gym.Env):
    """
    Обертка Unity среды, совместимая с интерфейсом Gym.
    """

    metadata = {"render.modes": ["human"]}

    @log_method_call(log_level=DEBUG, log_args=True)
    def __init__(
        self,
        file_name: str = None,
        worker_id: int = 0,
        time_scale: float = 20.0,
        seed: int = 0,
        no_graphics: bool = False,
        flatten_obs: bool = True,
        normalize_actions: bool = True,
        timeout_wait: int = 300,
        base_port: int = 5005,
        memory_buffer_size: int = 4096,
        device: str = "cuda",
    ) -> None:
        """
        Инициализирует обертку Gym для Unity среды.

        Args:
            file_name (str): Путь к файлу Unity-сценария.
            worker_id (int): Идентификатор рабочего процесса Unity.
            time_scale (float): Масштаб времени выполнения симуляции.
            seed (int): Сид для генерации случайных чисел.
            no_graphics (bool): Отключает графический интерфейс при запуске.
            flatten_obs (bool): Флаг, указывающий, нужно ли объединять несколько наблюдений в одно.
            normalize_actions (bool): Нормализует диапазон выходных действий до [-1, 1].
            timeout_wait (int): Время ожидания подключения к Unity.
            base_port (int): Базовый порт для связи с Unity.
            memory_buffer_size (int): Размер буфера памяти для передачи данных.
            device (str): Устройство ('cuda' или 'cpu') для обработки тензоров.
        """
        if not unity_available:
            raise ImportError(
                "Пакет mlagents_envs не установлен. Установите его для работы с Unity средой."
            )

        super(UnityToGymWrapper, self).__init__()

        # Создаем каналы для взаимодействия с Unity
        self.engine_config_channel = EngineConfigurationChannel()
        self.float_prop_channel = FloatPropertiesChannel()

        # Сохраняем параметры
        self.file_name = file_name
        self.worker_id = worker_id
        self.time_scale = time_scale
        self.seed = seed
        self.no_graphics = no_graphics
        self.flatten_obs = flatten_obs
        self.normalize_actions = normalize_actions
        self.timeout_wait = timeout_wait
        self.base_port = base_port
        self.memory_buffer_size = memory_buffer_size
        self.device = device_manager.get_device(device)  # Устройство для PyTorch

        # Инициализация AMP скейлера для смешанной точности
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        # Инициализируем среду Unity
        self.unity_env = None
        self.behavior_name = None

        # Сохраняем информацию о формах входных данных
        self.raw_observation_shapes = []

        # Пространства действий и наблюдений будут установлены при подключении
        self.observation_space = None
        self.action_space = None

        # Подключаемся к Unity
        self._connect()

        # Настраиваем среду
        self.engine_config_channel.set_configuration_parameters(
            time_scale=time_scale,
            width=1280,
            height=720,
            quality_level=5,
            target_frame_rate=-1,
            capture_frame_rate=60,
        )

        # Определяем пространства действий и наблюдений
        self._setup_spaces()

        logger.info(
            f"Инициализирована обертка Unity среды с worker_id={worker_id}, device={self.device}"
        )

    @log_method_call(log_level=DEBUG, log_args=False)
    def _connect(self):
        """
        Устанавливает соединение с Unity-средой с повторными попытками при неудаче.

        Raises:
            ConnectionError: Если не удалось установить соединение после всех попыток.
        """
        # Пытаемся подключиться к Unity с повторами при неудаче
        max_retries = 5
        retry_delay = 5  # секунд

        for attempt in range(max_retries):
            try:
                self.unity_env = UnityEnvironment(
                    file_name=self.file_name,
                    worker_id=self.worker_id,
                    seed=self.seed,
                    no_graphics=self.no_graphics,
                    timeout_wait=self.timeout_wait,
                    base_port=self.base_port,
                    side_channels=[self.engine_config_channel, self.float_prop_channel],
                )
                break
            except (UnityWorkerInUseException, UnityEnvironmentException) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Попытка {attempt + 1} подключения не удалась: {str(e)}. Повтор через {retry_delay} сек."
                    )
                    time.sleep(retry_delay)
                    self.worker_id += 1  # Пробуем другой worker_id
                else:
                    logger.error(
                        f"Не удалось подключиться к Unity после {max_retries} попыток: {str(e)}"
                    )
                    raise ConnectionError(f"Не удалось подключиться к Unity: {str(e)}")

        # Получаем имя поведения и спецификацию
        self.unity_env.reset()
        self.behavior_name = list(self.unity_env.behavior_specs.keys())[0]
        logger.info(f"Подключено к Unity среде. Поведение: {self.behavior_name}")

    @log_method_call(log_level=DEBUG, log_args=False)
    def _setup_spaces(self):
        """
        Настраивает пространства действий и наблюдений для Gym
        """
        # Получаем спецификацию поведения
        behavior_spec = self.unity_env.behavior_specs[self.behavior_name]

        # Сохраняем формы индивидуальных наблюдений для потенциального использования в экспорте ONNX
        self.raw_observation_shapes = []
        for obs_spec in behavior_spec.observation_specs:
            self.raw_observation_shapes.append(obs_spec.shape)

        logger.info(
            f"Обнаружены входные формы наблюдений: {self.raw_observation_shapes}"
        )

        # Определяем пространство наблюдений
        if self.flatten_obs:
            # Вычисляем общее количество наблюдений
            obs_size = 0
            for obs_spec in behavior_spec.observation_specs:
                obs_size += np.prod(obs_spec.shape)

            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(int(obs_size),), dtype=np.float32
            )
        else:
            # Если наблюдение только одно
            if len(behavior_spec.observation_specs) == 1:
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=behavior_spec.observation_specs[0].shape,
                    dtype=np.float32,
                )
            else:
                # Для множественных наблюдений используем Dict
                obs_spaces = {}
                for i, obs_spec in enumerate(behavior_spec.observation_specs):
                    obs_spaces[f"obs_{i}"] = spaces.Box(
                        low=-np.inf, high=np.inf, shape=obs_spec.shape, dtype=np.float32
                    )
                self.observation_space = spaces.Dict(obs_spaces)

        # Определяем пространство действий
        action_spec = behavior_spec.action_spec

        # Если действие континуальное
        if action_spec.continuous_size > 0:
            if self.normalize_actions:
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(action_spec.continuous_size,),
                    dtype=np.float32,
                )
            else:
                self.action_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(action_spec.continuous_size,),
                    dtype=np.float32,
                )

        # Если действие дискретное
        elif action_spec.discrete_size > 0:
            if action_spec.discrete_size == 1:
                self.action_space = spaces.Discrete(action_spec.discrete_branches[0])
            else:
                self.action_space = spaces.MultiDiscrete(action_spec.discrete_branches)

        logger.info(
            f"Настроены пространства: obs={self.observation_space}, action={self.action_space}"
        )

    @log_method_call(log_level=DEBUG, log_args=False)
    def reset(self):
        """
        Перезапускает среду Unity и возвращает начальное состояние.

        Returns:
            np.ndarray: Начальное наблюдение среды.
        """
        if self.unity_env is None:
            self._connect()

        try:
            # Сброс среды Unity
            self.unity_env.reset()

            # Получаем первое наблюдение
            decision_steps, _ = self.unity_env.get_steps(self.behavior_name)

            # Извлекаем и возвращаем наблюдение
            return self._extract_obs(decision_steps)
        except Exception as e:
            logger.error(f"Ошибка при сбросе среды: {str(e)}")
            # Пробуем переподключиться
            self._connect()
            decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
            return self._extract_obs(decision_steps)

    def _extract_obs(self, decision_steps):
        """
        Извлекает наблюдение из шага среды Unity.

        Args:
            decision_steps: Объект, содержащий текущие шаги среды.

        Returns:
            np.ndarray: Наблюдение агента в формате NumPy.
        """
        if decision_steps is None:
            logger.error("decision_steps равен None в _extract_obs")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        if len(decision_steps) == 0:
            logger.warning("Попытка извлечь наблюдение из пустого decision_steps")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        try:
            # Получаем индекс первого (и единственного) агента
            idx = 0

            # Проверяем наличие obs в decision_steps
            if not hasattr(decision_steps, "obs") or len(decision_steps.obs) == 0:
                logger.error("decision_steps не содержит наблюдений")
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            if self.flatten_obs:
                # Объединяем все наблюдения в плоский массив
                observations = []
                for obs in decision_steps.obs:
                    if obs is None or obs.size == 0:
                        logger.warning(f"Пустое наблюдение в decision_steps.obs")
                        continue
                    observations.append(obs[idx].flatten())

                if not observations:  # Если все наблюдения были пустыми
                    return np.zeros(self.observation_space.shape, dtype=np.float32)

                # Используем torch.no_grad() с тензорами на правильном устройстве
                with torch.no_grad():
                    np_obs = np.concatenate(observations)
                    tensor_obs = torch.as_tensor(
                        np_obs, dtype=torch.float32, device=self.device
                    )
                    return (
                        tensor_obs.cpu().numpy()
                        if self.device.type == "cuda"
                        else np_obs
                    )
            else:
                # Если только одно наблюдение и не нужно сжимать
                if len(decision_steps.obs) == 1:
                    if decision_steps.obs[0] is None or decision_steps.obs[0].size == 0:
                        logger.warning("Пустое наблюдение в decision_steps.obs[0]")
                        return np.zeros(self.observation_space.shape, dtype=np.float32)

                    with torch.no_grad():
                        np_obs = decision_steps.obs[0][idx]
                        tensor_obs = torch.as_tensor(
                            np_obs, dtype=torch.float32, device=self.device
                        )
                        return (
                            tensor_obs.cpu().numpy()
                            if self.device.type == "cuda"
                            else np_obs
                        )
                else:
                    # Несколько наблюдений, возвращаем словарь
                    obs_dict = {}
                    for i, obs in enumerate(decision_steps.obs):
                        if obs is None or obs.size == 0:
                            logger.warning(
                                f"Пустое наблюдение №{i} в decision_steps.obs"
                            )
                            obs_dict[f"obs_{i}"] = np.zeros(
                                self.observation_space[f"obs_{i}"].shape,
                                dtype=np.float32,
                            )
                        else:
                            with torch.no_grad():
                                np_obs = obs[idx]
                                tensor_obs = torch.as_tensor(
                                    np_obs, dtype=torch.float32, device=self.device
                                )
                                obs_dict[f"obs_{i}"] = (
                                    tensor_obs.cpu().numpy()
                                    if self.device.type == "cuda"
                                    else np_obs
                                )
                    return obs_dict
        except Exception as e:
            logger.error(f"Ошибка в _extract_obs: {str(e)}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    @log_method_call(log_level=DEBUG, log_args=False, log_return=False)
    def step(self, action):
        """
        Выполняет одно действие в среде и возвращает результат.

        Args:
            action (np.ndarray | torch.Tensor): Действие агента.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: \n
                - Новое наблюдение (obs),\n
                - Награда (reward),\n
                - Флаг завершения эпизода (done),\n
                - Дополнительная информация (info).\n
        """
        if self.unity_env is None:
            logger.error("Unity среда не инициализирована при вызове step")
            return np.zeros(self.observation_space.shape), 0.0, True, {}

        # Преобразуем действие в формат Unity
        with torch.no_grad():
            unity_action = self._prepare_action(action)

            # Отправляем действие в Unity
            self.unity_env.set_actions(self.behavior_name, unity_action)

            # Выполняем шаг в среде
            self.unity_env.step()

            # Получаем новое наблюдение, награду и флаг done
            decision_steps, terminal_steps = self.unity_env.get_steps(
                self.behavior_name
            )

            done = len(terminal_steps) > 0

            # Приоритет у terminal_steps, если агент завершил эпизод
            if done:
                reward = terminal_steps.reward[0]
                obs = self._extract_obs(terminal_steps)
                info = {"terminal": True}
            else:
                reward = decision_steps.reward[0]
                obs = self._extract_obs(decision_steps)
                info = {"terminal": False}

            return obs, float(reward), done, info

    def _prepare_action(self, action):
        """
        Преобразует действие из формата Gym в формат Unity.

        Args:
            action (np.ndarray | torch.Tensor | int): Действие в формате Gym.

        Returns:
            ActionTuple: Действие в формате Unity.
        """
        # Получаем спецификацию действия
        behavior_spec = self.unity_env.behavior_specs[self.behavior_name]
        action_spec = behavior_spec.action_spec

        # Преобразуем numpy массив в тензор для обработки на GPU
        if isinstance(action, np.ndarray):
            action_tensor = torch.as_tensor(
                action, dtype=torch.float32, device=self.device
            )
        elif isinstance(action, int):
            action_tensor = torch.tensor(
                [action], dtype=torch.int64, device=self.device
            )
        elif isinstance(action, torch.Tensor):
            action_tensor = action.to(self.device)
        else:
            raise ValueError(f"Неподдерживаемый тип действия: {type(action)}")

        # Для непрерывных действий
        if action_spec.continuous_size > 0:
            if self.normalize_actions and hasattr(self.action_space, "high"):
                # Размаскируем действие из [-1,1] до исходного диапазона, если оно нормализовано
                action_tensor = action_tensor.clamp(-1.0, 1.0)

            # Преобразуем тензор в numpy для Unity
            continuous_actions = action_tensor.cpu().numpy().reshape(1, -1)
            return ActionTuple(continuous=continuous_actions)

        # Для дискретных действий
        elif action_spec.discrete_size > 0:
            if isinstance(action, (int, np.int64, np.int32)):
                # Одно дискретное действие
                discrete_action = np.array([[action]], dtype=np.int32)
            else:
                # Несколько дискретных действий
                discrete_action = (
                    action_tensor.cpu().numpy().reshape(1, -1).astype(np.int32)
                )

            return ActionTuple(discrete=discrete_action)

    @log_method_call(log_level=DEBUG, log_args=False)
    def close(self):
        """
        Закрывает Unity среду
        """
        if self.unity_env is not None:
            self.unity_env.close()
            self.unity_env = None
            logger.info("Unity среда закрыта")

    def get_raw_observation_shapes(self) -> List[tuple]:
        """
        Возвращает исходные формы наблюдений из Unity среды (до объединения).

        Note:
            Эти формы могут быть использованы для ONNX экспорта.

        Returns:
            List[tuple]: Список кортежей, где каждый кортеж содержит форму одного наблюдения.
        """
        return self.raw_observation_shapes


# --- Многоагентная обёртка для Unity-среды ---
class MultiAgentUnityToGymWrapper(gym.Env):
    """
    Обертка для мультиагентной Unity среды в формате Gym
    """

    @log_method_call(log_level=DEBUG, log_args=True)
    def __init__(
        self,
        file_name: str = None,
        worker_id: int = 0,
        time_scale: float = 20.0,
        seed: int = 0,
        no_graphics: bool = False,
        flatten_obs: bool = True,
        normalize_actions: bool = True,
        timeout_wait: int = 300,
        base_port: int = 5005,
        memory_buffer_size: int = 4096,
    ) -> None:
        """
        Инициализирует мультиагентную обертку для Unity-среды.

        Args:
            file_name (str): Путь к файлу Unity-сценария.
            worker_id (int): Идентификатор рабочего процесса Unity.
            time_scale (float): Масштаб времени выполнения симуляции.
            seed (int): Сид для генерации случайных чисел.
            no_graphics (bool): Отключает графический интерфейс при запуске.
            flatten_obs (bool): Объединяет несколько наблюдений в одно.
            normalize_actions (bool): Нормализует действия до диапазона [-1, 1].
            timeout_wait (int): Время ожидания подключения к среде.
            base_port (int): Базовый порт для связи с Unity.
            memory_buffer_size (int): Размер буфера памяти для передачи данных.
        """
        log_method_call(
            logger,
            "MultiAgentUnityToGymWrapper.__init__",
            file_name=file_name,
            worker_id=worker_id,
            time_scale=time_scale,
        )

        if not unity_available:
            raise ImportError(
                "Пакет mlagents_envs не установлен. Установите его для работы с Unity средой."
            )

        super(MultiAgentUnityToGymWrapper, self).__init__()

        # Создаем каналы для взаимодействия с Unity
        self.engine_config_channel = EngineConfigurationChannel()
        self.float_properties = FloatPropertiesChannel()

        # Настройки среды
        self.file_name = file_name
        self.worker_id = worker_id
        self.time_scale = time_scale
        self.seed = seed
        self.no_graphics = no_graphics
        self.flatten_obs = flatten_obs
        self.normalize_actions = normalize_actions
        self.timeout_wait = timeout_wait
        self.base_port = base_port
        self.memory_buffer_size = memory_buffer_size

        # Unity среда
        self.unity_env = None
        self.behavior_name = None
        self.behavior_spec = None

        # Сохраняем информацию о формах входных данных
        self.raw_observation_shapes = []

        # Текущие агенты
        self.agent_ids = []

        # Пространства действий и наблюдений
        self.observation_space = None
        self.action_space = None

        # Кэш свойств
        self.properties = {}

        # Подключаемся к Unity
        self._connect()

        logger.info(
            f"Создана мультиагентная Gym обертка для Unity: file_name={file_name}, worker_id={worker_id}"
        )

    @log_method_call(log_level=DEBUG, log_args=True)
    def _connect(self, max_retries: int = 3, delay: float = 2.0) -> None:
        """
        Подключается к Unity-среде с повторными попытками.

        Args:
            max_retries (int): Максимальное количество попыток подключения.
            delay (float): Задержка между попытками в секундах.

        Raises:
            RuntimeError: Если все попытки подключения не удались.
        """
        log_method_call(
            logger,
            "MultiAgentUnityToGymWrapper._connect",
            max_retries=max_retries,
            delay=delay,
        )

        for attempt in range(max_retries):
            try:
                # Создаем среду Unity
                self.unity_env = UnityEnvironment(
                    file_name=self.file_name,
                    worker_id=self.worker_id,
                    seed=self.seed,
                    timeout_wait=self.timeout_wait,
                    side_channels=[self.engine_config_channel, self.float_properties],
                    base_port=self.base_port,
                    no_graphics=self.no_graphics,
                )

                # Настраиваем параметры среды, убираем no_graphics из параметров
                self.engine_config_channel.set_configuration_parameters(
                    time_scale=self.time_scale,
                    width=1280,
                    height=720,
                    quality_level=5,
                    target_frame_rate=-1,
                    capture_frame_rate=60,
                )

                # Сбрасываем среду для получения информации о поведениях
                self.unity_env.reset()
                self.behavior_name = list(self.unity_env.behavior_specs.keys())[0]
                self.behavior_spec = self.unity_env.behavior_specs[self.behavior_name]

                # Получаем список агентов
                decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
                self.agent_ids = list(decision_steps.agent_id)

                # Настраиваем пространства действий/наблюдений
                self._setup_spaces()

                logger.info(
                    f"Подключено к Unity среде, обнаружено {len(self.agent_ids)} агентов"
                )
                return

            except (UnityWorkerInUseException, UnityEnvironmentException) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Попытка {attempt + 1}/{max_retries} подключения к Unity не удалась: {str(e)}. Повторная попытка через {delay} сек..."
                    )
                    time.sleep(delay)
                    # Увеличиваем worker_id для следующей попытки
                    self.worker_id += 1
                else:
                    logger.error(
                        f"Не удалось подключиться к Unity после {max_retries} попыток: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Не удалось подключиться к Unity среде после {max_retries} попыток. Последняя ошибка: {str(e)}"
                    )

    def _setup_spaces(self) -> None:
        """
        Настраивает пространства действий и наблюдений для нескольких агентов.
        """
        # Получаем спецификацию среды
        observation_specs = self.behavior_spec.observation_specs
        action_spec = self.behavior_spec.action_spec

        # Сохраняем формы индивидуальных наблюдений для потенциального использования в экспорте ONNX
        self.raw_observation_shapes = []
        for obs_spec in observation_specs:
            self.raw_observation_shapes.append(obs_spec.shape)

        logger.info(
            f"Обнаружены входные формы наблюдений: {self.raw_observation_shapes}"
        )

        # Создаем пространство наблюдений
        if len(observation_specs) == 1 and not self.flatten_obs:
            # Простое пространство наблюдений
            obs_shape = observation_specs[0].shape
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
            )
        else:
            # Объединяем все наблюдения в одно пространство
            total_obs_size = sum(np.prod(spec.shape) for spec in observation_specs)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(int(total_obs_size),), dtype=np.float32
            )

        # Создаем пространство действий
        if action_spec.continuous_size > 0:
            # Непрерывные действия
            action_dim = action_spec.continuous_size
            if self.normalize_actions:
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
                )
            else:
                self.action_space = spaces.Box(
                    low=float("-inf"),
                    high=float("inf"),
                    shape=(action_dim,),
                    dtype=np.float32,
                )
        elif len(action_spec.discrete_branches) > 0:
            # Дискретные действия
            if len(action_spec.discrete_branches) == 1:
                # Одно дискретное действие
                self.action_space = spaces.Discrete(action_spec.discrete_branches[0])
            else:
                # Несколько дискретных действий
                self.action_space = spaces.MultiDiscrete(action_spec.discrete_branches)
        else:
            raise ValueError("В среде Unity не определены действия")

    @log_method_call(log_level=DEBUG, log_args=True)
    def reset(self, **kwargs) -> Tuple[Dict[int, np.ndarray], Dict]:
        """
        Сбрасывает мультиагентную среду и возвращает начальные наблюдения.

        Returns:
            Tuple[Dict[int, np.ndarray], Dict]: \n
                - Наблюдения для каждого агента по их ID. \n
                - Дополнительная информация. \n
        """
        log_method_call(logger, "MultiAgentUnityToGymWrapper.reset")

        if self.unity_env is None:
            raise RuntimeError("Среда не инициализирована")

        # Сбрасываем среду
        self.unity_env.reset()
        decision_steps, _ = self.unity_env.get_steps(self.behavior_name)

        # Обновляем список агентов
        self.agent_ids = list(decision_steps.agent_id)

        # Если агентов нет, возвращаем пустой словарь
        if not self.agent_ids:
            logger.warning("После сброса среды не обнаружено агентов")
            empty_dict = {}

            # Проверяем версию API для правильного возврата
            if hasattr(gym, "__version__") and gym.__version__ >= "0.26.0":
                return empty_dict, {}
            else:
                return empty_dict

        # Собираем наблюдения для каждого агента
        observations = {}
        try:
            for agent_id in self.agent_ids:
                observations[agent_id] = self._extract_obs(decision_steps, agent_id)

            # Проверяем версию API для правильного возврата
            if hasattr(gym, "__version__") and gym.__version__ >= "0.26.0":
                return observations, {}
            else:
                return observations
        except Exception as e:
            logger.error(f"Ошибка при извлечении наблюдений в reset(): {str(e)}")
            empty_dict = {
                agent_id: np.zeros(self.observation_space.shape, dtype=np.float32)
                for agent_id in self.agent_ids
            }

            # Проверяем версию API для правильного возврата
            if hasattr(gym, "__version__") and gym.__version__ >= "0.26.0":
                return empty_dict, {}
            else:
                return empty_dict

    @log_method_call(log_level=DEBUG, log_args=False, log_return=False)
    def step(
        self, actions: Dict[int, np.ndarray]
    ) -> Tuple[
        Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict
    ]:
        """
        Выполняет один шаг в мультиагентной среде.

        Args:
            actions (Dict[int, np.ndarray]): Действия для каждого агента по их ID.

        Returns:
            Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict]: \n
                - Новые наблюдения. \n
                - Награды. \n
                - Флаги завершения (dones). \n
                - Флаги обрезания (truncated) — новые в Gym v0.26+. \n
                - Дополнительная информация. \n
        """
        log_method_call(
            logger, "MultiAgentUnityToGymWrapper.step", action_count=len(actions)
        )

        if self.unity_env is None:
            raise RuntimeError("Среда не инициализирована")

        # Проверяем, что действия предоставлены для всех агентов
        if not self.agent_ids:
            logger.warning("Нет активных агентов для выполнения шага")
            empty_dict = {}
            # Для версии Gym >= 0.26.0 возвращаем 5 значений
            if hasattr(gym, "__version__") and gym.__version__ >= "0.26.0":
                return (
                    empty_dict,
                    empty_dict,
                    empty_dict,
                    empty_dict,
                    {"all_done": True},
                )
            else:
                return empty_dict, empty_dict, empty_dict, {"all_done": True}

        # Проверяем, что действия предоставлены для всех агентов
        missing_agents = [
            agent_id for agent_id in self.agent_ids if agent_id not in actions
        ]
        if missing_agents:
            logger.warning(f"Не предоставлены действия для агентов: {missing_agents}")
            # Создаем действия по умолчанию для отсутствующих агентов
            for agent_id in missing_agents:
                if isinstance(self.action_space, spaces.Discrete):
                    actions[agent_id] = 0
                else:
                    actions[agent_id] = np.zeros(
                        self.action_space.shape, dtype=np.float32
                    )

        try:
            # Подготавливаем и отправляем действия
            action_tuple = self._prepare_actions(actions)
            self.unity_env.set_actions(self.behavior_name, action_tuple)
            self.unity_env.step()

            # Получаем обновленные состояния
            decision_steps, terminal_steps = self.unity_env.get_steps(
                self.behavior_name
            )

            # Собираем информацию о результатах шага
            next_obs = {}
            rewards = {}
            dones = {}
            truncated = {}

            # Обрабатываем агентов, для которых эпизод завершился
            for agent_id in terminal_steps.agent_id:
                next_obs[agent_id] = self._extract_obs(terminal_steps, agent_id)
                rewards[agent_id] = terminal_steps.reward[agent_id]
                dones[agent_id] = True
                truncated[agent_id] = False

            # Обрабатываем оставшихся агентов
            for agent_id in decision_steps.agent_id:
                if agent_id not in dones:  # Если агент еще не обработан
                    next_obs[agent_id] = self._extract_obs(decision_steps, agent_id)
                    rewards[agent_id] = decision_steps.reward[agent_id]
                    dones[agent_id] = False
                    truncated[agent_id] = False

            # Обновляем список активных агентов
            self.agent_ids = list(decision_steps.agent_id)

            # Проверяем, завершились ли все эпизоды
            all_done = all(dones.values()) if dones else True

            # Для версии Gym >= 0.26.0 возвращаем 5 значений
            if hasattr(gym, "__version__") and gym.__version__ >= "0.26.0":
                return next_obs, rewards, dones, truncated, {"all_done": all_done}
            else:
                return next_obs, rewards, dones, {"all_done": all_done}

        except Exception as e:
            logger.error(f"Ошибка при выполнении шага: {str(e)}")
            # Создаем пустые словари для возврата в случае ошибки
            empty_obs = {
                agent_id: np.zeros(self.observation_space.shape, dtype=np.float32)
                for agent_id in self.agent_ids
            }
            empty_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            empty_dones = {agent_id: True for agent_id in self.agent_ids}
            empty_truncated = {agent_id: False for agent_id in self.agent_ids}

            # Для версии Gym >= 0.26.0 возвращаем 5 значений
            if hasattr(gym, "__version__") and gym.__version__ >= "0.26.0":
                return (
                    empty_obs,
                    empty_rewards,
                    empty_dones,
                    empty_truncated,
                    {"all_done": True, "error": str(e)},
                )
            else:
                return (
                    empty_obs,
                    empty_rewards,
                    empty_dones,
                    {"all_done": True, "error": str(e)},
                )

    def _extract_obs(
        self, steps: Union[DecisionSteps, TerminalSteps], agent_id: int
    ) -> np.ndarray:
        """
        Извлекает наблюдение конкретного агента из шага среды.

        Args:
            steps: Шаг среды, содержащий информацию о наблюдениях.
            agent_id (int): Идентификатор агента.

        Returns:
            np.ndarray: Наблюдение указанного агента.
        """
        # Найдем индекс агента в массиве ID
        idx = np.where(steps.agent_id == agent_id)[0]
        if len(idx) == 0:
            raise ValueError(f"Агент с ID {agent_id} не найден в текущем шаге")

        agent_idx = idx[0]

        if len(steps.obs) == 1 and not self.flatten_obs:
            # Простое наблюдение
            return steps.obs[0][agent_idx]
        else:
            # Объединяем все наблюдения
            observations = []
            for obs in steps.obs:
                observations.append(obs[agent_idx].flatten())

            return np.concatenate(observations)

    def _prepare_actions(self, actions: Dict[int, np.ndarray]) -> ActionTuple:
        """
        Преобразует действия из формата Gym в формат Unity.

        Args:
            actions (Dict[int, np.ndarray]): Действия агентов по их ID.

        Returns:
            ActionTuple: Действия в формате Unity.
        """
        action_spec = self.behavior_spec.action_spec
        num_agents = len(self.agent_ids)

        if action_spec.continuous_size > 0:
            # Непрерывные действия
            continuous_actions = np.zeros(
                (num_agents, action_spec.continuous_size), dtype=np.float32
            )

            for i, agent_id in enumerate(self.agent_ids):
                continuous_actions[i] = actions[agent_id]

            return ActionTuple(continuous=continuous_actions)
        else:
            # Дискретные действия
            num_branches = len(action_spec.discrete_branches)
            discrete_actions = np.zeros((num_agents, num_branches), dtype=np.int32)

            for i, agent_id in enumerate(self.agent_ids):
                agent_action = actions[agent_id]

                if num_branches == 1:
                    # Если одна ветка, то действие может быть скаляром
                    if isinstance(agent_action, (int, np.integer)):
                        discrete_actions[i, 0] = agent_action
                    else:
                        discrete_actions[i, 0] = agent_action[0]
                else:
                    # Несколько дискретных действий
                    discrete_actions[i] = agent_action

            return ActionTuple(discrete=discrete_actions)

    @log_method_call(log_level=DEBUG, log_args=False)
    def close(self) -> None:
        """
        Закрывает среду
        """
        if self.unity_env:
            try:
                self.unity_env.close()
                logger.info("Среда Unity успешно закрыта")
            except Exception as e:
                logger.error(f"Ошибка при закрытии среды Unity: {str(e)}")

    def set_property(self, key: str, value: float) -> None:
        """
        Устанавливает значение свойства в среде Unity.

        Args:
            key (str): Ключ свойства.
            value (float): Значение свойства.
        """
        self.float_properties.set_property(key, value)

    def get_property(self, key: str) -> float:
        """
        Получает значение свойства из среды Unity.

        Args:
            key (str): Ключ свойства.

        Returns:
            float: Текущее значение свойства.
        """
        return self.float_properties.get_property(key)

    def get_agent_ids(self) -> List[int]:
        """
        Возвращает список идентификаторов активных агентов.

        Returns:
            List[int]: Список ID агентов.
        """
        return self.agent_ids

    def __enter__(self):
        """
        Контекстный менеджер: вход
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Контекстный менеджер: выход
        """
        self.close()

    def get_raw_observation_shapes(self) -> List[tuple]:
        """
        Возвращает исходные формы наблюдений из Unity среды (до объединения).
        Эти формы могут быть использованы для ONNX экспорта.

        Returns:
            List[tuple]: Список форм наблюдений.
        """
        return self.raw_observation_shapes
