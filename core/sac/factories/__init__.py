import torch
import torch.nn as nn
from typing import (
    List,
    Dict,
    Tuple,
    Optional,
    Union,
    Any,
    Type,
    Callable,
    TypeVar,
    Generic,
)
import yaml
import json
import numpy as np
import functools
from dataclasses import dataclass, field
from pathlib import Path
import time
import inspect
import importlib

from core.sac.buffers import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    SequenceReplayBuffer,
    PrioritizedSequenceReplayBuffer,
    BaseReplayBuffer,
)
from core.sac.normalizers import (
    WelfordObservationNormalizer,
    BatchMeanStdNormalizer,
    DummyNormalizer,
)
from core.sac.activations import create_activation
from core.envs.env import UnityToGymWrapper, MultiAgentUnityToGymWrapper, UnityEnvConfig
from core.logging.logger import get_logger
from core.utils.device_utils import device_manager

from core.sac.agent import SACAgent
from core.sac.networks import (
    MLPPolicy,
    LSTMPolicy,
    DualMLPQNet,
    DualLSTMQNet,
    BasePolicy,
    BaseQNet,
)
from core.sac.interfaces.factory_interface import FactoryInterface

try:
    from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    class SklearnDecisionTreeRegressor:
        def __init__(self, *args, **kwargs):
            logger.warning("sklearn.tree.DecisionTreeRegressor не найден.")
            raise NotImplementedError("Sklearn недоступен.")


logger = get_logger("factories")

# Типизированный кэш для фабрик
T_co = TypeVar("T_co", covariant=True)


class FactoryCache(Generic[T_co]):
    """
    Простой кэш для объектов, создаваемых фабриками, с ограничением размера.
    Использует FIFO-стратегию (First-In, First-Out) для вытеснения при переполнении.

    Attributes:
        cache: Словарь, хранящий закэшированные объекты.
        max_size: Максимальный размер кэша.
        _keys_fifo: Список ключей, используемый для отслеживания порядка добавления.
    """

    def __init__(self, max_size: int = 10):
        """
        Инициализирует кэш с заданным максимальным размером.

        Args:
            max_size (int): Максимальное количество объектов, которые может хранить кэш.

        Raises:
            ValueError: Если передано значение max_size <= 0.
        """
        if max_size <= 0:
            raise ValueError("Максимальный размер кэша должен быть положительным.")
        self.cache: Dict[str, T_co] = {}
        self.max_size = max_size
        self._keys_fifo: List[str] = []
        logger.debug(f"Инициализирован FactoryCache с max_size={max_size}")

    def get(self, key: str) -> Optional[T_co]:
        """
        Получает объект из кэша по указанному ключу.

        Args:
            key (str): Ключ для поиска объекта.

        Returns:
            Optional[T_co]: Объект, если он найден, иначе None.
        """
        return self.cache.get(key)

    def put(self, key: str, obj: T_co) -> None:
        """
        Добавляет объект в кэш. Если кэш переполнен, удаляет самый старый элемент.

        Args:
            key (str): Ключ для сохраняемого объекта.
            obj (T_co): Объект, который нужно добавить в кэш.
        """
        if key in self.cache:
            # Объект уже в кэше, ничего делать не нужно
            return

        if len(self.cache) >= self.max_size:
            # Удаляем самый старый ключ
            key_to_remove = self._keys_fifo.pop(0)
            if key_to_remove in self.cache:
                del self.cache[key_to_remove]
                logger.debug(
                    f"Кэш переполнен ({self.max_size}). Удален старый ключ '{key_to_remove}'."
                )
            else:
                logger.warning(f"Ключ '{key_to_remove}' для удаления не найден в кэше.")

        self.cache[key] = obj
        self._keys_fifo.append(key)
        logger.debug(
            f"Объект добавлен в кэш по ключу '{key}'. Текущий размер: {len(self.cache)}.)"
        )

    def clear(self) -> None:
        """
        Очищает все содержимое кэша.
        """
        self.cache.clear()
        self._keys_fifo.clear()
        logger.debug("Кэш фабрики очищен.")


# Декоратор для кэширования результатов методов фабрик
def cache_factory_result(method: Callable[..., T_co]) -> Callable[..., T_co]:
    """
    Декоратор для кэширования результатов фабричных методов класса.
    Предполагает наличие атрибута `_cache` типа `FactoryCache`.

    Args:
        method (Callable[..., T_co]): Метод, результат которого будет кэшироваться.

    Returns:
        Callable[..., T_co]: Обёрнутая версия метода с кэшированием.
    """

    @functools.wraps(method)
    def wrapper(cls: Type, *args: Any, **kwargs: Any) -> T_co:
        # Проверяем наличие кэша у класса
        if not hasattr(cls, "_cache") or not isinstance(
            getattr(cls, "_cache", None), FactoryCache
        ):
            # Инициализируем кэш, если его нет
            logger.warning(
                f"Класс {cls.__name__} не имеет атрибута '_cache'. Создается новый кэш."
            )
            cls._cache = FactoryCache()

        # Формируем уникальный ключ на основе имени метода и его аргументов
        # Учитываем только hashable аргументы для ключа
        key_parts = [str(method.__name__)]
        try:
            hashable_args = [
                arg
                for arg in args
                if isinstance(arg, (int, float, str, bool, tuple, type(None)))
            ]
            hashable_kwargs = {
                k: v
                for k, v in kwargs.items()
                if isinstance(v, (int, float, str, bool, tuple, type(None)))
            }
            key_parts.extend(map(str, hashable_args))
            key_parts.extend(f"{k}={v}" for k, v in sorted(hashable_kwargs.items()))
            cache_key = "::".join(key_parts)
        except Exception as e:
            logger.warning(
                f"Ошибка при генерации ключа кэша для {method.__name__}: {e}. Кэширование отключено для этого вызова."
            )
            return method(cls, *args, **kwargs)

        cache = cls._cache

        # Пытаемся получить объект из кэша
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Если в кэше нет, создаем новый объект
        result = method(cls, *args, **kwargs)
        cache.put(cache_key, result)
        return result

    return wrapper


# Инициализация весов
def init_weights(m: nn.Module, init_w: float = 3e-3) -> None:
    """
    Инициализация весов для линейных слоев и LSTM.
    Использует Xavier Uniform для большинства слоев и ограниченный Uniform для выходных.

    Args:
        m (nn.Module): Модуль PyTorch, веса которого необходимо инициализировать.
        init_w (float): Диапазон значений для инициализации выходного слоя.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Инициализация forget gate bias в 1
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

    elif isinstance(m, nn.LayerNorm):
        if m.elementwise_affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# --- Фабрики --- #


class ActivationFactory(FactoryInterface):
    """
    Фабрика для создания экземпляров функций активации nn.Module.
    Использует `create_activation` из модуля `activations`.
    Предоставляет кэширование созданных активаций.
    """

    _cache: FactoryCache[nn.Module] = FactoryCache(max_size=15)

    @classmethod
    @cache_factory_result
    def create(cls, activation_type: str, **kwargs: Any) -> nn.Module:
        """
        Создаёт экземпляр функции активации по её имени.

        Args:
            activation_type (str): Название функции активации.
            **kwargs: Параметры, передаваемые в `create_activation`.

        Returns:
            nn.Module: Экземпляр функции активации.

        Raises:
            ValueError: Если указана неизвестная функция активации.
            Exception: Для других ошибок при создании активации.
        """
        try:
            # Делегируем создание модулю activations
            activation_module = create_activation(activation_type, **kwargs)
            logger.debug(
                f"Создана активация '{activation_type}' с параметрами: {kwargs}"
            )
            return activation_module
        except ValueError as e:
            logger.error(f"Ошибка при создании активации '{activation_type}': {e}")
            raise  # Перебрасываем исключение
        except Exception as e:
            logger.error(
                f"Неожиданная ошибка при создании активации '{activation_type}': {e}"
            )
            raise

    @classmethod
    def list_available(cls) -> List[str]:
        """
        Возвращает список доступных типов функций активации.

        Returns:
            List[str]: Список названий всех поддерживаемых активаций.
        """
        # Динамически получаем список из модуля activations, если возможно
        try:
            activations_module = importlib.import_module("data.models.activations")
            if hasattr(activations_module, "_available_activations") and isinstance(
                getattr(activations_module, "_available_activations"), dict
            ):
                # Доступ к приватному словарю - не лучший вариант, но работает
                return list(activations_module._available_activations.keys())
            else:
                # Запасной вариант - жестко закодированный список
                logger.warning(
                    "Не удалось динамически получить список активаций, используется жестко заданный."
                )
                # Ensure the list is comprehensive and matches activations.py
                return [
                    "relu",
                    "leaky_relu",
                    "elu",
                    "selu",
                    "tanh",
                    "sigmoid",
                    "gelu",
                    "swish",
                    "mish",
                    "softplus",
                    "identity",
                ]
        except ImportError:
            logger.error(
                "Не удалось импортировать модуль data.models.activations для получения списка активаций."
            )
            return []


@dataclass
class EnvContainer:
    """
    Контейнер для хранения окружения и метаданных о его использовании.

    Attributes:
        env: Экземпляр окружения.
        created_at: Временная метка создания окружения.
        last_used: Временная метка последнего использования окружения.
        use_count: Количество раз, которое окружение было использовано.
    """

    env: Any
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0

    def update_usage(self) -> None:
        """Обновляет время последнего использования и счетчик запусков."""
        self.last_used = time.time()
        self.use_count += 1


class EnvFactory:
    """
    Фабрика для создания и управления экземплярами окружений, в т.ч. Unity.
    Поддерживает загрузку конфигураций и кэширование созданных окружений.
    """

    def __init__(self) -> None:
        """
        Инициализирует фабрику окружений.
        """
        self.configs: Dict[str, UnityEnvConfig] = {}
        self.envs: Dict[str, EnvContainer] = {}
        logger.info("Инициализирована EnvFactory.")

    def register_config(self, config: UnityEnvConfig) -> None:
        """
        Регистрирует конфигурацию окружения.

        Args:
            config (UnityEnvConfig): Конфигурация окружения.

        Raises:
            TypeError: Если передан неверный тип конфигурации.
        """
        if not isinstance(config, UnityEnvConfig):
            raise TypeError("config должен быть экземпляром UnityEnvConfig")
        self.configs[config.env_name] = config
        logger.info(f"Конфигурация для окружения '{config.env_name}' зарегистрирована.")

    def load_config(
        self, config_file: Union[str, Path], env_name: Optional[str] = None
    ) -> UnityEnvConfig:
        """
        Загружает конфигурацию окружения из файла YAML или JSON.

        Args:
            config_file (Union[str, Path]): Путь к файлу конфигурации.
            env_name (Optional[str]): Имя окружения (необязательно).

        Returns:
            UnityEnvConfig: Загруженная конфигурация.

        Raises:
            FileNotFoundError: Если файл не найден.
            Exception: При ошибках загрузки конфигурации.
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

        try:
            # Загружаем как словарь
            loaded_dict = ConfigFactory.load_config(str(config_path))
            # Создаем объект UnityEnvConfig
            config = UnityEnvConfig()
            config.env_name = env_name or loaded_dict.get("env_name", config_path.stem)
            config.file_name = loaded_dict.get("file_name")
            config.port = loaded_dict.get("port", config.port)
            config.seed = loaded_dict.get("seed", config.seed)
            config.timeout_wait = loaded_dict.get("timeout_wait", config.timeout_wait)
            config.multi_agent = loaded_dict.get("multi_agent", config.multi_agent)
            # Добавляем остальные свойства
            for key, value in loaded_dict.items():
                if key not in [
                    "env_name",
                    "file_name",
                    "port",
                    "seed",
                    "timeout_wait",
                    "multi_agent",
                ]:
                    config.set_property(key, value)

            logger.info(f"Конфигурация '{config.env_name}' загружена из {config_path}.")
            self.register_config(config)
            return config
        except Exception as e:
            logger.error(
                f"Ошибка при загрузке конфигурации окружения из {config_path}: {e}"
            )
            raise

    def create_gym_env(
        self,
        env_name: str,
        config: Optional[Union[UnityEnvConfig, Dict[str, Any]]] = None,
    ) -> Union[UnityToGymWrapper, MultiAgentUnityToGymWrapper]:
        """
        Создаёт Gym-совместимую обёртку окружения Unity.

        Args:
            env_name (str): Имя окружения.
            config (Optional[Union[UnityEnvConfig, Dict[str, Any]]]): Конфигурация окружения.

        Returns:
            Union[UnityToGymWrapper, MultiAgentUnityToGymWrapper]: Созданное окружение.

        Raises:
            ValueError: Если конфигурация не найдена или некорректна.
            Exception: При ошибках инициализации окружения.
        """
        if env_name in self.envs:
            env_container = self.envs[env_name]
            env_container.update_usage()
            logger.info(
                f"Возвращено кэшированное окружение '{env_name}'. Использовано раз: {env_container.use_count}."
            )
            return env_container.env

        env_config: Optional[UnityEnvConfig] = None
        if isinstance(config, UnityEnvConfig):
            env_config = config
            if env_name not in self.configs or self.configs[env_name] != config:
                self.register_config(config)
        elif isinstance(config, dict):
            logger.debug(f"Создание UnityEnvConfig из словаря для '{env_name}'.")
            try:
                env_config = UnityEnvConfig()
                env_config.env_name = env_name
                for key, value in config.items():
                    if hasattr(env_config, key):
                        setattr(env_config, key, value)
                    elif key == "properties" and isinstance(value, dict):
                        for prop_key, prop_value in value.items():
                            env_config.set_property(prop_key, prop_value)
                    else:
                        env_config.set_property(key, value)
                self.register_config(env_config)
            except Exception as e:
                logger.error(f"Ошибка при создании UnityEnvConfig из словаря: {e}")
                raise ValueError(f"Некорректный словарь конфигурации для '{env_name}'.")
        elif config is None:
            if env_name not in self.configs:
                raise ValueError(
                    f"Конфигурация для окружения '{env_name}' не найдена и не передана."
                )
            env_config = self.configs[env_name]
        else:
            raise TypeError(
                "Параметр config должен быть UnityEnvConfig, dict или None."
            )

        if env_config is None:
            raise RuntimeError(
                f"Не удалось получить конфигурацию для окружения '{env_name}'."
            )

        is_multi_agent = getattr(env_config, "multi_agent", False)
        logger.info(
            f"Создание '{env_name}' (multi_agent={is_multi_agent}) с параметрами: {env_config.get_gym_kwargs()}"
        )

        try:
            if is_multi_agent:
                env = MultiAgentUnityToGymWrapper(**env_config.get_gym_kwargs())
                logger.info(f"Создано многоагентное gym-окружение '{env_name}'.")
            else:
                env = UnityToGymWrapper(**env_config.get_gym_kwargs())
                logger.info(f"Создано одноагентное gym-окружение '{env_name}'.")
        except Exception as e:
            logger.error(f"Ошибка при создании экземпляра окружения '{env_name}': {e}")
            raise

        self.envs[env_name] = EnvContainer(env=env)
        logger.info(f"Окружение '{env_name}' добавлено в кэш.")

        return env

    def get_env(self, env_name: str) -> Any:
        """
        Возвращает кэшированный экземпляр окружения.

        Args:
            env_name (str): Имя окружения.

        Returns:
            Any: Экземпляр окружения.

        Raises:
            ValueError: Если окружение не найдено в кэше.
        """
        if env_name not in self.envs:
            raise ValueError(
                f"Окружение '{env_name}' не найдено в кэше. Создайте его с помощью create_gym_env()."
            )

        env_container = self.envs[env_name]
        env_container.update_usage()
        logger.debug(f"Возвращено окружение '{env_name}' из кэша.")
        return env_container.env

    def close_env(self, env_name: str) -> None:
        """
        Закрывает окружение и удаляет его из кэша.

        Args:
            env_name (str): Имя окружения.
        """
        if env_name in self.envs:
            env_container = self.envs.pop(env_name)
            try:
                env_container.env.close()
                logger.info(
                    f"Окружение '{env_name}' успешно закрыто и удалено из кэша."
                )
            except Exception as e:
                logger.error(f"Ошибка при закрытии окружения '{env_name}': {e}")
        else:
            logger.warning(
                f"Попытка закрыть несуществующее в кэше окружение '{env_name}'."
            )

    def close_all_envs(self) -> None:
        """
        Закрывает все кэшированные окружения.
        """
        closed_count = 0
        for env_name in list(self.envs.keys()):
            self.close_env(env_name)
            closed_count += 1
        logger.info(f"Закрыто {closed_count} окружений.")

    def __del__(self) -> None:
        """
        Деструктор для автоматического закрытия окружений при уничтожении фабрики.
        """
        logger.debug("Вызван деструктор EnvFactory, закрытие оставшихся окружений...")
        self.close_all_envs()


class PolicyFactory(FactoryInterface):
    """
    Фабрика для создания сетей политики (MLP или LSTM).
    Предоставляет кэширование созданных политик.

    Attributes:
        _cache: Кэш для ранее созданных сетей политики.
    """

    _cache: FactoryCache[Union[MLPPolicy, LSTMPolicy]] = FactoryCache(max_size=10)

    @classmethod
    @cache_factory_result
    def create(
        cls,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation_fn: str = "relu",
        use_lstm: bool = False,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_spectral_norm: bool = False,
        use_fused_ops: bool = True,
        forget_bias: float = 1.0,
        cache_forward: bool = True,
        device: str = "cpu",
        use_amp: bool = True,
        custom_init=None,
    ) -> Union[MLPPolicy, LSTMPolicy]:
        """
        Создаёт сеть политики (MLP или LSTM) с заданными параметрами.

        Args:
            obs_dim (int): Размерность наблюдений.
            action_dim (int): Размерность действий.
            hidden_dim (int): Число нейронов в скрытых слоях.
            num_layers (int): Количество слоёв сети.
            activation_fn (str): Название функции активации.
            use_lstm (bool): Использовать ли LSTM вместо MLP.
            dropout (float): Вероятность отключения нейронов (для Dropout).
            use_layer_norm (bool): Использовать ли нормализацию слоёв.
            use_spectral_norm (bool): Применять ли спектральную нормализацию для стабилизации.
            use_fused_ops (bool): Использовать ли fused операции на CUDA.
            forget_bias (float): Bias для forget gate в LSTM (обычно 1.0).
            device (str): Устройство вычисления ('cpu' или 'cuda').

        Returns:
            Union[MLPPolicy, LSTMPolicy]: Созданная сеть политики.

        Raises:
            Exception: При ошибке создания сети.
        """
        target_device = device_manager.get_device(device)

        try:
            if use_lstm:
                policy = LSTMPolicy(
                    input_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    activation=activation_fn,
                    dropout_rate=dropout,
                    bidirectional=False,
                    use_layer_norm=use_layer_norm,
                )
                logger.info(
                    f"Создана LSTMPolicy: hidden={hidden_dim}, layers={num_layers}, device={target_device}"
                )
            else:
                hidden_dims = [hidden_dim] * num_layers
                policy = MLPPolicy(
                    input_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    activation=activation_fn,
                    dropout_rate=dropout,
                    use_layer_norm=use_layer_norm,
                )
                logger.info(
                    f"Создана MLPPolicy: hidden={hidden_dims}, device={target_device}"
                )

            policy = policy.to(target_device)

            return policy
        except Exception as e:
            logger.error(f"Ошибка при создании сети политики: {e}")
            raise


class QNetFactory(FactoryInterface):
    """
    Фабрика для создания двойных Q-сетей (Dual MLP или Dual LSTM).
    Предоставляет кэширование созданных Q-сетей.

    Attributes:
        _cache: Кэш для объектов класса.
        create: Создаёт экземпляр двойной Q-сети.
    """

    _cache: FactoryCache[Union[DualMLPQNet, DualLSTMQNet]] = FactoryCache(max_size=10)

    @classmethod
    @cache_factory_result
    def create(
        cls,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation_fn: str = "relu",
        use_lstm: bool = False,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        device: str = "cpu",
    ) -> Union[DualMLPQNet, DualLSTMQNet]:
        """
        Создает экземпляр двойной Q-сети (Dual MLP или Dual LSTM).

        Args:
            obs_dim (int): Размерность наблюдений.
            action_dim (int): Размерность действий.
            hidden_dim (int): Количество нейронов в скрытых слоях.
            num_layers (int): Количество слоев сети.
            activation_fn (str): Название функции активации.
            use_lstm (bool): Использовать ли LSTM вместо MLP.
            dropout (float): Вероятность отсечения весов.
            use_layer_norm (bool): Включить ли нормализацию слоёв.
            device (str): Устройство ('cpu' или 'cuda').

        Returns:
            Union[DualMLPQNet, DualLSTMQNet]: Созданная Q-сеть.

        Raises:
            Exception: Если произошла ошибка при создании Q-сети.
        """
        target_device = device_manager.get_device(device)

        try:
            if use_lstm:
                q_net = DualLSTMQNet(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    activation=activation_fn,
                    dropout_rate=dropout,
                    bidirectional=False,
                    use_layer_norm=use_layer_norm,
                )
                logger.info(
                    f"Создана DualLSTMQNet: hidden={hidden_dim}, layers={num_layers}, device={target_device}"
                )
            else:
                hidden_dims = [hidden_dim] * num_layers
                q_net = DualMLPQNet(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    activation=activation_fn,
                    dropout_rate=dropout,
                    use_layer_norm=use_layer_norm,
                )
                logger.info(
                    f"Создана DualMLPQNet: hidden={hidden_dims}, device={target_device}"
                )

            q_net = q_net.to(target_device)

            return q_net
        except Exception as e:
            logger.error(f"Ошибка при создании Q-сети: {e}")
            raise


class ReplayBufferFactory:
    """
    Фабрика для создания буферов воспроизведения опыта.
    Поддерживает стандартный, приоритезированный, последовательный и
    приоритезированный последовательный буферы.

    Attributes:
        create: Создаёт буфер воспроизведения нужного типа.
    """

    @staticmethod
    def create(
        capacity: int,
        obs_dim: Union[int, Tuple],
        action_dim: int,
        use_prioritized: bool = False,
        use_sequence: bool = False,
        sequence_length: int = 10,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing_steps: Optional[int] = None,
        epsilon: float = 1e-5,
        device: str = "cpu",
        storage_dtype: torch.dtype = torch.float32,
        tree_dtype: Optional[torch.dtype] = None,
    ) -> BaseReplayBuffer:
        """
        Создает буфер воспроизведения нужного типа.

        Args:
            capacity (int): Вместимость буфера.
            obs_dim (Union[int, Tuple]): Размерность наблюдений.
            action_dim (int): Размерность действий.
            use_prioritized (bool): Использовать ли приоритезированное воспроизведение.
            use_sequence (bool): Использовать ли последовательное воспроизведение.
            sequence_length (int): Длина последовательности для последовательного воспроизведения.
            alpha (float): Параметр приоритета выборки (для PrioritizedReplayBuffer).
            beta (float): Параметр важности выборки (для PrioritizedReplayBuffer).
            beta_annealing_steps (Optional[int]): Шаги для увеличения beta.
            epsilon (float): Малое значение для добавления к приоритетам.
            device (str): Устройство ('cpu' или 'cuda').
            storage_dtype (torch.dtype): Тип данных для хранения данных в буфере.
            tree_dtype (Optional[torch.dtype]): Тип данных для хранения дерева (SumTree/MinTree).

        Returns:
            BaseReplayBuffer: Созданный буфер воспроизведения.

        Raises:
            ValueError: Если емкость <= 0.
            Exception: Если произошла ошибка при создании буфера.
        """
        if capacity <= 0:
            raise ValueError("Емкость буфера (capacity) должна быть положительной.")
        target_device = device_manager.get_device(device)
        if tree_dtype is None:
            tree_dtype = storage_dtype
            logger.debug(
                f"tree_dtype не указан, используем storage_dtype: {storage_dtype}"
            )
        try:
            if use_sequence and use_prioritized:
                buffer = PrioritizedSequenceReplayBuffer(
                    capacity=capacity,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    sequence_length=sequence_length,
                    alpha=alpha,
                    beta=beta,
                    beta_annealing_steps=beta_annealing_steps,
                    epsilon=epsilon,
                    device=str(target_device),
                    storage_dtype=storage_dtype,
                    tree_dtype=tree_dtype,
                )
                logger.info(
                    f"Создан PrioritizedSequenceReplayBuffer: capacity={capacity}, seq_len={sequence_length}, device={target_device}, storage_dtype={storage_dtype}, tree_dtype={tree_dtype}"
                )
            elif use_sequence:
                buffer = SequenceReplayBuffer(
                    capacity=capacity,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    sequence_length=sequence_length,
                    device=str(target_device),
                    storage_dtype=storage_dtype,
                )
                logger.info(
                    f"Создан SequenceReplayBuffer: capacity={capacity}, seq_len={sequence_length}, device={target_device}, storage_dtype={storage_dtype}"
                )
            elif use_prioritized:
                buffer = PrioritizedReplayBuffer(
                    capacity=capacity,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    alpha=alpha,
                    beta=beta,
                    beta_annealing_steps=beta_annealing_steps,
                    epsilon=epsilon,
                    device=str(target_device),
                    storage_dtype=storage_dtype,
                    tree_dtype=tree_dtype,
                )
                logger.info(
                    f"Создан PrioritizedReplayBuffer: capacity={capacity}, alpha={alpha}, beta={beta}, device={target_device}, storage_dtype={storage_dtype}, tree_dtype={tree_dtype}"
                )
            else:
                buffer = ReplayBuffer(
                    capacity=capacity,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    device=str(target_device),
                    storage_dtype=storage_dtype,
                )
                logger.info(
                    f"Создан ReplayBuffer: capacity={capacity}, device={target_device}, storage_dtype={storage_dtype}"
                )
            return buffer
        except Exception as e:
            logger.error(f"Ошибка при создании буфера воспроизведения: {e}")
            raise


class NormalizerFactory(FactoryInterface):
    """
    Фабрика для создания нормализаторов наблюдений.
    Поддерживает Welford, BatchMeanStd и Dummy.

    Attributes:
        _normalizer_types (Dict[str, Type]): Словарь поддерживаемых нормализаторов.
        create: Создаёт нормализатор нужного типа.
        list_available: Возвращает список доступных типов нормализаторов.
    """

    _normalizer_types = {
        "batch_mean_std": BatchMeanStdNormalizer,
        "welford": WelfordObservationNormalizer,
        "dummy": DummyNormalizer,
    }

    @classmethod
    def create(
        cls,
        obs_dim: Union[int, Tuple],
        use_normalizer: bool = True,
        normalizer_type: str = "welford",
        clip_range: float = 5.0,
        epsilon: float = 1e-8,
    ) -> Union[WelfordObservationNormalizer, BatchMeanStdNormalizer, DummyNormalizer]:
        """
        Создает нормализатор наблюдений нужного типа.

        Args:
            obs_dim (Union[int, Tuple]): Размерность наблюдений.
            use_normalizer (bool): Включить ли нормализацию.
            normalizer_type (str): Тип нормализатора ("batch_mean_std", "welford", "dummy").
            clip_range (float): Диапазон ограничения значений после нормализации.
            epsilon (float): Малое значение для предотвращения деления на ноль.

        Returns:
            Union[WelfordObservationNormalizer, BatchMeanStdNormalizer, DummyNormalizer]: 
                Экземпляр указанного нормализатора.

        Raises:
            ValueError: Если `obs_dim` не соответствует требуемому типу.
            Exception: Если возникла ошибка при создании нормализатора.
        """
        if not use_normalizer:
            logger.info("Нормализация отключена, используется DummyNormalizer.")
            return DummyNormalizer()

        normalizer_type = normalizer_type.lower()

        if normalizer_type not in cls._normalizer_types:
            available = ", ".join(cls._normalizer_types.keys())
            logger.warning(
                f"Неизвестный тип нормализатора: '{normalizer_type}', доступные: {available}. Используется DummyNormalizer."
            )
            return DummyNormalizer()

        normalizer_class = cls._normalizer_types[normalizer_type]

        try:
            if normalizer_type == "batch_mean_std":
                shape = (obs_dim,) if isinstance(obs_dim, int) else obs_dim
                normalizer = normalizer_class(
                    shape=shape, clip_range=clip_range, epsilon=epsilon
                )
                logger.info(
                    f"Создан BatchMeanStdNormalizer: shape={shape}, clip={clip_range}, eps={epsilon}"
                )
            elif normalizer_type == "welford":
                if not isinstance(obs_dim, int):
                    raise ValueError(
                        "WelfordObservationNormalizer ожидает целочисленный obs_dim."
                    )
                normalizer = normalizer_class(
                    obs_dim=obs_dim, clip_range=clip_range, epsilon=epsilon
                )
                logger.info(
                    f"Создан WelfordObservationNormalizer: dim={obs_dim}, clip={clip_range}, eps={epsilon}"
                )
            else:
                normalizer = normalizer_class()
                logger.info("Создан DummyNormalizer.")

            return normalizer
        except Exception as e:
            logger.error(f"Ошибка при создании нормализатора '{normalizer_type}': {e}")
            raise

    @classmethod
    def list_available(cls) -> List[str]:
        """
        Возвращает список доступных типов нормализаторов.

        Returns:
            List[str]: Доступные типы нормализаторов.
        """
        return list(cls._normalizer_types.keys())


class OptimizerFactory(FactoryInterface):
    """
    Фабрика для создания оптимизаторов PyTorch.

    Attributes:
        _optimizers (Dict[str, torch.optim.Optimizer]): Словарь поддерживаемых оптимизаторов.
        create: Создаёт оптимизатор PyTorch по имени.
        get_available_optimizers: Возвращает список доступных оптимизаторов.
    """

    _optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
    }

    @staticmethod
    def create(
        params: Any,
        optimizer_type: str = "adam",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        """
        Создает оптимизатор PyTorch по его имени.

        Args:
            params (Any): Параметры модели, которые нужно оптимизировать.
            optimizer_type (str): Тип оптимизатора ("adam", "sgd" и др.).
            lr (float): Скорость обучения.
            weight_decay (float): Сила L2-регуляризации.
            **kwargs: Дополнительные параметры для оптимизатора.

        Returns:
            torch.optim.Optimizer: Созданный оптимизатор.

        Raises:
            Exception: Если возникает ошибка при создании оптимизатора.
        """
        optimizer_type = optimizer_type.lower()

        if optimizer_type not in OptimizerFactory._optimizers:
            available = ", ".join(OptimizerFactory._optimizers.keys())
            logger.warning(
                f"Неизвестный тип оптимизатора: '{optimizer_type}', доступные: {available}. Используется Adam."
            )
            optimizer_type = "adam"

        optimizer_class = OptimizerFactory._optimizers[optimizer_type]

        optimizer_args = {"lr": lr}

        sig = inspect.signature(optimizer_class.__init__)
        allowed_kwargs = {
            k
            for k, p in sig.parameters.items()
            if p.kind == inspect.Parameter.KEYWORD_ONLY
            or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            or k == "weight_decay"
        }

        if "weight_decay" in allowed_kwargs:
            optimizer_args["weight_decay"] = weight_decay

        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in allowed_kwargs and k not in ["lr", "weight_decay"]
        }
        optimizer_args.update(valid_kwargs)

        try:
            optimizer = optimizer_class(params, **optimizer_args)
            logger.info(
                f"Создан оптимизатор {optimizer_class.__name__} с параметрами: {optimizer_args}"
            )
            return optimizer
        except Exception as e:
            logger.error(
                f"Ошибка при создании оптимизатора '{optimizer_type}' с параметрами {optimizer_args}: {e}"
            )
            logger.warning("Попытка создать Adam как запасной вариант.")
            try:
                return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
            except Exception as fallback_e:
                logger.error(f"Не удалось создать даже запасной Adam: {fallback_e}")
                raise e

    @staticmethod
    def get_available_optimizers() -> List[str]:
        """
        Возвращает список доступных имен оптимизаторов.

        Returns:
            List[str]: Доступные имена оптимизаторов.
        """
        return list(OptimizerFactory._optimizers.keys())


class WarmStartFactory(FactoryInterface):
    """
    Фабрика для создания регрессоров для "теплого старта".
    Поддерживает sklearn DecisionTreeRegressor (если доступен) или None.

    Attributes:
        TYPES (List[str]): Список поддерживаемых типов регрессоров.
        create: Создаёт регрессор для "теплого старта".
        get_available_types: Возвращает список доступных типов регрессоров.
    """

    TYPES = ["decision_tree", "none"]

    @classmethod
    def create(
        cls,
        regressor_type: str = "decision_tree",
        max_depth: int = 10,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        """
        Создает регрессор для "теплого старта".

        Args:
            regressor_type (str): Тип регрессора ("decision_tree" или "none").
            max_depth (int): Максимальная глубина дерева решений.
            min_samples_split (int): Минимальное количество образцов для разделения узла.
            random_state (Optional[int]): Сид для случайного генератора.
            **kwargs: Дополнительные параметры.

        Returns:
            Optional[Any]: Объект регрессора или None, если теплый старт отключен.

        Raises:
            Exception: Если указан неподдерживаемый тип регрессора.
        """
        regressor_type = regressor_type.lower()

        if regressor_type == "none":
            logger.info("Warm Start отключен (тип 'none').")
            return None

        if regressor_type == "decision_tree":
            if not SKLEARN_AVAILABLE:
                logger.error(
                    "Sklearn недоступен, невозможно создать DecisionTreeRegressor для Warm Start."
                )
                return None
            else:
                try:
                    logger.info(
                        f"Создание sklearn.DecisionTreeRegressor: max_depth={max_depth}, min_samples_split={min_samples_split}, random_state={random_state}"
                    )
                    return SklearnDecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state,
                    )
                except Exception as e:
                    logger.error(
                        f"Ошибка при создании SklearnDecisionTreeRegressor: {e}"
                    )
                    return None
        else:
            available = ", ".join(cls.TYPES)
            logger.error(
                f"Неподдерживаемый тип регрессора для Warm Start: '{regressor_type}'. Доступные: {available}"
            )
            return None

    @classmethod
    def get_available_types(cls) -> List[str]:
        """
        Возвращает список доступных типов регрессоров для Warm Start.

        Returns:
            List[str]: Доступные типы регрессоров.
        """
        return cls.TYPES


class SACAgentFactory(FactoryInterface):
    """
    Фабрика для сборки и создания агента SAC из компонентов.
    Использует другие фабрики для создания сетей, буферов, оптимизаторов и т.д.
    """

    @staticmethod
    def create(
        obs_dim: Union[int, Tuple],
        action_dim: int,
        # Архитектура сетей
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation_fn: str = "relu",
        use_lstm: bool = False,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
        # Конфигурации компонентов (словари)
        buffer_config: Optional[dict] = None,
        normalizer_config: Optional[dict] = None,
        warm_start_config: Optional[dict] = None,
        # Параметры оптимизации
        optimizer_type: str = "adam",
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        weight_decay: float = 0.0,
        optimizer_kwargs: Optional[dict] = None,
        # Параметры SAC
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        learn_alpha: bool = True,
        target_entropy: Optional[float] = None,
        clip_grad_norm_actor: Optional[float] = 1.0,
        clip_grad_norm_critic: Optional[float] = 1.0,
        clip_grad_norm_alpha: Optional[float] = 1.0,
        device: str = "cpu",
        storage_dtype: str = "float32",
    ) -> SACAgent:
        """
        Создает и конфигурирует агента SAC на основе переданных параметров.

        Args:
            obs_dim (Union[int, Tuple]): Размерность наблюдений среды.
            action_dim (int): Размерность пространства действий.
            hidden_dim (int): Количество нейронов в скрытых слоях.
            num_layers (int): Количество слоев сети.
            activation_fn (str): Название функции активации.
            use_lstm (bool): Флаг использования LSTM вместо MLP.
            use_layer_norm (bool): Включить ли нормализацию слоёв в сети.
            dropout (float): Вероятность отсечения весов для Dropout.
            buffer_config (Optional[dict]): Конфигурация буфера воспроизведения.
            normalizer_config (Optional[dict]): Конфигурация нормализатора наблюдений.
            warm_start_config (Optional[dict]): Конфигурация теплого старта.
            optimizer_type (str): Тип оптимизатора ('adam', 'sgd' и др.).
            actor_lr (float): Скорость обучения актора.
            critic_lr (float): Скорость обучения критика.
            alpha_lr (float): Скорость обучения температурного коэффициента alpha.
            weight_decay (float): Коэффициент регуляризации весов.
            optimizer_kwargs (Optional[dict]): Дополнительные параметры оптимизатора.
            gamma (float): Коэффициент дисконтирования наград.
            tau (float): Параметр мягкого обновления целевых сетей.
            alpha (float): Температурный коэффициент энтропии.
            learn_alpha (bool): Обучать ли коэффициент alpha.
            target_entropy (Optional[float]): Целевая энтропия политики.
            clip_grad_norm_actor (Optional[float]): Ограничение градиента актора.
            clip_grad_norm_critic (Optional[float]): Ограничение градиента критика.
            clip_grad_norm_alpha (Optional[float]): Ограничение градиента alpha.
            device (str): Устройство вычисления ('cpu' или 'cuda').
            storage_dtype (str): Тип данных для хранения в буфере ('float32', 'float16').

        Returns:
            SACAgent: Настроенный и готовый к использованию агент SAC.

        Raises:
            Exception: Если произошла ошибка при создании любого из компонентов агента.
        """
        target_device = device_manager.get_device(device)
        kwargs = locals()  # Собираем все аргументы в словарь для удобства

        # Функция конвертации строкового dtype в torch.dtype
        def convert_dtype(dtype_str):
            if isinstance(dtype_str, torch.dtype):
                return dtype_str
            elif isinstance(dtype_str, str):
                dtype_map = {
                    "float16": torch.float16,
                    "float32": torch.float32,
                    "float64": torch.float64,
                    "bfloat16": torch.bfloat16
                    if hasattr(torch, "bfloat16")
                    else torch.float16,
                    "half": torch.float16,
                    "float": torch.float32,
                    "double": torch.float64,
                }
                return dtype_map.get(dtype_str.lower(), torch.float32)
            else:
                logger.warning(
                    f"Неизвестный тип dtype: {dtype_str}, используем float32"
                )
                return torch.float32

        try:
            # Конвертируем storage_dtype в torch.dtype
            torch_storage_dtype = convert_dtype(storage_dtype)
            logger.debug(
                f"Конвертирован storage_dtype из {storage_dtype} в {torch_storage_dtype}"
            )

            # Обновляем storage_dtype в buffer_config, если он указан
            effective_buffer_config = buffer_config if buffer_config is not None else {}

            # Если storage_dtype указан в buffer_config, конвертируем его
            if "storage_dtype" in effective_buffer_config:
                effective_buffer_config["storage_dtype"] = convert_dtype(
                    effective_buffer_config["storage_dtype"]
                )
            else:
                # Иначе используем общий storage_dtype
                effective_buffer_config["storage_dtype"] = torch_storage_dtype

            # Добавляем tree_dtype в конфигурацию буфера, чтобы он использовался для SumTree и MinTree
            if "tree_dtype" not in effective_buffer_config:
                effective_buffer_config["tree_dtype"] = torch_storage_dtype

            # 1. Сеть политики
            policy_net = PolicyFactory.create(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation_fn=activation_fn,
                use_lstm=use_lstm,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                device=str(target_device),
            )

            # 2. Q-сеть
            q_net = QNetFactory.create(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation_fn=activation_fn,
                use_lstm=use_lstm,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                device=str(target_device),
            )

            # 3. Буфер воспроизведения
            # Добавляем обязательные параметры, если их нет
            effective_buffer_config.setdefault("capacity", 1_000_000)
            effective_buffer_config.setdefault("obs_dim", obs_dim)
            effective_buffer_config.setdefault("action_dim", action_dim)
            effective_buffer_config.setdefault("device", device)
            # Добавляем параметры по умолчанию для PER/Sequence, если они включены
            if effective_buffer_config.get("use_prioritized", False):
                effective_buffer_config.setdefault("alpha", 0.6)
                effective_buffer_config.setdefault("beta", 0.4)
            if effective_buffer_config.get("use_sequence", False):
                effective_buffer_config.setdefault("sequence_length", 10)
            logger.debug(
                f"Итоговая конфигурация для ReplayBufferFactory: {effective_buffer_config}"
            )
            replay_buffer = ReplayBufferFactory.create(**effective_buffer_config)

            # 4. Нормализатор
            effective_normalizer_config = (
                normalizer_config if normalizer_config is not None else {}
            )
            normalizer_instance = NormalizerFactory.create(
                obs_dim=obs_dim,
                use_normalizer=effective_normalizer_config.get("use_normalizer", True),
                normalizer_type=effective_normalizer_config.get(
                    "normalizer_type", "welford"
                ),
                clip_range=effective_normalizer_config.get("clip_range", 5.0),
                epsilon=effective_normalizer_config.get("normalizer_epsilon", 1e-8),
            )
            if isinstance(normalizer_instance, nn.Module):
                normalizer_instance = normalizer_instance.to(target_device)

            # 5. Конфигурация Warm Start
            effective_warm_start_config = (
                warm_start_config if warm_start_config is not None else {}
            )
            use_warm_start_flag = effective_warm_start_config.get(
                "use_warm_start", False
            )
            warm_start_agent_config_dict = {
                "enabled": use_warm_start_flag
            }  # Передаем в агент

            if use_warm_start_flag:
                ws_type = effective_warm_start_config.get(
                    "warm_start_type", "decision_tree"
                )
                warm_start_regressor = WarmStartFactory.create(
                    regressor_type=ws_type,
                    max_depth=effective_warm_start_config.get(
                        "warm_start_max_depth", 10
                    ),
                    min_samples_split=effective_warm_start_config.get(
                        "warm_start_min_samples_split", 2
                    ),
                    random_state=effective_warm_start_config.get(
                        "warm_start_random_state", None
                    ),
                )
                if warm_start_regressor is None and ws_type != "none":
                    logger.warning(
                        f"Не удалось создать регрессор '{ws_type}' для Warm Start. Теплый старт будет отключен."
                    )
                    warm_start_agent_config_dict["enabled"] = False
                else:
                    # Заполняем словарь для передачи в SACAgent
                    warm_start_agent_config_dict.update(
                        {
                            "regressor_type": ws_type,
                            "steps": effective_warm_start_config.get(
                                "warm_start_steps", 10000
                            ),
                            "random_steps": effective_warm_start_config.get(
                                "warm_start_random_steps", 1000
                            ),
                            "max_depth": effective_warm_start_config.get(
                                "warm_start_max_depth", 10
                            ),
                            "min_samples_split": effective_warm_start_config.get(
                                "warm_start_min_samples_split", 2
                            ),
                            "random_state": effective_warm_start_config.get(
                                "warm_start_random_state", None
                            ),
                            "exploration_noise": effective_warm_start_config.get(
                                "warm_start_noise", 0.1
                            ),
                            "regressor": warm_start_regressor,
                        }
                    )
                    logger.info(f"Конфигурация Warm Start подготовлена: type={ws_type}")

            # 6. Создание оптимизаторов
            effective_optimizer_kwargs = (
                optimizer_kwargs if optimizer_kwargs is not None else {}
            )
            actor_optim = OptimizerFactory.create(
                params=policy_net.parameters(),
                optimizer_type=optimizer_type,
                lr=actor_lr,
                weight_decay=weight_decay,
                **effective_optimizer_kwargs,
            )
            critic_optim = OptimizerFactory.create(
                params=q_net.parameters(),
                optimizer_type=optimizer_type,
                lr=critic_lr,
                weight_decay=weight_decay,
                **effective_optimizer_kwargs,
            )

            # 7. Создание агента SAC
            # Передаем сюда только параметры, специфичные для SACAgent.__init__
            agent = SACAgent(
                policy_net=policy_net,
                q_net=q_net,
                replay_buffer=replay_buffer,
                actor_optimizer=actor_optim,
                critic_optimizer=critic_optim,
                policy_component=None,
                critic_component=None,
                alpha_component=None,
                normalizer_component=None,
                device=str(target_device),
                gamma=gamma,
                tau=tau,
                alpha_lr=alpha_lr,
                alpha_init=alpha,
                learn_alpha=learn_alpha,
                target_entropy=target_entropy,
                clip_grad_norm_actor=clip_grad_norm_actor,
                clip_grad_norm_critic=clip_grad_norm_critic,
                clip_grad_norm_alpha=clip_grad_norm_alpha,
                normalizer=normalizer_instance,
                normalize_obs=effective_normalizer_config.get("use_normalizer", True),
                clip_obs=effective_normalizer_config.get("clip_range", 5.0),
                obs_dim=obs_dim,
                action_dim=action_dim,
                warm_start_config=warm_start_agent_config_dict,
            )

            logger.info(
                f"Агент SAC успешно создан фабрикой SACAgentFactory на устройстве {target_device}."
            )
            return agent

        except Exception as e:
            logger.exception(
                f"Критическая ошибка при создании агента SAC: {e}", exc_info=True
            )
            raise


class ConfigFactory(FactoryInterface):
    """
    Фабрика для загрузки и сохранения конфигураций в форматах JSON и YAML.

    Attributes:
        load_config: Загружает конфигурацию из файла.
        save_config: Сохраняет конфигурацию в файл.
    """

    @staticmethod
    def load_config(path: Union[str, Path]) -> Dict[str, Any]:
        """
        Загружает конфигурацию из файла JSON или YAML.

        Args:
            path (Union[str, Path]): Путь к файлу конфигурации.

        Returns:
            Dict[str, Any]: Загруженная конфигурация в виде словаря.

        Raises:
            FileNotFoundError: Если указанный файл не существует.
            ValueError: Если формат файла не поддерживается.
            TypeError: Если содержимое файла не является словарём.
            Exception: Для других ошибок чтения файла.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

        suffix = config_path.suffix.lower()
        try:
            if suffix == ".json":
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            elif suffix in [".yaml", ".yml"]:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(
                    f"Неподдерживаемый формат файла конфигурации: {suffix}. Ожидался .json, .yaml или .yml."
                )

            if not isinstance(config, dict):
                raise TypeError(
                    f"Содержимое файла {config_path} не является словарем (dict)."
                )

            logger.info(f"Конфигурация успешно загружена из {config_path}.")
            return config
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации из {config_path}: {e}")
            raise

    @staticmethod
    def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
        """
        Сохраняет конфигурацию в файл JSON или YAML.

        Args:
            config (Dict[str, Any]): Конфигурация в виде словаря.
            path (Union[str, Path]): Путь для сохранения файла.

        Raises:
            ValueError: Если формат файла не поддерживается.
            Exception: Для других ошибок записи файла.
        """
        config_path = Path(path)

        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(
                f"Не удалось создать директорию для сохранения конфигурации {config_path.parent}: {e}"
            )
            raise

        suffix = config_path.suffix.lower()
        try:
            if suffix == ".json":
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            elif suffix in [".yaml", ".yml"]:
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(
                        config,
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False,
                    )
            else:
                raise ValueError(
                    f"Неподдерживаемый формат файла конфигурации для сохранения: {suffix}. Ожидался .json, .yaml или .yml."
                )

            logger.info(f"Конфигурация успешно сохранена в {config_path}.")
        except Exception as e:
            logger.error(f"Ошибка при сохранении конфигурации в {config_path}: {e}")
            raise


# Дополнительный класс для сериализации NumPy массивов в JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return super(NumpyEncoder, self).default(obj)
