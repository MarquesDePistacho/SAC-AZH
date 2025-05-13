import os
import time
import torch
import numpy as np
import random
from typing import Dict, Any, Optional
from collections import defaultdict

from core.training.config import expand_config, save_config
from core.sac.factories import SACAgentFactory, EnvFactory
from core.logging.logger import (
    get_logger,
    log_method_call,
    log_tensor_info,
    DEBUG,
    TRACE,
)
from core.utils.device_utils import device_manager
from core.training.mlflow import MetricsTracker, save_metrics_to_json, log_artifact, TRAINING_EXPERIMENT

logger = get_logger("sac_trainer")


class SACTrainer:
    """
    Класс SACTrainer координирует процесс обучения агента Soft Actor-Critic (SAC).
    Обеспечивает инициализацию окружения, агента, логгера, а также запуск основного цикла обучения.
    """

    @log_method_call()
    def __init__(self, config: Dict[str, Any], env: Optional[Any] = None):
        """
        Инициализация SACTrainer.

        Args:
            config (Dict[str, Any]): Конфигурация эксперимента. Должна содержать:\n
                - `agent_config`: параметры агента\n
                - `env_config`: параметры среды обучения\n
                - `logging_config`: настройки логгирования\n
                - `training_loop_config`: параметры тренировочного цикла\n
            env (Optional[Any]): Готовое окружение gym-совместимой среды. 
                                Если None — создается через EnvFactory.
        """
        logger.info("Инициализация SACTrainer...")

        # Расширяем конфигурацию дефолтными значениями
        self.config = expand_config(config)
        logger.debug(f"Полная конфигурация после расширения: {self.config}")

        # Устанавливаем seed для воспроизводимости
        seed = self.config.get("seed")
        if seed is not None:
            self._set_seed(seed)
        else:
            logger.warning(
                "Seed не указан в конфигурации, результаты могут быть невоспроизводимы."
            )

        # Устанавливаем устройство для вычислений
        device_name = self.config.get("device", "auto")
        self.device = device_manager.get_device(device_name)
        logger.info(f"Установлено устройство для вычислений: {self.device}")

        # --- Создание или использование окружения --- #
        self.env = env
        if self.env is None:
            logger.info("Окружение не передано, попытка создать с помощью EnvFactory.")
            if "env_config" in self.config:
                env_factory = EnvFactory()  # Используем фабрику окружений
                env_config = self.config["env_config"]
                env_name = env_config.get("env_name", "default_env")
                logger.debug(
                    f"Создание окружения '{env_name}' с конфигурацией: {env_config}"
                )
                try:
                    self.env = env_factory.create_gym_env(
                        env_name=env_name, config=env_config
                    )
                    logger.info(f"Окружение '{env_name}' успешно создано.")
                except Exception as e:
                    logger.error(
                        f"Ошибка при создании окружения '{env_name}' через EnvFactory: {e}",
                        exc_info=True,
                    )
                    raise ValueError(f"Не удалось создать окружение: {e}")
            else:
                logger.error(
                    "Не указана 'env_config' в конфигурации и не передано готовое окружение."
                )
                raise ValueError(
                    "Не указана конфигурация окружения ('env_config') и не передано готовое окружение."
                )
        else:
            logger.info("Используется переданный экземпляр окружения.")

        # --- Получение размерностей --- #
        try:
            self.obs_dim = (
                self.env.observation_space.shape[0]
                if hasattr(self.env.observation_space, "shape")
                else self.env.observation_space.n
            )
            self.action_dim = (
                self.env.action_space.shape[0]
                if hasattr(self.env.action_space, "shape")
                else self.env.action_space.n
            )
            logger.info(
                f"Размерности окружения: obs_dim={self.obs_dim}, action_dim={self.action_dim}"
            )
        except AttributeError as e:
            logger.error(
                f"Не удалось получить размерности из пространства окружения: {e}. Убедитесь, что окружение соответствует gym API."
            )
            raise AttributeError(f"Ошибка получения размерностей окружения: {e}")

        # Добавляем размерности в конфигурацию (может быть полезно для агента)
        self.config["agent_config"] = self.config.get("agent_config", {})
        self.config["agent_config"]["obs_dim"] = self.obs_dim
        self.config["agent_config"]["action_dim"] = self.action_dim

        # --- Создание агента --- #
        try:
            self.agent = self._create_agent()
        except Exception as e:
            logger.error(f"Ошибка при создании агента: {e}", exc_info=True)
            raise

        # --- Инициализация логгера MLflow --- #
        try:
            # Извлекаем из конфига параметры для MLflow
            experiment_name = self.config.get("mlflow_experiment_name", TRAINING_EXPERIMENT)
            run_name = self.config.get("mlflow_run_name", f"train_{time.strftime('%Y%m%d_%H%M%S')}")
            
            # Инициализируем трекер метрик
            self.metrics_tracker = MetricsTracker(
                config=self.config,
                experiment_name=experiment_name
            )
            
            # Проверяем, что инициализация прошла успешно
            if getattr(self.metrics_tracker, "run_id", None) is None:
                self.metrics_tracker = None  # Устанавливаем в None, чтобы избежать ошибок
                logger.warning("MetricsTracker не инициализирован, логирование MLflow будет отключено.")
            else:
                logger.info(
                    f"MLflow метрики успешно инициализированы (run_id: {self.metrics_tracker.run_id})."
                )
        except Exception as e:
            logger.error(f"Ошибка при инициализации MLflow метрик: {e}", exc_info=True)
            self.metrics_tracker = None  # Убедимся, что metrics_tracker None при любой ошибке

        # --- Инициализация счетчиков и переменных состояния --- #
        self.total_steps = 0
        self.total_episodes = 0
        self.start_time = time.time()
        self.best_reward = -float("inf")
        self.checkpoint_counter = 0

        # --- Сохранение итоговой конфигурации --- #
        save_dir = self.config.get("save_dir", "results")
        try:
            os.makedirs(save_dir, exist_ok=True)
            config_path = os.path.join(save_dir, "config.json")
            save_config(self.config, config_path)  # Используем функцию из config.py
            logger.info(f"Итоговая конфигурация сохранена в {config_path}")
        except Exception as e:
            logger.error(
                f"Ошибка при сохранении конфигурации в {save_dir}: {e}", exc_info=True
            )

        logger.info("SACTrainer успешно инициализирован.")

    @log_method_call()
    def _set_seed(self, seed: int) -> None:
        """
        Устанавливает начальное значение (seed) для генераторов случайных чисел

        Args:
            seed (int): Сид для random, numpy, PyTorch и CUDA, если доступно.
        """
        logger.info(f"Установка seed: {seed}")
        try:
            os.environ["PYTHONHASHSEED"] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # Для multi-GPU
                # Эти флаги могут замедлить обучение, но обеспечивают детерминизм
                if self.config.get("cudnn_deterministic", True):
                    torch.backends.cudnn.deterministic = True
                    logger.info("torch.backends.cudnn.deterministic установлен в True.")
                else:
                    torch.backends.cudnn.deterministic = False
                if self.config.get("cudnn_benchmark", False):
                    torch.backends.cudnn.benchmark = True
                    logger.info(
                        "torch.backends.cudnn.benchmark установлен в True (может улучшить производительность, но снизить детерминизм)."
                    )
                else:
                    torch.backends.cudnn.benchmark = False
            logger.info(f"Seed {seed} успешно установлен для всех библиотек.")
        except Exception as e:
            logger.error(f"Ошибка при установке seed {seed}: {e}", exc_info=True)

    @log_method_call()
    def _create_agent(
        self,
    ) -> Any:  # Возвращаемый тип зависит от агента, лучше Any или BaseAgent
        """
        Создает экземпляр обучаемого агента SAC с использованием SACAgentFactory.
        Извлекает необходимые параметры из конфигурации self.config['agent_config'].
    
        Returns:
            Any: Инициализированный агент SAC.
        """
        logger.info("Создание агента SAC с помощью SACAgentFactory...")
        agent_config = self.config.get("agent_config")
        if not agent_config or not isinstance(agent_config, dict):
            logger.error(
                "'agent_config' отсутствует или не является словарем в основной конфигурации."
            )
            raise ValueError("'agent_config' не найден в конфигурации.")

        # Передаем obs_dim и action_dim, которые уже должны быть в agent_config
        if "obs_dim" not in agent_config or "action_dim" not in agent_config:
            logger.error(
                "obs_dim или action_dim отсутствуют в agent_config. Они должны были быть добавлены в __init__."
            )
            raise ValueError("obs_dim или action_dim не найдены в agent_config.")

        # Передаем buffer_config отдельно, если он есть верхнего уровня
        buffer_config = self.config.get(
            "buffer_config", agent_config.get("buffer_config")
        )

        # Добавляем buffer_config в agent_config, если он определен
        if buffer_config:
            agent_config["buffer_config"] = buffer_config

        logger.debug(f"Параметры для SACAgentFactory: {agent_config}")

        try:
            # Добавляем устройство
            agent_config["device"] = str(self.device)

            # Используем фабрику для создания агента
            agent = SACAgentFactory.create(**agent_config)
            logger.info(f"Агент SAC успешно создан на устройстве {self.device}.")
            return agent
        except TypeError as e:
            logger.error(
                f"Ошибка TypeError при вызове SACAgentFactory.create: {e}. Проверьте параметры в agent_config.",
                exc_info=True,
            )
            raise ValueError(f"Неверные параметры для создания агента: {e}")
        except Exception as e:
            logger.error(
                f"Неожиданная ошибка при создании агента через SACAgentFactory: {e}",
                exc_info=True,
            )
            raise

    @log_method_call(log_level=DEBUG, log_args=False, log_return=False)
    def _sample_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """
        Выбирает действие на основе текущего наблюдения.

        Args:
            obs (np.ndarray): Входное наблюдение (observation), shape зависит от среды.
            deterministic (bool): Флаг, указывающий, нужно ли брать детерминированное действие.

        Returns:
            np.ndarray: Вектор действия, соответствующий пространству действий среды.
        """
        # logger.trace(f"Получение действия для наблюдения shape: {obs.shape}, детерминированный: {deterministic}") # Заменено декоратором
        try:
            action = self.agent.act(obs, deterministic=deterministic)
            # logger.trace(f"Получено действие shape: {action.shape}") # Заменено декоратором
            # Проверяем тип действия перед проверкой на NaN
            if isinstance(action, torch.Tensor):
                if torch.isnan(action.cpu()).any():
                    logger.warning(f"Агент вернул NaN действие для obs: {obs}")
                    action = self.env.action_space.sample()  # Как запасной вариант
            else:
                if np.isnan(action).any():
                    logger.warning(f"Агент вернул NaN действие для obs: {obs}")
                    action = self.env.action_space.sample()  # Как запасной вариант
            return action
        except Exception as e:
            logger.error(f"Ошибка при получении действия от агента: {e}", exc_info=True)
            # Возвращаем случайное действие в случае ошибки
            return self.env.action_space.sample()

    @log_method_call(log_level=DEBUG)
    def _update_agent(self, batch_size: int) -> Dict[str, float]:
        """
        Выполняет шаги обновления параметров агента (политики, Q-сетей, alpha).
        Вызывает метод `agent.perform_updates()` заданное количество раз (`updates_per_step`).
        
        Note:
            Обработка AMP (GradScaler) должна происходить внутри `agent.perform_updates()`.

        Args:
            batch_size (int): Размер батча данных для взаимодействия с буфером.

        Returns:
            Dict[str, float]: Метрики обучения (потери политики, critic'а, alpha).
        """
        # logger.debug("Обновление агента...")
        n_updates = self.config.get("updates_per_step", 1)
        # logger.debug(f"Количество обновлений за шаг: {n_updates}")

        all_metrics = {}
        # Переключаем агента в режим обучения (важно для dropout, batchnorm и т.д.)
        self.agent.train()

        # Дебаг информация о буфере - выводим каждые log_interval шагов
        log_interval = self.config.get("log_interval", 1000)
        if self.total_steps % log_interval == 0 and hasattr(
            self.agent, "replay_buffer"
        ):
            self._debug_buffer_sample(batch_size)

        for i in range(n_updates):
            try:
                # logger.debug(f"Запуск agent.perform_updates() #{i+1}/{n_updates}...")
                # Передаем оба аргумента: num_updates и batch_size
                metrics = self.agent.perform_updates(
                    num_updates=n_updates, batch_size=batch_size
                )
                if metrics:  # agent.perform_updates может вернуть None или {}
                    all_metrics.update(metrics)  # Обновляем метрики
                # logger.debug(f"agent.perform_updates() #{i+1} завершен.")
            except Exception as e:
                logger.error(
                    f"Ошибка при выполнении agent.perform_updates() #{i + 1}: {e}",
                    exc_info=True,
                )
                # Можно пропустить обновление или прервать обучение
                # break # Прерываем цикл обновлений при ошибке

        # Переключаем агента обратно в режим оценки
        self.agent.eval()

        if not all_metrics:
            logger.debug("Обновление агента не вернуло метрик.")

        return all_metrics  # Возвращаем метрики от последнего успешного обновления

    def _debug_buffer_sample(self, batch_size: int) -> None:
        """
        Получает образец данных из буфера для отладки и логирует информацию о типах и значениях.
        Помогает обнаруживать проблемы с несовместимостью типов данных.

        Args:
            batch_size (int): Количество переходов для выборки.
        """
        try:
            # Проверяем, что буфер имеет метод sample
            if not hasattr(self.agent.replay_buffer, "sample"):
                logger.debug("Буфер реплеев не имеет метода sample, дебаг невозможен")
                return

            # Пробуем получить образец данных
            batch = self.agent.replay_buffer.sample(batch_size)
            if not isinstance(batch, dict):
                logger.debug(f"Буфер вернул не словарь: {type(batch)}")
                return

            # Логируем информацию о типах и размерах
            dtype_info = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    min_val = value.min().item() if value.numel() > 0 else "N/A"
                    max_val = value.max().item() if value.numel() > 0 else "N/A"
                    zeros_percent = (
                        (value == 0).float().mean().item() * 100
                        if value.numel() > 0
                        else 0
                    )
                    has_nan = (
                        torch.isnan(value).any().item() if value.numel() > 0 else False
                    )
                    has_inf = (
                        torch.isinf(value).any().item() if value.numel() > 0 else False
                    )

                    dtype_info[key] = (
                        f"torch.{value.dtype}, shape={tuple(value.shape)}, device={value.device}, "
                        f"min={min_val:.4f}, max={max_val:.4f}, zeros={zeros_percent:.1f}%, "
                        f"has_nan={has_nan}, has_inf={has_inf}"
                    )
                elif isinstance(value, np.ndarray):
                    min_val = value.min() if value.size > 0 else "N/A"
                    max_val = value.max() if value.size > 0 else "N/A"
                    zeros_percent = (value == 0).mean() * 100 if value.size > 0 else 0
                    has_nan = np.isnan(value).any() if value.size > 0 else False
                    has_inf = np.isinf(value).any() if value.size > 0 else False

                    dtype_info[key] = (
                        f"numpy.{value.dtype}, shape={value.shape}, "
                        f"min={min_val}, max={max_val}, zeros={zeros_percent:.1f}%, "
                        f"has_nan={has_nan}, has_inf={has_inf}"
                    )
                else:
                    dtype_info[key] = f"{type(value)}"

            # Проверка на несоответствие типов данных
            tensor_keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor)]

            # Проверка на несоответствие устройств
            devices = [batch[k].device for k in tensor_keys]
            if len(set(devices)) > 1:
                logger.warning(
                    f"[Буфер] Обнаружены разные устройства в батче: {set(devices)}"
                )

                # Более подробная информация о том, какие ключи на каких устройствах
                device_groups = {}
                for k in tensor_keys:
                    device = str(batch[k].device)
                    if device not in device_groups:
                        device_groups[device] = []
                    device_groups[device].append(k)

                for device, keys in device_groups.items():
                    logger.warning(f"[Буфер] Устройство {device}: {', '.join(keys)}")

        except Exception as e:
            logger.error(f"Ошибка при дебаге буфера: {e}")

    @log_method_call()
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """
        Сохраняет текущее состояние агента (и нормализатора) в чекпоинт.
        Создает файлы вида `model_name_step.pth` и `model_name_best.pth`.

        Args:
            is_best (bool): Если True — сохраняет как best модель,
                            иначе обычный чекпоинт по номеру шага.
        """
        save_dir = self.config.get("save_dir", "results")
        model_name = self.config.get("model_name", "sac_agent")
        file_extension = ".pth"  # Используем .pth для PyTorch state_dict

        try:
            os.makedirs(save_dir, exist_ok=True)

            if is_best:
                checkpoint_path = os.path.join(
                    save_dir, f"{model_name}_best{file_extension}"
                )
                logger.info(
                    f"Сохранение лучшего чекпоинта (Награда: {self.best_reward:.2f}) в {checkpoint_path}..."
                )
            else:
                checkpoint_path = os.path.join(
                    save_dir, f"{model_name}_{self.total_steps}{file_extension}"
                )
                logger.info(
                    f"Сохранение чекпоинта на шаге {self.total_steps} в {checkpoint_path}..."
                )

            # Сохраняем состояние агента (включая нормализатор, если он есть)
            self.agent.save(checkpoint_path)
            logger.info(f"Чекпоинт успешно сохранен: {checkpoint_path}")

            self.checkpoint_counter += 1

            # Удаляем старые чекпоинты, если нужно
            if not is_best and self.config.get("keep_last_n_checkpoints", 0) > 0:
                self._cleanup_old_checkpoints()

            # Экспортируем лучшую модель в ONNX, если нужно
            if is_best and self.config.get("export_onnx", False):
                self._export_onnx()

        except Exception as e:
            logger.error(
                f"Ошибка при сохранении чекпоинта в {save_dir}: {e}", exc_info=True
            )

    @log_method_call()
    def _load_checkpoint(self, path: str) -> None:
        """
        Загружает веса агента из файла чекпоинта.

        Args:
            path (str): Путь к файлу чекпоинта.
        """
        logger.info(f"Попытка загрузки чекпоинта из: {path}")
        if not os.path.exists(path):
            logger.error(f"Файл чекпоинта не найден: {path}")
            return

        try:
            self.agent.load(path, map_location=self.device)
            # Можно также загрузить состояние оптимизаторов, если оно сохраняется
            # и total_steps из метаданных, если нужно продолжить с того же места.
            logger.info(
                f"Чекпоинт успешно загружен из {path}. Агент переведен на устройство {self.device}."
            )
            # Если чекпоинт содержит информацию о шагах/эпизодах, можно ее восстановить
            # state_dict = torch.load(path, map_location='cpu') # Загружаем на CPU для проверки метаданных
            # self.total_steps = state_dict.get('total_steps', self.total_steps)
            # self.total_episodes = state_dict.get('total_episodes', self.total_episodes)
            # logger.info(f"Восстановлено состояние: total_steps={self.total_steps}, total_episodes={self.total_episodes}")
        except FileNotFoundError:
            logger.error(f"Файл чекпоинта не найден при загрузке: {path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке чекпоинта из {path}: {e}", exc_info=True)

    @log_method_call()
    def _cleanup_old_checkpoints(self) -> None:
        """
        Удаляет старые чекпоинты, оставляя только `keep_last_n_checkpoints` последних.
        Не удаляет чекпоинты `_best.pth`.
        """
        keep_last_n = self.config.get("keep_last_n_checkpoints", 0)
        if keep_last_n <= 0:
            return

        save_dir = self.config.get("save_dir", "results")
        model_name = self.config.get("model_name", "sac_agent")
        file_extension = ".pth"

        logger.debug(
            f"Очистка старых чекпоинтов в {save_dir}, оставляем последние {keep_last_n}..."
        )

        try:
            checkpoints = []
            for filename in os.listdir(save_dir):
                # Ищем файлы вида model_name_ЧИСЛО.pth
                if (
                    filename.startswith(f"{model_name}_")
                    and filename.endswith(file_extension)
                    and "_best" not in filename
                ):
                    checkpoint_path = os.path.join(save_dir, filename)
                    try:
                        # Извлекаем номер шага из имени файла
                        step_str = filename[len(model_name) + 1 : -len(file_extension)]
                        step = int(step_str)
                        checkpoints.append((step, checkpoint_path))
                    except ValueError:
                        logger.warning(
                            f"Не удалось извлечь номер шага из имени файла: {filename}"
                        )
                        continue

            # Сортируем по номеру шага (по возрастанию, чтобы удалить самые старые)
            checkpoints.sort(key=lambda x: x[0])

            # Определяем количество чекпоинтов для удаления
            num_to_delete = len(checkpoints) - keep_last_n

            if num_to_delete > 0:
                logger.debug(
                    f"Найдено {len(checkpoints)} чекпоинтов, будет удалено {num_to_delete}."
                )
                # Удаляем самые старые чекпоинты
                for step, path in checkpoints[:num_to_delete]:
                    try:
                        os.remove(path)
                        logger.info(f"Удален старый чекпоинт (шаг {step}): {path}")
                    except OSError as e:
                        logger.warning(f"Не удалось удалить чекпоинт {path}: {e}")
            else:
                logger.debug(
                    f"Найдено {len(checkpoints)} чекпоинтов, удаление не требуется."
                )

        except FileNotFoundError:
            logger.warning(f"Директория для очистки чекпоинтов не найдена: {save_dir}")
        except Exception as e:
            logger.error(
                f"Ошибка при очистке старых чекпоинтов в {save_dir}: {e}", exc_info=True
            )

    @log_method_call()
    def _export_onnx(self) -> None:
        """
        Экспортирует обученную модель (обычно политику) в формат ONNX.
        Параметры экспорта берутся из конфигурации.
        """
        logger.info("Экспорт модели в ONNX...")
        save_dir = self.config.get("save_dir", "results")
        export_dir = self.config.get(
            "export_dir", os.path.join(save_dir, "onnx_export")
        )
        model_name = self.config.get("model_name", "sac_agent")
        onnx_filename = f"{model_name}_best.onnx"  # Обычно экспортируют лучшую модель
        export_path = os.path.join(export_dir, onnx_filename)

        try:
            os.makedirs(export_dir, exist_ok=True)

            # Определяем входную форму для ONNX модели
            input_shape = self.config.get("onnx_input_shape")

            # Если не указана форма входа, пытаемся получить её из среды
            if input_shape is None:
                # Пробуем получить необработанные формы наблюдений из среды
                if hasattr(self.env, "get_raw_observation_shapes") and callable(
                    getattr(self.env, "get_raw_observation_shapes")
                ):
                    raw_shapes = self.env.get_raw_observation_shapes()
                    if raw_shapes and len(raw_shapes) > 0:
                        logger.info(f"Получены формы наблюдений из среды: {raw_shapes}")
                        # Используем эти формы для экспорта
                        input_shape = raw_shapes
                    else:
                        logger.warning("Среда вернула пустой список форм наблюдений")
                        # Используем размерность плоского наблюдения
                        input_shape = (self.obs_dim,)
                        logger.warning(
                            f"'onnx_input_shape' не указан, используется {input_shape}"
                        )
                else:
                    # Среда не поддерживает получение исходных форм
                    if self.config.get("agent_config", {}).get("use_lstm", False):
                        # Пример для LSTM: батч=1, длина послед=1
                        input_shape = (1, 1, self.obs_dim)
                        logger.warning(
                            f"'onnx_input_shape' не указан для LSTM, используется {input_shape}"
                        )
                    else:
                        input_shape = (self.obs_dim,)  # Для MLP
                        logger.warning(
                            f"'onnx_input_shape' не указан для MLP, используется {input_shape}"
                        )

            # Экспортируем модель
            if hasattr(self.agent, "export_to_onnx"):
                exported_path = self.agent.export_to_onnx(
                    obs_shape=input_shape, export_dir=export_dir, filename=onnx_filename
                )

                if exported_path:
                    logger.info(f"Модель успешно экспортирована в {exported_path}")
                else:
                    logger.error("Не удалось экспортировать модель")
            else:
                logger.error("Агент не поддерживает экспорт в ONNX")
        except Exception as e:
            logger.error(f"Ошибка при экспорте модели: {str(e)}", exc_info=True)

    @log_method_call()
    def train(self) -> Dict[str, Any]:
        """
        Запускает основной цикл обучения SAC-агента.

        Returns:
            Dict[str, Any]: Результаты обучения:\n
                - total_steps: общее число выполненных шагов\n
                - total_episodes: завершённые эпизоды\n
                - best_reward: лучшая награда за эпизод\n
                - training_time_seconds: общее время обучения\n
        """
        logger.info("=== Начало сессии обучения SAC ===")
        self.start_time = time.time()  # Перезапускаем таймер

        # --- Загрузка чекпоинта (если указан) --- #
        load_path = self.config.get("load_checkpoint_path")
        if load_path:
            self._load_checkpoint(load_path)
            # total_steps и best_reward могли быть восстановлены внутри _load_checkpoint

        # --- Получение параметров из конфига --- #
        max_steps = self.config.get("max_steps", 1_000_000)
        episode_max_steps = self.config.get("episode_max_steps", 1000)
        batch_size = self.config.get("batch_size", 256)
        update_after = self.config.get(
            "update_after", 1000
        )  # Шаг, после которого начинаются обновления
        start_steps = self.config.get(
            "start_steps", 1000
        )  # Шаги со случайными действиями
        log_interval = self.config.get(
            "log_interval", 1000
        )  # Как часто логировать метрики обучения
        save_interval = self.config.get(
            "save_interval", 10000
        )  # Как часто сохранять чекпоинты
        reward_scale = self.config.get("reward_scale", 1.0)
        use_lstm = self.config.get("agent_config", {}).get("use_lstm", False)
        clear_cache_interval = self.config.get("clear_cuda_cache_interval_episodes", 10)

        logger.info(
            f"Параметры обучения: max_steps={max_steps}, batch_size={batch_size}, update_after={update_after}, start_steps={start_steps}"
        )

        # --- Инициализация цикла --- #
        episode_return = 0.0
        episode_length = 0
        try:
            obs = self.env.reset()
            logger.info("Окружение сброшено для начала обучения.")
        except Exception as e:
            logger.error(f"Ошибка при первом сбросе окружения: {e}", exc_info=True)
            raise RuntimeError(f"Не удалось сбросить окружение: {e}")

        # Сбрасываем скрытые состояния LSTM перед началом
        if use_lstm:
            try:
                self.agent.reset_hidden()
            except AttributeError:
                logger.warning(
                    "Агент не имеет метода reset_hidden(), хотя use_lstm=True."
                )

        # Устанавливаем режим оценки для начального сбора данных
        self.agent.eval()

        # --- Основной цикл обучения --- #
        logger.info("Запуск основного цикла обучения...")
        while self.total_steps < max_steps:
            try:
                # --- Шаг взаимодействия с окружением --- #
                # Выбор действия (случайное на старте, иначе - агентом)
                # Режим eval используется внутри select_action если deterministic=True
                action = self._sample_action(obs, deterministic=False)

                # Шаг в окружении
                next_obs, reward, done, info = self.env.step(action)
                log_tensor_info(
                    logger, "Шаг окружения: next_obs", next_obs, level=TRACE
                )
                log_tensor_info(logger, "Шаг окружения: reward", reward, level=TRACE)
                log_tensor_info(logger, "Шаг окружения: done", done, level=TRACE)

                # Масштабирование награды
                scaled_reward = reward * reward_scale

                episode_return += (
                    reward  # Сохраняем оригинальную награду для логгирования
                )
                episode_length += 1

                # Проверка на максимальную длину эпизода
                force_done = episode_length >= episode_max_steps
                if force_done:
                    done = True  # Завершаем эпизод принудительно
                    logger.debug(
                        f"Эпизод {self.total_episodes} завершен принудительно по длине ({episode_max_steps})."
                    )

                # --- Добавление в буфер --- #
                try:
                    # Вызываем метод add() у буфера
                    self.agent.replay_buffer.add(
                        obs, action, scaled_reward, next_obs, done
                    )
                except AttributeError:
                    logger.error(
                        "Агент или его replay_buffer не имеет метода add(). Проверьте реализацию буфера.",
                        exc_info=True,
                    )
                except Exception as e:
                    logger.error(
                        f"Ошибка при добавлении опыта в буфер: {e}", exc_info=True
                    )

                # Обновляем текущее наблюдение
                obs = next_obs
                self.total_steps += 1

                # --- Обновление агента --- #
                if (
                    self.total_steps >= update_after
                    and self.agent.replay_buffer.can_sample(batch_size)
                ):
                    # Обновление агента и получение метрик
                    train_metrics = self._update_agent(batch_size)
                    
                    # Сохраняем метрики обучения для последующего логирования в конце эпизода
                    if train_metrics:
                        # Обновляем средние значения метрик для текущего эпизода
                        for key, value in train_metrics.items():
                            if not hasattr(self, '_episode_train_metrics'):
                                self._episode_train_metrics = defaultdict(list)
                            self._episode_train_metrics[key].append(value)

                # --- Обработка конца эпизода --- #
                if done:
                    logger.info(
                        f"Эпизод {self.total_episodes} завершен на шаге {self.total_steps}. Награда: {episode_return:.2f}, Длина: {episode_length}."
                    )
                    
                    # Подготавливаем все метрики эпизода
                    episode_metrics = {
                        "episode/return": episode_return,
                        "episode/length": episode_length,
                        "episode/steps_per_second": episode_length
                        / (time.time() - self.start_time),
                    }
                    
                    # Добавляем средние значения метрик обучения за эпизод
                    if hasattr(self, '_episode_train_metrics'):
                        for key, values in self._episode_train_metrics.items():
                            if values:  # Проверяем, что есть значения
                                episode_metrics[f"train/{key}"] = np.mean(values)
                        # Очищаем метрики для следующего эпизода
                        self._episode_train_metrics.clear()
                    
                    # Добавляем метрики буфера
                    if hasattr(self.agent, "replay_buffer"):
                        buffer = self.agent.replay_buffer
                        buffer_capacity = getattr(buffer, "capacity", 1)
                        buffer_size = getattr(buffer, "_size", 0)
                        buffer_fill_ratio = buffer_size / buffer_capacity
                        episode_metrics["buffer/fill_ratio"] = buffer_fill_ratio
                        episode_metrics["buffer/size"] = buffer_size
                        
                        if hasattr(buffer, "beta"):
                            episode_metrics["buffer/beta"] = buffer.beta
                    
                    # Добавляем метрики alpha
                    if hasattr(self.agent, "alpha") and self.agent.alpha is not None:
                        episode_metrics["alpha/value"] = self.agent.alpha.alpha_numpy
                        episode_metrics["alpha/target_entropy"] = getattr(
                            self.agent.alpha, "target_entropy", 0.0
                        )
                    
                    # Логируем все метрики через metrics_tracker
                    if self.metrics_tracker is not None:
                        self.metrics_tracker.log_episode_metrics(
                            episode_metrics,
                            episode=self.total_episodes,
                            step=self.total_steps
                        )

                    # Обновление лучшей награды и сохранение чекпоинта
                    if episode_return > self.best_reward:
                        logger.info(
                            f"*** Новая лучшая награда: {episode_return:.2f} (предыдущая: {self.best_reward:.2f}) на эпизоде {self.total_episodes} ***"
                        )
                        self.best_reward = episode_return
                        if self.config.get("save_checkpoints", True):
                            self._save_checkpoint(is_best=True)

                    # Периодическое сохранение чекпоинта
                    if (
                        self.config.get("save_checkpoints", True)
                        and self.total_episodes > 0
                        and self.total_episodes % save_interval == 0
                    ):
                        self._save_checkpoint(is_best=False)

                    # Очистка CUDA кэша
                    if (
                        torch.cuda.is_available()
                        and clear_cache_interval > 0
                        and self.total_episodes % clear_cache_interval == 0
                    ):
                        logger.debug(
                            f"Очистка CUDA кэша на эпизоде {self.total_episodes}..."
                        )
                        device_manager.clear_cuda_cache()

                    # Сброс окружения и состояния эпизода
                    obs = self.env.reset()
                    episode_return = 0.0
                    episode_length = 0
                    self.total_episodes += 1

                    # Сброс скрытых состояний LSTM
                    if use_lstm:
                        self.agent.reset_hidden()

                    # Вывод прогресса в консоль (реже, чем логгирование метрик)
                    if (
                        self.total_episodes
                        % self.config.get("print_interval_episodes", 10)
                        == 0
                    ):
                        elapsed = time.time() - self.start_time
                        steps_sec = self.total_steps / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Эпизод: {self.total_episodes}, Шаг: {self.total_steps}/{max_steps}, "
                            f"Средняя награда (эпизод): {episode_metrics['episode/return']:.2f}, "
                            f"Длина: {episode_metrics['episode/length']}, "
                            f"Время: {elapsed:.1f}с, Шагов/сек: {steps_sec:.1f}"
                        )

            except KeyboardInterrupt:
                logger.warning(
                    "Обнаружено прерывание (KeyboardInterrupt). Завершение обучения..."
                )
                break  # Выход из цикла while
            except Exception as e:
                logger.error(
                    f"Критическая ошибка в основном цикле обучения на шаге {self.total_steps}: {e}",
                    exc_info=True,
                )
                # Можно добавить логику для попытки восстановления или просто прервать
                break  # Прерываем обучение при критической ошибке

        # --- Завершение обучения --- #
        end_time = time.time()
        total_training_time = end_time - self.start_time
        logger.info("=== Обучение завершено ===")
        logger.info(f"Всего шагов: {self.total_steps}")
        logger.info(f"Всего эпизодов: {self.total_episodes}")
        logger.info(f"Лучшая достигнутая награда: {self.best_reward:.2f}")
        logger.info(f"Общее время обучения: {total_training_time:.2f} секунд")

        # --- Финальное сохранение и экспорт --- #
        if self.config.get("save_checkpoints", True):
            logger.info("Сохранение финального чекпоинта...")
            self._save_checkpoint(is_best=False)  # Сохраняем как обычный чекпоинт

        if self.config.get("export_onnx", False):
            logger.info("Экспорт финальной (лучшей) модели в ONNX...")
            # Обычно экспортируют лучшую модель, поэтому сначала загрузим ее, если она есть
            best_checkpoint_path = os.path.join(
                self.config.get("save_dir", "results"),
                f"{self.config.get('model_name', 'sac_agent')}_best.pth",
            )
            if os.path.exists(best_checkpoint_path):
                logger.info(
                    f"Загрузка лучшего чекпоинта {best_checkpoint_path} для экспорта..."
                )
                self._load_checkpoint(best_checkpoint_path)
            else:
                logger.warning(
                    f"Лучший чекпоинт {best_checkpoint_path} не найден. Экспортируется текущее состояние агента."
                )
            self._export_onnx()

        # --- Закрытие ресурсов --- #
        self.close()

        # --- Возврат результатов --- #
        results = {
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "best_reward": self.best_reward,
            "training_time_seconds": total_training_time,
        }
        logger.info(f"Результаты обучения: {results}")
        return results

    @log_method_call()
    def close(self) -> None:
        """
        Закрывает тренер, освобождая ресурсы: закрывает окружение и логгер.
        """
        logger.info("Закрытие SACTrainer и освобождение ресурсов...")
        try:
            # Закрываем окружение
            if self.env is not None and hasattr(self.env, "close"):
                logger.debug("Закрытие окружения...")
                self.env.close()
                self.env = None
                logger.info("Окружение закрыто.")
        except Exception as e:
            logger.error(f"Ошибка при закрытии окружения: {e}", exc_info=True)

        try:
            # Закрываем MLflow трекер метрик
            if self.metrics_tracker is not None and hasattr(self.metrics_tracker, "close"):
                logger.debug("Закрытие MLflow трекера метрик...")
                self.metrics_tracker.close()
                self.metrics_tracker = None
                logger.info("MLflow трекер метрик закрыт.")
        except Exception as e:
            logger.error(f"Ошибка при закрытии MLflow трекера метрик: {e}", exc_info=True)

        try:
            # Очищаем память CUDA (на всякий случай)
            if torch.cuda.is_available():
                logger.debug("Очистка CUDA кэша при закрытии...")
                device_manager.clear_cuda_cache()
        except Exception as e:
            logger.error(f"Ошибка при очистке CUDA кэша: {e}", exc_info=True)

        logger.info("SACTrainer успешно закрыт.")


# --- Вспомогательная функция для запуска обучения --- #
@log_method_call()
def train_agent(config: Dict[str, Any], env: Optional[Any] = None) -> Dict[str, Any]:
    """
    Удобная обёртка для инициализации и запуска обучения через SACTrainer.

    Args:
        config (Dict[str, Any]): Полная конфигурация для тренера.
        env (Optional[Any]): Готовый инстанс среды Gym.

    Returns:
        Dict[str, Any]: Результаты обучения или информация об ошибке:\n
            - status: статус выполнения ('interrupted', 'error', ...)\n
            - error: описание ошибки (если есть)\n
            - total_steps: общее число шагов, если были\n
            - best_reward: лучшая достигнутая награда\n
    """
    trainer: Optional[SACTrainer] = None
    try:
        logger.info("Запуск процесса обучения через train_agent...")
        trainer = SACTrainer(config, env)
        results = trainer.train()
        logger.info("Процесс обучения train_agent успешно завершен.")
        return results
    except KeyboardInterrupt:
        logger.warning("Обучение прервано пользователем (KeyboardInterrupt).")
        return {"status": "interrupted"}
    except Exception as e:
        logger.error(
            f"Критическая ошибка во время обучения в train_agent: {e}", exc_info=True
        )
        return {"status": "error", "error": str(e)}
    finally:
        # Гарантированное закрытие тренера, если он был создан
        if trainer is not None:
            logger.info("Закрытие тренера в блоке finally функции train_agent.")
            trainer.close()
