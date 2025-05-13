import os
import json
import torch
import numpy as np
from typing import Dict, Any, Optional

# Значения по умолчанию для конфигурации SAC
DEFAULT_CONFIG = {
    # --- Общие параметры Trainer & Logger ---
    "seed": 42,
    "exp_name": "sac_default",
    "device": "cuda",
    "save_dir": "results",
    "model_name": "sac_agent",
    "save_checkpoints": True,
    "keep_last_n_checkpoints": 3,
    "load_checkpoint_path": None,
    "log_interval": 1000,
    "save_interval": 10000,
    "print_interval_episodes": 10,
    "eval_interval": 5000,
    "eval_episodes": 10,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    # --- Параметры MLflow (для TrainingLogger) ---
    "mlflow_tracking_uri": None,
    "mlflow_experiment_name": "SAC",
    "mlflow_run_name": None,
    "log_system_info": True,
    "log_memory_usage": True,
    # --- Параметры обучения (для SACTrainer) ---
    "max_steps": 1_000_000,
    "episode_max_steps": 1000,
    "batch_size": 256,
    "update_after": 1000,
    "start_steps": 1000,
    "updates_per_step": 1,
    "reward_scale": 1.0,
    "clear_cuda_cache_interval_episodes": 0,  # По умолчанию отключено
    # --- Конфигурация агента (для SACAgentFactory) ---
    "agent_config": {
        "obs_dim": None,
        "action_dim": None,
        "device": "cuda",
        # -- Архитектура сетей --
        "hidden_dim": 256,
        "num_layers": 2,
        "activation_fn": "relu",
        "use_lstm": False,
        "use_layer_norm": False,
        "dropout": 0.0,
        # -- Параметры оптимизации --
        "optimizer_type": "adam",
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "alpha_lr": 3e-4,
        "weight_decay": 0.0,
        "optimizer_kwargs": {},
        # -- Гиперпараметры SAC --
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
        "learn_alpha": True,
        "target_entropy": None,
        "clip_grad_norm_actor": 1.0,
        "clip_grad_norm_critic": 1.0,
        "clip_grad_norm_alpha": 1.0,
        # -- Конфигурация буфера (вложенная) --
        "buffer_config": {
            "capacity": 1_000_000,
            "use_prioritized": False,
            "alpha": 0.6,
            "beta": 0.4,
            "beta_annealing_steps": None,
            "epsilon": 1e-5,
            "use_sequence": False,
            "sequence_length": 10,
        },
        # -- Конфигурация нормализатора (вложенная) --
        "normalizer_config": {
            "use_normalizer": False,
            "normalizer_type": "welford",
            "clip_range": 10.0,
            "normalizer_epsilon": 1e-8,
        },
        # -- Конфигурация Warm Start (вложенная) --
        "warm_start_config": {
            "use_warm_start": False,
            "warm_start_type": "decision_tree",
            "warm_start_steps": 10000,
            "warm_start_random_steps": 1000,
            "warm_start_max_depth": 10,
            "warm_start_min_samples_split": 2,
            "warm_start_random_state": 42,
            "warm_start_noise": 0.1,
        },
    },
    # --- Параметры экспорта ONNX (для SACTrainer) ---
    "export_onnx": False,
    "export_dir": "onnx_export",
    "onnx_input_shape": None,
    # --- Конфигурация окружения (для EnvFactory) ---
    "env_config": {
        "env_name": "Pendulum-v1",  # Пример для Gym
        "file_name": None,
        "worker_id": 0,
        "base_port": 5004,
        "seed": None,
        "side_channels": [],
        "timeout_wait": 60,
        "no_graphics": False,
        # Параметры специфичные для Unity, которые могут не быть в Gym
        "time_scale": None,
        "flatten_obs": None,
        "normalize_actions": None,
    },
}


# Дополнительный класс для сериализации NumPy массивов и torch.dtype в JSON
class CustomEncoder(json.JSONEncoder):
    """
    Дополнительный класс для сериализации NumPy массивов и torch.dtype в JSON.
    """
    def default(self, obj: Any) -> Any:
        """
        Переопределённый метод сериализации объектов в JSON.

        Args:
            obj (Any): Объект для сериализации.

        Returns:
            Any: Сериализованное представление объекта.
        """
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
        elif isinstance(obj, torch.dtype):
            # Конвертируем torch.dtype в строку
            return str(obj).split(".")[-1]
        return super(CustomEncoder, self).default(obj)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Загружает конфигурацию из JSON-файла.

    Args:
        config_path (str): Путь к файлу конфигурации.

    Returns:
        Dict[str, Any]: Загруженная конфигурация.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Сохраняет конфигурацию в JSON-файл с использованием пользовательского кодировщика.

    Args:
        config (Dict[str, Any]): Конфигурация для сохранения.
        config_path (str): Путь, куда нужно сохранить конфигурацию.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, cls=CustomEncoder)


def expand_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Дополняет переданную конфигурацию значениями по умолчанию.

    Args:
        config (Dict[str, Any]): Пользовательская конфигурация.

    Returns:
        Dict[str, Any]: Расширенная конфигурация с дефолтными значениями.
    """
    # Создаем копию дефолтной конфигурации
    expanded_config = DEFAULT_CONFIG.copy()

    # Обновляем дефолтные значения пользовательскими
    if "env_config" in config and "env_config" in expanded_config:
        # Специальная обработка вложенного словаря env_config
        expanded_config["env_config"].update(config.get("env_config", {}))

        # Удаляем из основного словаря, чтобы избежать повторного обновления
        config_copy = config.copy()
        if "env_config" in config_copy:
            del config_copy["env_config"]

        # Обновляем остальные параметры
        expanded_config.update(config_copy)
    else:
        # Просто обновляем конфигурацию
        expanded_config.update(config)

    # Создаем директории, если их нет
    for dir_key in ["save_dir", "export_dir"]:
        if dir_key in expanded_config:
            os.makedirs(expanded_config[dir_key], exist_ok=True)

    return expanded_config


def create_experiment_dir(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Создаёт директории для эксперимента и обновляет пути в конфигурации.

    Args:
        config (Dict[str, Any]): Исходная конфигурация.

    Returns:
        Dict[str, Any]: Обновлённая конфигурация с указанием новых путей.
    """
    # Расширяем конфигурацию дефолтными значениями
    config = expand_config(config)

    # Получаем имя эксперимента
    exp_name = config.get("exp_name", "sac_default")

    # Создаем директории для эксперимента
    for dir_key in ["save_dir", "export_dir"]:
        if dir_key in config:
            base_dir = config[dir_key]
            exp_dir = os.path.join(base_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            config[dir_key] = exp_dir

    # Сохраняем конфигурацию в директорию результатов
    if "save_dir" in config:
        config_path = os.path.join(config["save_dir"], "config.json")
        save_config(config, config_path)

    return config


def get_config(
    config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Получает конфигурацию с возможностью переопределения параметров.

    Args:
        config_path (Optional[str]): Путь к файлу конфигурации (если есть).
        overrides (Optional[Dict[str, Any]]): Перечень параметров, которые нужно переопределить.

    Returns:
        Dict[str, Any]: Итоговая конфигурация после применения всех изменений.
    """
    # Начинаем с дефолтной конфигурации
    config = DEFAULT_CONFIG.copy()

    # Если указан путь к файлу, загружаем оттуда
    if config_path and os.path.exists(config_path):
        file_config = load_config(config_path)
        config.update(file_config)

    # Применяем переопределения
    if overrides:
        # Специальная обработка вложенного словаря env_config
        if "env_config" in overrides and "env_config" in config:
            config["env_config"].update(overrides.get("env_config", {}))

            # Удаляем из переопределений, чтобы избежать повторного обновления
            overrides_copy = overrides.copy()
            if "env_config" in overrides_copy:
                del overrides_copy["env_config"]

            # Обновляем остальные параметры
            config.update(overrides_copy)
        else:
            # Просто обновляем конфигурацию
            config.update(overrides)

    return config


def get_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлекает конфигурацию окружения из основной конфигурации.

    Args:
        config (Dict[str, Any]): Полная конфигурация.

    Returns:
        Dict[str, Any]: Только часть конфигурации, связанная с окружением.
    """
    # Получаем конфигурацию окружения
    env_config = config.get("env_config", {})

    # Добавляем параметры, которые могут быть нужны для окружения
    env_config["episode_max_steps"] = config.get("episode_max_steps", 1000)

    return env_config


def print_config(config: Dict[str, Any]) -> None:
    """
    Выводит конфигурацию в консоль в структурированном виде.

    Args:
        config (Dict[str, Any]): Конфигурация для вывода.
    """
    print("\n=== Конфигурация ===")

    # Группы параметров для лучшей организации вывода
    groups = {
        "Общие параметры": [
            "seed",
            "exp_name",
            "device",
            "save_dir",
            "model_name",
            "save_checkpoints",
            "keep_last_n_checkpoints",
            "load_checkpoint_path",
            "log_interval",
            "save_interval",
            "print_interval_episodes",
            "eval_interval",
            "eval_episodes",
            "cudnn_deterministic",
            "cudnn_benchmark",
        ],
        "Параметры обучения": [
            "max_steps",
            "episode_max_steps",
            "batch_size",
            "update_after",
            "start_steps",
            "updates_per_step",
            "reward_scale",
            "clear_cuda_cache_interval_episodes",
        ],
        "Параметры агента": [
            "agent_config.obs_dim",
            "agent_config.action_dim",
            "agent_config.device",
            "agent_config.hidden_dim",
            "agent_config.num_layers",
            "agent_config.activation_fn",
            "agent_config.use_lstm",
            "agent_config.use_layer_norm",
            "agent_config.dropout",
            "agent_config.optimizer_type",
            "agent_config.actor_lr",
            "agent_config.critic_lr",
            "agent_config.alpha_lr",
            "agent_config.weight_decay",
            "agent_config.optimizer_kwargs",
            "agent_config.gamma",
            "agent_config.tau",
            "agent_config.alpha",
            "agent_config.learn_alpha",
            "agent_config.target_entropy",
            "agent_config.clip_grad_norm_actor",
            "agent_config.clip_grad_norm_critic",
            "agent_config.clip_grad_norm_alpha",
            "agent_config.buffer_config.capacity",
            "agent_config.buffer_config.use_prioritized",
            "agent_config.buffer_config.alpha",
            "agent_config.buffer_config.beta",
            "agent_config.buffer_config.beta_annealing_steps",
            "agent_config.buffer_config.epsilon",
            "agent_config.buffer_config.use_sequence",
            "agent_config.buffer_config.sequence_length",
            "agent_config.normalizer_config.use_normalizer",
            "agent_config.normalizer_config.normalizer_type",
            "agent_config.normalizer_config.clip_range",
            "agent_config.normalizer_config.normalizer_epsilon",
            "agent_config.warm_start_config.use_warm_start",
            "agent_config.warm_start_config.warm_start_type",
            "agent_config.warm_start_config.warm_start_steps",
            "agent_config.warm_start_config.warm_start_random_steps",
            "agent_config.warm_start_config.warm_start_max_depth",
            "agent_config.warm_start_config.warm_start_min_samples_split",
            "agent_config.warm_start_config.warm_start_random_state",
            "agent_config.warm_start_config.warm_start_noise",
        ],
        "Параметры экспорта": ["export_onnx", "export_dir", "onnx_input_shape"],
        "Параметры среды": ["env_config"],
    }

    # Выводим параметры по группам
    for group_name, param_keys in groups.items():
        print(f"\n{group_name}:")
        for key in param_keys:
            if key in config:
                value = config[key]
                # Специальная обработка для env_config
                if key == "env_config" and isinstance(value, dict):
                    print(f"  {key}:")
                    for env_key, env_value in value.items():
                        print(f"    {env_key}: {env_value}")
                else:
                    print(f"  {key}: {value}")

    print("\n===================\n")


def load_or_create_config(
    config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Загружает конфигурацию из файла или создаёт новую с возможностью переопределения.

    Args:
        config_path (Optional[str]): Путь к файлу конфигурации (если есть).
        overrides (Optional[Dict[str, Any]]): Перечень параметров, которые нужно переопределить.

    Returns:
        Dict[str, Any]: Итоговая конфигурация.
    """
    # Если путь к файлу указан и файл существует, загружаем оттуда
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        # Расширяем дефолтными значениями
        config = expand_config(config)
    else:
        # Иначе используем дефолтную конфигурацию
        config = DEFAULT_CONFIG.copy()

    # Применяем переопределения
    if overrides:
        # Специальная обработка вложенного словаря env_config
        if "env_config" in overrides and "env_config" in config:
            config["env_config"].update(overrides.get("env_config", {}))

            # Удаляем из переопределений, чтобы избежать повторного обновления
            overrides_copy = overrides.copy()
            if "env_config" in overrides_copy:
                del overrides_copy["env_config"]

            # Обновляем остальные параметры
            config.update(overrides_copy)
        else:
            # Просто обновляем конфигурацию
            config.update(overrides)

    # Создаем директории для эксперимента
    config = create_experiment_dir(config)

    return config
