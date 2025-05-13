import os
import time
import json
import torch
import psutil
import platform
import gc
import numpy as np
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
from threading import Lock

# Проверяем доступность MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    raise ImportError("MLflow не установлен. Установите его командой: pip install mlflow")

from core.logging.logger import get_logger

logger = get_logger("mlflow_logger")

# Константы для названий экспериментов
TRAINING_EXPERIMENT = "SAC_Training"
EVAL_EXPERIMENT = "SAC_Evaluation"
BENCHMARK_EXPERIMENT = "SAC_Benchmarks"

# Кэш активных сессий для предотвращения конфликтов
_active_runs = {}
_mlflow_lock = Lock()

def setup_experiment(experiment_name: Optional[str] = None, run_name: Optional[str] = None) -> str:
    """
    Настраивает MLflow эксперимент и создает новую сессию.
    
    Args:
        experiment_name: Название эксперимента (если None, используется TRAINING_EXPERIMENT)
        run_name: Название запуска (если None, генерируется автоматически)
        
    Returns:
        str: ID запущенной сессии MLflow
    """
    with _mlflow_lock:
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow не доступен, логирование отключено")
            return None
        
        # Завершаем текущий активный запуск, если он есть
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()
            logger.debug(f"Завершен активный запуск MLflow: {active_run.info.run_id}")
        
        # Определяем название эксперимента
        if experiment_name is None:
            experiment_name = TRAINING_EXPERIMENT
        
        # Получаем или создаем эксперимент
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            logger.info(f"Создан новый эксперимент MLflow: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        
        # Генерируем название запуска, если не указано
        if run_name is None:
            run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Запускаем новую сессию
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id
        _active_runs[run_id] = {"experiment_name": experiment_name, "run_name": run_name}
        
        logger.info(f"Запущен MLflow эксперимент: {experiment_name}, run: {run_name}, ID: {run_id}")
        return run_id

def end_run(run_id: Optional[str] = None) -> None:
    """
    Завершает сессию MLflow.
    
    Args:
        run_id: ID сессии для завершения. Если None, завершается текущая активная сессия.
    """
    with _mlflow_lock:
        if not MLFLOW_AVAILABLE:
            return
        
        if run_id is not None and run_id in _active_runs:
            active_run = mlflow.active_run()
            if active_run and active_run.info.run_id != run_id:
                # Сохраняем текущий запуск
                current_run_id = active_run.info.run_id
                mlflow.end_run()
                # Переключаемся на указанный запуск и завершаем его
                mlflow.start_run(run_id=run_id)
                mlflow.end_run()
                # Восстанавливаем предыдущий запуск
                if current_run_id in _active_runs:
                    mlflow.start_run(run_id=current_run_id)
            else:
                mlflow.end_run()
            
            del _active_runs[run_id]
            logger.info(f"Завершен MLflow запуск: {run_id}")
        else:
            # Завершаем текущий активный запуск
            active_run = mlflow.active_run()
            if active_run:
                run_id = active_run.info.run_id
                mlflow.end_run()
                if run_id in _active_runs:
                    del _active_runs[run_id]
                logger.info(f"Завершен активный MLflow запуск: {run_id}")

def log_params(params: Dict[str, Any], run_id: Optional[str] = None) -> None:
    """
    Логирует параметры в MLflow.
    
    Args:
        params: Словарь параметров для логирования
        run_id: ID сессии MLflow (если None, используется текущая сессия)
    """
    if not MLFLOW_AVAILABLE:
        return
    
    try:
        if run_id is not None and mlflow.active_run() and mlflow.active_run().info.run_id != run_id:
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.log_params(params)
        else:
            mlflow.log_params(params)
    except Exception as e:
        logger.error(f"Ошибка при логировании параметров: {e}")

def log_system_metrics() -> Dict[str, float]:
    """
    Собирает системные метрики и метрики оборудования.
    
    Returns:
        Dict[str, float]: Словарь метрик
    """
    metrics = {}
    try:
        # Системная память
        memory = psutil.virtual_memory()
        metrics["system/memory_used_gb"] = memory.used / (1024**3)
        metrics["system/memory_percent"] = memory.percent
        
        # Память процесса
        process = psutil.Process(os.getpid())
        metrics["process/memory_gb"] = process.memory_info().rss / (1024**3)
        metrics["process/cpu_percent"] = process.cpu_percent()
        
        # Общая загрузка CPU
        metrics["system/cpu_percent"] = psutil.cpu_percent(interval=None)
        
        # Память CUDA
        if torch.cuda.is_available():
            metrics["cuda/memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            metrics["cuda/memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            
            # Подробная информация по всем CUDA устройствам
            for i in range(torch.cuda.device_count()):
                stats = torch.cuda.memory_stats(i)
                metrics[f"cuda_{i}/memory_active_gb"] = stats.get("active.all.current", 0) / (1024**3)
                metrics[f"cuda_{i}/memory_allocated_gb"] = stats.get("allocated.all.current", 0) / (1024**3)
                
                # Утилизация GPU (если доступно)
                try:
                    if hasattr(torch.cuda, 'utilization'):
                        metrics[f"cuda_{i}/gpu_utilization"] = torch.cuda.utilization(i)
                except:
                    pass
    except Exception as e:
        logger.error(f"Ошибка при сборе системных метрик: {e}")
    
    return metrics

def log_metrics(metrics: Dict[str, float], step: int, run_id: Optional[str] = None) -> None:
    """
    Логирует метрики в MLflow.
    
    Args:
        metrics: Словарь метрик для логирования
        step: Шаг (для оси X на графиках)
        run_id: ID сессии MLflow (если None, используется текущая сессия)
    """
    if not MLFLOW_AVAILABLE:
        return
    
    try:
        # Преобразуем все значения в float для MLflow
        clean_metrics = {k: float(v) for k, v in metrics.items()}
        
        if run_id is not None and mlflow.active_run() and mlflow.active_run().info.run_id != run_id:
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.log_metrics(clean_metrics, step=step)
        else:
            mlflow.log_metrics(clean_metrics, step=step)
    except Exception as e:
        logger.error(f"Ошибка при логировании метрик: {e}")

def log_memory_usage(step: int, run_id: Optional[str] = None) -> None:
    """
    Логирует информацию об использовании памяти.
    
    Args:
        step: Шаг (для оси X на графиках)
        run_id: ID сессии MLflow (если None, используется текущая сессия)
    """
    if not MLFLOW_AVAILABLE:
        return
    
    try:
        memory_metrics = {}
        
        # Системная память
        memory = psutil.virtual_memory()
        memory_metrics["system_memory_used_gb"] = memory.used / (1024**3)
        memory_metrics["system_memory_percent"] = memory.percent
        
        # Память процесса
        process = psutil.Process(os.getpid())
        memory_metrics["process_memory_gb"] = process.memory_info().rss / (1024**3)
        
        # Память CUDA
        if torch.cuda.is_available():
            memory_metrics["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            memory_metrics["cuda_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            
            # Подробная информация по всем CUDA устройствам
            for i in range(torch.cuda.device_count()):
                stats = torch.cuda.memory_stats(i)
                memory_metrics[f"cuda_{i}_memory_active_gb"] = stats.get("active.all.current", 0) / (1024**3)
                memory_metrics[f"cuda_{i}_memory_allocated_gb"] = stats.get("allocated.all.current", 0) / (1024**3)
        
        log_metrics(memory_metrics, step, run_id)
    except Exception as e:
        logger.error(f"Ошибка при логировании использования памяти: {e}")

def log_artifact(local_path: str, run_id: Optional[str] = None) -> None:
    """
    Логирует артефакт в MLflow.
    
    Args:
        local_path: Путь к локальному файлу для логирования
        run_id: ID сессии MLflow (если None, используется текущая сессия)
    """
    if not MLFLOW_AVAILABLE:
        return
    
    try:
        if run_id is not None and mlflow.active_run() and mlflow.active_run().info.run_id != run_id:
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.log_artifact(local_path)
        else:
            mlflow.log_artifact(local_path)
    except Exception as e:
        logger.error(f"Ошибка при логировании артефакта {local_path}: {e}")

def save_metrics_to_json(
    raw_metrics: Dict[str, List], 
    episode_metrics: Dict[str, List], 
    config: Dict[str, Any], 
    filepath: str,
    total_episodes: int,
    total_steps: int,
    training_time: float,
    run_id: Optional[str] = None
) -> None:
    """
    Сохраняет метрики в JSON файл и логирует его как артефакт.
    
    Args:
        raw_metrics: Словарь метрик по шагам [(step, value), ...]
        episode_metrics: Словарь метрик по эпизодам [(step, value), ...]
        config: Конфигурация обучения
        filepath: Путь для сохранения JSON-файла
        total_episodes: Общее число эпизодов
        total_steps: Общее число шагов
        training_time: Время обучения в секундах
        run_id: ID сессии MLflow (если None, используется текущая сессия)
    """
    try:
        # Подготовка данных для сохранения
        data_to_save = {
            "raw_metrics": raw_metrics,
            "episode_metrics": episode_metrics,
            "total_episodes": total_episodes,
            "total_steps": total_steps,
            "training_time": training_time,
            "config": config
        }
        
        # Убедимся, что директория существует
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Сохраняем JSON файл
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        # Логируем как артефакт
        log_artifact(filepath, run_id)
        logger.info(f"Метрики сохранены в {filepath}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении метрик в JSON: {e}")

class MetricsTracker:
    """
    Класс для отслеживания метрик обучения с поддержкой MLflow.
    Оптимизирован для частого обновления метрик без излишней нагрузки на систему.
    """
    
    def __init__(self, config: Dict[str, Any], experiment_name: Optional[str] = None):
        """
        Инициализирует трекер метрик.
        
        Args:
            config: Конфигурация обучения
            experiment_name: Название эксперимента MLflow (если None, используется TRAINING_EXPERIMENT)
        """
        self.config = config
        self.raw_metrics = defaultdict(list)  # Метрики по шагам [(step, value), ...]
        self.episode_metrics = defaultdict(list)  # Метрики по эпизодам [(episode, value), ...]
        self.steps = []  # Шаги, на которых производилось логирование
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.total_episodes = 0
        self.total_steps = 0  # Счетчик общего количества шагов
        self.lock = Lock()  # Для потокобезопасности
        
        # Создаем директорию для результатов
        self.save_dir = config.get("save_dir", "results")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Инициализация MLflow
        run_name = config.get("run_name", f"train_{time.strftime('%Y%m%d_%H%M%S')}")
        self.run_id = setup_experiment(experiment_name, run_name)
        
        # Логирование конфигурации
        if self.run_id:
            log_params(config, self.run_id)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, episode: Optional[int] = None) -> None:
        """
        Логирует метрики обучения. Теперь только сохраняет метрики в буфер,
        без немедленного логирования в MLflow.
        
        Args:
            metrics: Словарь метрик для логирования
            step: Шаг обучения
            episode: Номер эпизода (опционально)
        """
        with self.lock:
            # Обновляем счетчик общего количества шагов
            self.total_steps = max(self.total_steps, step)
            
            # Добавляем шаг
            if step not in self.steps:
                self.steps.append(step)
            
            # Обновляем счетчик эпизодов
            if episode is not None:
                self.total_episodes = max(self.total_episodes, episode + 1)
            
            # Добавляем метрики о времени
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            metrics["time_elapsed"] = elapsed_time
            metrics["time_step"] = current_time - self.last_log_time
            self.last_log_time = current_time
            
            # Сохраняем метрики в буфер
            for key, value in metrics.items():
                self.raw_metrics[key].append((step, value))
    
    def log_episode_metrics(self, metrics: Dict[str, float], episode: int, step: Optional[int] = None) -> None:
        """
        Логирует все метрики эпизода, включая системные метрики.
        
        Args:
            metrics: Словарь метрик для логирования
            episode: Номер эпизода
            step: Шаг обучения (опционально) для использования как X-координата
        """
        with self.lock:
            # Обновляем счетчик эпизодов
            self.total_episodes = max(self.total_episodes, episode + 1)
            
            # Используем шаг как X-координату, если указан, иначе используем total_steps
            x_coord_step = step if step is not None else self.total_steps
            
            # Собираем системные метрики
            system_metrics = log_system_metrics()
            
            # Объединяем все метрики
            all_metrics = {
                **metrics,  # Метрики обучения
                **system_metrics,  # Системные метрики
            }
            
            # Сохраняем метрики в буфер эпизодов
            for key, value in all_metrics.items():
                self.episode_metrics[key].append((x_coord_step, value))
            
            # Логируем все метрики в MLflow с шагом как осью X
            if self.run_id:
                log_metrics(all_metrics, x_coord_step, self.run_id)
    
    def save(self, filepath: Optional[str] = None) -> None:
        """
        Сохраняет метрики в JSON файл.
        
        Args:
            filepath: Путь для сохранения (если None, генерируется автоматически)
        """
        if filepath is None:
            timestr = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_dir, f"metrics_{timestr}.json")
        
        save_metrics_to_json(
            {k: v for k, v in self.raw_metrics.items()},
            {k: v for k, v in self.episode_metrics.items()},
            self.config,
            filepath,
            self.total_episodes,
            self.total_steps,
            time.time() - self.start_time,
            self.run_id
        )
    
    def close(self) -> None:
        """
        Закрывает трекер метрик и освобождает ресурсы.
        """
        # Сохраняем метрики
        self.save()
        
        # Завершаем MLflow сессию
        end_run(self.run_id)
        
        # Очищаем буферы
        self.raw_metrics.clear()
        self.episode_metrics.clear()
        self.steps.clear()
        
        # Запускаем сборку мусора
        gc.collect()
        
        logger.info("Трекер метрик закрыт")

# Функции для бенчмарков 
def benchmark_run(experiment_name: str = BENCHMARK_EXPERIMENT, run_name: Optional[str] = None) -> Any:
    """
    Создает декоратор для запуска бенчмарка в MLflow.
    
    Args:
        experiment_name: Название эксперимента
        run_name: Название запуска (если None, генерируется автоматически)
    
    Returns:
        Декоратор для функции бенчмарка
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Получаем имя запуска из аргументов или генерируем
            actual_run_name = run_name
            if actual_run_name is None:
                device = kwargs.get('device', 'cpu')
                actual_run_name = f"benchmark_{func.__name__}_{device}"
            
            # Настраиваем MLflow и запускаем бенчмарк
            run_id = setup_experiment(experiment_name, actual_run_name)
            try:
                # Логируем параметры бенчмарка
                params = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}
                log_params(params, run_id)
                
                # Запускаем бенчмарк
                return func(*args, **kwargs)
            finally:
                # Закрываем MLflow сессию
                end_run(run_id)
        return wrapper
    return decorator 