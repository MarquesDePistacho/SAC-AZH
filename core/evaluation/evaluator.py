import os
import time
import json
import numpy as np
from typing import Dict, Any

from core.training.config import expand_config
from core.sac.factories import SACAgentFactory, EnvFactory
from core.logging.logger import get_logger

# Проверяем доступность MLflow
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    raise ImportError(
        "MLflow не установлен. Установите его командой: pip install mlflow"
    )

logger = get_logger("evaluator")


class SACEvaluator:
    """
    Класс для оценки SAC-агентов.

    Attributes:
        config (Dict[str, Any]): Конфигурация агента и среды.
        env (gym.Env): Среда Gym для взаимодействия с агентом.
        obs_dim (int): Размерность пространства наблюдений.
        action_dim (int): Размерность пространства действий.
        agent (Optional[SACAgent]): Загруженный агент для оценки.
    """

    def __init__(self, config: Dict[str, Any], env=None):
        """
        Конструктор класса.

        Attributes:
            config (Dict[str, Any]): Конфигурация агента и среды.
            env (gym.Env): Среда Gym для взаимодействия с агентом.
            obs_dim (int): Размерность пространства наблюдений.
            action_dim (int): Размерность пространства действий.
            agent (Optional[SACAgent]): Загруженный агент для оценки.
        """
        # Расширяем конфигурацию дефолтными значениями
        self.config = expand_config(config)

        # Создаем или используем переданное окружение
        self.env = env
        if self.env is None:
            # Если окружение не передано, пытаемся создать его
            if "env_config" in self.config:
                env_factory = EnvFactory()
                env_config = self.config["env_config"]
                self.env = env_factory.create_gym_env(
                    env_name=env_config.get("env_name", "default"), config=env_config
                )
            else:
                raise ValueError(
                    "Не указана конфигурация окружения и не передано готовое окружение"
                )

        # Получаем размерности пространств наблюдений и действий
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

        # Создаем или загружаем агента
        self.agent = None
        if self.config.get("model_path"):
            self._load_agent(self.config["model_path"])

        logger.info(
            f"Инициализирован оценщик SAC для среды {self.config['env_config'].get('env_name', 'unknown')}"
        )

    def _load_agent(self, model_path: str) -> None:
        """
        Загружает модель агента из указанного пути.

        Args:
            model_path (str): Путь к файлу сохраненной модели.
        """
        # Создаем агента с помощью фабрики
        agent = SACAgentFactory.create(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.get("hidden_dim", 256),
            num_layers=self.config.get("num_layers", 2),
            activation_fn=self.config.get("activation_fn", "relu"),
            use_lstm=self.config.get("use_lstm", False),
            device=self.config.get("device", "cpu"),
        )

        # Загружаем модель
        try:
            agent.load(model_path)
            self.agent = agent
            logger.info(f"Модель успешно загружена из {model_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise

    def evaluate(
        self, n_episodes: int = 10, deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Оценивает производительность агента на заданном количестве эпизодов.

        Args:
            n_episodes (int): Количество эпизодов для оценки. По умолчанию 10.
            deterministic (bool): Использовать ли детерминированный выбор действий. По умолчанию True.

        Returns:
            Dict[str, Any]: Статистика по результатам оценки (средняя награда, время выполнения и т.д.).
        """
        if self.agent is None:
            raise ValueError(
                "Агент не загружен. Используйте метод _load_agent перед оценкой."
            )

        logger.info(f"Начало оценки агента на {n_episodes} эпизодах")

        # Подготовка к MLflow логированию
        if MLFLOW_AVAILABLE:
            # Устанавливаем или создаем эксперимент
            experiment_name = self.config.get(
                "mlflow_experiment_name", "SAC_Evaluation"
            )
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)

            # Создаем новый запуск
            run_name = self.config.get(
                "mlflow_run_name", f"eval-{time.strftime('%Y%m%d-%H%M%S')}"
            )
            mlflow.start_run(run_name=run_name)

            # Логируем параметры
            eval_params = {
                "n_episodes": n_episodes,
                "deterministic": deterministic,
                "env_name": self.config["env_config"].get("env_name", "unknown"),
                "model_path": self.config.get("model_path", "unknown"),
            }
            mlflow.log_params(eval_params)

        # Статистика для отслеживания
        returns = []
        lengths = []
        start_time = time.time()

        # Запускаем оценку
        for episode in range(n_episodes):
            episode_return = 0
            episode_length = 0
            done = False
            obs = self.env.reset()

            # Сбрасываем скрытые состояния для LSTM
            if self.config.get("use_lstm", False):
                self.agent.reset_hidden()

            # Проходим эпизод
            while not done:
                # Получаем действие от агента
                action = self.agent.select_action(obs, deterministic=deterministic)

                # Делаем шаг в окружении
                next_obs, reward, done, info = self.env.step(action)

                # Обновляем статистику
                episode_return += reward
                episode_length += 1

                # Обновляем наблюдение
                obs = next_obs

            # Сохраняем результаты эпизода
            returns.append(episode_return)
            lengths.append(episode_length)

            # Логируем в MLflow
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics(
                    {
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                    },
                    step=episode,
                )

            logger.info(
                f"Эпизод {episode + 1}/{n_episodes}: Награда = {episode_return:.2f}, Длина = {episode_length}"
            )

        # Рассчитываем статистику
        total_time = time.time() - start_time
        results = {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "min_return": np.min(returns),
            "max_return": np.max(returns),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "returns": returns,
            "lengths": lengths,
            "total_time": total_time,
            "n_episodes": n_episodes,
            "deterministic": deterministic,
        }

        # Логируем итоговые метрики в MLflow
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(
                {
                    "mean_return": results["mean_return"],
                    "std_return": results["std_return"],
                    "min_return": results["min_return"],
                    "max_return": results["max_return"],
                    "mean_length": results["mean_length"],
                    "std_length": results["std_length"],
                    "total_time": total_time,
                }
            )

            # Сохраняем результаты как артефакт
            results_file = f"evaluation_results_{time.strftime('%Y%m%d-%H%M%S')}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            mlflow.log_artifact(results_file)
            os.remove(results_file)  # Удаляем временный файл

            # Закрываем MLflow run
            mlflow.end_run()

        # Выводим итоговую статистику
        logger.info(f"Оценка завершена за {total_time:.2f}с")
        logger.info(
            f"Средняя награда: {results['mean_return']:.2f} ± {results['std_return']:.2f}"
        )
        logger.info(f"Средняя длина эпизода: {results['mean_length']:.2f}")

        return results

    def close(self):
        """
        Закрывает среду и освобождает ресурсы.
        """
        if hasattr(self.env, "close"):
            self.env.close()


def evaluate_agent(
    config: Dict[str, Any], n_episodes: int = 10, deterministic: bool = True
) -> Dict[str, Any]:
    """
    Функция для запуска оценки агента.

    Args:
        config (Dict[str, Any]): Конфигурация агента и окружения.
        n_episodes (int): Количество эпизодов для оценки.
        deterministic (bool): Использовать ли детерминированный выбор действий.

    Returns:
        Dict[str, Any]: Результаты оценки агента.
    """
    evaluator = SACEvaluator(config)

    try:
        results = evaluator.evaluate(n_episodes, deterministic)
        return results
    finally:
        evaluator.close()
