import torch
import torch.nn as nn
import torch.onnx
import os
import onnx
from typing import Tuple, Optional, Union, List, Any

from core.logging.logger import get_logger
from core.sac.networks import BasePolicy, LSTMPolicy

logger = get_logger("export")

# Определяем минимальное и максимальное значение для log_std для стабильности
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class OnnxablePolicy(nn.Module):
    """
    Класс-обертка для конвертации политики в формат ONNX, совместимый с Unity ML-Agents.
    Принимает базовую политику и action_dim, возвращает 5 тензоров, ожидаемых Barracuda.
    """

    def __init__(self, policy: BasePolicy, action_dim: int = 2):
        """
        Конструктор класса.

        Args:
            policy (BasePolicy): Политика, которую необходимо экспортировать.
            action_dim (int): Размерность действий. По умолчанию равно 2.
        """
        super().__init__()
        if not isinstance(policy, BasePolicy):
            logger.warning(
                f"Переданная политика типа {type(policy).__name__} не является наследником BasePolicy."
            )
        self.policy = policy
        self.action_dim = action_dim
        self.is_lstm = isinstance(policy, LSTMPolicy)

        # Устанавливаем memory_size для LSTM-политик
        self.memory_size_value = 0
        if self.is_lstm and hasattr(policy, "hidden_size"):
            num_layers = (
                policy.lstm.num_layers if hasattr(policy.lstm, "num_layers") else 1
            )
            self.memory_size_value = 2 * num_layers * policy.hidden_size
            logger.info(
                f"LSTM обнаружена: memory_size={self.memory_size_value} (layers={num_layers}, hidden={policy.hidden_size})"
            )

        # Регистрируем буферы как тензоры формы [1] для совместимости с Unity
        self.register_buffer(
            "version_number", torch.tensor([3.0], dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "memory_size",
            torch.tensor([float(self.memory_size_value)], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "continuous_action_output_shape",
            torch.tensor([float(self.action_dim)], dtype=torch.float32),
            persistent=True,
        )

        logger.info(
            f"OnnxablePolicy инициализирована для {type(policy).__name__} с action_dim={action_dim}"
        )

    def forward(
        self,
        obs_0: torch.Tensor,
        obs_1: torch.Tensor,
        obs_2: torch.Tensor,
        obs_3: torch.Tensor,
        obs_4: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Прямой проход, возвращающий кортеж тензоров в формате, ожидаемом Barracuda.
        Принимает 5 входных тензоров наблюдений.

        Args:
            obs_0 (torch.Tensor): Первый тензор наблюдения.
            obs_1 (torch.Tensor): Второй тензор наблюдения.
            obs_2 (torch.Tensor): Третий тензор наблюдения.
            obs_3 (torch.Tensor): Четвертый тензор наблюдения.
            obs_4 (torch.Tensor): Пятый тензор наблюдения.

        Returns:
            Tuple[torch.Tensor]: Кортеж из 5 тензоров, подходящих для Unity ML-Agents.
        """
        with torch.no_grad():
            # Проверяем наличие размерности батча
            has_batch_dim = len(obs_0.shape) > 1
            if not has_batch_dim:
                obs_0 = obs_0.unsqueeze(0)
                obs_1 = obs_1.unsqueeze(0)
                obs_2 = obs_2.unsqueeze(0)
                obs_3 = obs_3.unsqueeze(0)
                obs_4 = obs_4.unsqueeze(0)
                logger.info(
                    f"Добавлена размерность батча к входным тензорам: {obs_0.shape}"
                )

            # Объединяем все наблюдения
            combined_obs = torch.cat([obs_0, obs_1, obs_2, obs_3, obs_4], dim=1)
            logger.info(f"Объединенный тензор наблюдений: shape={combined_obs.shape}")

            # Преобразуем для LSTM, если нужно
            working_obs = combined_obs
            if self.is_lstm and len(combined_obs.shape) == 2:
                working_obs = combined_obs.unsqueeze(1)

            # Получаем вывод политики
            policy_output = (
                self.policy(working_obs)
                if not self.is_lstm
                else self.policy(working_obs, None)
            )

            # Извлекаем mean и log_std
            if isinstance(policy_output, tuple) and len(policy_output) >= 2:
                mean, log_std = policy_output[:2]
            else:
                logger.warning("Используем заглушки для mean/log_std")
                mean = torch.zeros(
                    combined_obs.shape[0], self.action_dim, device=obs_0.device
                )
                log_std = torch.zeros_like(mean)

            # Ограничиваем log_std
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = log_std.exp()

            # Детерминированные и стохастические действия
            deterministic_continuous_actions = torch.tanh(mean)
            torch.manual_seed(42)  # Фиксированный seed для воспроизводимости
            noise = torch.randn_like(mean)
            continuous_actions = torch.tanh(mean + std * noise)

            # Возвращаем буферы напрямую как константы
            return (
                self.version_number,  # [1]
                self.memory_size,  # [1]
                continuous_actions,  # [batch, action_dim]
                self.continuous_action_output_shape,  # [1]
                deterministic_continuous_actions,  # [batch, action_dim]
            )


def export_model(
    model: nn.Module, obs_shapes: Union[list, tuple], onnx_path: str, action_dim: int
) -> Optional[str]:
    """
    Экспортирует модель политики в формат ONNX, совместимый с Unity ML-Agents.

    Args:
        model (nn.Module): Модель политики, которую нужно экспортировать.
        obs_shapes (Union[list, tuple]): Список или кортеж форм наблюдений.
        onnx_path (str): Путь для сохранения экспортированной модели.
        action_dim (int): Размерность действий.

    Returns:
        Optional[str]: Путь к сохраненной модели, если экспорт прошел успешно, иначе None.
    """
    original_device = (
        next(model.parameters()).device if model.parameters() else torch.device("cpu")
    )
    try:
        # Проверяем и корректируем obs_shapes
        if not isinstance(obs_shapes, list) or len(obs_shapes) != 5:
            logger.warning(
                f"Ожидается 5 форм наблюдений, получено {len(obs_shapes)}. Корректируем..."
            )
            if (
                isinstance(obs_shapes, tuple)
                and len(obs_shapes) == 1
                and obs_shapes[0] == 66
            ):
                obs_shapes = [(15,), (12,), (12,), (15,), (12,)]
                logger.info(f"Форма входа 66 разделена на 5: {obs_shapes}")
            else:
                while len(obs_shapes) < 5:
                    obs_shapes.append(obs_shapes[-1])
                obs_shapes = obs_shapes[:5]
                logger.info(f"Скорректированные входы: {obs_shapes}")

        # Создаем директорию
        export_dir = os.path.dirname(onnx_path)
        if export_dir and not os.path.exists(export_dir):
            os.makedirs(export_dir)
            logger.info(f"Создана директория: {export_dir}")

        # Перемещаем модель на CPU и переключаем в режим оценки
        model = model.cpu().eval()

        # Оборачиваем модель
        onnx_policy = OnnxablePolicy(model, action_dim=action_dim)
        onnx_policy.eval()

        # Создаем dummy-входы
        batch_dim = 1
        dummy_inputs = [
            torch.zeros((batch_dim, shape[0]), dtype=torch.float32)
            if len(shape) == 1
            else torch.zeros(shape, dtype=torch.float32)
            for shape in obs_shapes
        ]

        # Экспортируем модель
        torch.onnx.export(
            onnx_policy,
            tuple(dummy_inputs),
            onnx_path,
            export_params=True,
            opset_version=9,
            input_names=["obs_0", "obs_1", "obs_2", "obs_3", "obs_4"],
            output_names=[
                "version_number",
                "memory_size",
                "continuous_actions",
                "continuous_action_output_shape",
                "deterministic_continuous_actions",
            ],
            dynamic_axes={
                "obs_0": {0: "batch"},
                "obs_1": {0: "batch"},
                "obs_2": {0: "batch"},
                "obs_3": {0: "batch"},
                "obs_4": {0: "batch"},
                "continuous_actions": {0: "batch"},
                "deterministic_continuous_actions": {0: "batch"},
            },
            do_constant_folding=True,  # Включаем сворачивание констант
            verbose=False,
            keep_initializers_as_inputs=False,
        )

        logger.info(f"Модель успешно экспортирована в {onnx_path}")
        check_onnx_model(onnx_path)
        return onnx_path

    except Exception as e:
        logger.error(f"Ошибка экспорта в {onnx_path}: {str(e)}", exc_info=True)
        return None
    finally:
        if original_device is not None:
            model.to(original_device)
            logger.info(f"Модель возвращена на устройство: {original_device}")


def check_onnx_model(onnx_path: str) -> None:
    """
    Проверяет ONNX модель и выводит базовую информацию.

    Args:
        onnx_path (str): Путь к файлу ONNX модели.
    """
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"Проверка ONNX модели {onnx_path} прошла успешно.")
        # Дополнительная проверка выходов
        for output in onnx_model.graph.output:
            logger.debug(f"Выход: {output.name}, тип: {output.type}")
    except Exception as e:
        logger.error(f"Ошибка проверки модели {onnx_path}: {e}", exc_info=True)
