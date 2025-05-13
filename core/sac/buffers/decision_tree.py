import numpy as np
import json
import pickle
from typing import Dict, Any, Optional, Union, Tuple
import os

from core.logging.logger import get_logger

logger = get_logger("decision_tree")

# --- Узел дерева решений ---
class Node:
    """
    Класс, представляющий узел дерева решений.

    Attributes:
        feature_idx (Optional[int]): Индекс признака для разделения.
        threshold (Optional[float]): Пороговое значение для разделения.
        value (Optional[np.ndarray]): Значение для листового узла.
        left (Optional['Node']): Левый дочерний узел.
        right (Optional['Node']): Правый дочерний узел.
    """
    def __init__(self, 
                feature_idx: Optional[int] = None, 
                threshold: Optional[float] = None, 
                value: Optional[np.ndarray] = None, 
                left: Optional['Node'] = None, 
                right: Optional['Node'] = None):
        """
        Инициализация узла дерева решений.

        Args:
            feature_idx (Optional[int]): Индекс признака для разделения.
            threshold (Optional[float]): Пороговое значение для разделения.
            value (Optional[np.ndarray]): Значение для листового узла.
            left (Optional['Node']): Левый дочерний узел.
            right (Optional['Node']): Правый дочерний узел.
        """
        # Индекс признака для разделения
        self.feature_idx = feature_idx
        # Пороговое значение для разделения
        self.threshold = threshold
        # Значение для листа
        self.value = value
        # Левый дочерний узел
        self.left = left
        # Правый дочерний узел
        self.right = right
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Конвертация узла в словарь для сериализации.

        Returns:
            Dict[str, Any]: Словарь с данными узла.
        """
        node_dict = {
            'feature_idx': self.feature_idx,
            'threshold': float(self.threshold) if self.threshold is not None else None,
            'value': self.value.tolist() if self.value is not None else None,
            'left': self.left.to_dict() if self.left is not None else None,
            'right': self.right.to_dict() if self.right is not None else None
        }
        return node_dict
    
    @classmethod
    def from_dict(cls, node_dict: Dict[str, Any]) -> 'Node':
        """
        Создание узла из словаря.

        Args:
            node_dict (Dict[str, Any]): Словарь с данными узла.

        Returns:
            Node: Восстановленный узел.
        """
        # Рекурсивно создаем левый и правый дочерние узлы
        left = cls.from_dict(node_dict['left']) if node_dict['left'] is not None else None
        right = cls.from_dict(node_dict['right']) if node_dict['right'] is not None else None
        
        # Преобразуем значение в numpy массив
        value = np.array(node_dict['value']) if node_dict['value'] is not None else None
        
        # Создаем узел
        return cls(
            feature_idx=node_dict['feature_idx'],
            threshold=node_dict['threshold'],
            value=value,
            left=left,
            right=right
        )

# --- Дерево решений для регрессии ---
class DecisionTreeRegressor:
    """
    Регрессионная модель на основе дерева решений.

    Attributes:
        max_depth (int): Максимальная глубина дерева.
        min_samples_split (int): Минимальное количество образцов для разделения.
        max_features (Optional[Union[int, float]]): Максимальное количество признаков для выбора.
        random_state (Optional[int]): Сид для генератора случайных чисел.
        root (Optional[Node]): Корень дерева.
        n_features_ (Optional[int]): Размерность входных данных.
        n_outputs_ (Optional[int]): Размерность выходных данных.
    """

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, 
                max_features: Optional[Union[int, float]] = None, random_state: Optional[int] = None):
        """
        Инициализация регрессора дерева решений.

        Args:
            max_depth (int): Максимальная глубина дерева.
            min_samples_split (int): Минимальное количество образцов для разделения.
            max_features (Optional[Union[int, float]]): Максимальное количество признаков.
            random_state (Optional[int]): Сид для генератора случайных чисел.
        """
        # Максимальная глубина дерева
        self.max_depth = max_depth
        # Минимальное количество образцов для разделения
        self.min_samples_split = min_samples_split
        # Максимальное количество признаков для выбора при разделении
        self.max_features = max_features
        # Корень дерева
        self.root = None
        # Генератор случайных чисел
        self.random_state = np.random.RandomState(random_state)
        # Размерность входных данных
        self.n_features_ = None
        # Размерность выходных данных
        self.n_outputs_ = None
    
    def _calculate_mse(self, y: np.ndarray) -> float:
        """
        Вычисление среднеквадратичной ошибки (MSE).

        Args:
            y (np.ndarray): Целевые значения.

        Returns:
            float: MSE для целевых значений.
        """
        # MSE = среднее квадрата отклонений от среднего
        return np.mean(np.square(y - np.mean(y, axis=0)))
    
    def _select_features(self, n_features: int) -> np.ndarray:
        """
        Выбор подмножества признаков для рассмотрения при разделении.

        Args:
            n_features (int): Общее количество доступных признаков.

        Returns:
            np.ndarray: Индексы выбранных признаков.
        """
        # Если max_features не задано, используем все признаки
        if self.max_features is None:
            return np.arange(n_features)
        
        # Вычисляем количество признаков
        if isinstance(self.max_features, int):
            n_selected = min(self.max_features, n_features)
        else:  # Если задано как доля
            n_selected = max(1, int(self.max_features * n_features))
        
        # Выбираем случайные признаки
        return self.random_state.choice(n_features, size=n_selected, replace=False)
    
    def _calculate_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[
            Optional[int], Optional[float], Optional[np.ndarray], 
            Optional[np.ndarray], Optional[float]]:
        """
        Поиск лучшего разбиения по признакам и порогам.

        Args:
            X (np.ndarray): Входные данные.
            y (np.ndarray): Целевые значения.

        Returns:
            Tuple[Optional[int], ...]: Информация о лучшем разбиении.
        """
        # Получаем размеры данных
        n_samples, n_features = X.shape
        
        # Базовый случай: недостаточно образцов для разделения
        if n_samples < self.min_samples_split:
            return None, None, None, None, None
        
        # Выбираем признаки для рассмотрения
        features = self._select_features(n_features)
        
        # Начальные значения
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        best_mse_gain = -float('inf')
        
        # Исходная ошибка (до разделения)
        parent_mse = self._calculate_mse(y)
        
        # Перебираем все признаки
        for feature_idx in features:
            # Получаем уникальные значения признака
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Если только одно значение, пропускаем
            if len(thresholds) == 1:
                continue
            
            # Перебираем все пороговые значения
            for threshold in thresholds:
                # Разделяем данные
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]
                
                # Проверяем, что оба подмножества не пустые
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                # Вычисляем взвешенную сумму MSE для левого и правого подмножеств
                left_mse = self._calculate_mse(y[left_indices])
                right_mse = self._calculate_mse(y[right_indices])
                
                # Взвешиваем ошибки по размеру подмножеств
                n_left, n_right = len(left_indices), len(right_indices)
                weighted_mse = (n_left * left_mse + n_right * right_mse) / n_samples
                
                # Вычисляем прирост информации как разницу с исходной ошибкой
                mse_gain = parent_mse - weighted_mse
                
                # Обновляем лучшее разделение
                if mse_gain > best_mse_gain:
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices
                    best_mse_gain = mse_gain
        
        return best_feature, best_threshold, best_left_indices, best_right_indices, best_mse_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Рекурсивное построение дерева решений.

        Args:
            X (np.ndarray): Входные данные.
            y (np.ndarray): Целевые значения.
            depth (int): Текущая глубина дерева.

        Returns:
            Node: Корневой узел построенного дерева.
        """
        # Получаем размер данных
        n_samples, n_features = X.shape
        
        # Базовый случай 1: достигнута максимальная глубина
        if depth >= self.max_depth:
            leaf_value = np.mean(y, axis=0)
            return Node(value=leaf_value)
        
        # Базовый случай 2: недостаточно образцов для разделения
        if n_samples < self.min_samples_split:
            leaf_value = np.mean(y, axis=0)
            return Node(value=leaf_value)
        
        # Базовый случай 3: все целевые значения одинаковы
        if np.all(y == y[0]):
            leaf_value = y[0]
            return Node(value=leaf_value)
        
        # Находим лучшее разделение
        best_feature, best_threshold, best_left_indices, best_right_indices, best_gain = self._calculate_best_split(X, y)
        
        # Если не удалось найти разделение, создаем лист
        if best_feature is None:
            leaf_value = np.mean(y, axis=0)
            return Node(value=leaf_value)
        
        # Создаем левое и правое поддеревья
        left_subtree = self._build_tree(X[best_left_indices], y[best_left_indices], depth + 1)
        right_subtree = self._build_tree(X[best_right_indices], y[best_right_indices], depth + 1)
        
        # Возвращаем узел с найденным разделением
        return Node(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        """
        Обучение модели дереву решений.

        Args:
            X (np.ndarray): Входные данные.
            y (np.ndarray): Целевые значения.

        Returns:
            DecisionTreeRegressor: Обученная модель.
        """
        # Конвертируем входные данные в numpy массивы
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Если y одномерный, добавляем ось
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Сохраняем размерности
        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
        
        # Строим дерево
        self.root = self._build_tree(X, y)
        
        # Возвращаем self для возможности цепочки вызовов
        return self
    
    def _predict_sample(self, x: np.ndarray, node: Node) -> np.ndarray:
        """
        Предсказание для одного образца.

        Args:
            x (np.ndarray): Один образец входных данных.
            node (Node): Текущий узел дерева.

        Returns:
            np.ndarray: Предсказанное значение.
        """
        # Если узел - лист, возвращаем его значение
        if node.value is not None:
            return node.value
        
        # Определяем, в какую ветвь идти
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание для набора образцов.

        Args:
            X (np.ndarray): Входные данные.

        Returns:
            np.ndarray: Предсказанные значения.
        """
        # Проверяем, что дерево обучено
        if self.root is None:
            raise ValueError("Дерево еще не обучено, вызовите fit() перед predict()")
        
        # Конвертируем входные данные в numpy массив
        X = np.asarray(X)
        
        # Предсказываем для каждого образца
        predictions = np.array([self._predict_sample(x, self.root) for x in X])
        
        # Сжимаем размерность, если выход одномерный
        if self.n_outputs_ == 1:
            predictions = predictions.ravel()
            
        return predictions
    
    def get_tree_depth(self) -> int:
        """
        Получение фактической глубины дерева.

        Returns:
            int: Глубина дерева.
        """
        def _get_depth(node: Optional[Node]) -> int:
            if node is None or node.value is not None:
                return 0
            return 1 + max(_get_depth(node.left), _get_depth(node.right))
        
        if self.root is None:
            return 0
        return _get_depth(self.root)
    
    def get_n_leaves(self) -> int:
        """
        Получение количества листовых узлов.

        Returns:
            int: Количество листов дерева.
        """
        def _count_leaves(node: Optional[Node]) -> int:
            if node is None:
                return 0
            if node.value is not None:
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)
        
        if self.root is None:
            return 0
        return _count_leaves(self.root)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализация модели в словарь.

        Returns:
            Dict[str, Any]: Словарь с данными модели.
        """
        if self.root is None:
            raise ValueError("Дерево еще не обучено, вызовите fit() перед to_dict()")
        
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'max_features': self.max_features,
            'n_features_': self.n_features_,
            'n_outputs_': self.n_outputs_,
            'tree': self.root.to_dict()
        }
    
    @classmethod
    def from_dict(cls, tree_dict: Dict[str, Any]) -> 'DecisionTreeRegressor':
        """
        Создание модели из словаря.

        Args:
            tree_dict (Dict[str, Any]): Словарь с данными модели.

        Returns:
            DecisionTreeRegressor: Восстановленная модель.
        """
        # Создаем регрессор
        tree = cls(
            max_depth=tree_dict['max_depth'],
            min_samples_split=tree_dict['min_samples_split'],
            max_features=tree_dict['max_features']
        )
        
        # Восстанавливаем размерности
        tree.n_features_ = tree_dict['n_features_']
        tree.n_outputs_ = tree_dict['n_outputs_']
        
        # Восстанавливаем дерево
        tree.root = Node.from_dict(tree_dict['tree'])
        
        return tree
    
    def save(self, path: str, format: str = "pickle") -> None:
        """
        Сохранение модели в файл.

        Args:
            path (str): Путь к файлу.
            format (str): Формат сохранения ('pickle' или 'json').
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if format.lower() == "pickle":
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        elif format.lower() == "json":
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}. Используйте 'pickle' или 'json'")
    
    @classmethod
    def load(cls, path: str, format: str = None) -> 'DecisionTreeRegressor':
        """
        Загрузка модели из файла.

        Args:
            path (str): Путь к файлу.
            format (str): Формат файла ('pickle' или 'json').

        Returns:
            DecisionTreeRegressor: Загруженная модель.
        """
        # Определяем формат по расширению файла, если не указан
        if format is None:
            if path.endswith('.pkl') or path.endswith('.pickle'):
                format = 'pickle'
            elif path.endswith('.json'):
                format = 'json'
            else:
                raise ValueError("Не удалось определить формат файла, укажите явно через аргумент format")
        
        # Загружаем модель
        if format.lower() == 'pickle':
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif format.lower() == 'json':
            with open(path, 'r') as f:
                tree_dict = json.load(f)
                return cls.from_dict(tree_dict)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}. Используйте 'pickle' или 'json'")