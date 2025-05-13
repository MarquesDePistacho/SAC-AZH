import torch
from torch.amp import autocast
import numpy as np
from typing import Any, Dict, Optional, Union, Tuple, List

from core.sac.device import device_manager
from core.sac.batching.batching import BatchFetcher
from core.sac.losses import compute_critic_target, compute_critic_loss, compute_actor_loss
from core.sac.plugins.plugin_base import AgentPlugin, PLUGIN_REGISTRY
from core.sac.buffers.segment_tree import check_bfloat16_support
from core.sac.components import PolicyComponent, CriticComponent, AlphaComponent, NormalizerComponent


class CoreSACAgent:
    """
    Базовый SAC-агент: включает инициализацию компонентов, плагины и инфраструктуру обновления.
    Содержит только приватные методы; публичные методы определены в agent_api.py
    """
    def __init__(
        self,
        policy_net: Any,
        q_net: Any,
        replay_buffer: Any,
        replay_loader_workers: int = 0,
        actor_optimizer: Optional[Any] = None,
        critic_optimizer: Optional[Any] = None,
        policy_component: Optional[PolicyComponent] = None,
        critic_component: Optional[CriticComponent] = None,
        alpha_component: Optional[AlphaComponent] = None,
        normalizer_component: Optional[NormalizerComponent] = None,
        device: str = "cpu",
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_lr: float = 3e-4,
        alpha_init: float = 0.2,
        learn_alpha: bool = True,
        target_entropy: Optional[float] = None,
        clip_grad_norm_actor: Optional[float] = 1.0,
        clip_grad_norm_critic: Optional[float] = 1.0,
        clip_grad_norm_alpha: Optional[float] = 1.0,
        normalizer: Optional[Any] = None,
        normalize_obs: bool = True,
        clip_obs: float = 10.0,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        warm_start_config: Optional[Dict[str, Any]] = None,
        use_amp: Optional[bool] = None,
        use_bfloat16: Optional[bool] = None,
        adaptive_exploration: bool = True,
        exploration_ma_decay: float = 0.99,
        min_log_std: float = -2.0,
        max_log_std: float = 2.0,
    ):
        # Сохраняем параметры для сохранения состояния
        self.gamma = gamma
        self.tau = tau
        # Инициализация устройства и precision
        self._setup_device_and_precision(device, use_amp, use_bfloat16)
        # Определение размерностей
        self._setup_dims(policy_net, obs_dim, action_dim)
        # Инициализация компонентов
        self._init_components(
            policy_net, q_net,
            actor_optimizer, critic_optimizer,
            policy_component, critic_component,
            alpha_component, normalizer_component,
            gamma, tau, alpha_lr, alpha_init, learn_alpha,
            target_entropy, clip_grad_norm_actor,
            clip_grad_norm_critic, clip_grad_norm_alpha,
            normalizer, normalize_obs, clip_obs,
            replay_buffer, replay_loader_workers
        )
        # Инициализация плагинов
        self._init_plugins(adaptive_exploration, min_log_std, max_log_std,
                           exploration_ma_decay, warm_start_config)
        # Счетчик и скрытые состояния
        self.update_counter = 0
        self._policy_hidden = None

    def _setup_device_and_precision(
        self, device: str, use_amp: Optional[bool], use_bfloat16: Optional[bool]
    ) -> None:
        """
        Настраивает устройство (CPU/GPU), поддержку AMP и bfloat16.
        """
        # Определяем устройство
        self.device = device_manager.get_device(device)
        # Поддержка bfloat16
        self.supports_bfloat16 = check_bfloat16_support(self.device)
        # Автоматическое смешанное прецизионное обучение (AMP)
        if use_amp is None:
            self.enable_amp = self.device.type == "cuda"
        else:
            self.enable_amp = use_amp and (self.device.type == "cuda")
        # Использование bfloat16 вместо float16, если поддерживается
        if use_bfloat16 is None:
            self.use_bfloat16 = self.supports_bfloat16
        else:
            self.use_bfloat16 = use_bfloat16 and self.supports_bfloat16
        # Выбор dtype для AMP
        if self.enable_amp:
            self.amp_dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
        else:
            self.amp_dtype = torch.float32

    def _setup_dims(
        self, policy_net: Any, obs_dim: Optional[int], action_dim: Optional[int]
    ) -> None:
        """
        Определяет и сохраняет размерности входа и выхода (obs_dim, action_dim) из сети или аргументов.
        """
        # Определяем размерность наблюдений
        if obs_dim is None:
            obs_dim = self._get_network_input_dim(policy_net)
            # логируем, если автоопределено
            try:
                from core.logging.logger import get_logger
            except ImportError:
                pass
        self._obs_dim = obs_dim
        # Определяем размерность действий
        if action_dim is None:
            action_dim = self._get_network_action_dim(policy_net)
        self._action_dim = action_dim

    def _init_components(
        self,
        policy_net: Any,
        q_net: Any,
        actor_optimizer: Optional[Any],
        critic_optimizer: Optional[Any],
        policy_component: Optional[PolicyComponent],
        critic_component: Optional[CriticComponent],
        alpha_component: Optional[AlphaComponent],
        normalizer_component: Optional[NormalizerComponent],
        gamma: float,
        tau: float,
        alpha_lr: float,
        alpha_init: float,
        learn_alpha: bool,
        target_entropy: Optional[float],
        clip_grad_norm_actor: float,
        clip_grad_norm_critic: float,
        clip_grad_norm_alpha: float,
        normalizer: Optional[Any],
        normalize_obs: bool,
        clip_obs: float,
        replay_buffer: Any,
        replay_loader_workers: int,
    ) -> None:
        """
        Инициализирует основные компоненты: policy, critic, alpha, normalizer и DataLoader.
        """
        # Инициализация компонента Policy
        self.policy = policy_component or PolicyComponent(
            policy_net,
            actor_optimizer,
            clip_grad_norm=clip_grad_norm_actor,
            device=self.device,
        )
        # Инициализация компонента Critic
        self.critic = critic_component or CriticComponent(
            q_net,
            critic_optimizer,
            tau=tau,
            clip_grad_norm=clip_grad_norm_critic,
            device=self.device,
        )
        # Инициализация компонента Alpha
        self.alpha = alpha_component or AlphaComponent(
            action_dim=self._action_dim,
            initial_alpha=alpha_init,
            learn_alpha=learn_alpha,
            target_entropy=target_entropy,
            clip_grad_norm=clip_grad_norm_alpha,
            device=self.device,
        )
        # Инициализация Normalizer
        if normalizer_component:
            self.normalizer = normalizer_component
        else:
            # Выбираем реализацию нормализатора
            from core.sac.normalizers import WelfordObservationNormalizer, DummyNormalizer
            norm_obj = WelfordObservationNormalizer(self._obs_dim) if normalize_obs else DummyNormalizer(self._obs_dim)
            self.normalizer = NormalizerComponent(
                normalizer=norm_obj,
                clip_obs=clip_obs,
                device=self.device,
                obs_dim=self._obs_dim,
            )
        # Настройка replay buffer и DataLoader
        self.replay_buffer = replay_buffer
        self._use_dataloader = replay_loader_workers > 0
        self._dataloader_workers = replay_loader_workers
        self._dataloader_batch_size = None
        self._dataloader = None
        self._dataloader_iter = None

    def _init_plugins(
        self,
        adaptive_exploration: bool,
        min_log_std: float,
        max_log_std: float,
        exploration_ma_decay: float,
        warm_start_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Инициализирует и конфигурирует все зарегистрированные плагины.
        """
        self.plugins: List[AgentPlugin] = []
        for name, plugin_cls in PLUGIN_REGISTRY.items():
            # Собираем конфиг для плагина
            cfg: Dict[str, Any] = {}
            if name == 'adaptive_explorer':
                cfg = {'min_log_std': min_log_std, 'max_log_std': max_log_std, 'exploration_ma_decay': exploration_ma_decay}
            elif name == 'warm_start' and warm_start_config:
                cfg = warm_start_config
            # Создаём плагин
            plugin = plugin_cls(self, **cfg)
            self.plugins.append(plugin)
        # Хук on_init для всех плагинов
        for plugin in self.plugins:
            plugin.on_init()

    def _compile_critical_functions(self) -> None:
        """
        JIT-компиляция критических функций: policy.sample и critic.forward на CUDA.
        """
        if torch.jit.is_scripting() or self.device.type != "cuda":
            return
        try:
            if hasattr(self.policy, "sample") and not isinstance(
                self.policy.sample, torch.jit.ScriptFunction
            ):
                test_obs = torch.zeros((1, self._obs_dim), device=self.device)
                self.policy.sample(test_obs)
            if hasattr(self.critic, "forward") and not isinstance(
                self.critic.forward, torch.jit.ScriptFunction
            ):
                test_obs = torch.zeros((1, self._obs_dim), device=self.device)
                test_action = torch.zeros((1, self._action_dim), device=self.device)
                self.critic.forward(test_obs, test_action)
        except Exception:
            pass

    def _fetch_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Получает батч из replay_buffer, используя DataLoader если включено.
        """
        if self._use_dataloader:
            if self._dataloader is None or self._dataloader_batch_size != batch_size:
                dataset = BatchFetcher(self.replay_buffer, self._dataloader_workers)
                self._dataloader = dataset
                self._dataloader_batch_size = batch_size
            return self._dataloader.next_batch(batch_size)
        else:
            return self.replay_buffer.sample(batch_size)

    def _update_components(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Выполняет обновление сущностей агента: critic, actor, alpha, soft update и обновление приоритетов.
        Вызывает хуки before_update и after_update плагинов.
        """
        # Хук перед обновлением
        for plugin in self.plugins:
            plugin.before_update(batch)
        # Извлекаем приоритетные данные
        tree_indices = batch.get('tree_indices', None)
        importance_weights = batch.get(
            'is_weights', torch.ones(batch['rewards'].shape, device=self.device)
        )
        # 1. Critic update
        critic_loss, td_errors = self._compute_critic_loss(batch, self.alpha.alpha)
        weighted_critic_loss = (critic_loss * importance_weights).mean()
        critic_loss_val = self.critic.update(weighted_critic_loss)
        # 2. Actor update
        policy_loss, log_probs = self._compute_actor_loss(batch)
        weighted_policy_loss = (policy_loss * importance_weights).mean()
        policy_loss_val = self.policy.update(weighted_policy_loss)
        # 3. Alpha update
        alpha_loss = self.alpha.update(log_probs)
        # 4. Soft update target critic
        if hasattr(self.critic, 'target_update_interval') and self.update_counter % self.critic.target_update_interval == 0:
            self.critic.soft_update()
        # 5. Обновление приоритетов (PER)
        if tree_indices is not None and hasattr(self.replay_buffer, 'update_priorities'):
            td = torch.nan_to_num(td_errors.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            self.replay_buffer.update_priorities(tree_indices, td, dynamic_alpha_fn=lambda _: self.alpha.alpha.item())
        # Формируем метрики
        metrics: Dict[str, Any] = {
            'loss/critic': critic_loss_val,
            'loss/policy': policy_loss_val,
            'loss/alpha': alpha_loss,
            'alpha': self.alpha.alpha.item(),
            'td_error_mean': td_errors.mean().item(),
            'td_error_std': td_errors.std().item(),
        }
        # Хук после обновления
        for plugin in self.plugins:
            plugin.after_update(metrics, batch)
        return metrics

    def _compute_critic_loss(self, batch: Dict[str, torch.Tensor], alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes SAC critic loss and TD errors using current and target networks.
        """
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        # Sample next actions and log_probs without gradient
        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.policy.sample(next_obs)
            # Get target Q values
            q1_next, q2_next, _, _ = self.critic.target_forward(next_obs, next_action)
            next_q_values = torch.min(q1_next, q2_next)
        # Compute critic targets
        target_q = compute_critic_target(rewards, next_q_values, next_log_prob, dones, self.gamma, alpha)
        # Current Q estimates
        q1, q2, _, _ = self.critic.forward(obs, actions)
        # Compute losses and td errors
        critic_loss, td_errors = compute_critic_loss(q1, q2, target_q)
        return critic_loss, td_errors

    def _compute_actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes SAC actor loss and returns log probabilities.
        """
        obs = batch['obs']
        # Sample new actions and get log_prob
        action, log_prob, _, _ = self.policy.sample(obs)
        # Evaluate critic for these actions
        q1, q2, _, _ = self.critic.forward(obs, action)
        q_values = torch.min(q1, q2)
        # Compute actor loss
        policy_loss = compute_actor_loss(log_prob, q_values, self.alpha.alpha)
        return policy_loss, log_prob

    def train(self, mode: bool = True) -> 'CoreSACAgent':
        """
        Переключает агента и все компоненты в режим train или eval.
        """
        self.training = mode
        self.policy.train(mode)
        self.critic.train(mode)
        # Alpha компонент тоже может иметь режим train/eval
        try:
            self.alpha.train(mode)
        except Exception:
            pass
        return self

    def select_action(self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool=False, evaluate: bool=False) -> torch.Tensor:
        """
        Выбирает действие как тензор на основе наблюдения, обрабатывая плагины.
        """
        # Попытка перехвата действия через плагины
        for plugin in self.plugins:
            alt = plugin.before_action(obs, deterministic)
            if alt is not None:
                action = alt
                break
        else:
            # Подготовка и нормализация входа
            obs_tensor = self._prepare_observation(obs)
            obs_norm = self.normalizer.normalize(obs_tensor)
            # Временный режим оценки
            prev_mode = getattr(self, 'training', True)
            if evaluate and prev_mode:
                self.policy.train(False)
            # Выбор действия и скрытого состояния
            with torch.no_grad(), autocast(device_type=self.device.type, enabled=self.enable_amp, dtype=self.amp_dtype):
                action, _, hidden, _ = self.policy.sample(obs_norm, self._policy_hidden, deterministic=deterministic)
            self._policy_hidden = hidden
            # Восстановление режима
            if evaluate and prev_mode:
                self.policy.train(prev_mode)
        # Постобработка через плагины
        for plugin in self.plugins:
            action = plugin.after_action(action, obs)
        return action

    def act(self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool=False) -> Union[np.ndarray, torch.Tensor]:
        """
        Выбирает действие и возвращает его в формате входного наблюдения.
        """
        is_np = isinstance(obs, np.ndarray)
        action = self.select_action(obs, deterministic=deterministic, evaluate=True)
        if is_np:
            return action.cpu().numpy()
        return action

    def update(self, batch_size: int) -> Optional[Dict[str, float]]:
        """
        Выполняет одно обновление параметров агента.
        """
        # Очистка кеша CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        # Проверка буфера
        if not self.replay_buffer.can_sample(batch_size):
            return None
        # Переключаем в режим обучения, если нужно
        if not getattr(self, 'training', True):
            self.train(True)
        # Сэмплируем батч
        batch = self._fetch_batch(batch_size)
        # Нормализация наблюдений
        batch['obs'] = self.normalizer.normalize(batch['obs'])
        batch['next_obs'] = self.normalizer.normalize(batch['next_obs'])
        # Обновляем компоненты и получаем метрики
        metrics = self._update_components(batch)
        # Увеличиваем счетчик
        self.update_counter += 1
        return metrics

    def perform_updates(self, num_updates: int, batch_size: int) -> Optional[Dict[str, float]]:
        """
        Выполняет несколько обновлений подряд и возвращает средние метрики.
        """
        metrics_sum: Dict[str, float] = {}
        valid = 0
        for _ in range(num_updates):
            m = self.update(batch_size)
            if m is not None:
                valid += 1
                for k, v in m.items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v
        if valid == 0:
            return None
        return {k: metrics_sum[k] / valid for k in metrics_sum}

    def export_to_onnx(self, obs_shape: Union[tuple, List[tuple]], export_dir: str, filename: str = "policy.onnx") -> Optional[str]:
        """
        Экспортирует политику в ONNX формат через export_model.
        """
        import os
        from core.utils.export_policy import export_model
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, filename)
        # Режим оценки
        self.train(False)
        # Экспорт
        return export_model(self.policy.policy_net, obs_shape, path, self._action_dim)

    def save(self, path: str) -> None:
        """
        Сохраняет состояние агента и компонентов в файл.
        """
        import os
        save_dir = os.path.dirname(path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        state = {
            'update_counter': self.update_counter,
            'obs_dim': self._obs_dim,
            'action_dim': self._action_dim,
            'gamma': self.gamma,
            'tau': self.tau,
            'policy': self.policy.get_state_dict(),
            'critic': self.critic.get_state_dict(),
            'alpha': self.alpha.get_state_dict(),
            'normalizer': self.normalizer.get_state_dict() if self.normalizer else None,
        }
        if hasattr(self, 'warm_starter') and self.warm_starter:
            state['warm_starter'] = self.warm_starter.get_state_dict()
        torch.save(state, path)

    def load(self, path: str, map_location: Optional[Union[str, torch.device]] = None) -> None:
        """
        Загружает состояние агента из файла.
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"State file not found: {path}")
        state = torch.load(path, map_location=map_location or self.device)
        self.update_counter = state.get('update_counter', self.update_counter)
        self.policy.load_state_dict(state['policy'])
        self.critic.load_state_dict(state['critic'])
        self.alpha.load_state_dict(state['alpha'])
        if state.get('normalizer') and self.normalizer:
            self.normalizer.load_state_dict(state['normalizer'])
        if state.get('warm_starter') and hasattr(self, 'warm_starter'):
            self.warm_starter.load_state_dict(state['warm_starter'])

    def eval(self) -> 'CoreSACAgent':
        """
        Переключает агента в режим оценки.
        """
        return self.train(False)

    def reset_hidden(self) -> None:
        """
        Сбрасывает скрытое состояние LSTM.
        """
        try:
            self.policy.reset_hidden()
            self.critic.reset_hidden()
        except AttributeError:
            pass

    def start_parallel(self, env: Any, batch_size: int, max_update_steps: Optional[int] = None) -> None:
        """
        Запускает параллельные потоки producer и consumer через ParallelAgentRunner.
        """
        self._parallel_runner = ParallelAgentRunner(self, env, batch_size, max_update_steps)
        self._parallel_runner.start()

    def stop_parallel(self) -> None:
        """
        Останавливает потоки producer и consumer через ParallelAgentRunner.
        """
        if hasattr(self, '_parallel_runner') and self._parallel_runner:
            self._parallel_runner.stop()

    def _clear_cuda_cache(self) -> None:
        """
        Очищает кеш CUDA, если используется GPU.
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _sanitize_tensor(
        self, tensor: torch.Tensor, name: str, clamp_range: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """
        Очищает тензор от NaN/Inf и ограничивает значение, если указан clamp_range.
        """
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
        if clamp_range is not None:
            tensor = torch.clamp(tensor, clamp_range[0], clamp_range[1])
        return tensor

    def _prepare_observation(
        self, obs: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Преобразует наблюдение в torch.Tensor на нужном устройстве.
        """
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device, dtype=torch.float32, non_blocking=True)
        return torch.tensor(obs, dtype=torch.float32, device=self.device)


class ParallelAgentRunner:
    def __init__(self, agent, env, batch_size, max_update_steps=None):
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self.max_update_steps = max_update_steps
        self._running = False
        self._prod = None
        self._cons = None

    def start(self):
        self._running = True
        import threading
        def _producer():
            obs = self.env.reset()
            while self._running:
                action = self.agent.act(obs)
                next_obs, reward, done, _ = self.env.step(action)
                self.agent.replay_buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs if not done else self.env.reset()
        def _consumer():
            steps = 0
            while self._running:
                if self.agent.replay_buffer.can_sample(self.batch_size):
                    self.agent.update(self.batch_size)
                    steps += 1
                    if self.max_update_steps and steps >= self.max_update_steps:
                        self._running = False
        self._prod = threading.Thread(target=_producer, daemon=True)
        self._cons = threading.Thread(target=_consumer, daemon=True)
        self._prod.start()
        self._cons.start()

    def stop(self):
        self._running = False
        try:
            if self._prod:
                self._prod.join()
        except Exception:
            pass
        try:
            if self._cons:
                self._cons.join()
        except Exception:
            pass