import threading
import time
from typing import Any

class ParallelRunner:
    """
    Плагин для разделения работы агента на producer и consumer потоки: сбор данных и обновление.
    """
    def __init__(self, agent, env, batch_size, max_update_steps=None):
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self.max_update_steps = max_update_steps
        self.stop_event = threading.Event()
        self.producer_thread = None
        self.consumer_thread = None
        self.steps = 0

    def start(self):
        """Запускает потоки producer и consumer"""
        self.stop_event.clear()
        self.producer_thread = threading.Thread(target=self._producer, daemon=True)
        self.consumer_thread = threading.Thread(target=self._consumer, daemon=True)
        self.producer_thread.start()
        self.consumer_thread.start()

    def stop(self):
        """Останавливает потоки"""
        self.stop_event.set()
        if self.producer_thread:
            self.producer_thread.join()
        if self.consumer_thread:
            self.consumer_thread.join()

    def _producer(self):
        """Собирает переходы и добавляет в буфер"""
        obs = self.env.reset()
        while not self.stop_event.is_set():
            action = self.agent.act(obs)
            next_obs, reward, done, info = self.env.step(action)
            self.agent.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs if not done else self.env.reset()
            time.sleep(0)  

    def _consumer(self):
        """Выполняет обновления агента"""
        while not self.stop_event.is_set():
            if self.agent.replay_buffer.can_sample(self.batch_size):
                self.agent.update(self.batch_size)
                self.steps += 1
                if self.max_update_steps and self.steps >= self.max_update_steps:
                    break
            time.sleep(0)  