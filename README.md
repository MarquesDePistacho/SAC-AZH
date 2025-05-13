# 🧠 THE DEVELOPMENT OF AN ADAPTIVE SELF-MANAGEMENT SYSTEM FOR A COMPETITIVE AGENT BASED ON REINFORCEMENT LEARNING

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLflow](https://img.shields.io/badge/MLflow-1.23.0-cyan)](https://mlflow.org/)

**Проект реализует адаптивную систему управления на базе алгоритма SAC (Soft Actor-Critic) с поддержкой:**
- 🚀 Плагинов для расширения функциональности
- 🧮 Специализированных CPU-буферов
- 🔄 Интеграции с Unity ML-Agents
- 📊 Визуализации обучения через MLflow

## 📖 Теоретические основы
**Soft Actor-Critic (SAC)** - алгоритм обучения с подкреплением, сочетающий:
- **Энтропийную регуляризацию** для баланса исследования/эксплуатации
- **Q-функцию с двойными сетями** для стабильности обучения
- **Политику стохастических действий** для адаптивности

## 🛠 Установка
### Вариант 1: Conda
```bash
conda env create -f env-cpu.yml  # или env-gpu.yml для версии с CUDA
conda activate rl-race-[cpu/gpu]
```
### Вариант 2: Pip
```bash
# Базовая установка
pip install sac_azh

# Дополнительные режимы:
pip install sac_azh[cpu]    # Версия с поддержкой CPU
pip install sac_azh[gpu]    # Версия с поддержкой CUDA
pip install sac_azh[all]    # Все зависимости
```

## 🏗 Архитектура системы
![alt text](/docs/images/core.png)

## 📈 Визуализация прогресса
```bash
mlflow ui  # Открыть http://localhost:5000
```

## 🧑💻 Участники 

<div align="center">

### 🌟 Команда разработчиков

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/MarquesDePistacho">
        👨💻<br/>
        <b>Аждарьян А.А.</b>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/sofya-lyapunova">
        👩💻<br/>
        <b>Ляпунова С.А.</b>
      </a>
    </td>
  </tr>
</table>
</div>

## 🤝 Руководство по Contributing
### 🛠 Работа с кодом
- Для новых функций используйте ветки: feat/feature-name
- Исправления ошибок: fix/bug-description
- Форматируйте код через black и isort
### 🗂 Управление билдами Unity
- Сохраняйте билды в builds/{OS}/
- Используйте сжатие ZIP для экономии пространства
- Для LFS-файлов: git lfs track "*.onnx"

## 📚 Полезные ресурсы
- [Netron](https://netron.app/): Визуализация нейросетевых моделей
- [MLflow Docs](https://mlflow.org/docs/latest/index.html): Документация по трекингу экспериментов
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents): Официальный репозиторий ML-Agents

## Лицензия
Этот проект распространяется под лицензией [MIT](https://opensource.org/licenses/MIT)