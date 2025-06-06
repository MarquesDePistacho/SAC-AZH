# Документация пользователя

Данная документация предоставляет руководство пользователя для работы с модулем обучения агентов на основе обучения с подкреплением. В этом модуле представлены инструменты для создания, обучения и оценки агентов в рамках многозадачного обучения с подкреплением, а также включает возможности для экспорта моделей в формат ONNX для дальнейшего использования или инференса.

Проект состоит из нескольких ключевых компонентов, включая настройку окружения, функции для обучения и оценки агентов, а также классы для работы с моделями и экспорта данных. В этой документации описано, как установить зависимости, необходимы для работы модуля, а также как запустить обучение агента на пользовательском устройстве.

**Структура проекта**:

```plaintext
rl-race/ 
├── builds/             # Сборки Unity-среды для Linux, MacOS, Windows
├── core/               # Реализация агентов, обучения, логирования и среды
├── notebooks/           # Jupyter-ноутбуки для запуска обучения и оценки
├── docs/               # Пользовательская документация
├── env-cpu.yml         # Зависимости для запуска без GPU
├── env-gpu.yml         # Зависимости для запуска с GPU
└── README.md           # README для проекта
```
