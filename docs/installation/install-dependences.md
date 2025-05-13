# Настройка окружения и запуск проекта

## Установка Python 3.9

Рекомендуется использовать один из следующих методов для установки Python 3.9.

---

### Вариант 1: Установка через Anaconda

1. Создание нового окружения с Python 3.9:

```bash
conda create -n py39 python=3.9
```

2. Активация окружения:

```bash
conda activate py39
```

3. Установка Jupyter и ipykernel (если используется Jupyter):

```bash
conda install jupyter ipykernel
python -m ipykernel install --user --name=py39 --display-name "Python 3.9"
```

---

### Вариант 2: Установка через pyenv

1. Установка pyenv:

```bash
# macOS
brew install pyenv

# Linux (Ubuntu)
curl https://pyenv.run | bash
```

2. Добавление в конфигурацию shell:

Для bash:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

Для zsh:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

3. Применение настроек:

```bash
source ~/.bashrc  # или source ~/.zshrc
```

4. Установка Python 3.9:

```bash
pyenv install 3.9.18
pyenv local 3.9.18
```

5. Создание и активация виртуального окружения:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## Установка зависимостей проекта

В зависимости от конфигурации выбирается один из файлов зависимостей.

- Для CPU:

```bash
conda env update -f env-cpu.yml
```

- Для GPU:

```bash
conda env update -f env-gpu.yml
```

Затем окружение активируется:

```bash
conda activate rl-race
```

---

## Установка виртуального дисплея (если требуется)

- macOS:

```bash
brew install xvfb
```

- Linux:

```bash
sudo apt install xvfb
```

---

## Unity-сборки

Сборки среды находятся в папке `builds/` и распределены по платформам:

```plaintext
builds/
├── Linux/
├── MacOS/
└── Windows/
```

Для запуска среды используется путь к соответствующей сборке, передаваемый в `core.envs.env`.
