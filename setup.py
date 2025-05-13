from setuptools import setup, find_packages

# Базовые зависимости
install_requires = [
    'torch>=1.7.0',
    'numpy>=1.21.0',
    'mlflow>=1.23.0',
]

# Дополнительные зависимости для CPU
cpu_extras = [
    'numpy==1.23.1',
    'scikit-learn==0.24.2',
    'onnx==1.17.0',
    'gym==0.21.0',
    'pettingzoo==1.15.0',
    'protobuf==3.19.1',
    'matplotlib==3.9.4',
    'mlflow==1.23.0',
    'mlagents_envs @ git+https://github.com/Unity-Technologies/ml-agents.git@release_20#egg=mlagents_envs&subdirectory=ml-agents-envs',
    'torch==1.12.1',
    'torchvision==0.13.1',
    'torchaudio==0.12.1',
]

# Дополнительные зависимости для GPU
gpu_extras = [
    'numpy==1.21.1',
    'scikit-learn==0.24.2',
    'onnx==1.17.0',
    'gym==0.21.0',
    'pettingzoo==1.15.0',
    'protobuf==3.19.1',
    'matplotlib==3.4.3',
    'mlflow==1.23.0',
    'mlagents_envs @ git+https://github.com/Unity-Technologies/ml-agents.git@release_20#egg=mlagents_envs&subdirectory=ml-agents-envs',
    'torch==1.12.1+cu116',
    'torchvision==0.13.1+cu116',
    'torchaudio==0.12.1',
]

# Зависимости для документации
docs_extras = [
    'mkdocs',
    'mkdocs-material',
    'mkdocstrings-python',
    'pymdown-extensions',
    'callouts',
    'mkdocs-redirects',
    'mkdocs-drawio',
]

# Объединённые опциональные зависимости
extras_require = {
    'cpu': cpu_extras,
    'gpu': gpu_extras,
    'docs': docs_extras,
}
# всё сразу
extras_require['all'] = cpu_extras + gpu_extras + docs_extras

# Читаем long_description из README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="sac_azh",
    version="0.0.2",
    description="Уникальная реализация SAC с плагинами и CPU-буферами",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Azh",
    author_email="aaazhdaryan@miem.hse.ru",
    url="https://git.miem.hse.ru/1584/rl-race",
    packages=find_packages(where="core"),
    package_dir={'': 'core'},
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'sac-train=core.training.trainer:train_agent',
        ],
    },
) 