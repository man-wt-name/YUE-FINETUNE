# YuE - Инструкции по установке

## Системные требования

### Обязательные системные зависимости

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y build-essential make python3-dev
```

#### CentOS/RHEL:
```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3-devel
```

#### macOS:
```bash
xcode-select --install
```

### Python зависимости

1. **Создайте виртуальное окружение:**
```bash
python3 -m venv yue_env
source yue_env/bin/activate  # Linux/macOS
# или
yue_env\Scripts\activate  # Windows
```

2. **Установите зависимости:**
```bash
pip install -r requirements.txt
```

3. **Скомпилируйте C++ ускорители:**
```bash
cd finetune/core/datasets
make
cd ../../..
```

## Проверка установки

### Тест базовой функциональности:
```bash
python -c "import torch; import transformers; import librosa; print('Все основные библиотеки установлены')"
```

### Тест C++ ускорителей:
```bash
python -c "from finetune.core.datasets import helpers; print('C++ ускорители скомпилированы')"
```

### Полный тест импортов:
```bash
python test_imports.py
```

## Запуск проекта

### Использование обертки (рекомендуется):
```bash
python run_yue.py pipeline --help
```

### Прямой запуск:
```bash
cd finetune
python new_yue_trainer.py pipeline --help
```

## Настройка для GPU (опционально)

### CUDA 11.8:
```bash
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1:
```bash
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Устранение проблем

### Ошибка "ModuleNotFoundError: No module named 'finetune'"

Эта ошибка возникает из-за неправильных путей импорта. Решения:

1. **Используйте обертку (рекомендуется):**
```bash
python run_yue.py pipeline --help
```

2. **Установите переменную PYTHONPATH:**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python finetune/new_yue_trainer.py pipeline --help
```

3. **Запустите из корневой директории:**
```bash
cd YuE-main
python finetune/new_yue_trainer.py pipeline --help
```

### Ошибка компиляции C++:
- Убедитесь, что установлен `build-essential` (Linux) или Xcode (macOS)
- Проверьте версию pybind11: `pip install --upgrade pybind11`

### Ошибки с аудио библиотеками:
```bash
# Ubuntu/Debian
sudo apt install -y libsndfile1-dev

# macOS
brew install libsndfile
```

### Проблемы с NLTK:
```python
import nltk
nltk.download('punkt')
```

## Структура проекта после установки

```
YuE-main/
├── finetune/
│   ├── core/datasets/helpers.*.so  # Скомпилированные ускорители
│   └── ...
├── requirements.txt
├── INSTALL.md
├── test_imports.py                  # Тест импортов
├── run_yue.py                      # Обертка для запуска
└── ...
```

## Быстрый старт

После установки можно запустить тестовый пайплайн:

```bash
cd YuE-main
python run_yue.py pipeline --help
```

## Примеры использования

### Конвертация аудио:
```bash
python run_yue.py convert --input-dir /path/to/audio --output-dir /path/to/output
```

### Предобработка данных:
```bash
python run_yue.py preprocess --data-dir /path/to/npy --output-dir /path/to/output
```

### Обучение модели:
```bash
python run_yue.py train --model /path/to/model --data-dir /path/to/data --output-dir /path/to/output
```

### Полный пайплайн:
```bash
python run_yue.py pipeline --input-dir /path/to/audio --model /path/to/model --output-dir /path/to/output
``` 