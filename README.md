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
└── ...
```

## Быстрый старт

После установки можно запустить тестовый пайплайн:

```bash
cd YuE-main
python finetune/new_yue_trainer.py pipeline --help
``` 
