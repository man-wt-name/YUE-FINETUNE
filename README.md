# YuE - Multimodal Language Model Finetuning Framework

YuE - это фреймворк для файнтюнинга мультимодальных языковых моделей с поддержкой аудио. Проект основан на PyTorch и Transformers, с оптимизированной системой обработки данных и поддержкой LoRA для эффективного обучения.

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Установите системные зависимости
sudo apt update && sudo apt install -y build-essential make python3-dev

# Создайте виртуальное окружение
python3 -m venv yue_env
source yue_env/bin/activate

# Установите Python зависимости
pip install -r requirements.txt

# Скомпилируйте C++ ускорители
cd finetune/core/datasets && make && cd ../../..
```

### 2. Проверка установки

```bash
# Проверка структуры проекта
python check_project_structure.py

# Тест импортов
python test_imports.py

# Тест импорта CodecManipulator
python test_codec_import.py

# Проверка CLI
python run_yue.py pipeline --help
```

### 3. Запуск пайплайна

```bash
# Полный пайплайн: от аудио до обученной модели
python run_yue.py pipeline \
    --input-dir /path/to/audio/files \
    --model /path/to/base/model \
    --output-dir /path/to/output \
    --epochs 3 \
    --batch-size 4

# По умолчанию используется semanticodec для максимального качества
# Для других кодеков используйте --codec-type:
# --codec-type xcodec     # Высокое качество, быстрая обработка
# --codec-type dac16k     # Быстрая обработка, базовое качество
# --codec-type semanticodec # Максимальное качество (по умолчанию)
```

## 📁 Структура проекта

```
YuE-main/
├── finetune/                    # Основной модуль файнтюнинга
│   ├── core/                   # Ядро системы
│   │   ├── arguments.py        # Парсинг аргументов
│   │   ├── datasets/          # Система датасетов
│   │   ├── tokenizer/         # Токенизаторы
│   │   └── preprocess_*.py   # Предобработка данных
│   ├── pipeline/              # Пайплайн обработки
│   ├── scripts/               # Скрипты обучения
│   └── new_yue_trainer.py    # Главный CLI интерфейс
├── inference/                 # Модуль инференса
├── requirements.txt           # Python зависимости
├── INSTALL.md                # Подробные инструкции установки
├── test_imports.py           # Тест импортов
└── run_yue.py               # Обертка для запуска
```

## 🔧 Основные возможности

### Аудио обработка
- Конвертация аудио файлов в numpy массивы
- Поддержка различных форматов (wav, mp3, flac)
- Нормализация и сегментация аудио
- Детекция вокала/инструментала
- **semanticodec** по умолчанию для максимального качества аудио

### Предобработка данных
- Интеграция с XCodec и другими кодеками
- Создание JSONL и mmap файлов
- Эффективная система кэширования
- Поддержка многопроцессорной обработки

### Обучение моделей
- LoRA для параметр-эффективного обучения
- Поддержка DeepSpeed для распределенного обучения
- Интеграция с Hugging Face Transformers
- Мониторинг через Weights & Biases

### Датасеты
- Оптимизированные MMap датасеты
- Поддержка смешивания нескольких датасетов
- C++ ускорители для быстрой обработки
- Автоматическое кэширование индексов

## 📖 Примеры использования

### Конвертация аудио
```bash
python run_yue.py convert \
    --input-dir /path/to/audio \
    --output-dir /path/to/npy \
    --sr 16000 \
    --mono \
    --normalize
```

### Предобработка данных
```bash
python run_yue.py preprocess \
    --data-dir /path/to/npy \
    --output-dir /path/to/processed \
    --codec-type semanticodec \
    --workers 4
```

### Обучение модели
```bash
python run_yue.py train \
    --model /path/to/model \
    --data-dir /path/to/data \
    --output-dir /path/to/output \
    --epochs 3 \
    --batch-size 4 \
    --lora-r 8 \
    --lora-alpha 16
```

## 🛠️ Устранение проблем

### Ошибка импорта модулей
```bash
# Используйте обертку
python run_yue.py pipeline --help

# Или установите PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
python finetune/new_yue_trainer.py pipeline --help
```

### Проблемы с C++ компиляцией
```bash
# Убедитесь, что установлены системные зависимости
sudo apt install -y build-essential make python3-dev

# Переустановите pybind11
pip install --upgrade pybind11
```

### Ошибки с аудио библиотеками
```bash
# Ubuntu/Debian
sudo apt install -y libsndfile1-dev

# macOS
brew install libsndfile
```

## 📚 Документация

- [Подробные инструкции установки](INSTALL.md)
- [Список зависимостей](requirements.txt)
- [Конфигурация DeepSpeed](finetune/config/ds_config_zero2.json)

## 🤝 Поддержка

При возникновении проблем:

1. Проверьте [INSTALL.md](INSTALL.md) для решения типичных проблем
2. Запустите `python test_imports.py` для диагностики
3. Убедитесь, что все системные зависимости установлены

## 📄 Лицензия

Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved. 