#!/usr/bin/env python3
"""
Тестовый скрипт для проверки импортов в проекте YuE
"""

import sys
import os

# Добавляем путь к проекту в sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Тестирует все основные импорты проекта."""
    print("Тестирование импортов...")
    
    try:
        # Тест основных библиотек
        print("✓ Импорт основных библиотек...")
        import torch
        import transformers
        import numpy as np
        import librosa
        import soundfile
        print("  - torch:", torch.__version__)
        print("  - transformers:", transformers.__version__)
        print("  - numpy:", np.__version__)
        
        # Тест модулей проекта
        print("\n✓ Импорт модулей проекта...")
        
        # Тест core модулей
        from finetune.core.arguments import parse_args
        print("  - core.arguments: OK")
        
        from finetune.core.datasets.indexed_dataset import MMapIndexedDataset
        print("  - core.datasets.indexed_dataset: OK")
        
        from finetune.core.datasets.gpt_dataset import GPTDataset
        print("  - core.datasets.gpt_dataset: OK")
        
        # Тест pipeline модулей
        from finetune.pipeline.new_config import create_pipeline_config
        print("  - pipeline.new_config: OK")
        
        from finetune.pipeline.new_steps import AudioConverter, DataPreprocessor
        print("  - pipeline.new_steps: OK")
        
        # Тест tokenizer
        from finetune.core.tokenizer.mmtokenizer import _MMSentencePieceTokenizer
        print("  - core.tokenizer.mmtokenizer: OK")
        
        print("\n✅ Все импорты успешны!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        return False

def test_cpp_helpers():
    """Тестирует компиляцию C++ ускорителей."""
    print("\nТестирование C++ ускорителей...")
    
    try:
        from finetune.core.datasets.utils import compile_helpers
        compile_helpers()
        print("✅ C++ ускорители скомпилированы успешно!")
        return True
    except Exception as e:
        print(f"❌ Ошибка компиляции C++ ускорителей: {e}")
        print("  Убедитесь, что установлены build-essential и python3-dev")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("YuE - Тест импортов")
    print("=" * 50)
    
    imports_ok = test_imports()
    cpp_ok = test_cpp_helpers()
    
    print("\n" + "=" * 50)
    if imports_ok and cpp_ok:
        print("🎉 Все тесты пройдены! Проект готов к использованию.")
    else:
        print("⚠️  Некоторые тесты не пройдены. Проверьте установку зависимостей.")
    print("=" * 50) 