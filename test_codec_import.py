#!/usr/bin/env python3
"""
Тестовый скрипт для проверки импорта CodecManipulator
"""

import sys
import os
import librosa
import numpy as np

def test_codec_import():
    """Тестирует импорт CodecManipulator."""
    print("Тестирование импорта CodecManipulator...")
    
    # Добавляем путь к проекту в sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    try:
        # Тест 1: Прямой импорт из tools
        print("✓ Тест 1: Прямой импорт из tools...")
        tools_path = os.path.join(project_root, "finetune", "tools")
        sys.path.insert(0, tools_path)
        
        from codecmanipulator import CodecManipulator
        print("  - CodecManipulator успешно импортирован")
        
        # Тест 2: Создание экземпляра
        print("✓ Тест 2: Создание экземпляра...")
        manipulator = CodecManipulator("semanticodec", 0, 2)
        print("  - Экземпляр CodecManipulator успешно создан")
        
        # Тест 3: Импорт из pipeline
        print("✓ Тест 3: Импорт из pipeline...")
        pipeline_path = os.path.join(project_root, "finetune", "pipeline")
        sys.path.insert(0, pipeline_path)
        
        # Добавляем путь к tools для pipeline
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        
        from new_steps import DataPreprocessor
        print("  - DataPreprocessor успешно импортирован")
        
        print("\n✅ Все тесты импорта прошли успешно!")
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return False

if __name__ == "__main__":
    success = test_codec_import()
    sys.exit(0 if success else 1) 