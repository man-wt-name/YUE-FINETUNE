#!/usr/bin/env python3
"""
Скрипт для проверки структуры проекта YuE и исправления путей импорта
"""

import os
import sys
import importlib.util

def check_file_exists(filepath):
    """Проверяет существование файла."""
    return os.path.exists(filepath)

def check_import(module_name, filepath):
    """Проверяет возможность импорта модуля."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True
    except Exception as e:
        print(f"  ❌ Ошибка импорта {module_name}: {e}")
        return False

def main():
    """Основная функция проверки структуры проекта."""
    print("🔍 Проверка структуры проекта YuE...")
    
    # Получаем корневую директорию проекта
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Список критических файлов и директорий
    critical_paths = [
        "finetune/__init__.py",
        "finetune/core/__init__.py",
        "finetune/core/datasets/__init__.py",
        "finetune/core/tokenizer/__init__.py",
        "finetune/pipeline/__init__.py",
        "finetune/tools/__init__.py",
        "finetune/scripts/__init__.py",
        "finetune/new_yue_trainer.py",
        "finetune/pipeline/new_config.py",
        "finetune/pipeline/new_steps.py",
        "finetune/tools/codecmanipulator.py",
        "requirements.txt"
    ]
    
    print("\n📁 Проверка критических файлов:")
    missing_files = []
    
    for path in critical_paths:
        full_path = os.path.join(project_root, path)
        if check_file_exists(full_path):
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path} - НЕ НАЙДЕН")
            missing_files.append(path)
    
    if missing_files:
        print(f"\n⚠️  Отсутствуют критические файлы: {len(missing_files)}")
        for file in missing_files:
            print(f"    - {file}")
        return False
    
    print("\n✅ Все критические файлы найдены!")
    
    # Проверка импортов
    print("\n🔧 Проверка импортов:")
    
    # Тест 1: CodecManipulator
    codec_path = os.path.join(project_root, "finetune", "tools", "codecmanipulator.py")
    if check_import("codecmanipulator", codec_path):
        print("  ✅ CodecManipulator импортируется корректно")
    else:
        print("  ❌ Проблема с импортом CodecManipulator")
        return False
    
    # Тест 2: new_steps
    steps_path = os.path.join(project_root, "finetune", "pipeline", "new_steps.py")
    if check_import("new_steps", steps_path):
        print("  ✅ new_steps импортируется корректно")
    else:
        print("  ❌ Проблема с импортом new_steps")
        return False
    
    # Тест 3: new_config
    config_path = os.path.join(project_root, "finetune", "pipeline", "new_config.py")
    if check_import("new_config", config_path):
        print("  ✅ new_config импортируется корректно")
    else:
        print("  ❌ Проблема с импортом new_config")
        return False
    
    # Проверка PYTHONPATH
    print("\n🔧 Проверка PYTHONPATH:")
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    print(f"  Текущий PYTHONPATH: {current_pythonpath}")
    
    # Рекомендуемые пути для PYTHONPATH
    recommended_paths = [
        project_root,
        os.path.join(project_root, "finetune"),
        os.path.join(project_root, "finetune", "tools"),
        os.path.join(project_root, "finetune", "pipeline")
    ]
    
    print("\n📋 Рекомендуемые пути для PYTHONPATH:")
    for path in recommended_paths:
        if os.path.exists(path):
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path} - НЕ СУЩЕСТВУЕТ")
    
    print("\n✅ Проверка структуры проекта завершена успешно!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 