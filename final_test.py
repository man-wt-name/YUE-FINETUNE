#!/usr/bin/env python3
"""
Финальный тест для проверки всех исправлений импортов в проекте YuE
"""

import sys
import os
import subprocess
import importlib.util

def run_command(cmd, description):
    """Запускает команду и возвращает результат."""
    print(f"\n🔧 {description}")
    print(f"Команда: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Успешно")
            return True
        else:
            print(f"❌ Ошибка: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Таймаут выполнения")
        return False
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return False

def test_import_module(module_path, module_name):
    """Тестирует импорт модуля."""
    print(f"\n🔧 Тест импорта {module_name}")
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            print(f"❌ Не удалось создать spec для {module_name}")
            return False
        
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            print(f"❌ Loader не найден для {module_name}")
            return False
            
        spec.loader.exec_module(module)
        print(f"✅ {module_name} импортируется успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка импорта {module_name}: {e}")
        return False

def main():
    """Основная функция тестирования."""
    print("🚀 Финальный тест проекта YuE")
    print("=" * 50)
    
    # Получаем корневую директорию проекта
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Список тестов
    tests = []
    
    # Тест 1: Проверка структуры проекта
    tests.append(("python check_project_structure.py", "Проверка структуры проекта"))
    
    # Тест 2: Тест импортов
    tests.append(("python test_imports.py", "Тест основных импортов"))
    
    # Тест 3: Тест импорта CodecManipulator
    tests.append(("python test_codec_import.py", "Тест импорта CodecManipulator"))
    
    # Тест 4: Проверка CLI
    tests.append(("python run_yue.py pipeline --help", "Проверка CLI интерфейса"))
    
    # Тест 5: Проверка отдельных модулей
    print("\n🔧 Тест импорта отдельных модулей:")
    
    # Пути к модулям
    modules_to_test = [
        (os.path.join(project_root, "finetune", "tools", "codecmanipulator.py"), "codecmanipulator"),
        (os.path.join(project_root, "finetune", "pipeline", "new_steps.py"), "new_steps"),
        (os.path.join(project_root, "finetune", "pipeline", "new_config.py"), "new_config"),
        (os.path.join(project_root, "finetune", "new_yue_trainer.py"), "new_yue_trainer")
    ]
    
    # Добавляем пути в sys.path для корректного импорта
    finetune_path = os.path.join(project_root, "finetune")
    tools_path = os.path.join(finetune_path, "tools")
    pipeline_path = os.path.join(finetune_path, "pipeline")
    
    for path in [project_root, finetune_path, tools_path, pipeline_path]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Тестируем импорт модулей
    for module_path, module_name in modules_to_test:
        if os.path.exists(module_path):
            test_import_module(module_path, module_name)
        else:
            print(f"❌ Файл не найден: {module_path}")
    
    # Запускаем команды
    print("\n🔧 Запуск тестовых команд:")
    passed_tests = 0
    total_tests = len(tests)
    
    for cmd, description in tests:
        if run_command(cmd, description):
            passed_tests += 1
    
    # Итоговый результат
    print("\n" + "=" * 50)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 50)
    print(f"Пройдено тестов: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("\n✅ Проект YuE готов к использованию!")
        print("\n📋 Следующие шаги:")
        print("1. Подготовьте аудио файлы для обучения")
        print("2. Выберите базовую модель (например, microsoft/DialoGPT-medium)")
        print("3. Запустите полный пайплайн:")
        print("   python run_yue.py pipeline --input-dir /path/to/audio --model /path/to/model --output-dir ./output")
        return True
    else:
        print(f"⚠️  ПРОЙДЕНО {passed_tests} ИЗ {total_tests} ТЕСТОВ")
        print("\n🔧 Для решения проблем:")
        print("1. Проверьте установку зависимостей: pip install -r requirements.txt")
        print("2. Убедитесь, что все файлы проекта на месте")
        print("3. Проверьте права доступа к файлам")
        print("4. Обратитесь к INSTALL.md для подробных инструкций")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 