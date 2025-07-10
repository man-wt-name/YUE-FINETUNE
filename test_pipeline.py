#!/usr/bin/env python3
"""
Тестовый скрипт для проверки команды pipeline с пропуском шагов
"""

import subprocess
import sys
import os

def test_pipeline_convert_only():
    """Тестирует команду pipeline только с конвертацией аудио."""
    print("🧪 Тест команды pipeline только с конвертацией...")
    
    # Команда для тестирования
    cmd = [
        "python", "finetune/new_yue_trainer.py", "pipeline",
        "--input-dir", "/kaggle/input/strukalo",
        "--output-dir", "/kaggle/working/",
        "--skip-steps", "preprocess", "dataset", "count", "train"
    ]
    
    print(f"Выполняется команда: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Команда выполнена успешно!")
            print("📋 Вывод:")
            print(result.stdout)
        else:
            print("❌ Команда завершилась с ошибкой!")
            print("📋 Ошибка:")
            print(result.stderr)
            print("📋 Вывод:")
            print(result.stdout)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Команда превысила лимит времени (5 минут)")
        return False
    except Exception as e:
        print(f"💥 Неожиданная ошибка: {e}")
        return False

def test_pipeline_help():
    """Тестирует справку команды pipeline."""
    print("🧪 Тест справки команды pipeline...")
    
    cmd = ["python", "finetune/new_yue_trainer.py", "pipeline", "--help"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Справка отображается корректно!")
            print("📋 Справка:")
            print(result.stdout)
        else:
            print("❌ Ошибка при отображении справки!")
            print(result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"💥 Неожиданная ошибка: {e}")
        return False

def main():
    """Основная функция тестирования."""
    print("🚀 Тестирование исправленной команды pipeline")
    print("=" * 50)
    
    tests = [
        ("Справка команды", test_pipeline_help),
        ("Только конвертация", test_pipeline_convert_only),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔧 Тест: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} - ПРОЙДЕН")
        else:
            print(f"❌ {test_name} - ПРОВАЛЕН")
    
    print("\n" + "=" * 50)
    print(f"📊 РЕЗУЛЬТАТЫ: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("\n✅ Команда pipeline теперь работает корректно!")
        print("\n📋 Пример использования:")
        print("python finetune/new_yue_trainer.py pipeline \\")
        print("  --input-dir /path/to/audio \\")
        print("  --output-dir ./output \\")
        print("  --skip-steps preprocess dataset count train")
        return True
    else:
        print("⚠️  НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 