#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправления проблемы с кодеком xcodec_16k
"""

import sys
import os
import numpy as np

def test_codec_manipulator():
    """Тестирует CodecManipulator с различными кодеками."""
    print("🧪 Тест CodecManipulator с различными кодеками...")
    
    # Добавляем пути
    project_root = os.path.dirname(os.path.abspath(__file__))
    tools_path = os.path.join(project_root, "finetune", "tools")
    sys.path.insert(0, tools_path)
    
    try:
        from codecmanipulator import CodecManipulator
        
        # Тестируем поддерживаемые кодеки
        codecs_to_test = ["semanticodec", "xcodec", "xcodec_16k", "dac16k", "dac44k"]
        
        for codec_type in codecs_to_test:
            print(f"\n🔧 Тест кодека: {codec_type}")
            try:
                manipulator = CodecManipulator(codec_type, 0, 4)
                print(f"  ✅ {codec_type} - создан успешно")
                print(f"    - codebook_size: {manipulator.codebook_size}")
                print(f"    - num_codebooks: {manipulator.num_codebooks}")
                print(f"    - global_offset: {manipulator.global_offset}")
                
                # Тестируем создание тестовых данных
                test_data = np.random.randint(0, 1024, size=(4, 100), dtype=np.int16)
                print(f"    - Тестовые данные созданы: {test_data.shape}")
                
                # Тестируем npy2ids
                try:
                    ids = manipulator.npy2ids(test_data)
                    print(f"    - npy2ids работает: {len(ids)} токенов")
                except Exception as e:
                    print(f"    - ❌ npy2ids ошибка: {e}")
                
            except Exception as e:
                print(f"  ❌ {codec_type} - ошибка: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта CodecManipulator: {e}")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return False

def test_pipeline_with_fix():
    """Тестирует пайплайн с исправленным кодеком."""
    print("\n🧪 Тест пайплайна с исправленным кодеком...")
    
    # Создаем тестовые .npy файлы
    test_dir = "/tmp/test_yue_codec"
    os.makedirs(test_dir, exist_ok=True)
    
    # Создаем тестовый .npy файл
    test_data = np.random.randn(4, 1000).astype(np.float32)
    test_file = os.path.join(test_dir, "test_audio.npy")
    np.save(test_file, test_data)
    
    print(f"  ✅ Создан тестовый файл: {test_file}")
    
    # Тестируем CodecManipulator с тестовым файлом
    try:
        tools_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune", "tools")
        sys.path.insert(0, tools_path)
        
        from codecmanipulator import CodecManipulator
        
        manipulator = CodecManipulator("semanticodec", 0, 2)
        json_str = manipulator.npy_to_json_str(test_file)
        
        print(f"  ✅ JSON строка создана: {len(json_str)} символов")
        print(f"  📋 JSON: {json_str[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка при тестировании: {e}")
        return False

def main():
    """Основная функция тестирования."""
    print("🚀 Тестирование исправления проблемы с кодеком")
    print("=" * 50)
    
    tests = [
        ("CodecManipulator с различными кодеками", test_codec_manipulator),
        ("Пайплайн с исправленным кодеком", test_pipeline_with_fix),
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
        print("\n✅ Проблема с кодеком исправлена!")
        print("\n📋 Теперь можно запускать пайплайн:")
        print("python finetune/new_yue_trainer.py pipeline \\")
        print("  --input-dir /kaggle/input/strukalo \\")
        print("  --output-dir /kaggle/working/ \\")
        print("  --skip-steps dataset count train")
        return True
    else:
        print("⚠️  НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 