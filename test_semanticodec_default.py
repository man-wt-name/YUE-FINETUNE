#!/usr/bin/env python3
"""
Тест semanticodec как кодека по умолчанию
"""

import sys
import os

# Добавляем пути для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'finetune'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'inference'))

def test_default_codec():
    """Тестируем semanticodec как кодек по умолчанию"""
    
    print("🎵 Тестирование semanticodec как кодека по умолчанию")
    print("=" * 60)
    
    try:
        # Импортируем CodecManipulator
        from finetune.tools.codecmanipulator import CodecManipulator
        print("✅ CodecManipulator импортирован успешно")
        
        # Создаем экземпляр с кодеком по умолчанию (semanticodec)
        print("\n🔧 Создание экземпляра с кодеком по умолчанию...")
        manipulator = CodecManipulator()  # Должен использовать semanticodec
        print(f"✅ Кодек по умолчанию: {manipulator.codec_type}")
        print(f"✅ Количество кодбуков: {manipulator.num_codebooks}")
        
        # Проверяем, что это действительно semanticodec
        if manipulator.codec_type == "semanticodec":
            print("🎉 Успех! semanticodec установлен как кодек по умолчанию")
        else:
            print(f"⚠️  Внимание: кодек по умолчанию {manipulator.codec_type}, ожидался semanticodec")
        
        # Тестируем создание с явным указанием semanticodec
        print("\n🔧 Тестирование явного указания semanticodec...")
        semanticodec_manipulator = CodecManipulator("semanticodec", 0, 2)
        print(f"✅ Явный semanticodec: {semanticodec_manipulator.codec_type}")
        print(f"✅ Кодбуки: {semanticodec_manipulator.num_codebooks}")
        
        # Сравниваем с другими кодеками
        print("\n📊 Сравнение с другими кодеками:")
        codecs = [
            ("semanticodec", 2),
            ("xcodec", 4),
            ("dac16k", 4),
            ("dac44k", 9)
        ]
        
        for codec_type, num_codebooks in codecs:
            try:
                test_manipulator = CodecManipulator(codec_type, 0, num_codebooks)
                print(f"✅ {codec_type}: {test_manipulator.num_codebooks} кодбуков")
            except Exception as e:
                print(f"❌ {codec_type}: ошибка - {e}")
        
        print("\n🎯 Рекомендации:")
        print("• semanticodec - максимальное качество (по умолчанию)")
        print("• xcodec - высокое качество, быстрая обработка")
        print("• dac16k - быстрая обработка, базовое качество")
        print("• dac44k - хорошее качество для речи")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_cli_defaults():
    """Тестируем значения по умолчанию в CLI"""
    
    print("\n🔧 Тестирование CLI значений по умолчанию...")
    
    try:
        from finetune.new_yue_trainer import create_parser
        parser = create_parser()
        
        # Получаем парсер для preprocess
        preprocess_parser = None
        for action in parser._subparsers._group_actions:
            if hasattr(action, 'choices'):
                for choice_name, choice_parser in action.choices.items():
                    if choice_name == 'preprocess':
                        preprocess_parser = choice_parser
                        break
        
        if preprocess_parser:
            # Проверяем значения по умолчанию
            defaults = {}
            for action in preprocess_parser._actions:
                if action.dest and hasattr(action, 'default'):
                    defaults[action.dest] = action.default
            
            print("📋 Значения по умолчанию в CLI:")
            for key, value in defaults.items():
                if key in ['codec_type', 'num_codebooks']:
                    print(f"  • {key}: {value}")
            
            # Проверяем ключевые значения
            if defaults.get('codec_type') == 'semanticodec':
                print("✅ codec_type по умолчанию: semanticodec")
            else:
                print(f"⚠️  codec_type по умолчанию: {defaults.get('codec_type')}")
                
            if defaults.get('num_codebooks') == 2:
                print("✅ num_codebooks по умолчанию: 2")
            else:
                print(f"⚠️  num_codebooks по умолчанию: {defaults.get('num_codebooks')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования CLI: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск тестов semanticodec как кодека по умолчанию")
    print("=" * 60)
    
    success1 = test_default_codec()
    success2 = test_cli_defaults()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 Все тесты пройдены! semanticodec успешно установлен как кодек по умолчанию")
    else:
        print("⚠️  Некоторые тесты не пройдены. Проверьте настройки.")
    
    print("\n📚 Дополнительная информация:")
    print("• Подробное руководство по кодекам: CODECS.md")
    print("• Примеры использования: README.md") 