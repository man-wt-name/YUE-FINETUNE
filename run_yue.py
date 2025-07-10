#!/usr/bin/env python3
"""
Обертка для запуска YuE trainer с правильными путями
"""

import sys
import os
import subprocess

def main():
    """Запускает YuE trainer с правильными путями."""
    
    # Получаем путь к текущему скрипту
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Путь к new_yue_trainer.py
    trainer_path = os.path.join(script_dir, "finetune", "new_yue_trainer.py")
    
    # Проверяем, что файл существует
    if not os.path.exists(trainer_path):
        print(f"❌ Файл {trainer_path} не найден!")
        sys.exit(1)
    
    # Добавляем текущую директорию в PYTHONPATH
    env = os.environ.copy()
    
    # Добавляем пути к finetune и его подмодулям
    finetune_path = os.path.join(script_dir, "finetune")
    tools_path = os.path.join(finetune_path, "tools")
    pipeline_path = os.path.join(finetune_path, "pipeline")
    
    pythonpath_parts = [script_dir, finetune_path, tools_path, pipeline_path]
    
    if 'PYTHONPATH' in env:
        pythonpath_parts.append(env['PYTHONPATH'])
    
    env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
    
    # Запускаем trainer с переданными аргументами
    try:
        result = subprocess.run([sys.executable, trainer_path] + sys.argv[1:], 
                              env=env, 
                              cwd=script_dir)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 