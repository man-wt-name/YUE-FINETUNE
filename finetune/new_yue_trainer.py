#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import logging
import json
import time
import tempfile

# --- ИСПРАВЛЕНИЕ: Импортируем из исправленных модулей ---
from finetune.pipeline.new_config import create_pipeline_config
from finetune.pipeline.new_steps import (
    AudioConverter,
    DataPreprocessor,
    DatasetPreparer,
    TokenCounter,
    Trainer,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('YuE-CLI')

class YuETrainer:
    """Основной класс для управления процессом файнтюнинга моделей YuE.
    
    Этот класс предоставляет интерфейс командной строки для выполнения различных
    шагов процесса файнтюнинга, включая конвертацию аудио, предобработку данных,
    подготовку датасета, подсчет токенов и обучение модели. Также поддерживается
    выполнение полного пайплайна обработки и обучения.
    
    Attributes:
        project_root (str): Корневая директория проекта.
        output_dir (str): Директория для сохранения результатов.
        parser (argparse.ArgumentParser): Парсер аргументов командной строки.
        cache_dir (str): Директория для кэширования промежуточных результатов.
    """
    
    def __init__(self):
        """Инициализация тренера YuE."""
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = None
        self.parser = self.setup_parser() # Сохраняем парсер для доступа к defaults
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".yue_cache")
        
    def setup_parser(self):
        """Настройка парсера аргументов командной строки с расширенными опциями.
        
        Создает и настраивает парсер аргументов командной строки с подкомандами
        для различных шагов процесса файнтюнинга.
        
        Returns:
            argparse.ArgumentParser: Настроенный парсер аргументов.
        """
        parser = argparse.ArgumentParser(
            description='YuE Trainer - утилита для файнтюнинга моделей YuE',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Глобальные параметры
        parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Уровень логирования')

        subparsers = parser.add_subparsers(dest='command', help='Команды', required=True)
        
        # --------------------------------------------------
        # Конвертер аудио
        # --------------------------------------------------
        convert_parser = subparsers.add_parser('convert', help='Конвертация аудио файлов')
        convert_parser.add_argument('--input-dir', type=str, required=True, help='Директория с исходными аудио файлами')
        convert_parser.add_argument('--output-dir', type=str, help='Куда сохранить *.npy')
        convert_parser.add_argument('--sr', type=int, default=16000, help='Частота дискретизации в Гц')
        convert_parser.add_argument('--mono', action='store_true', default=True, help='Преобразовать в моно')
        convert_parser.add_argument('--normalize', action='store_true', default=True, help='Нормализовать амплитуду')
        convert_parser.add_argument('--max-amplitude', type=float, default=0.95, help='Максимальная амплитуда после нормализации')
        convert_parser.add_argument('--trim-silence', action='store_true', help='Удалять начальную/конечную тишину')
        convert_parser.add_argument('--segment-length', type=float, help='Длина сегмента (сек)')
        convert_parser.add_argument('--min-segment-length', type=float, default=1.0, help='Минимальная длина сегмента (сек)')
        convert_parser.add_argument('--detect-vocals', action='store_true', help='Пытаться определить тип трека (Vocals/Instrumental)')

        # --------------------------------------------------
        # Предобработка
        # --------------------------------------------------
        preprocess_parser = subparsers.add_parser('preprocess', help='Преобразование *.npy в JSONL + mmap')
        preprocess_parser.add_argument('--data-dir', required=True, help='Директория с *.npy')
        preprocess_parser.add_argument('--output-dir', help='Куда писать выходные файлы')
        preprocess_parser.add_argument('--config', help='Путь к mixture_cfg.yml (если есть)')
        preprocess_parser.add_argument('--codec-type', default='xcodec_16k')
        preprocess_parser.add_argument('--num-codebooks', type=int, default=4)
        preprocess_parser.add_argument('--workers', type=int, default=4)
        preprocess_parser.add_argument('--stage', default='both', help='Этап XCodec (encode/decode/both)')
        preprocess_parser.add_argument('--shuffle', action='store_true')

        # --------------------------------------------------
        # Подготовка датасета
        # --------------------------------------------------
        dataset_parser = subparsers.add_parser('dataset', help='Подготовка датасета для обучения')
        dataset_parser.add_argument('--data-dir', required=True)
        dataset_parser.add_argument('--output-dir')
        dataset_parser.add_argument('--cache-dir')
        dataset_parser.add_argument('--seed', type=int, default=42)
        dataset_parser.add_argument('--sequence-length', type=int, default=2048)
        dataset_parser.add_argument('--enable-shuffle', action='store_true')
        dataset_parser.add_argument('--split', default='960,30,10')
        dataset_parser.add_argument('--micro-batch-size', type=int, default=1)
        dataset_parser.add_argument('--global-batch-size', type=int)
        dataset_parser.add_argument('--blend-weights')

        # --------------------------------------------------
        # Подсчёт токенов
        # --------------------------------------------------
        count_parser = subparsers.add_parser('count', help='Подсчёт токенов в mmap датасете')
        count_parser.add_argument('--data-dir', required=True)
        count_parser.add_argument('--save-stats', action='store_true', help='Сохранить подробную статистику в JSON')

        # --------------------------------------------------
        # Обучение
        # --------------------------------------------------
        train_parser = subparsers.add_parser('train', help='Запуск обучения модели')
        train_parser.add_argument('--model', required=True, help='Путь к исходной модели или хаб-имя')
        train_parser.add_argument('--data-dir', required=True)
        train_parser.add_argument('--output-dir')
        train_parser.add_argument('--epochs', type=int, default=3)
        train_parser.add_argument('--batch-size', type=int, default=4)
        train_parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
        train_parser.add_argument('--lr', type=float, default=5e-4)
        train_parser.add_argument('--weight-decay', type=float, default=0.01)
        train_parser.add_argument('--warmup-ratio', type=float, default=0.03)
        train_parser.add_argument('--lr-scheduler', default='cosine')
        train_parser.add_argument('--logging-steps', type=int, default=10)
        train_parser.add_argument('--save-steps', type=int, default=100)
        train_parser.add_argument('--eval-steps', type=int, default=100)
        train_parser.add_argument('--save-total-limit', type=int, default=2)
        train_parser.add_argument('--optimizer', default='adamw_torch_fused')
        train_parser.add_argument('--lora-r', type=int, default=8)
        train_parser.add_argument('--lora-alpha', type=int, default=16)
        train_parser.add_argument('--lora-dropout', type=float, default=0.05)
        train_parser.add_argument('--lora-target-modules', nargs='*', default=["q_proj", "k_proj", "v_proj", "o_proj"])
        train_parser.add_argument('--fp16', action='store_true')
        train_parser.add_argument('--bf16', action='store_true')
        train_parser.add_argument('--gradient-checkpointing', action='store_true')
        train_parser.add_argument('--model-max-length', type=int)
        train_parser.add_argument('--resume-from-checkpoint')
        train_parser.add_argument('--report-to')
        train_parser.add_argument('--run-name')
        train_parser.add_argument('--deepspeed')

        # --------------------------------------------------
        # Полный пайплайн
        # --------------------------------------------------
        pipeline_parser = subparsers.add_parser('pipeline', help='Запуск всех шагов последовательно')
        pipeline_parser.add_argument('--input-dir', required=True, help='Директория с исходными аудио файлами')
        pipeline_parser.add_argument('--model', required=True, help='Путь к исходной модели или хаб-имя')
        pipeline_parser.add_argument('--output-dir', help='Директория для результатов')
        pipeline_parser.add_argument('--config', help='Пользовательский файл mixture_cfg.yml')
        pipeline_parser.add_argument('--pipeline-config', help='Путь к JSON-файлу с конфигурацией пайплайна')
        pipeline_parser.add_argument('--epochs', type=int, default=3)
        pipeline_parser.add_argument('--batch-size', type=int, default=4)
        pipeline_parser.add_argument('--sr', type=int, default=16000, help='Частота дискретизации для конвертера')
        pipeline_parser.add_argument('--lora-r', type=int, default=8)
        pipeline_parser.add_argument('--lora-alpha', type=int, default=16)
        pipeline_parser.add_argument('--fp16', action='store_true')
        pipeline_parser.add_argument('--bf16', action='store_true')
        pipeline_parser.add_argument('--skip-steps', nargs='*', choices=['convert', 'preprocess', 'dataset', 'count', 'train'], help='Какие шаги пропустить')

        # --------------------------------------------------
        return parser

    def _get_command_defaults(self, command_name, required_args):
        """Вспомогательная функция для получения словаря аргументов по умолчанию для команды.
        
        Создает минимальный список аргументов, чтобы парсер не падал при попытке
        получить значения по умолчанию для команды.
        
        Args:
            command_name (str): Имя команды (convert, preprocess, dataset, count, train).
            required_args (list): Список обязательных аргументов для команды.
            
        Returns:
            dict: Словарь аргументов по умолчанию для команды.
        """
        # Создаем минимальный список аргументов, чтобы парсер не падал
        dummy_args = [command_name] + required_args
        defaults = self.parser.parse_args(dummy_args)
        return vars(defaults)

    def _get_cache_path(self, step_name, args_hash):
        """Получает путь к кэшированному результату для шага пайплайна.
        
        Args:
            step_name (str): Имя шага пайплайна.
            args_hash (str): Хеш аргументов шага.
            
        Returns:
            str: Путь к файлу кэша.
        
        Raises:
            OSError: Если директория кэша недоступна для записи.
        """
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Проверяем права на запись, создав временный файл
            test_file_path = os.path.join(self.cache_dir, f".write_test_{time.time()}")
            with open(test_file_path, 'w') as f:
                f.write("test")
            os.remove(test_file_path)
                
            return os.path.join(self.cache_dir, f"{step_name}_{args_hash}.json")
        except (OSError, PermissionError) as e:
            logger.error(f"Нет доступа к директории кэша {self.cache_dir}: {e}")
            # Используем временную директорию в качестве запасного варианта
            fallback_dir = os.path.join(tempfile.gettempdir(), "yue_cache")
            logger.warning(f"Используем резервную директорию для кэша: {fallback_dir}")
            os.makedirs(fallback_dir, exist_ok=True)
            return os.path.join(fallback_dir, f"{step_name}_{args_hash}.json")
    
    def _hash_args(self, args_dict):
        """Создает хеш из словаря аргументов.
        
        Сортирует ключи для стабильного хеширования и создает SHA-256-хеш
        из JSON-представления словаря аргументов.
        
        Args:
            args_dict (dict): Словарь аргументов.
            
        Returns:
            str: Хеш аргументов в виде шестнадцатеричной строки.
        """
        import hashlib
        # Сортируем ключи для стабильного хеширования
        args_str = json.dumps(args_dict, sort_keys=True)
        return hashlib.sha256(args_str.encode()).hexdigest()[:16]  # Берем первые 16 символов для совместимости с длиной MD5
    
    def _cache_result(self, step_name, args_dict, result):
        """Кэширует результат шага пайплайна.
        
        Сохраняет результат выполнения шага пайплайна в файл кэша
        вместе с аргументами и временной меткой.
        
        Args:
            step_name (str): Имя шага пайплайна.
            args_dict (dict): Словарь аргументов шага.
            result (dict): Результат выполнения шага.
        """
        args_hash = self._hash_args(args_dict)
        cache_path = self._get_cache_path(step_name, args_hash)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'args': args_dict,
                    'result': result,
                    'timestamp': time.time()
                }, f, indent=2)
            logger.debug(f"Результат шага {step_name} сохранен в кэше: {cache_path}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить результат в кэше: {e}")
    
    def _get_cached_result(self, step_name, args_dict, max_age_hours=24):
        """Получает кэшированный результат для шага пайплайна.
        
        Проверяет наличие кэшированного результата для шага пайплайна
        с заданными аргументами и возвращает его, если он не устарел.
        
        Args:
            step_name (str): Имя шага пайплайна.
            args_dict (dict): Словарь аргументов шага.
            max_age_hours (int, optional): Максимальный возраст кэша в часах.
            
        Returns:
            dict or None: Кэшированный результат или None, если кэш не найден или устарел.
        """
        args_hash = self._hash_args(args_dict)
        cache_path = self._get_cache_path(step_name, args_hash)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Проверяем возраст кэша
            if time.time() - cache_data['timestamp'] > max_age_hours * 3600:
                logger.debug(f"Кэш для шага {step_name} устарел")
                return None
                
            # Проверяем соответствие аргументов
            cached_args = cache_data.get('args', {})
            if cached_args != args_dict:
                logger.debug(f"Аргументы в кэше не соответствуют текущим аргументам")
                # Находим различающиеся аргументы для лога
                diff_args = []
                for key in set(cached_args) | set(args_dict):
                    if key not in cached_args:
                        diff_args.append(f"Отсутствует в кэше: {key}")
                    elif key not in args_dict:
                        diff_args.append(f"Отсутствует в текущих аргументах: {key}")
                    elif cached_args[key] != args_dict[key]:
                        diff_args.append(f"Различие в {key}: кэш={cached_args[key]}, текущее={args_dict[key]}")
                logger.debug(f"Различия в аргументах: {', '.join(diff_args)}")
                return None
            
            logger.info(f"Найден кэшированный результат для шага {step_name}")
            return cache_data['result']
        except Exception as e:
            logger.warning(f"Ошибка при чтении кэша: {e}")
            return None

    def run_pipeline(self, args):
        """Выполнение полного пайплайна обработки и обучения.
        
        Последовательно выполняет все шаги пайплайна:
        1. Конвертация аудио - преобразование аудиофайлов в формат numpy (.npy)
        2. Предобработка данных - преобразование .npy в JSONL и mmap-формат
        3. Подготовка датасета - создание датасета для обучения
        4. Подсчет токенов - анализ количества токенов в датасете
        5. Обучение модели - файнтюнинг модели на подготовленном датасете
        
        Поддерживает пропуск шагов с помощью аргумента --skip-steps,
        кэширование промежуточных результатов для ускорения повторных запусков
        и обработку ошибок на каждом шаге с сохранением лога ошибок.
        
        Args:
            args (argparse.Namespace): Аргументы командной строки.
            
        Returns:
            None
        
        Raises:
            SystemExit: Если произошла критическая ошибка, препятствующая продолжению пайплайна.
        """
        logger.info("Запуск полного пайплайна обработки и обучения...")
        
        try:
            if not args.output_dir:
                args.output_dir = os.path.join(os.getcwd(), "yue_pipeline_output")
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Создаем файл для логирования ошибок пайплайна
            error_log_path = os.path.join(args.output_dir, "pipeline_errors.log")
            
            # Сохраняем конфигурацию пайплайна
            pipeline_config = create_pipeline_config(args)
            config_path = os.path.join(args.output_dir, "pipeline_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_config, f, indent=2)
            logger.info(f"Конфигурация пайплайна сохранена в {config_path}")
            
            skip_steps = args.skip_steps or []
            
            # Словарь для хранения результатов каждого шага
            step_results = {}
            
            # Добавляем флаг для использования кэша
            use_cache = True
            
            # --- КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Логика сборки аргументов для каждого шага ---

            # Шаг 1: Конвертация аудио
            converted_dir = os.path.join(args.output_dir, "converted")
            if 'convert' not in skip_steps:
                try:
                    logger.info("Шаг 1: Конвертация аудио")
                    # 1. Получаем defaults из парсера
                    convert_defaults = self._get_command_defaults('convert', ['--input-dir', '.'])
                    # 2. Обновляем их конфигом из pipeline_config.json
                    convert_defaults.update(pipeline_config.get("convert", {}))
                    # 3. Устанавливаем динамические/внешние значения
                    convert_defaults['input_dir'] = args.input_dir
                    convert_defaults['output_dir'] = converted_dir
                    
                    # Проверяем наличие кэшированного результата
                    cached_result = None
                    if use_cache:
                        cached_result = self._get_cached_result('convert', convert_defaults)
                    
                    if cached_result:
                        # Используем кэшированный результат
                        converted_dir = cached_result.get('output_dir', converted_dir)
                        logger.info(f"Используем кэшированный результат для шага конвертации: {converted_dir}")
                        step_results['convert'] = {
                            'status': 'cached',
                            'output_dir': converted_dir
                        }
                    else:
                        # 4. Создаем корректный объект Namespace
                        convert_args = argparse.Namespace(**convert_defaults)
                        converted_dir = AudioConverter().run(convert_args)
                        step_results['convert'] = {
                            'status': 'success',
                            'output_dir': converted_dir
                        }
                        
                        # Кэшируем результат
                        if use_cache:
                            self._cache_result('convert', convert_defaults, {
                                'output_dir': converted_dir
                            })
                except Exception as e:
                    error_msg = f"Ошибка при конвертации аудио: {str(e)}"
                    logger.error(error_msg)
                    with open(error_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"{error_msg}\n")
                    step_results['convert'] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    if not os.path.exists(converted_dir):
                        logger.error("Критическая ошибка: не удалось создать директорию для конвертированных файлов")
                        return
            else:
                logger.info("Шаг 1: Конвертация аудио (пропущено)")
                step_results['convert'] = {'status': 'skipped'}
            
            # Шаг 2: Предобработка данных
            processed_dir = os.path.join(args.output_dir, "processed")
            if 'preprocess' not in skip_steps:
                try:
                    logger.info("Шаг 2: Предобработка данных")
                    preprocess_defaults = self._get_command_defaults('preprocess', ['--data-dir', '.'])
                    preprocess_defaults.update(pipeline_config.get("preprocess", {}))
                    preprocess_defaults['data_dir'] = converted_dir
                    preprocess_defaults['output_dir'] = processed_dir
                    preprocess_defaults['config'] = args.config
                    
                    # Проверяем наличие кэшированного результата
                    cached_result = None
                    if use_cache:
                        cached_result = self._get_cached_result('preprocess', preprocess_defaults)
                    
                    if cached_result:
                        # Используем кэшированный результат
                        processed_dir = cached_result.get('output_dir', processed_dir)
                        logger.info(f"Используем кэшированный результат для шага предобработки: {processed_dir}")
                        step_results['preprocess'] = {
                            'status': 'cached',
                            'output_dir': processed_dir
                        }
                    else:
                        preprocess_args = argparse.Namespace(**preprocess_defaults)
                        processed_dir = DataPreprocessor().run(preprocess_args, self.project_root)
                        step_results['preprocess'] = {
                            'status': 'success',
                            'output_dir': processed_dir
                        }
                        
                        # Кэшируем результат
                        if use_cache:
                            self._cache_result('preprocess', preprocess_defaults, {
                                'output_dir': processed_dir
                            })
                except Exception as e:
                    error_msg = f"Ошибка при предобработке данных: {str(e)}"
                    logger.error(error_msg)
                    with open(error_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"{error_msg}\n")
                    step_results['preprocess'] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    if not os.path.exists(processed_dir):
                        logger.error("Критическая ошибка: не удалось создать директорию для обработанных файлов")
                        return
            else:
                logger.info("Шаг 2: Предобработка данных (пропущено)")
                step_results['preprocess'] = {'status': 'skipped'}

            # Шаг 3: Подготовка датасета
            dataset_dir = os.path.join(args.output_dir, "dataset")
            if 'dataset' not in skip_steps:
                try:
                    logger.info("Шаг 3: Подготовка датасета")
                    dataset_defaults = self._get_command_defaults('dataset', ['--data-dir', '.'])
                    dataset_defaults.update(pipeline_config.get("dataset", {}))
                    dataset_defaults['data_dir'] = processed_dir
                    dataset_defaults['output_dir'] = dataset_dir
                    
                    # Проверяем наличие кэшированного результата
                    cached_result = None
                    if use_cache:
                        cached_result = self._get_cached_result('dataset', dataset_defaults)
                    
                    if cached_result:
                        # Используем кэшированный результат
                        dataset_dir = cached_result.get('output_dir', dataset_dir)
                        logger.info(f"Используем кэшированный результат для шага подготовки датасета: {dataset_dir}")
                        step_results['dataset'] = {
                            'status': 'cached',
                            'output_dir': dataset_dir
                        }
                    else:
                        dataset_args = argparse.Namespace(**dataset_defaults)
                        dataset_dir = DatasetPreparer().run(dataset_args)
                        step_results['dataset'] = {
                            'status': 'success',
                            'output_dir': dataset_dir
                        }
                        
                        # Кэшируем результат
                        if use_cache:
                            self._cache_result('dataset', dataset_defaults, {
                                'output_dir': dataset_dir
                            })
                except Exception as e:
                    error_msg = f"Ошибка при подготовке датасета: {str(e)}"
                    logger.error(error_msg)
                    with open(error_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"{error_msg}\n")
                    step_results['dataset'] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    if not os.path.exists(dataset_dir):
                        logger.error("Критическая ошибка: не удалось создать директорию для датасета")
                        return
            else:
                logger.info("Шаг 3: Подготовка датасета (пропущено)")
                step_results['dataset'] = {'status': 'skipped'}
            
            # Шаг 4: Подсчет токенов
            if 'count' not in skip_steps:
                try:
                    logger.info("Шаг 4: Подсчет токенов")
                    count_defaults = self._get_command_defaults('count', ['--data-dir', '.'])
                    count_defaults.update(pipeline_config.get("count", {}))
                    count_defaults['data_dir'] = processed_dir
                    
                    # Проверяем наличие кэшированного результата
                    cached_result = None
                    if use_cache:
                        cached_result = self._get_cached_result('count', count_defaults)
                    
                    if cached_result:
                        # Используем кэшированный результат
                        logger.info(f"Используем кэшированный результат для шага подсчета токенов")
                        step_results['count'] = {
                            'status': 'cached',
                            'token_stats': cached_result.get('token_stats', {})
                        }
                    else:
                        count_args = argparse.Namespace(**count_defaults)
                        token_stats = TokenCounter().run(count_args, self.project_root)
                        step_results['count'] = {
                            'status': 'success',
                            'token_stats': token_stats
                        }
                        
                        # Кэшируем результат
                        if use_cache:
                            self._cache_result('count', count_defaults, {
                                'token_stats': token_stats
                            })
                except Exception as e:
                    error_msg = f"Ошибка при подсчете токенов: {str(e)}"
                    logger.error(error_msg)
                    with open(error_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"{error_msg}\n")
                    step_results['count'] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    # Подсчет токенов не является критическим шагом, продолжаем выполнение
            else:
                logger.info("Шаг 4: Подсчет токенов (пропущено)")
                step_results['count'] = {'status': 'skipped'}
            
            # Шаг 5: Запуск обучения
            if 'train' not in skip_steps:
                try:
                    logger.info("Шаг 5: Запуск обучения")
                    model_dir = os.path.join(args.output_dir, "model")
                    train_defaults = self._get_command_defaults('train', ['--model', '.', '--data-dir', '.'])
                    train_defaults.update(pipeline_config.get("train", {}))
                    train_defaults['model'] = args.model
                    train_defaults['data_dir'] = processed_dir
                    train_defaults['output_dir'] = model_dir
                    train_defaults['config'] = args.config
                    
                    # Обучение всегда выполняем заново, не используем кэш
                    train_args = argparse.Namespace(**train_defaults)
                    model_dir = Trainer().run(train_args, self.project_root)
                    step_results['train'] = {
                        'status': 'success',
                        'output_dir': model_dir
                    }
                except Exception as e:
                    error_msg = f"Ошибка при обучении модели: {str(e)}"
                    logger.error(error_msg)
                    with open(error_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"{error_msg}\n")
                    step_results['train'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            else:
                logger.info("Шаг 5: Запуск обучения (пропущено)")
                step_results['train'] = {'status': 'skipped'}
            
            # Сохраняем результаты выполнения пайплайна
            results_path = os.path.join(args.output_dir, "pipeline_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(step_results, f, indent=2)
            
            # Проверяем, были ли ошибки
            errors = [step for step, result in step_results.items() if result.get('status') == 'error']
            if errors:
                logger.warning(f"Пайплайн завершен с ошибками в шагах: {', '.join(errors)}")
                logger.warning(f"Подробности в файле {error_log_path}")
            else:
                logger.info("Полный пайплайн успешно завершен!")
            
            logger.info(f"Результаты сохранены в {args.output_dir}")
            
        except Exception as e:
            logger.error(f"Критическая ошибка при выполнении пайплайна: {str(e)}")
            logger.exception("Подробная информация об ошибке:")
            return
    
    def check_disk_space(self, path, required_mb=100):
        """Проверяет наличие свободного места на диске.
        
        Args:
            path (str): Путь к директории, где будут сохраняться файлы.
            required_mb (int): Требуемое количество свободного места в МБ.
            
        Returns:
            bool: True, если достаточно места, иначе False.
        """
        try:
            import shutil
            free_bytes = shutil.disk_usage(path).free
            free_mb = free_bytes / (1024 * 1024)  # Конвертируем в МБ
            
            if free_mb < required_mb:
                logger.warning(
                    f"Недостаточно свободного места на диске: {free_mb:.2f} МБ доступно, "
                    f"требуется минимум {required_mb} МБ"
                )
                return False
                
            logger.debug(f"Доступно {free_mb:.2f} МБ на диске, требуется {required_mb} МБ")
            return True
        except Exception as e:
            logger.warning(f"Не удалось проверить свободное место на диске: {e}")
            return True  # В случае ошибки предполагаем, что места достаточно
    
    def _cleanup_old_caches(self, max_age_hours=168):  # Значение по умолчанию - одна неделя
        """Очищает устаревшие кэш-файлы.
        
        Args:
            max_age_hours (int, optional): Максимальный возраст кэша в часах. 
                                          По умолчанию 168 (одна неделя).
        
        Returns:
            tuple: (int, int) - количество удаленных файлов и общее количество файлов
        """
        if not os.path.exists(self.cache_dir):
            return 0, 0
            
        now = time.time()
        max_age_seconds = max_age_hours * 3600
        
        deleted_count = 0
        total_count = 0
        
        try:
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                    
                total_count += 1
                filepath = os.path.join(self.cache_dir, filename)
                
                try:
                    # Проверяем возраст файла
                    file_age = now - os.path.getmtime(filepath)
                    
                    # Если файл старше указанного возраста, то пытаемся его открыть и проверить
                    if file_age > max_age_seconds:
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                cache_data = json.load(f)
                                
                            # Проверяем возраст кэша по временной метке внутри файла
                            if now - cache_data.get('timestamp', 0) > max_age_seconds:
                                os.remove(filepath)
                                deleted_count += 1
                                logger.debug(f"Удален устаревший кэш-файл: {filename}")
                        except (json.JSONDecodeError, IOError) as e:
                            # Если файл поврежден, удаляем его
                            os.remove(filepath)
                            deleted_count += 1
                            logger.warning(f"Удален поврежденный кэш-файл {filename}: {e}")
                except OSError as e:
                    logger.warning(f"Ошибка при проверке файла {filename}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Очистка кэша: удалено {deleted_count} из {total_count} файлов")
            
            return deleted_count, total_count
        except Exception as e:
            logger.warning(f"Ошибка при очистке кэша: {e}")
            return 0, total_count
    
    def main(self):
        args = self.parser.parse_args()
        
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        if args.output_dir:
            self.output_dir = args.output_dir
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Проверяем наличие свободного места на диске
            if args.command == 'pipeline':
                # Для полного пайплайна требуется больше места
                required_mb = 2000  # 2 ГБ
            elif args.command == 'train':
                # Для обучения требуется много места
                required_mb = 1000  # 1 ГБ
            else:
                # Для остальных команд
                required_mb = 500  # 500 МБ
                
            if not self.check_disk_space(self.output_dir, required_mb):
                logger.error(f"Недостаточно свободного места на диске для выполнения команды {args.command}")
                logger.error(f"Требуется минимум {required_mb} МБ свободного места")
                logger.error("Освободите место на диске или укажите другую директорию с помощью --output-dir")
                sys.exit(1)
        
        command_map = {
            'convert': AudioConverter,
            'preprocess': DataPreprocessor,
            'dataset': DatasetPreparer,
            'count': TokenCounter,
            'train': Trainer,
        }

        if args.command == 'pipeline':
            self.run_pipeline(args)
        elif args.command in command_map:
            handler = command_map[args.command]()
            if args.command in ['preprocess', 'count', 'train']:
                handler.run(args, self.project_root)
            else:
                handler.run(args)
        else:
            self.parser.print_help()
            sys.exit(1)

        # Очищаем устаревшие кэш-файлы
        self._cleanup_old_caches()


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    trainer = YuETrainer()
    trainer.main() 