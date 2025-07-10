import logging
import os
import subprocess
import sys
import json
import shutil
import yaml
from pathlib import Path
import time

# Импортируем tqdm для отображения прогресс-баров
try:
    from tqdm import tqdm
except ImportError:
    print("Рекомендуется установить библиотеку tqdm для отображения прогресс-баров.")
    print("Установите её с помощью: pip install tqdm")
    # Создаем заглушку для tqdm
    def tqdm(iterable=None, **kwargs):
        return iterable or range(0)

# Импортируем numpy с проверкой
try:
    import numpy as np
except ImportError:
    print("Для работы требуется библиотека numpy.")
    print("Установите её с помощью: pip install numpy")
    sys.exit(1)

# Импортируем библиотеки для работы с аудио с проверкой
try:
    import librosa
    import soundfile as sf
except ImportError:
    # Не выходим из программы сразу, так как эти библиотеки нужны только для AudioConverter
    librosa = None
    sf = None

# Импортируем собственный модуль с проверкой
try:
    from ..tools.codecmanipulator import CodecManipulator
except ImportError as e:
    print(f"Ошибка импорта CodecManipulator: {e}")
    print("Убедитесь, что структура проекта соответствует ожидаемой.")
    sys.exit(1)

logger = logging.getLogger(__name__)

# Добавляем цветное логирование, если поддерживается
try:
    import colorlog
    
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    )
    
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
except ImportError:
    pass  # Если colorlog не установлен, используем обычное логирование


class PipelineStep:
    """Базовый класс для всех шагов пайплайна обработки данных.
    
    Определяет общий интерфейс для всех классов обработки, включая методы
    для проверки зависимостей, валидации аргументов и выполнения шага.
    """
    
    def __init__(self, name=None):
        """Инициализация шага пайплайна.
        
        Args:
            name (str, optional): Имя шага. Если не указано, используется имя класса.
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.{self.name}")
    
    def check_dependencies(self):
        """Проверка наличия необходимых зависимостей для выполнения шага.
        
        Returns:
            bool: True, если все зависимости доступны, иначе False.
        """
        return True
    
    def validate_args(self, args):
        """Валидация аргументов перед выполнением шага.
        
        Args:
            args (argparse.Namespace): Аргументы командной строки.
            
        Returns:
            bool: True, если аргументы валидны, иначе False.
            
        Raises:
            ValueError: Если аргументы не валидны.
        """
        return True
    
    def check_disk_space(self, path, required_mb=100):
        """Проверка наличия свободного места на диске.
        
        Args:
            path (str): Путь к директории, где будут сохраняться файлы.
            required_mb (int, optional): Требуемое количество свободного места в МБ.
            
        Returns:
            bool: True, если достаточно места, иначе False.
        """
        try:
            import shutil
            free_bytes = shutil.disk_usage(path).free
            free_mb = free_bytes / (1024 * 1024)  # Конвертируем в МБ
            
            if free_mb < required_mb:
                self.logger.warning(
                    f"Недостаточно свободного места на диске: {free_mb:.2f} МБ доступно, "
                    f"требуется минимум {required_mb} МБ"
                )
                return False
                
            self.logger.debug(f"Доступно {free_mb:.2f} МБ на диске, требуется {required_mb} МБ")
            return True
        except Exception as e:
            self.logger.warning(f"Не удалось проверить свободное место на диске: {e}")
            return True  # В случае ошибки предполагаем, что места достаточно
    
    def run(self, args, project_root=None):
        """Запуск шага пайплайна.
        
        Args:
            args (argparse.Namespace): Аргументы командной строки.
            project_root (str, optional): Корневая директория проекта.
            
        Returns:
            str: Путь к директории с результатами выполнения шага.
            
        Raises:
            NotImplementedError: Если метод не переопределен в дочернем классе.
        """
        raise NotImplementedError("Метод run() должен быть реализован в дочернем классе")


class AudioConverter(PipelineStep):
    """Конвертация аудио файлов в формат для обучения."""
    
    def __init__(self):
        super().__init__(name="AudioConverter")
    
    def check_dependencies(self):
        """Проверка наличия необходимых зависимостей."""
        if librosa is None or sf is None:
            self.logger.error("Для конвертации аудио требуются библиотеки librosa и soundfile.")
            self.logger.error("Установите их с помощью: pip install librosa soundfile")
            return False
        return True
    
    def validate_args(self, args):
        """Валидация аргументов перед выполнением шага."""
        if not os.path.exists(args.input_dir):
            raise ValueError(f"Директория с исходными аудио файлами не существует: {args.input_dir}")
        
        if args.segment_length and args.segment_length <= 0:
            raise ValueError(f"Длина сегмента должна быть положительным числом: {args.segment_length}")
        
        if args.min_segment_length <= 0:
            raise ValueError(f"Минимальная длина сегмента должна быть положительным числом: {args.min_segment_length}")
        
        if args.segment_length and args.min_segment_length > args.segment_length:
            raise ValueError(
                f"Минимальная длина сегмента ({args.min_segment_length}) "
                f"не может быть больше длины сегмента ({args.segment_length})"
            )
        
        return True

    def run(self, args, project_root=None):
        """Запуск конвертации аудио файлов."""
        # Проверяем зависимости
        if not self.check_dependencies():
            sys.exit(1)
        
        # Валидируем аргументы
        try:
            self.validate_args(args)
        except ValueError as e:
            self.logger.error(f"Ошибка в аргументах: {e}")
            sys.exit(1)
            
        self.logger.info("Начинаем конвертацию аудио файлов...")
        
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir or os.path.join(input_dir, "converted"))
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Проверяем наличие свободного места на диске
        # Примерно оцениваем требуемое место: 50 МБ на каждый аудиофайл
        audio_files = [f for f in input_dir.glob("**/*.*") if f.suffix.lower() in ['.mp3', '.wav', '.flac']]
        audio_files_count = len(audio_files)
        required_mb = audio_files_count * 50
        if not self.check_disk_space(output_dir, required_mb):
            self.logger.warning(f"Продолжение может привести к нехватке места на диске")
        
        self.logger.info(f"Найдено {audio_files_count} аудиофайлов для обработки")
        
        # Группируем файлы по базовому имени
        audio_files_dict = {}
        
        # Используем прогресс-бар для отображения процесса сканирования файлов
        self.logger.info("Сканирование и классификация аудиофайлов...")
        for file in tqdm(audio_files, desc="Сканирование файлов", unit="файл"):
            if file.suffix.lower() in ['.mp3', '.wav', '.flac']:
                base_name = file.stem
                
                if args.detect_vocals:
                    self.logger.debug(f"Автоматическое определение типа для {file.name} не реализовано, используется имя файла")
                
                # Улучшенное логирование для прозрачности
                if "vocals" in base_name.lower() or "vocal" in base_name.lower():
                    file_type = "Vocals"
                    self.logger.debug(f"Файл '{file.name}' классифицирован как 'Vocals' по имени.")
                elif "instrumental" in base_name.lower() or "inst" in base_name.lower():
                    file_type = "Instrumental"
                    self.logger.debug(f"Файл '{file.name}' классифицирован как 'Instrumental' по имени.")
                else:
                    file_type = "Full"
                    self.logger.debug(f"Файл '{file.name}' классифицирован как 'Full' (общий).")

                
                key = base_name.split(".")[0]
                if key not in audio_files_dict:
                    audio_files_dict[key] = {}
                
                audio_files_dict[key][file_type] = file
        
        # Обрабатываем файлы с отображением прогресса
        total_files = sum(len(files) for files in audio_files_dict.values())
        processed_files = 0
        
        self.logger.info(f"Начинаем обработку {total_files} файлов...")
        progress_bar = tqdm(total=total_files, desc="Обработка аудио", unit="файл")
        
        for key, files in audio_files_dict.items():
            self.logger.debug(f"Обработка файлов для {key}...")
            
            def process_audio_file(file_path, output_base_name):
                self.logger.debug(f"Конвертация: {file_path}")
                try:
                    # Измеряем время загрузки и обработки
                    start_time = time.time()
                    audio, sr = librosa.load(file_path, sr=args.sr, mono=args.mono)
                    load_time = time.time() - start_time
                    self.logger.debug(f"  Аудио загружено за {load_time:.2f} сек, длина: {len(audio)/sr:.2f} сек")
                    
                    if args.normalize:
                        max_amp = np.max(np.abs(audio))
                        if max_amp > 0:
                            scale_factor = args.max_amplitude / max_amp
                            audio = audio * scale_factor
                            self.logger.debug(f"  Применена нормализация с коэффициентом {scale_factor:.4f}")
                    
                    if args.trim_silence:
                        trim_start = time.time()
                        non_silent_intervals = librosa.effects.split(audio, top_db=20, frame_length=2048, hop_length=512)
                        if len(non_silent_intervals) > 0:
                            audio = np.concatenate([audio[s:e] for s, e in non_silent_intervals])
                            trim_time = time.time() - trim_start
                            self.logger.debug(f"  Удалена тишина за {trim_time:.2f} сек, новая длина: {len(audio)/sr:.2f} сек")
                    
                    if args.segment_length:
                        segment_samples = int(args.segment_length * sr)
                        segments = []
                        for i in range(0, len(audio), segment_samples):
                            segment = audio[i:i+segment_samples]
                            if len(segment) >= int(args.min_segment_length * sr):
                                segments.append((segment, f"{output_base_name}.seg{len(segments)+1}"))
                        
                        if segments:
                            self.logger.debug(f"  Разделено на {len(segments)} сегментов")
                            return segments
                        else:
                            self.logger.warning(f"  Не удалось создать сегменты минимальной длины")
                            return [(audio, output_base_name)]
                    
                    return [(audio, output_base_name)]
                except Exception as e:
                    self.logger.error(f"Ошибка при конвертации {file_path}: {e}")
                    return []

            for file_type, file_path in files.items():
                base_name = f"{key}.{file_type}" if file_type != "Full" else key
                segments = process_audio_file(file_path, base_name)
                
                # Обновляем прогресс-бар для каждого обработанного файла
                processed_files += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"текущий": f"{key}.{file_type}"})
                
                for audio_segment, segment_name in segments:
                    output_path = output_dir / f"{segment_name}.npy"
                    np.save(output_path, audio_segment)
                    self.logger.debug(f"  Сохранено в {output_path}")
        
        # Закрываем прогресс-бар
        progress_bar.close()
        
        self.logger.info(f"Конвертация завершена. Обработано {processed_files} файлов.")
        self.logger.info(f"Результаты сохранены в {output_dir}")
        return str(output_dir)


class DataPreprocessor(PipelineStep):
    """Запуск предобработки данных."""
    
    def __init__(self):
        super().__init__(name="DataPreprocessor")
    
    def validate_args(self, args):
        """Валидация аргументов перед выполнением шага."""
        if not os.path.exists(args.data_dir):
            raise ValueError(f"Директория с данными не существует: {args.data_dir}")
        
        if args.config and not os.path.exists(args.config):
            self.logger.warning(f"Указанный конфигурационный файл не существует: {args.config}")
            self.logger.warning("Будет создан стандартный конфигурационный файл")
        
        return True

    def _create_mixture_config(self, data_dir, blend_weights=None, split="960,30,10"):
        """Создаёт YAML-конфиг для скрипта `parse_mixture.py`.

        Он указывает, где находятся jsonl-файлы и как делить датасет
        на train/val/test. Реализация заимствована из старого `steps.py`.
        """
        import yaml

        config_path = os.path.join(data_dir, "mixture_cfg.yml")

        # По умолчанию один источник с весом 1.0 — все jsonl в подпапке `jsonl`.
        data_paths = [{"path": os.path.join(data_dir, "jsonl"), "weight": 1.0}]

        # Если пользователь передал кастомные веса бленда — учитываем.
        if blend_weights:
            try:
                # Ожидаем строку вида "0.5,0.5" или список чисел.
                if isinstance(blend_weights, str):
                    weights = [float(w) for w in blend_weights.split(",")]
                elif isinstance(blend_weights, list):
                    weights = [float(w) for w in blend_weights]
                else:
                    self.logger.warning(f"Неподдерживаемый формат весов бленда: {type(blend_weights)}. Используем вес 1.0.")
                    weights = [1.0]
                
                # Нормализуем веса, чтобы их сумма была равна 1.0
                weight_sum = sum(weights)
                if weight_sum <= 0:
                    self.logger.warning("Сумма весов бленда <= 0. Используем вес 1.0.")
                    weights = [1.0]
                else:
                    weights = [w / weight_sum for w in weights]
                
                # Применяем веса к источникам данных
                # Предполагаем, что у нас может быть несколько поддиректорий с jsonl-файлами
                jsonl_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and "jsonl" in d.lower()]
                
                if len(jsonl_dirs) == 0:
                    # Если нет поддиректорий, используем основную директорию jsonl
                    jsonl_dirs = ["jsonl"]
                
                # Если количество весов не соответствует количеству директорий, корректируем
                if len(weights) != len(jsonl_dirs):
                    if len(weights) > len(jsonl_dirs):
                        weights = weights[:len(jsonl_dirs)]
                        self.logger.warning(f"Слишком много весов. Используем первые {len(jsonl_dirs)}: {weights}")
                    else:
                        # Дополняем недостающие веса равномерно
                        remaining_weight = 1.0 - sum(weights)
                        additional_weights = [remaining_weight / (len(jsonl_dirs) - len(weights))] * (len(jsonl_dirs) - len(weights))
                        weights.extend(additional_weights)
                        self.logger.warning(f"Недостаточно весов. Дополнено до {len(jsonl_dirs)}: {weights}")
                
                # Создаем список источников с весами
                data_paths = []
                for i, dir_name in enumerate(jsonl_dirs):
                    data_paths.append({
                        "path": os.path.join(data_dir, dir_name),
                        "weight": weights[i]
                    })
                
                self.logger.info(f"Настроены веса бленда: {[{p['path']: p['weight']} for p in data_paths]}")
            except Exception as e:
                self.logger.error(f"Ошибка при обработке весов бленда: {e}. Используем вес 1.0.")
                data_paths = [{"path": os.path.join(data_dir, "jsonl"), "weight": 1.0}]

        mixture_config = {
            "data_path_prefix": data_dir,
            "split": split,
            "mixture": [
                {
                    "name": "custom_dataset",
                    "weights": 1.0,
                    "datasets": data_paths,
                }
            ],
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(mixture_config, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"Конфигурационный файл микро-смеси сохранён в {config_path}")
        return config_path
        
    def run(self, args, project_root=None):
        """Запуск предобработки данных."""
        # Валидируем аргументы
        try:
            self.validate_args(args)
        except ValueError as e:
            self.logger.error(f"Ошибка в аргументах: {e}")
            sys.exit(1)
            
        self.logger.info("Начинаем предобработку данных...")
        
        data_dir = args.data_dir
        output_dir = args.output_dir or os.path.join(data_dir, "processed")
        os.makedirs(output_dir, exist_ok=True)
        
        # Проверяем наличие свободного места на диске
        # Примерно оцениваем требуемое место: 100 МБ на каждый NPY файл
        npy_files_count = len(list(Path(data_dir).glob("**/*.npy")))
        required_mb = npy_files_count * 100
        if not self.check_disk_space(output_dir, required_mb):
            self.logger.warning(f"Продолжение может привести к нехватке места на диске")
        
        jsonl_dir = os.path.join(data_dir, "jsonl")
        os.makedirs(jsonl_dir, exist_ok=True)
        
        self.logger.info("Подготовка JSONL файлов из NPY...")
        npy_files = list(Path(data_dir).glob("**/*.npy"))
        
        manipulator = CodecManipulator(
            codec_type=args.codec_type,
            n_quantizer=args.num_codebooks
        )
        
        for npy_file in npy_files:
            base_name = npy_file.stem
            jsonl_file = os.path.join(jsonl_dir, f"{base_name}.{args.codec_type}.jsonl")
            self.logger.info(f"Преобразование {npy_file} в {jsonl_file}...")
            try:
                json_str = manipulator.npy_to_json_str(str(npy_file))
                with open(jsonl_file, 'w') as f:
                    f.write(json_str)
            except Exception as e:
                self.logger.error(f"Ошибка при преобразовании {npy_file}: {e}")
        
        # Создаем конфигурационный файл микро-смеси, если не указан пользовательский
        if not args.config:
            self.logger.info("Создание конфигурационного файла микро-смеси...")
            mixture_config_path = self._create_mixture_config(
                data_dir=output_dir,
                blend_weights=getattr(args, "blend_weights", None),
                split=getattr(args, "split", "960,30,10")
            )
            args.config = mixture_config_path
        else:
            # Если пользователь указал свой конфиг, копируем его в выходную директорию
            if os.path.exists(args.config):
                target_config = os.path.join(output_dir, os.path.basename(args.config))
                shutil.copy2(args.config, target_config)
                self.logger.info(f"Пользовательский конфигурационный файл скопирован в {target_config}")
                args.config = target_config
            else:
                self.logger.warning(f"Указанный конфигурационный файл {args.config} не найден. Создаем стандартный.")
                mixture_config_path = self._create_mixture_config(
                    data_dir=output_dir,
                    blend_weights=getattr(args, "blend_weights", None),
                    split=getattr(args, "split", "960,30,10")
                )
                args.config = mixture_config_path

        # Проверяем, что project_root не None, иначе выбрасываем исключение с понятным сообщением
        if project_root is None:
            raise ValueError("project_root не задан. Пожалуйста, передайте корректный путь к корню проекта.")

        preprocess_script = os.path.join(str(project_root), "core", "preprocess_data_conditional_xcodec.py")

        # Безопасно достаём значения — если их нет, подставляем дефолты.
        workers = getattr(args, "workers", 4)
        stage = str(getattr(args, "stage", "both"))

        preprocess_cmd = [
            sys.executable, preprocess_script,
            "--input-json-dir", jsonl_dir,
            "--output-prefix", os.path.join(output_dir, os.path.basename(jsonl_dir)),
            "--dataset-impl", "mmap",
            "--tokenizer-type", "MMSentencePieceTokenizer",
            "--workers", str(workers),
            "--stage", stage,
        ]

        if getattr(args, "shuffle", False):
            preprocess_cmd.append("--shuffle")
        
        self.logger.info(f"Выполнение команды: {' '.join(preprocess_cmd)}")
        try:
            subprocess.run(preprocess_cmd, check=True)
            self.logger.info("Предобработка данных завершена успешно")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ошибка при предобработке данных: {e}")
            sys.exit(1)
        
        return output_dir

class DatasetPreparer(PipelineStep):
    """Подготовка датасета для обучения."""
    
    def __init__(self):
        super().__init__(name="DatasetPreparer")
    
    def validate_args(self, args):
        """Валидация аргументов перед выполнением шага."""
        if not os.path.exists(args.data_dir):
            raise ValueError(f"Директория с данными не существует: {args.data_dir}")
        
        try:
            split_parts = [int(x) for x in args.split.split(",")]
            if len(split_parts) != 3:
                raise ValueError("Параметр split должен содержать 3 значения (train,val,test)")
        except Exception as e:
            self.logger.warning(f"Ошибка при разборе параметра split '{args.split}': {e}")
            self.logger.warning("Будет использовано соотношение по умолчанию: 96% train, 3% val, 1% test")
        
        return True
    
    def run(self, args, project_root=None):
        """Подготовка датасета для обучения."""
        # Валидируем аргументы
        try:
            self.validate_args(args)
        except ValueError as e:
            self.logger.error(f"Ошибка в аргументах: {e}")
            sys.exit(1)
            
        self.logger.info("Начинаем подготовку датасета...")
        
        data_dir = args.data_dir
        output_dir = args.output_dir or os.path.join(data_dir, "dataset")
        os.makedirs(output_dir, exist_ok=True)
        
        # Проверяем наличие свободного места на диске
        # Примерно оцениваем требуемое место: 50 МБ на каждый бинарный файл
        bin_files_count = len(list(Path(data_dir).glob("**/*.bin")))
        required_mb = bin_files_count * 50
        if not self.check_disk_space(output_dir, required_mb):
            self.logger.warning(f"Продолжение может привести к нехватке места на диске")
        
        # Проверяем наличие необходимых файлов
        bin_files = list(Path(data_dir).glob("**/*.bin"))
        idx_files = list(Path(data_dir).glob("**/*.idx"))
        
        if not bin_files or not idx_files:
            self.logger.error(f"В директории {data_dir} не найдены необходимые файлы .bin и .idx")
            sys.exit(1)
            
        # Создаем конфигурационный файл для датасета
        config_path = os.path.join(output_dir, "dataset_config.json")
        
        # Разбираем параметр split (train,val,test)
        try:
            split_parts = [int(x) for x in args.split.split(",")]
            if len(split_parts) != 3:
                raise ValueError("Ожидается 3 значения (train,val,test)")
            split_sum = sum(split_parts)
            train_ratio = split_parts[0] / split_sum
            val_ratio = split_parts[1] / split_sum
            test_ratio = split_parts[2] / split_sum
        except Exception as e:
            self.logger.error(f"Ошибка при разборе параметра split '{args.split}': {e}")
            self.logger.info("Используем соотношение по умолчанию: 96% train, 3% val, 1% test")
            train_ratio, val_ratio, test_ratio = 0.96, 0.03, 0.01
        
        # Создаем конфигурацию датасета
        dataset_config = {
            "data_dir": data_dir,
            "output_dir": output_dir,
            "cache_dir": args.cache_dir or os.path.join(output_dir, "cache"),
            "sequence_length": args.sequence_length,
            "seed": args.seed,
            "enable_shuffle": args.enable_shuffle,
            "split": {
                "train": train_ratio,
                "validation": val_ratio,
                "test": test_ratio
            },
            "micro_batch_size": args.micro_batch_size,
            "global_batch_size": args.global_batch_size or args.micro_batch_size * 4
        }
        
        # Если указаны веса для бленда, добавляем их в конфигурацию
        if args.blend_weights:
            try:
                blend_weights = [float(x) for x in args.blend_weights.split(",")]
                dataset_config["blend_weights"] = blend_weights
                self.logger.info(f"Используются веса для бленда: {blend_weights}")
            except Exception as e:
                self.logger.error(f"Ошибка при разборе весов бленда '{args.blend_weights}': {e}")
        
        # Сохраняем конфигурацию
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_config, f, indent=2)
        
        self.logger.info(f"Конфигурация датасета сохранена в {config_path}")
        
        # Создаем символические ссылки на файлы данных в выходной директории
        for bin_file in bin_files:
            target_path = os.path.join(output_dir, bin_file.name)
            try:
                # Удаляем существующую ссылку, если она есть
                if os.path.exists(target_path):
                    if os.path.islink(target_path):
                        os.unlink(target_path)
                    else:
                        os.remove(target_path)
                # Создаем символическую ссылку
                os.symlink(bin_file, target_path)
                self.logger.debug(f"Создана символическая ссылка: {target_path} -> {bin_file}")
            except (OSError, PermissionError) as e:
                self.logger.error(f"Ошибка при создании символической ссылки {target_path}: {e}")
                # Пытаемся скопировать файл вместо создания ссылки
                try:
                    import shutil
                    shutil.copy2(bin_file, target_path)
                    self.logger.warning(f"Не удалось создать символическую ссылку, файл скопирован: {target_path}")
                except Exception as copy_err:
                    self.logger.error(f"Также не удалось скопировать файл: {copy_err}")
        
        for idx_file in idx_files:
            target_path = os.path.join(output_dir, idx_file.name)
            try:
                # Удаляем существующую ссылку, если она есть
                if os.path.exists(target_path):
                    if os.path.islink(target_path):
                        os.unlink(target_path)
                    else:
                        os.remove(target_path)
                # Создаем символическую ссылку
                os.symlink(idx_file, target_path)
                self.logger.debug(f"Создана символическая ссылка: {target_path} -> {idx_file}")
            except (OSError, PermissionError) as e:
                self.logger.error(f"Ошибка при создании символической ссылки {target_path}: {e}")
                # Пытаемся скопировать файл вместо создания ссылки
                try:
                    import shutil
                    shutil.copy2(idx_file, target_path)
                    self.logger.warning(f"Не удалось создать символическую ссылку, файл скопирован: {target_path}")
                except Exception as copy_err:
                    self.logger.error(f"Также не удалось скопировать файл: {copy_err}")
        
        self.logger.info(f"Подготовка датасета завершена. Результаты сохранены в {output_dir}")
        return output_dir

class TokenCounter(PipelineStep):
    """Запуск подсчета токенов."""
    
    def __init__(self):
        super().__init__(name="TokenCounter")
    
    def validate_args(self, args):
        """Валидация аргументов перед выполнением шага."""
        if not os.path.exists(args.data_dir):
            raise ValueError(f"Директория с данными не существует: {args.data_dir}")
        
        return True
    
    def run(self, args, project_root=None):
        """Запуск подсчета токенов."""
        # Валидируем аргументы
        try:
            self.validate_args(args)
        except ValueError as e:
            self.logger.error(f"Ошибка в аргументах: {e}")
            sys.exit(1)
            
        self.logger.info("Начинаем подсчет токенов...")
        
        data_dir = args.data_dir
        # --- ИСПРАВЛЕНИЕ: Используем новый, исправленный скрипт ---
        if project_root is None:
            self.logger.error("Не указан корневой каталог проекта")
            sys.exit(1)
            
        count_script = os.path.join(project_root, "tools", "new_count_mmap_token.py")
        
        bin_files = list(Path(data_dir).glob("**/*.bin"))
        if not bin_files:
            self.logger.warning(f"Не найдены бинарные файлы (.bin) в {data_dir}. Шаг подсчета токенов будет пропущен.")
            return
        
        token_stats = {}
        
        for bin_file in bin_files:
            idx_file = bin_file.with_suffix(".idx")
            if idx_file.exists():
                self.logger.info(f"Подсчет токенов для {bin_file.name}...")
                # Используем более надежный способ получения пути без расширения
                bin_path_without_ext = os.path.splitext(str(bin_file))[0]
                count_cmd = [sys.executable, count_script, "--path", bin_path_without_ext]
                
                try:
                    result = subprocess.run(count_cmd, check=True, stdout=subprocess.PIPE, text=True, encoding='utf-8')
                    stats = json.loads(result.stdout)
                    
                    if stats.get("status") == "success":
                        token_count = stats.get("total_tokens", 0)
                        token_stats[bin_file.name] = token_count
                        self.logger.info(f"  Найдено токенов: {token_count:,}")
                    else:
                        self.logger.error(f"Ошибка при подсчете токенов для {bin_file.name}: {stats.get('error')}")

                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Критическая ошибка при запуске скрипта подсчета токенов для {bin_file.name}: {e}")
                except json.JSONDecodeError:
                    self.logger.error(f"Не удалось декодировать JSON-ответ от скрипта подсчета токенов.")

        if args.save_stats and token_stats:
            stats_file = os.path.join(data_dir, "token_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(token_stats, f, indent=2)
            self.logger.info(f"Статистика токенов сохранена в {stats_file}")
        
        if token_stats:
            total_tokens = sum(token_stats.values())
            self.logger.info(f"Общее количество токенов во всех файлах: {total_tokens:,}")
        
        self.logger.info("Подсчет токенов завершен")
        return data_dir


class Trainer(PipelineStep):
    """Запуск обучения модели."""
    
    def __init__(self):
        super().__init__(name="Trainer")
    
    def validate_args(self, args):
        """Валидация аргументов перед выполнением шага."""
        if not os.path.exists(args.data_dir):
            raise ValueError(f"Директория с данными не существует: {args.data_dir}")
        
        if args.epochs <= 0:
            raise ValueError(f"Количество эпох должно быть положительным числом: {args.epochs}")
        
        if args.batch_size <= 0:
            raise ValueError(f"Размер батча должен быть положительным числом: {args.batch_size}")
        
        if args.lr <= 0:
            raise ValueError(f"Скорость обучения должна быть положительным числом: {args.lr}")
        
        return True
    
    def run(self, args, project_root=None):
        """Запуск обучения модели."""
        # Валидируем аргументы
        try:
            self.validate_args(args)
        except ValueError as e:
            self.logger.error(f"Ошибка в аргументах: {e}")
            sys.exit(1)
            
        self.logger.info("Начинаем обучение модели...")
        
        model_path = args.model
        data_dir = args.data_dir
        output_dir = args.output_dir or os.path.join(os.getcwd(), "model_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Проверяем наличие свободного места на диске
        # Примерно оцениваем требуемое место: 1 ГБ на модель + 100 МБ на каждую эпоху
        required_mb = 1000 + args.epochs * 100
        if not self.check_disk_space(output_dir, required_mb):
            self.logger.warning(f"Продолжение может привести к нехватке места на диске")
        
        # Проверяем наличие необходимых файлов
        bin_files = list(Path(data_dir).glob("**/*.bin"))
        idx_files = list(Path(data_dir).glob("**/*.idx"))
        
        if not bin_files or not idx_files:
            self.logger.error(f"В директории {data_dir} не найдены необходимые файлы .bin и .idx")
            sys.exit(1)
        
        # Сохраняем конфигурацию обучения
        train_config = {
            "model": model_path,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler": args.lr_scheduler,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "save_total_limit": args.save_total_limit,
            "optimizer": args.optimizer,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": args.lora_target_modules,
            "fp16": args.fp16,
            "bf16": args.bf16,
            "gradient_checkpointing": args.gradient_checkpointing
        }
        
        config_path = os.path.join(output_dir, "train_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(train_config, f, indent=2)
        
        self.logger.info(f"Конфигурация обучения сохранена в {config_path}")
        
        # Подготавливаем команду для запуска обучения
        if project_root is None:
            self.logger.error("Не указан корневой каталог проекта")
            sys.exit(1)
            
        train_script = os.path.join(project_root, "scripts", "train_lora.py")
        
        train_cmd = [
            sys.executable, train_script,
            "--model", model_path,
            "--data-path", data_dir,
            "--output-dir", output_dir,
            "--num-train-epochs", str(args.epochs),
            "--per-device-train-batch-size", str(args.batch_size),
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--learning-rate", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--warmup-ratio", str(args.warmup_ratio),
            "--lr-scheduler-type", args.lr_scheduler,
            "--logging-steps", str(args.logging_steps),
            "--save-steps", str(args.save_steps),
            "--eval-steps", str(args.eval_steps),
            "--save-total-limit", str(args.save_total_limit),
            "--lora-r", str(args.lora_r),
            "--lora-alpha", str(args.lora_alpha),
            "--lora-dropout", str(args.lora_dropout),
        ]
        
        # Добавляем опциональные аргументы
        if args.lora_target_modules:
            for module in args.lora_target_modules:
                train_cmd.extend(["--lora-target-modules", module])
        
        if args.fp16:
            train_cmd.append("--fp16")
        
        if args.bf16:
            train_cmd.append("--bf16")
        
        if args.gradient_checkpointing:
            train_cmd.append("--gradient-checkpointing")
        
        if args.model_max_length:
            train_cmd.extend(["--model-max-length", str(args.model_max_length)])
        
        if args.resume_from_checkpoint:
            train_cmd.extend(["--resume-from-checkpoint", args.resume_from_checkpoint])
        
        if args.report_to:
            train_cmd.extend(["--report-to", args.report_to])
        
        if args.run_name:
            train_cmd.extend(["--run-name", args.run_name])
        
        if args.deepspeed:
            train_cmd.extend(["--deepspeed", args.deepspeed])
        
        # Запускаем процесс обучения
        self.logger.info(f"Запуск обучения с командой: {' '.join(train_cmd)}")
        try:
            subprocess.run(train_cmd, check=True)
            self.logger.info("Обучение модели успешно завершено!")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ошибка при обучении модели: {e}")
            sys.exit(1)
        
        self.logger.info(f"Обученная модель сохранена в {output_dir}")
        return output_dir 