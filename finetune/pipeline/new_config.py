import json
import logging
import os

logger = logging.getLogger(__name__)

def create_pipeline_config(args):
    """Создание конфигурации для полного пайплайна"""
    config = {
        "convert": {
            "sr": args.sr,
            "mono": True,
            "normalize": True,
            # Параметр "format" был удален, т.к. конвертер жестко сохраняет в .npy
        },
        "preprocess": {
            "codec_type": "semanticodec",
            "num_codebooks": 2,
            "codebook_size": 2048,
            "workers": 4,
            "stage": "both",
            "instruction_dropout_rate": 0.0,
            "to_lower": False
        },
        "dataset": {
            "split": "960,30,10",
            "sequence_length": 2048,
            "seed": 42,
            "enable_shuffle": True
        },
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": 0.0005,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "optimizer": "adamw_torch_fused",
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "fp16": args.fp16,
            "bf16": args.bf16,
            "gradient_checkpointing": True
        }
    }
    
    # Если указан файл конфигурации пайплайна, загружаем его и обновляем наш конфиг
    if args.pipeline_config and os.path.exists(args.pipeline_config):
        try:
            with open(args.pipeline_config, 'r') as f:
                user_config = json.load(f)
            
            # Обновляем каждую секцию конфигурации
            for section, section_config in user_config.items():
                if section in config:
                    config[section].update(section_config)
                else:
                    config[section] = section_config
                    
            logger.info(f"Загружена пользовательская конфигурация из {args.pipeline_config}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке пользовательской конфигурации: {e}")
    
    return config 