import argparse
import json
import os
import sys
# tqdm выводит прогресс-бар в stdout по умолчанию, что ломает JSON-парсинг.
# Перенаправляем его в stderr.
from tqdm import tqdm

# Добавляем путь к родительской директории для импорта core.datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from core.datasets.indexed_dataset import MMapIndexedDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the mmap files (without .bin or .idx extension)")
    return parser.parse_args()


def count_tokens_in_dataset(dataset_path):
    """Быстро подсчитывает количество токенов в MMapIndexedDataset.

    Сначала пытаемся воспользоваться вектором ``sequence_lengths`` (O(1)).
    Если по какой-то причине он отсутствует, переходим к медленному перебору с tqdm.
    """
    try:
        dataset = MMapIndexedDataset(dataset_path)

        # Оптимальный путь: просто суммируем длины всех последовательностей.
        try:
            return int(dataset.sequence_lengths.sum())
        except Exception:
            pass  # перейдём к медленной стратегии ниже

        # Фоллбэк: итеративный подсчёт (медленнее, но надёжнее).
        token_count = 0
        for doc_ids in tqdm(
            dataset,
            desc=f"Counting tokens in {os.path.basename(dataset_path)}",
            file=sys.stderr,
        ):
            token_count += doc_ids.shape[0]
        return token_count
    except Exception as e:
        # В случае ошибки выводим информацию в stderr и возвращаем None
        print(f"Error processing dataset at {dataset_path}: {e}", file=sys.stderr)
        return None

def main():
    """Главная функция скрипта."""
    args = get_args()
    
    mmap_prefix_path = args.path
    
    # Проверяем наличие .bin и .idx файлов
    bin_file = f"{mmap_prefix_path}.bin"
    idx_file = f"{mmap_prefix_path}.idx"
    
    if not os.path.exists(bin_file) or not os.path.exists(idx_file):
        error_message = f"Error: Required files not found. Searched for {bin_file} and {idx_file}."
        print(json.dumps({"error": error_message}), file=sys.stdout)
        sys.exit(1)

    total_tokens = count_tokens_in_dataset(mmap_prefix_path)

    # Формируем и выводим результат в формате JSON
    if total_tokens is not None:
        result = {
            "file": os.path.basename(mmap_prefix_path),
            "total_tokens": total_tokens,
            "status": "success"
        }
    else:
        result = {
            "file": os.path.basename(mmap_prefix_path),
            "error": "Failed to count tokens.",
            "status": "error"
        }
        
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main() 