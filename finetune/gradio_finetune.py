import os
import sys
import shutil
import json
import yaml
import threading
import traceback
import numpy as np
import gradio as gr
import soundfile as sf
import torchaudio
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from datetime import datetime
import torch
from omegaconf import OmegaConf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../inference/xcodec_mini_infer')))
from models.soundstream_hubert_new import SoundStream

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def safe_log(logbox, msg):
    print(msg)
    if logbox is not None:
        logbox.append(msg)

def create_workspace_structure(workspace_path, logbox=None):
    try:
        os.makedirs(workspace_path, exist_ok=True)
        subdirs = ["npy", "jsonl", "mmap", "logs", "output", "models"]
        for sub in subdirs:
            os.makedirs(os.path.join(workspace_path, sub), exist_ok=True)
        safe_log(logbox, f"Структура папок создана в {workspace_path}")
        return f"Структура папок создана в {workspace_path}"
    except Exception as e:
        err = f"Ошибка при создании структуры: {e}\n{traceback.format_exc()}"
        safe_log(logbox, err)
        return err

def xcodec_encode_audio(audio_path, model, device, target_bw=4.0, sample_rate=16000):
    import torchaudio
    wav, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.to(device)
    if wav.dim() == 2:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    wav = wav.unsqueeze(0)  # (1, 1, T)
    with torch.no_grad():
        codes = model.encode(wav, target_bw=target_bw)
    codes = codes.cpu().numpy().astype(np.int16)
    return codes

def convert_audio_to_npy(audio_files, output_dir, codec_type, n_threads, logbox=None, xcodec_params=None):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    use_xcodec = codec_type == "xcodec"
    if use_xcodec:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_path = os.path.abspath("../inference/xcodec_mini_infer/final_ckpt/config.yaml")
        ckpt_path = os.path.abspath("../inference/xcodec_mini_infer/final_ckpt/ckpt_00360000.pth")
        config = OmegaConf.load(config_path)
        model = SoundStream(**config.generator.config)
        param_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(param_dict['codec_model'])
        model.eval()
        model.to(device)
        target_bw = xcodec_params.get('target_bw', 4.0) if xcodec_params else 4.0
        sample_rate = xcodec_params.get('sample_rate', 16000) if xcodec_params else 16000
    def process_file(audio_path):
        try:
            if use_xcodec:
                codes = xcodec_encode_audio(audio_path, model, device, target_bw=target_bw, sample_rate=sample_rate)
                base = os.path.splitext(os.path.basename(audio_path))[0]
                npy_path = os.path.join(output_dir, base + ".npy")
                np.save(npy_path, codes)
                return f"OK (xcodec): {audio_path} -> {npy_path}"
            else:
                data, sr = sf.read(audio_path)
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                base = os.path.splitext(os.path.basename(audio_path))[0]
                npy_path = os.path.join(output_dir, base + ".npy")
                np.save(npy_path, data)
                return f"OK: {audio_path} -> {npy_path}"
        except Exception as e:
            return f"ERROR: {audio_path}: {e}"
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futs = [executor.submit(process_file, f.name if hasattr(f, 'name') else f) for f in audio_files]
        for fut in as_completed(futs):
            res = fut.result()
            safe_log(logbox, res)
            results.append(res)
    return "\n".join(results)

def prepare_jsonl(audio_npy_dir, jsonl_path, logbox=None):
    # Пример: для каждого npy создаем jsonl с dummy-метаданными
    try:
        npy_files = glob(os.path.join(audio_npy_dir, "*.npy"))
        with open(jsonl_path, "w", encoding="utf-8") as fout:
            for npy in npy_files:
                entry = {
                    "id": os.path.splitext(os.path.basename(npy))[0],
                    "codec": npy,
                    "audio_length_in_sec": 0,  # Можно вычислить по shape/sr
                    "genres": "pop",
                    "splitted_lyrics": {"segmented_lyrics": []},
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        safe_log(logbox, f"JSONL создан: {jsonl_path} ({len(npy_files)} файлов)")
        return f"JSONL создан: {jsonl_path} ({len(npy_files)} файлов)"
    except Exception as e:
        err = f"Ошибка при создании JSONL: {e}\n{traceback.format_exc()}"
        safe_log(logbox, err)
        return err

def run_preprocess_data(jsonl_path, tokenizer_model, codec_type, order, output_prefix, logbox=None):
    # Запуск скрипта preprocess_data_conditional_xcodec.py через subprocess
    import subprocess
    try:
        cmd = [
            sys.executable, "core/preprocess_data_conditional_xcodec.py",
            "--input", jsonl_path,
            "--json-keys", "text", "codec",
            "--tokenizer-type", "MMSentencePieceTokenizer",
            "--tokenizer-model", tokenizer_model,
            "--codec-type", codec_type,
            "--order", order,
            "--output-prefix", output_prefix,
        ]
        safe_log(logbox, f"Запуск: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, cwd="finetune", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            safe_log(logbox, line.strip())
        proc.wait()
        if proc.returncode == 0:
            return "Предобработка завершена успешно"
        else:
            return f"Ошибка предобработки, код {proc.returncode}"
    except Exception as e:
        err = f"Ошибка запуска предобработки: {e}\n{traceback.format_exc()}"
        safe_log(logbox, err)
        return err

def run_finetune(params, logbox=None):
    # Запуск обучения через bash-скрипт run_finetune.sh
    import subprocess
    try:
        # Сохраняем параметры в окружение
        env = os.environ.copy()
        for k, v in params.items():
            env[k] = str(v)
        cmd = ["bash", "scripts/run_finetune.sh"]
        safe_log(logbox, f"Запуск: {' '.join(cmd)} с параметрами {params}")
        proc = subprocess.Popen(cmd, cwd="finetune", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            safe_log(logbox, line.strip())
        proc.wait()
        if proc.returncode == 0:
            return "Обучение завершено успешно"
        else:
            return f"Ошибка обучения, код {proc.returncode}"
    except Exception as e:
        err = f"Ошибка запуска обучения: {e}\n{traceback.format_exc()}"
        safe_log(logbox, err)
        return err

# === GRADIO UI ===

global_log = []
def get_log():
    return "\n".join(global_log[-100:])

def logbox_append(msg):
    global_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def clear_log():
    global_log.clear()
    return ""

with gr.Blocks() as demo:
    gr.Markdown("# YuE Finetune: Графический интерфейс")
    with gr.Tab("1. Подготовка окружения"):
        workspace = gr.Textbox(label="Рабочая директория", value="./finetune_workspace", info="Папка, где будут храниться все данные и результаты")
        btn_init = gr.Button("Создать структуру")
        log1 = gr.Textbox(label="Лог", lines=5, interactive=False)
        btn_init.click(lambda w: create_workspace_structure(w, logbox=global_log), inputs=workspace, outputs=log1).then(get_log, None, log1)
    with gr.Tab("2. Конвертация аудио"):
        audio_files = gr.Files(label="Аудиофайлы (.wav, .flac, .mp3)")
        output_dir = gr.Textbox(label="Папка для .npy", value="./finetune_workspace/npy")
        codec_type = gr.Dropdown(["xcodec", "dac16k", "dac44k"], label="Тип кодека", value="xcodec", info="Рекомендуется xcodec")
        n_threads = gr.Slider(1, 16, value=4, label="Потоков", info="Чем больше, тем быстрее обработка")
        xcodec_bw = gr.Number(label="xcodec target_bw (Kbps)", value=4.0, info="Битрейт кодека (рекомендуется 4.0)")
        xcodec_sr = gr.Number(label="xcodec sample_rate", value=16000, info="Частота дискретизации (обычно 16000)")
        btn_convert = gr.Button("Конвертировать")
        log2 = gr.Textbox(label="Лог", lines=10, interactive=False)
        def convert_audio_ui(files, out, codec, n, bw, sr):
            params = {'target_bw': bw, 'sample_rate': int(sr)} if codec == 'xcodec' else None
            return convert_audio_to_npy(files, out, codec, int(n), logbox=global_log, xcodec_params=params)
        btn_convert.click(convert_audio_ui,
                         inputs=[audio_files, output_dir, codec_type, n_threads, xcodec_bw, xcodec_sr], outputs=log2).then(get_log, None, log2)
    with gr.Tab("3. Подготовка датасета"):
        audio_npy_dir = gr.Textbox(label="Папка с .npy", value="./finetune_workspace/npy")
        jsonl_path = gr.Textbox(label="Путь для jsonl", value="./finetune_workspace/jsonl/dataset.jsonl")
        btn_jsonl = gr.Button("Создать JSONL")
        log3 = gr.Textbox(label="Лог", lines=10, interactive=False)
        btn_jsonl.click(lambda d, j: prepare_jsonl(d, j, logbox=global_log), inputs=[audio_npy_dir, jsonl_path], outputs=log3).then(get_log, None, log3)
        gr.Markdown("""
        <small>Для реального обучения требуется вручную заполнить поля splitted_lyrics, genres и др. в jsonl-файле.</small>
        """)
    with gr.Tab("4. Предобработка для обучения"):
        jsonl_path2 = gr.Textbox(label="Путь к jsonl", value="./finetune_workspace/jsonl/dataset.jsonl")
        tokenizer_model = gr.Textbox(label="Путь к tokenizer.model", value="./tokenizer.model")
        codec_type2 = gr.Dropdown(["xcodec", "dac16k", "dac44k"], label="Тип кодека", value="xcodec")
        order = gr.Dropdown(["textfirst", "audiofirst"], label="Порядок", value="textfirst", info="textfirst для text2audio")
        output_prefix = gr.Textbox(label="Префикс выходных файлов", value="./finetune_workspace/mmap/dataset")
        btn_preproc = gr.Button("Запустить предобработку")
        log4 = gr.Textbox(label="Лог", lines=10, interactive=False)
        btn_preproc.click(lambda j, t, c, o, p: run_preprocess_data(j, t, c, o, p, logbox=global_log),
                          inputs=[jsonl_path2, tokenizer_model, codec_type2, order, output_prefix], outputs=log4).then(get_log, None, log4)
    with gr.Tab("5. Настройка и запуск обучения"):
        gr.Markdown("<b>Все параметры ниже будут переданы в окружение для run_finetune.sh</b>")
        params = {}
        params["DATA_PATH"] = gr.Textbox(label="DATA_PATH (mmap)", value="./finetune_workspace/mmap/dataset_text_document.bin")
        params["DATA_CACHE_PATH"] = gr.Textbox(label="DATA_CACHE_PATH", value="./finetune_workspace/mmap/")
        params["TOKENIZER_MODEL_PATH"] = gr.Textbox(label="TOKENIZER_MODEL_PATH", value="./tokenizer.model")
        params["MODEL_NAME"] = gr.Textbox(label="MODEL_NAME", value="m-a-p/YuE-s1-7B-anneal-en-cot")
        params["MODEL_CACHE_DIR"] = gr.Textbox(label="MODEL_CACHE_DIR", value="./finetune_workspace/models/")
        params["OUTPUT_DIR"] = gr.Textbox(label="OUTPUT_DIR", value="./finetune_workspace/output/")
        params["LORA_R"] = gr.Number(label="LoRA Rank (LORA_R)", value=64, info="64 — стандарт, больше — больше памяти")
        params["LORA_ALPHA"] = gr.Number(label="LoRA Alpha (LORA_ALPHA)", value=32)
        params["LORA_DROPOUT"] = gr.Number(label="LoRA Dropout (LORA_DROPOUT)", value=0.1)
        params["LORA_TARGET_MODULES"] = gr.Textbox(label="LoRA Target Modules", value="q_proj k_proj v_proj o_proj")
        params["PER_DEVICE_TRAIN_BATCH_SIZE"] = gr.Number(label="Batch size на устройство", value=1)
        params["NUM_TRAIN_EPOCHS"] = gr.Number(label="Эпох обучения", value=10)
        params["USE_WANDB"] = gr.Checkbox(label="Использовать WandB", value=True)
        params["WANDB_API_KEY"] = gr.Textbox(label="WandB API Key", value="<your_wandb_api_key>")
        params["NUM_GPUS"] = gr.Number(label="Число GPU", value=2, info="Для ускорения обучения")
        btn_train = gr.Button("Запустить обучение")
        log5 = gr.Textbox(label="Лог", lines=20, interactive=False)
        def collect_params(*vals):
            keys = list(params.keys())
            return {k: v for k, v in zip(keys, vals)}
        btn_train.click(lambda *vals: run_finetune(collect_params(*vals), logbox=global_log),
                       inputs=list(params.values()), outputs=log5).then(get_log, None, log5)
    with gr.Row():
        btn_clear = gr.Button("Очистить лог")
        logbox = gr.Textbox(label="Общий лог", lines=10, interactive=False)
        btn_clear.click(clear_log, None, logbox)
        gr.Markdown("<small>Все ошибки и предупреждения отображаются здесь и в консоли.</small>")

demo.launch() 