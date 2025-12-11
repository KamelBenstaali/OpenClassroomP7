""" 
Benchmark DistilBERT vers AI Edge LiteRT (CPU) :
- Charge le package HF local (tokenizer + hf_model).
- Convertit le modèle PyTorch en LiteRT via ai_edge_torch (backend TFLite).
- Sauvegarde un .tflite dans ce dossier puis exécute une inférence via ai_edge_litert pour mesurer pic RSS.
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple
import warnings

# Définir les variables d'env avant d'importer transformers/torch pour éviter torchvision/TF.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["TRANSFORMERS_NO_TF"] = "1"  # force Transformers à ignorer TensorFlow
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent
DISTILBERT_ROOT = ROOT.parent / "Model_4_DISTILBERT" / "distilbert_model_package"
LITERT_PATH = ROOT / "distilbert_ai_edge.tflite"


def human_bytes(num: int) -> str:
    step = 1024.0
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(num) < step:
            return f"{num:.1f} {unit}"
        num /= step
    return f"{num:.1f} PiB"


def directory_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run_with_peak_rss(func: Callable[[], None], interval: float = 0.2) -> Tuple[Optional[object], int]:
    proc = psutil.Process(os.getpid())
    peak = 0
    running = True

    def monitor():
        nonlocal peak, running
        while running:
            try:
                rss = proc.memory_info().rss
            except psutil.Error:
                break
            peak = max(peak, rss)
            time.sleep(interval)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    try:
        result = func()
    finally:
        running = False
        t.join()
    return result, peak


def ensure_litert(sample_text: str, seq_len: int) -> Optional[Path]:
    def has_signature(path: Path) -> bool:
        try:
            from ai_edge_litert.interpreter import Interpreter
            Interpreter(str(path)).get_signature_runner()
            return True
        except Exception:
            return False

    if LITERT_PATH.exists() and has_signature(LITERT_PATH):
        return LITERT_PATH

    # Imports différés pour respecter les variables d'env ci-dessus.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import import_utils

    # Forcer la désactivation de torchvision pour éviter l'operator nms manquant.
    import_utils._torchvision_available = False
    import_utils.is_torchvision_available = lambda: False  # type: ignore[assignment]
    import ai_edge_torch

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_ROOT / "tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_ROOT / "hf_model")
    model.eval()

    enc = tokenizer(sample_text, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    LITERT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        edge_model = ai_edge_torch.convert(
            model,
            sample_args=(input_ids, attention_mask),
            strict_export="auto",
        )
        edge_model.export(str(LITERT_PATH))
        return LITERT_PATH if LITERT_PATH.exists() and has_signature(LITERT_PATH) else None
    except Exception as exc:
        raise RuntimeError(f"ai_edge_torch.convert a échoué : {exc}") from exc


def bench_litert(sample_text: str, interval: float, seq_len: int) -> Tuple[int, int]:
    litert_path = ensure_litert(sample_text, seq_len)
    if not litert_path:
        raise RuntimeError("Could not build LiteRT model.")

    def run_litert():
        from ai_edge_litert.interpreter import Interpreter
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_ROOT / "tokenizer")
        enc = tokenizer(sample_text, padding="max_length", truncation=True, max_length=seq_len, return_tensors="np")
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        interpreter = Interpreter(str(litert_path))
        signatures = interpreter.get_signature_list()
        signature_key = "serving_default" if "serving_default" in signatures else next(iter(signatures))
        input_names = signatures[signature_key]["inputs"]
        feed = {
            input_names[0]: input_ids,
            input_names[1]: attention_mask,
        }
        signature_runner = interpreter.get_signature_runner(signature_key=signature_key)
        outputs = signature_runner(**feed)
        _ = outputs.get("logits")

    _, peak = run_with_peak_rss(run_litert, interval=interval)
    size_bytes = litert_path.stat().st_size
    return size_bytes, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-text", type=str, default="I love how friendly this app is!")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    base_disk = directory_size_bytes(DISTILBERT_ROOT)
    try:
        size_bytes, peak = bench_litert(args.sample_text, args.interval, args.seq_len)
        print("=== DistilBERT (PyTorch LiteRT) ===")
        print(f"Model size : {human_bytes(size_bytes)}")
        print(f"Backend    : LiteRT (ai_edge_litert runtime)")
        print(f"Peak RSS   : {human_bytes(peak)}")
    except Exception as exc:
        print("=== DistilBERT (PyTorch LiteRT) ===")
        print(f"Model size : {human_bytes(base_disk)}")
        print(f"Error      : {exc}")


if __name__ == "__main__":
    main()
