"""
Benchmark DistilBERT (HF -> ONNX -> int8) for disk footprint.
Pipeline du script complet:
- Charge le package local `Model_4_DISTILBERT/distilbert_model_package` (tokenizer + hf_model).
- Exporte en ONNX dans ce dossier (pas de fallback HF).
- Quantification int8 : dynamique (poids) ou statique full int8 (poids + activations) via onnxruntime.quantization.
"""
from __future__ import annotations
import argparse
import os
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parent  # .../Model_4_DISTILBERT_onnx
MODEL_DIR = ROOT.parent / "Model_4_DISTILBERT" / "distilbert_model_package"
ONNX_EXPORT_DIR = ROOT  # export alongside this script


def human_bytes(num: int) -> str:
    step = 1024.0
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(num) < step:
            return f"{num:.1f} {unit}"
        num /= step
    return f"{num:.1f} PiB"


def directory_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def quantize_onnx_dynamic(onnx_path: Path, target: Path) -> Path:
    """Quantification dynamique (poids int8, activations float)."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as exc:
        raise RuntimeError("onnxruntime.quantization manquant (pip install onnxruntime).") from exc
    quantize_dynamic(
        str(onnx_path),
        str(target),
        weight_type=QuantType.QInt8,
        per_channel=False,
        use_external_data_format=True,
        extra_options={"DisableShapeInference": True},
    )
    return target


def quantize_onnx_static(
    onnx_path: Path,
    target: Path,
    tokenizer,
    calib_texts: List[str],
    seq_len: int,
) -> Path:
    """Quantification statique full int8 (poids + activations) avec calibration."""
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except Exception as exc:
        raise RuntimeError("onnxruntime.quantization manquant (pip install onnxruntime).") from exc

    class TokenDataReader(CalibrationDataReader):
        def __init__(self, tokenizer, texts: List[str], seq_len: int):
            enc = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_tensors="np",
            )
            self._samples = []
            for i in range(len(texts)):
                sample = {k: v[i : i + 1] for k, v in enc.items()}
                self._samples.append(sample)
            self._iter = iter(self._samples)

        def get_next(self):
            return next(self._iter, None)

        def rewind(self):
            self._iter = iter(self._samples)

    reader = TokenDataReader(tokenizer, calib_texts, seq_len)
    quantize_static(
        str(onnx_path),
        str(target),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return target


def ensure_onnx(
    model_root: Path,
    sample_text: str,
    quant_mode: str,
    seq_len: int,
    calib_texts: Optional[List[str]] = None,
    opset: int = 17,
) -> Optional[Path]:
    quant_mode = quant_mode.lower()
    target_map = {
        "static": ONNX_EXPORT_DIR / "model-int8-static.onnx",
        "dynamic": ONNX_EXPORT_DIR / "model-int8-dynamic.onnx",
        "none": ONNX_EXPORT_DIR / "model-fp32.onnx",
    }
    if quant_mode not in target_map:
        raise ValueError(f"quant_mode inconnu: {quant_mode}")

    target_path = target_map[quant_mode]
    if target_path.exists() and quant_mode != "none":
        return target_path

    ONNX_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import import_utils
    except ImportError:
        return None

    import_utils._torchvision_available = False
    import_utils.is_torchvision_available = lambda: False  # type: ignore[assignment]

    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(model_root / "hf_model")
    model.eval()

    def export_fp32(export_path: Path) -> Path:
        inputs = tokenizer(
            sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        input_names, input_tensors, dynamic_axes = [], [], {}
        for name in ["input_ids", "attention_mask", "token_type_ids"]:
            tensor = inputs.get(name)
            if tensor is None:
                continue
            input_names.append(name)
            input_tensors.append(tensor)
            dynamic_axes[name] = {0: "batch", 1: "sequence"}
        dynamic_axes["logits"] = {0: "batch"}
        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    tuple(input_tensors),
                    str(export_path),
                    input_names=input_names,
                    output_names=["logits"],
                    dynamic_axes=dynamic_axes,
                    opset_version=opset,
                    do_constant_folding=True,
                )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ONNX export failed: missing dependency (install onnxscript: pip install onnxscript)"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"ONNX export failed: {exc}") from exc
        return export_path

    if quant_mode == "none":
        fp32_path = target_path
        if not fp32_path.exists():
            export_fp32(fp32_path)
        return fp32_path

    calib = calib_texts or [sample_text]
    with TemporaryDirectory() as tmpdir:
        fp32_tmp = Path(tmpdir) / "model-fp32-temp.onnx"
        export_fp32(fp32_tmp)
        if quant_mode == "dynamic":
            return quantize_onnx_dynamic(fp32_tmp, target_path)
        return quantize_onnx_static(fp32_tmp, target_path, tokenizer, calib, seq_len)


def bench_distilbert(
    model_root: Path,
    sample_text: str,
    prune_fp32: bool,
    quant_mode: str,
    seq_len: int,
    calib_texts: List[str],
) -> Tuple[Path, int]:
    onnx_path = ensure_onnx(model_root, sample_text, quant_mode, seq_len, calib_texts)
    if not onnx_path:
        raise RuntimeError("Could not export or find ONNX model.")
    if prune_fp32 and quant_mode in ("static", "dynamic"):
        for p in ONNX_EXPORT_DIR.glob("*.onnx"):
            if "int8" not in p.stem:
                try:
                    p.unlink()
                except Exception:
                    pass

    size_bytes = onnx_path.stat().st_size
    data_path = onnx_path.with_suffix(".onnx.data")
    if data_path.exists():
        size_bytes += data_path.stat().st_size
    return onnx_path, size_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-text", type=str, default="I love how friendly this app is!")
    parser.add_argument("--prune-fp32-onnx", action="store_true")
    parser.add_argument(
        "--quant-mode",
        choices=["static", "dynamic", "none"],
        default="dynamic",
        help="static = full int8 (poids + activations, calibration requise), dynamic = poids int8, none = fp32.",
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Longueur max pour padding/truncation.")
    parser.add_argument(
        "--calib-text",
        action="append",
        help="Texte de calibration supplémentaire pour la quantification statique (répéter l'option).",
    )
    args = parser.parse_args()

    calib_texts = args.calib_text or []
    model_root = MODEL_DIR
    disk = directory_size_bytes(model_root)
    try:
        backend_path, model_bytes = bench_distilbert(
            model_root,
            args.sample_text,
            args.prune_fp32_onnx,
            args.quant_mode,
            args.seq_len,
            calib_texts,
        )
        print("=== DistilBERT (HF/ONNX) ===")
        print(f"Model size : {human_bytes(model_bytes)}")
        print(f"Backend    : onnx {args.quant_mode} ({backend_path.name})")
    except Exception as exc:
        print("=== DistilBERT (HF/ONNX) ===")
        print(f"Model size : {human_bytes(disk)}")
        print(f"Error      : {exc}")


if __name__ == "__main__":
    main()
