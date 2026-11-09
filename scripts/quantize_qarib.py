"""Compare QARiB fp32 with dynamic int8 quantization (validation only).

Dynamic quantization converts the Linear layers to int8 for CPU inference.
No retraining. Reports validation Macro-F1, agreement with fp32, size and latency.
"""

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aratox_nlp.config import ROOT, ensure_output_dirs
from aratox_nlp.constants import TOXIC_LABELS
from aratox_nlp.data import load_processed
from aratox_nlp.inference import load_qarib, pick_device, qarib_probs
from aratox_nlp.metrics import multilabel_metrics


def quantized(model):
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else torch.backends.quantized.engine
    return torch.ao.quantization.quantize_dynamic(model.cpu(), {torch.nn.Linear}, dtype=torch.qint8)


def size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1e6


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def latency_ms(qarib, texts, batch_size):
    device = qarib[2]
    qarib_probs(qarib, texts[:8], batch_size=batch_size)
    sync(device)
    started = time.perf_counter()
    qarib_probs(qarib, texts, batch_size=batch_size)
    sync(device)
    return (time.perf_counter() - started) / len(texts) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency-n", type=int, default=500)
    args = ap.parse_args()
    ensure_output_dirs()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    val = load_processed("validation", ROOT / "data/processed")
    texts = val.text_norm.astype(str).tolist()
    y = val[TOXIC_LABELS].astype(int).values
    timing_texts = texts[: args.latency_n]

    accel = pick_device()
    tok, model, _ = load_qarib(device=accel)
    variants = {f"fp32 ({accel.type})": (tok, model, accel)}
    tok_cpu, model_cpu, cpu = load_qarib(device=torch.device("cpu"))
    variants["fp32 (cpu)"] = (tok_cpu, model_cpu, cpu)
    variants["int8 dynamic (cpu)"] = (tok_cpu, quantized(load_qarib(device=torch.device("cpu"))[1]), cpu)

    reference = None
    rows = []
    for name, qarib in variants.items():
        probs = qarib_probs(qarib, texts, batch_size=32)
        pred = (probs >= 0.5).astype(int)
        reference = pred if reference is None else reference
        metrics = multilabel_metrics(y, pred, TOXIC_LABELS)
        rows.append({
            "variant": name,
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "label_agreement_with_fp32": float((pred == reference).mean()),
            "size_mb": size_mb(qarib[1]),
            "ms_per_sample_batch16": latency_ms(qarib, timing_texts, 16),
            "ms_per_sample_batch1": latency_ms(qarib, timing_texts[:100], 1),
            "n_eval": len(texts),
        })
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "outputs/metrics/qarib_quantization_validation.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
