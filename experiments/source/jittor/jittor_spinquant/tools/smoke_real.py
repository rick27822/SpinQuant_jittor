# coding=utf-8
# Real-model smoke test: load TinyLlama-1.1B via the Jittor loader and run a
# forward (no rotation / no quantization) to validate loader.py + model.py on
# real HF shapes.
#
# Run on the GPU box:
#   python -m jittor_spinquant.tools.smoke_real
import os
import time

import numpy as np
import jittor as jt

jt.flags.use_cuda = 1

MODEL_DIR = os.environ.get("TINYLLAMA_DIR", "/root/autodl-tmp/tinyllama")

from ..io.loader import load_llama_model


def main():
    t0 = time.time()
    print(f"Loading {MODEL_DIR} (float16) ...")
    model = load_llama_model(MODEL_DIR, dtype="float16")
    print(f"  loaded in {time.time() - t0:.1f}s")

    rng = np.random.default_rng(0)
    input_ids = jt.array(rng.integers(0, 32000, size=(1, 64)).astype("int64"))
    t1 = time.time()
    out = model(input_ids)
    logits = out["logits"]
    print(f"  forward in {time.time() - t1:.2f}s  logits shape {tuple(logits.shape)}")
    lmax = float(logits.max().item())
    lmin = float(logits.min().item())
    lsum = float(logits.sum().item())
    print(f"  logits max={lmax:.3f} min={lmin:.3f} sum={lsum:.1f}")
    assert np.all(np.isfinite(logits.numpy())), "logits not finite!"
    print("REAL-MODEL FORWARD OK")


if __name__ == "__main__":
    main()
