# coding=utf-8
# Jittor port of SpinQuant -- model / tokenizer loading (single-card).
#
# This module loads a HuggingFace Llama-family checkpoint into the Jittor
# LlamaForCausalLM defined in model.py.  We use a "mixed" loading strategy:
#   * transformers is used for the config + tokenizer, and
#   * the raw weight tensors are read with safetensors / torch (whichever is
#     available) and converted to NumPy, then copied into the Jittor model via
#     `Module.load_parameters` (which automatically converts torch tensors /
#     NumPy arrays into Jittor Vars).
#
# Distributed / FSDP / ExecuTorch code paths are intentionally excluded.

import os
import glob

import numpy as np

from ..core.model import LlamaForCausalLM


# --------------------------------------------------------------------------- #
# Config + tokenizer
# --------------------------------------------------------------------------- #
def get_config(input_model, token=None):
    """Load the HF config for *input_model* (no weights are read yet)."""
    import transformers

    return transformers.AutoConfig.from_pretrained(input_model, token=token)


def load_tokenizer(input_model, token=None, model_max_length=2048):
    from transformers import LlamaTokenizerFast

    return LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=input_model,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=token,
    )


# --------------------------------------------------------------------------- #
# Raw weight loading -> NumPy
# --------------------------------------------------------------------------- #
def _load_safetensors_numpy_raw(path):
    """Read a .safetensors file into {name: np.ndarray} using only numpy.

    The upstream ``safetensors.numpy`` reader cannot decode BF16 (numpy has no
    brain-float type), and torch is not a hard dependency of the Jittor env.
    We parse the safetensors container directly and expand BF16 words into the
    high half of fp32 via a bitshift (bf16 == fp32 with the low 16 bits zeroed).
    """
    import json
    import mmap
    import struct

    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))

    out = {}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = meta["dtype"]
            shape = tuple(meta["shape"])
            s0, s1 = meta["data_offsets"]
            raw = mm[base + s0: base + s1]
            if dtype == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                arr = (u16 << 16).view(np.float32).reshape(shape)
            elif dtype == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
            elif dtype == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            elif dtype == "F64":
                arr = np.frombuffer(raw, dtype=np.float64).reshape(shape)
            elif dtype == "I64":
                arr = np.frombuffer(raw, dtype=np.int64).reshape(shape)
            elif dtype == "I32":
                arr = np.frombuffer(raw, dtype=np.int32).reshape(shape)
            else:
                raise ValueError(f"Unsupported safetensors dtype {dtype}")
            out[name] = arr
        mm.close()
    return out


def _load_weights_numpy(input_model, token=None, dtype="float16"):
    """Return a dict {name: np.ndarray} of the checkpoint weights.

    Reads .safetensors directly (torch-free, BF16-aware).  A torch-backed .bin
    loader is kept as a fallback for environments that have torch installed.
    Arrays are normalised to the requested *dtype* (fp16/fp32); BF16 weights
    are first expanded to fp32 and then downcast if fp16 was requested.
    """
    if dtype not in ("float16", "float32"):
        raise ValueError("Supported working dtypes are float16 and float32; BF16 checkpoint storage is decoded to FP32.")
    if not os.path.isdir(input_model):
        raise FileNotFoundError(f"Model directory not found: {input_model}. Download the HuggingFace checkpoint first.")
    files = sorted(glob.glob(os.path.join(input_model, "*.safetensors")))
    bin_files = sorted(glob.glob(os.path.join(input_model, "*.bin")))

    state = {}
    if files:
        for f in files:
            state.update(_load_safetensors_numpy_raw(f))
    elif bin_files:
        import torch

        for f in bin_files:
            part = torch.load(f, map_location="cpu", weights_only=True)
            part = {k: v.float().numpy() if hasattr(v, "detach") else v for k, v in part.items()}
            state.update(part)
    else:
        raise FileNotFoundError(f"No .safetensors / .bin weights found under {input_model}")

    # Normalise dtypes so Jittor receives something it handles well.
    out = {}
    for k, v in state.items():
        arr = np.asarray(v)
        arr = arr.astype(dtype, copy=False)
        out[k] = arr
    return out


# --------------------------------------------------------------------------- #
# Build + populate the Jittor model
# --------------------------------------------------------------------------- #
def build_jittor_model(config):
    """Instantiate the Jittor LlamaForCausalLM from a HF config object."""
    return LlamaForCausalLM(config)


def load_llama_model(input_model, token=None, dtype="float16",
                     for_rotation=False):
    """Load a HF Llama checkpoint into a Jittor LlamaForCausalLM.

    *input_model* must be a local HuggingFace checkpoint directory.

    Handles the Llama-v3.2 `tie_word_embeddings` incompatibility by un-tying the
    head and cloning the embedding weights into lm_head (identical to the
    original SpinQuant entry points).
    """
    config = get_config(input_model, token=token)

    # Normalise HF attribute names to what model.py expects.  The HF Llama config
    # exposes `num_attention_heads` (no `num_heads` / `head_dim`), while the
    # Jittor model reads `config.num_heads` / `config.head_dim`.  The synthetic
    # test config already carries these, so we only fill them in for real HF
    # configs.
    if not hasattr(config, "num_heads"):
        config.num_heads = config.num_attention_heads
    if not hasattr(config, "head_dim"):
        config.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

    # Llama v3.2 specific: SpinQuant is not compatible with tie_word_embeddings;
    # clone lm_head from embed_tokens.
    process_word_embeddings = False
    if getattr(config, "tie_word_embeddings", False):
        config.tie_word_embeddings = False  # keep a local copy un-tied
        process_word_embeddings = True

    model = build_jittor_model(config)

    state = _load_weights_numpy(input_model, token=token, dtype=dtype)
    # Jittor's load_parameters resolves dotted keys (incl. ModuleList indices)
    # and auto-converts NumPy / torch tensors into Vars.
    model.load_parameters(state)

    if process_word_embeddings:
        # Clone the (already loaded) embedding weights into the head.
        emb = model.model.embed_tokens.weight
        model.lm_head.weight.assign(emb.numpy())

    model.config.use_cache = False
    return model


# Convenience classmethod-style wrapper kept close to the original
# `LlamaForCausalLM.from_pretrained` API.
def from_pretrained(pretrained_model_name_or_path, token=None, dtype="float16"):
    return load_llama_model(
        pretrained_model_name_or_path, token=token, dtype=dtype
    )
