# coding=utf-8
# Jittor port of SpinQuant -- command-line arguments (single-card).
#
# A simplified, dependency-light port of utils/process_args.py.  We drop the
# HuggingFace Trainer / FSDP integration and expose everything through a plain
# argparse parser.  `process_args_ptq` returns three lightweight namespace
# objects (model_args, training_args, ptq_args) so the entry points read almost
# identically to the original.

import argparse
import os
import math


class _NS:
    """A tiny attribute-style namespace (avoids dataclass boilerplate)."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __repr__(self):
        return "NS(" + ", ".join(f"{k}={v!r}" for k, v in vars(self).items()) + ")"


def _parser():
    p = argparse.ArgumentParser(description="SpinQuant (Jittor, single-card)")

    # --- model paths ----------------------------------------------------- #
    p.add_argument("--input_model", type=str, required=True,
                   help="Local HuggingFace model directory (download weights first).")
    p.add_argument("--output_rotation_path", type=str, default="./rotate_out",
                   help="Where to save the learned R1/R2 checkpoint (R.npy).")
    p.add_argument("--optimized_rotation_path", type=str, default=None,
                   help="Load a previously learned R1/R2 .npy checkpoint.")
    p.add_argument("--access_token", type=str, default=None,
                   help="HuggingFace access token (for gated repos).")

    # --- training / rotation hyperparameters ------------------------------ #
    p.add_argument("--model_max_length", type=int, default=2048)
    p.add_argument("--bsz", type=int, default=1, help="Eval batch size.")
    p.add_argument("--learning_rate", type=float, default=2e-4,
                   help="Learning rate for rotation (SGDG).")
    p.add_argument("--rotate_iters", type=int, default=1000,
                   help="Number of optimization steps for rotation training.")
    p.add_argument("--fp16", action="store_true", default=False,
                   help="Use FP16 base weights (default FP32). Rotations stay FP32.")
    p.add_argument("--allow_synthetic_data", action="store_true",
                   help="Smoke tests only: allow random tokens if WikiText-2 fails.")
    p.add_argument("--allow_unquantized_training", action="store_true",
                   help="Smoke tests only: allow training with all quantizers disabled.")
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--lr_scheduler_type", choices=["constant", "cosine"], default="constant",
                   help="Constant preserves the historical Jittor training protocol.")
    p.add_argument("--seed", type=int, default=0)

    # --- rotation flags --------------------------------------------------- #
    p.add_argument("--rotate", action="store_true", default=False,
                   help="Bake R1/R2 (and R4 Hadamard) into the weights.")
    p.add_argument("--rotate_mode", type=str, default="hadamard",
                   choices=["hadamard", "random"])
    p.add_argument("--fp32_had", action="store_true", default=False,
                   help="Apply the online Hadamard in FP32.")

    # --- activation quantization ------------------------------------------ #
    p.add_argument("--a_bits", type=int, default=16)
    p.add_argument("--a_groupsize", type=int, default=-1)
    p.add_argument("--a_asym", action="store_true", default=False)
    p.add_argument("--a_clip_ratio", type=float, default=1.0)

    # --- weight quantization ---------------------------------------------- #
    p.add_argument("--w_bits", type=int, default=16)
    p.add_argument("--w_groupsize", type=int, default=-1)
    p.add_argument("--w_asym", action="store_true", default=False)
    p.add_argument("--w_quant_method", choices=["rtn", "gptq"], default="rtn",
                   help="PTQ weight method; RTN remains the default.")
    p.add_argument("--w_rtn", action="store_true", default=False,
                   help="Compatibility alias for --w_quant_method rtn.")
    p.add_argument("--w_clip", action="store_true", default=False)
    p.add_argument("--nsamples", type=int, default=128,
                   help="Maximum calibration batches for GPTQ (not training steps).")
    p.add_argument("--percdamp", type=float, default=0.01,
                   help="GPTQ diagonal damping fraction.")
    p.add_argument("--act_order", action="store_true",
                   help="Process GPTQ weight columns by descending Hessian diagonal.")
    p.add_argument("--gptq_blocksize", type=int, default=128)
    p.add_argument("--static_groups", action="store_true",
                   help="GPTQ: fit groups before column reordering; requires w_groupsize > 0.")

    # --- general --------------------------------------------------------- #
    p.add_argument("--int8_down_proj", action="store_true", default=False)

    # --- KV-cache quantization ------------------------------------------- #
    p.add_argument("--v_bits", type=int, default=16)
    p.add_argument("--v_groupsize", type=int, default=None,
                   help="Default: head_dim. Use -1 for per-token V quantization.")
    p.add_argument("--v_asym", action="store_true", default=False)
    p.add_argument("--v_clip_ratio", type=float, default=1.0)
    p.add_argument("--k_bits", type=int, default=16)
    p.add_argument("--k_groupsize", type=int, default=None,
                   help="Default: head_dim. Use -1 for per-token K quantization.")
    p.add_argument("--k_asym", action="store_true", default=False)
    p.add_argument("--k_pre_rope", action="store_true", default=False,
                   help="Pre-RoPE K-cache quantization (NOT supported).")
    p.add_argument("--k_clip_ratio", type=float, default=1.0)

    # --- save / load quantized model ------------------------------------- #
    p.add_argument("--load_qmodel_path", type=str, default=None)
    p.add_argument("--save_qmodel_path", type=str, default=None)
    return p


def process_args_ptq(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)

    if args.w_rtn and args.w_quant_method != "rtn":
        parser.error("--w_rtn conflicts with --w_quant_method gptq.")
    args.w_rtn = args.w_quant_method == "rtn"
    if args.nsamples <= 0 or args.gptq_blocksize <= 0:
        parser.error("--nsamples and --gptq_blocksize must be positive.")
    if not math.isfinite(args.percdamp) or args.percdamp < 0:
        parser.error("--percdamp must be finite and nonnegative.")
    if args.static_groups and args.w_groupsize <= 0:
        parser.error("--static_groups requires --w_groupsize > 0.")

    if args.k_pre_rope:
        parser.error("Pre-RoPE K quantization is not supported.")
    for field in ("w_bits", "a_bits", "k_bits", "v_bits"):
        if not 2 <= getattr(args, field) <= 16:
            parser.error(f"--{field} must be between 2 and 16 (16 disables fake quantization).")
    for field in ("bsz", "rotate_iters", "logging_steps"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field} must be positive.")
    if args.model_max_length < 2:
        parser.error("--model_max_length must be at least 2.")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive.")
    for field in ("w_groupsize", "a_groupsize", "v_groupsize", "k_groupsize"):
        value = getattr(args, field)
        if value is not None and value != -1 and value <= 0:
            parser.error(f"--{field} must be -1 or a positive group size.")
    for field in ("a_clip_ratio", "k_clip_ratio", "v_clip_ratio"):
        if not 0 < getattr(args, field) <= 1:
            parser.error(f"--{field} must be in (0, 1].")
    if args.optimized_rotation_path:
        if not args.rotate:
            parser.error("--optimized_rotation_path requires --rotate.")
        args.optimized_rotation_path = os.path.expanduser(args.optimized_rotation_path)
        if not os.path.isfile(args.optimized_rotation_path):
            parser.error(f"Rotation checkpoint not found: {args.optimized_rotation_path}")
    if args.load_qmodel_path and args.save_qmodel_path:
        parser.error("Cannot load and save a quantized model in the same invocation.")

    model_args = _NS(
        input_model=args.input_model,
        output_rotation_path=args.output_rotation_path,
        optimized_rotation_path=args.optimized_rotation_path,
        access_token=args.access_token,
    )
    training_args = _NS(
        model_max_length=args.model_max_length,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.learning_rate,
        output_dir="./outputs",
        rotate_iters=args.rotate_iters,
    )
    # ptq_args carries the quantization / rotation flags directly.
    ptq_args = args
    ptq_args.bsz = args.bsz
    return model_args, training_args, ptq_args


def validate_rotation_training(args):
    """Reject an unquantized rotation objective except in an explicit smoke test."""
    if args.w_quant_method != "rtn":
        raise ValueError("GPTQ is a post-training step; rotation training uses RTN/STE. "
                         "Select --w_quant_method gptq when preparing the trained model.")
    quantized = min(args.w_bits, args.a_bits, args.k_bits, args.v_bits) < 16
    if not quantized and not args.allow_unquantized_training:
        raise ValueError("Rotation training needs a low-bit quantization target. "
                         "Use the W4A4KV4 training script, or explicitly pass "
                         "--allow_unquantized_training for a smoke test.")
    if args.optimized_rotation_path:
        raise ValueError("Training does not resume R.npy checkpoints; "
                         "--optimized_rotation_path is an evaluation option.")


def resolve_quantization_groups(args, config):
    """Resolve default K groups and reject unsupported K/V grouping."""
    head_dim = config.hidden_size // config.num_attention_heads
    if args.k_groupsize is None:
        args.k_groupsize = head_dim
    if args.v_groupsize is None:
        args.v_groupsize = head_dim
    if args.k_bits < 16 and args.k_groupsize not in (-1, head_dim):
        raise ValueError(f"K groupsize must be -1 or head_dim={head_dim}.")
    if args.v_bits < 16 and args.v_groupsize not in (-1, head_dim):
        raise ValueError(f"V quantization uses head_dim={head_dim}; other groups are unsupported.")
