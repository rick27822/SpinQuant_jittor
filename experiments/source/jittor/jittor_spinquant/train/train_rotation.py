# coding=utf-8
# Jittor port of optimize_rotation.py -- rotate the model (learns R1 / R2).
#
# Single-card only: the original FSDP / HuggingFace Trainer machinery is dropped
# and replaced by a tiny gradient-descent loop using `jt.grad` + the SGDG (Cayley)
# optimizer.  All base weights stay frozen; only R1 / R2 are optimized.

import os
import json
import time
import math
import logging

import numpy as np

from ..arguments import process_args_ptq, validate_rotation_training, resolve_quantization_groups
log = logging.getLogger("spinquant")


def _attach_weight_quantizers(model, ptq_args):
    """Attach a configured WeightQuantizer to every (non-head) QuantizeLinear.

    Mirrors train_utils/rtn_utils.rtn_fwrd: the quantizer is *attached* (not
    baked) so that, during the training forward, QuantizeLinear.execute
    fake-quantizes the (R1-rotated) weight with a straight-through gradient.
    This is what lets R1 / R2 "see" the weight-quantization error and learn to
    reduce it.  The quantizers live on the *inner* QuantizeLinear (the
    ``.module`` of each ActQuantWrapper), so add_actquant must run first.
    """
    import jittor.nn as nn
    from ..quant.quantization import (
        WeightQuantizer, find_qlayers, QuantizeLinear,
    )

    for layer in model.model.layers:
        subset = find_qlayers(layer, layers=[nn.Linear, QuantizeLinear])
        for name in subset:
            bits = ptq_args.w_bits
            if "lm_head" in name:
                continue  # lm_head is never weight-quantized
            if getattr(ptq_args, "int8_down_proj", False) and "down_proj" in name:
                bits = 8
            q = WeightQuantizer()
            q.configure(
                bits, perchannel=True, sym=not ptq_args.w_asym,
                mse=ptq_args.w_clip, weight_groupsize=ptq_args.w_groupsize,
            )
            subset[name].quantizer = q


def _configure_act_quantizers(model, ptq_args):
    """Configure the per-token activation / v-cache quantizers.

    Mirrors the a_bits / v_bits block of train_utils.main.prepare_model.  The
    down_proj online Hadamard is applied *inline* in LlamaMLP.execute (when R1
    is not None), so we deliberately leave ``online_full_had`` False here to
    avoid a double application.
    """
    from ..quant import quantization as quant_utils

    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])

    down_proj_groupsize = -1
    if ptq_args.a_groupsize > 0:
        inter = model.config.intermediate_size
        hidden = model.config.hidden_size
        if inter % ptq_args.a_groupsize == 0:
            down_proj_groupsize = ptq_args.a_groupsize
        else:
            group_num = int(hidden / ptq_args.a_groupsize)
            down_proj_groupsize = inter // group_num

    num_heads = model.config.num_attention_heads
    model_dim = model.config.hidden_size
    head_dim = model_dim // num_heads

    for name in qlayers:
        layer_input_bits = ptq_args.a_bits
        layer_groupsize = ptq_args.a_groupsize
        layer_a_sym = not ptq_args.a_asym
        layer_a_clip = ptq_args.a_clip_ratio

        if "v_proj" in name and ptq_args.v_bits < 16:  # V-cache precision
            qlayers[name].out_quantizer.configure(
                bits=ptq_args.v_bits, groupsize=ptq_args.v_groupsize,
                sym=not ptq_args.v_asym, clip_ratio=ptq_args.v_clip_ratio,
            )
        if "o_proj" in name:
            layer_groupsize = head_dim
        if "lm_head" in name:  # skip lm_head quantization
            layer_input_bits = 16
        if "down_proj" in name:
            if getattr(ptq_args, "int8_down_proj", False):
                layer_input_bits = 8
            layer_groupsize = down_proj_groupsize

        qlayers[name].quantizer.configure(
            bits=layer_input_bits, groupsize=layer_groupsize,
            sym=layer_a_sym, clip_ratio=layer_a_clip,
        )


def _prepare_for_rotation(model, ptq_args):
    """Prepare the model for rotation training -- mirrors prepare_model.

    The official ``optimize_rotation.py`` calls ``prepare_model`` *before*
    attaching R1 / R2.  Three steps are essential and were missing here:

      1. ``fuse_layer_norms`` folds each RMSNorm's learnable scale into the
         adjacent linear weights and resets its affine scale to one, so RMSNorm
         *commutes* with R1.  Without it a random orthogonal R1 does NOT cancel
         at init (loss ~= ln(vocab) instead of the baseline).
      2. The R4 Hadamard is baked into down_proj (input side) so it cancels the
         online Hadamard applied in LlamaMLP.execute during training.
      3. Weight / activation / KV quantizers are attached so R1 / R2 are
         optimised *against the quantized model* (otherwise the loss is
         independent of R1 and the gradient vanishes -- no learning).

    R1 / R2 themselves stay online / learnable (they are NOT baked here).
    """
    from ..core import fuse_norm as fuse_norm_utils
    from ..math import hadamard as hadamard_utils
    from ..quant import quantization as quant_utils
    from ..rotation import rotation_utils

    # (1) Make RMSNorm scale-free so the R1 rotation is exact at init.
    fuse_norm_utils.fuse_layer_norms(model)

    # (2) Bake R4 into down_proj (cancels the online Hadamard in the MLP).
    for layer in model.model.layers:
        hadamard_utils.apply_exact_had_to_linear(
            layer.mlp.down_proj, had_dim=-1, output=False
        )

    # (3a) Wrap every linear so activations / v-cache can be quantized.
    quant_utils.add_actquant(model)
    # (3b) Attach weight quantizers (fake-quant with STE during training).
    if ptq_args.w_bits < 16:
        _attach_weight_quantizers(model, ptq_args)
    # (3c) Configure activation / v-cache quantizers.
    if ptq_args.a_bits < 16 or ptq_args.v_bits < 16:
        _configure_act_quantizers(model, ptq_args)
    # (3d) K-cache (KV) quantizer wrapper around RoPE output.
    if ptq_args.k_bits < 16:
        rotation_utils.add_qk_rotation_wrapper(
            model,
            k_bits=ptq_args.k_bits,
            k_groupsize=ptq_args.k_groupsize,
            k_sym=not ptq_args.k_asym,
            k_clip_ratio=ptq_args.k_clip_ratio,
        )


def _calibration_batches(args, tokenizer, vocab_size):
    """Return an iterable of {input_ids, labels} batches for rotation training.

    Real-data failures stop training by default. Random tokens are allowed
    only with --allow_synthetic_data and are labelled in logs and metadata.
    """
    seqlen = args.model_max_length
    try:
        from ..io.data_utils import CustomJsonDataset, _load_wt2

        traindata = _load_wt2()["train"]
        ds = CustomJsonDataset(traindata, tokenizer, block_size=seqlen)
        if not len(ds):
            raise ValueError("WikiText-2 produced no calibration blocks.")
        args.calibration_source = "wikitext2"
        log.info("Loaded %d calibration blocks from WikiText-2.", len(ds))
        return ds
    except Exception as e:
        if not args.allow_synthetic_data:
            raise RuntimeError("WikiText-2 preparation failed. Check WIKITEXT2_DIR "
                               "and the datasets dependency. Random-token fallback is "
                               "disabled; --allow_synthetic_data is for smoke tests only.") from e
        log.warning("SMOKE TEST ONLY: using synthetic random-token data: %s", e)
        args.calibration_source = "synthetic_smoke_only"
        n = 256
        batches = []
        rng = np.random.default_rng(args.seed)
        for _ in range(n):
            ids = rng.integers(0, vocab_size, size=(1, seqlen)).astype("int64")
            batches.append({"input_ids": ids, "labels": ids})
        return batches


def train():
    model_args, training_args, ptq_args = process_args_ptq()
    validate_rotation_training(ptq_args)
    # Parse --help and reject invalid objectives before initializing Jittor/CUDA.
    import jittor as jt
    from ..io.loader import load_llama_model, load_tokenizer
    from ..rotation.rotation_utils import add_rotate_module
    from ..rotation.optimizer import SGDG
    from ..misc import get_logger, set_seed
    get_logger("spinquant")
    set_seed(ptq_args.seed)
    # Run on GPU (eval_ptq.py does this; without it Jittor silently stays on CPU
    # and rotation training is orders of magnitude too slow).
    jt.flags.use_cuda = 1

    # Rotation training runs in fp32.  Two reasons:
    #   * Jittor's CPU MKL matmul only supports float32, and the Cayley
    #     backward graph is materialised on CPU when we call `.numpy()` on the
    #     gradient inside the SGDG optimizer -- an fp16/fp32-mixed graph fails
    #     to compile there.
    #   * The original SpinQuant optimises the rotations in fp32/bf16 anyway;
    #     the base weights are only ~4.4GB in fp32, trivial on an 80GB A800.
    #
    # 7B single-card exception: online rotation creates a *second* copy of
    # every linear weight (weight @ R1), so fp32 Mistral needs ~28GB weights
    # + ~28GB rotated copies and OOMs a 96GB card.  `--fp16` loads the base
    # weights as float16 (halving both copies).  R1/R2 themselves stay fp32
    # (they come from random_hadamard_matrix), so the Cayley/NumPy optimizer
    # is unaffected.  Opt-in only: TinyLlama keeps the original fp32 path.
    dtype = "float16" if getattr(ptq_args, "fp16", False) else "float32"
    log.info("rotation training base-weight dtype = %s", dtype)
    model = load_llama_model(
        model_args.input_model, token=model_args.access_token, dtype=dtype,
    )
    resolve_quantization_groups(ptq_args, model.config)
    tokenizer = load_tokenizer(model_args.input_model, token=model_args.access_token,
                               model_max_length=training_args.model_max_length)

    # Prepare the model exactly like the official prepare_model (fuse norms,
    # bake R4, attach weight/activation/KV quantizers) BEFORE attaching the
    # learnable R1 / R2 -- otherwise the online rotation does not cancel at
    # init and provides no gradient signal.
    _prepare_for_rotation(model, ptq_args)

    trainable_parameters = add_rotate_module(model, rotate_mode=ptq_args.rotate_mode)
    num_layers = model.config.num_hidden_layers
    log.info("R1 + %d R2 matrices prepared for training (stiefel=Cayley).", num_layers)

    optimizer = SGDG(trainable_parameters, lr=training_args.learning_rate,
                     stiefel=True)

    batches = _calibration_batches(ptq_args, tokenizer, model.config.vocab_size)
    model.train()

    n_steps = training_args.rotate_iters
    log.info("Training %d steps on %d blocks; scheduler=%s, data=%s.",
             n_steps, len(batches), ptq_args.lr_scheduler_type, ptq_args.calibration_source)
    train_started = time.perf_counter()
    for step in range(n_steps):
        factor = (0.5 * (1.0 + math.cos(math.pi * step / n_steps))
                  if ptq_args.lr_scheduler_type == "cosine" else 1.0)
        optimizer.lr = training_args.learning_rate * factor
        batch = batches[step % len(batches)]
        # Both the synthetic fallback (2-D: (1, seq)) and CustomJsonDataset
        # (1-D block) must end up as (batch, seq) for the embedding lookup.
        inp = np.asarray(batch["input_ids"])
        if inp.ndim == 1:
            inp = inp[None, :]
        lab = np.asarray(batch["labels"])
        if lab.ndim == 1:
            lab = lab[None, :]
        input_ids = jt.array(inp)
        labels = jt.array(lab)
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        # retain_graph=False frees the backward graph immediately (jt.grad
        # defaults to retain_graph=True, which would pin ~90GB and OOM the
        # optimizer's CPU fetch of the tiny R1/R2 gradients).
        grads = jt.grad(loss, trainable_parameters, retain_graph=False)
        loss_val = float(loss.item())
        # Drop the forward graph (logits, saved activations) before the
        # optimizer copies gradients to CPU for the NumPy Cayley update.
        del out, loss, input_ids, labels
        optimizer.step(grads)
        del grads
        if step % ptq_args.logging_steps == 0 or step == n_steps - 1:
            log.info("step %4d / %4d  loss = %.4f  lr = %.8g", step, n_steps, loss_val, optimizer.lr)

    # Save the learned rotations (keys match rotation_utils.rotate_model).
    os.makedirs(model_args.output_rotation_path, exist_ok=True)
    R_dict = {"R1": model.R1.weight.numpy()}
    for i in range(num_layers):
        R2 = model.model.layers[i].self_attn.R2.weight
        R_dict[f"model.layers.{i}.self_attn.R2"] = R2.numpy()
    out_path = os.path.join(model_args.output_rotation_path, "R.npy")
    np.save(out_path, R_dict)
    metadata = {k: v for k, v in vars(ptq_args).items() if k != "access_token"}
    metadata.update(base_weight_dtype=dtype, steps=n_steps, calibration_blocks=len(batches),
                    train_seconds=time.perf_counter() - train_started)
    with open(os.path.join(model_args.output_rotation_path, "training_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved learned rotations -> %s", out_path)


if __name__ == "__main__":
    train()
