# coding=utf-8
# Jittor port of eval_utils/rotation_utils.py
#
# Rotation utilities for SpinQuant:
#   * random_orthogonal_matrix / get_orthogonal_matrix
#       Build the offline rotation matrices: a random orthogonal matrix (via
#       NumPy QR) or a randomized Hadamard matrix.
#   * rotate_embeddings / rotate_attention_inputs / rotate_attention_output /
#     rotate_mlp_input / rotate_mlp_output / rotate_head / rotate_ov_proj /
#     rotate_model
#       Bake R1 (and the R4 Hadamard / R2) into the weights so that an ordinary
#       (non-rotated) forward is mathematically equivalent to the online-rotated
#       forward used during rotation training.
#   * QKRotationWrapper / add_qk_rotation_wrapper
#       K-cache (KV) quantization: apply the Hadamard to q/k and quantize k.
#       In the original PyTorch code this is installed via a forward-method
#       monkeypatch; in Jittor we instead set `self_attn.qk_rotation` and apply
#       it inside LlamaAttention.execute (see model.py).  Functionally identical
#       and avoids fragile bytecode-level monkeypatching.
#   * RotateModule / add_rotate_module
#       Training-side wiring that attaches the learnable R1 / R2 as parameters
#       and returns the trainable parameter list.
#
# ---------------------------------------------------------------------------
# Rotation conventions (MUST match model.py and quant_linear.py exactly)
# ---------------------------------------------------------------------------
#   input-side rotation of a Linear  -> weight @ R1     (QuantizeLinear transpose=False)
#   output-side rotation of a Linear -> R1.T @ weight   (QuantizeLinear transpose=True)
#   embedding / lm_head             -> weight @ R1
#   residual stream at the model    -> rotated by R1 at the input and by R1.T at
#                                     the output (LlamaModel / LlamaForCausalLM).
#
# The offline baking below performs the *absorbing* transform of the online
# rotation, so a weight-baked model run with R1=R2=None produces the same result
# as the online-rotated model.  The matrix multiplications are done in NumPy
# (CPU / float64) -- numerically identical to the original torch.cuda/float64
# code and keeps the one-shot operation out of the Jittor autograd graph.

import math
import logging
import time

import numpy as np
import jittor as jt
import jittor.nn as nn

from ..quant.quantization import ActQuantizer
from ..math import hadamard as hadamard_utils

log = logging.getLogger("spinquant")


# --------------------------------------------------------------------------- #
# Orthogonal / Hadamard matrix generation
# --------------------------------------------------------------------------- #
def random_orthogonal_matrix(size):
    """Random orthogonal matrix of order *size* via NumPy QR (+ sign fix)."""
    random_matrix = np.random.randn(size, size).astype(np.float64)
    q, r = np.linalg.qr(random_matrix)
    q *= np.sign(np.diag(r)).reshape(1, -1)
    return q


def get_orthogonal_matrix(size, mode):
    if mode == "random":
        return random_orthogonal_matrix(size)
    elif mode == "hadamard":
        # random_hadamard_matrix returns a jt.Var; materialize to NumPy.
        return hadamard_utils.random_hadamard_matrix(size).numpy()
    else:
        raise ValueError(f"Unknown mode {mode}")


# --------------------------------------------------------------------------- #
# Offline weight baking
# --------------------------------------------------------------------------- #
def _bake(weight_var, M, side):
    """In-place bake of a rotation matrix *M* into a linear layer's weight/bias.

    weight_var : a jt.Var of shape (out, in) -- or (out,) for a bias.
    M          : rotation matrix, NumPy (n, n) float64 (or a jt.Var, converted).
    side       : "right" -> W' = W @ M    (rotate the *input* side)
                 "left"  -> W' = M.T @ W   (rotate the *output* side)
    """
    if isinstance(M, jt.Var):
        M = M.numpy()
    W = weight_var.numpy().astype(np.float64)
    dtype = weight_var.dtype
    M64 = M.astype(np.float64)
    if side == "right":
        W_new = W @ M64
    else:  # left
        W_new = M64.T @ W
    weight_var.assign(jt.array(W_new).cast(dtype))


def rotate_embeddings(model, R1):
    W = model.model.embed_tokens
    _bake(W.weight, R1, "right")


def rotate_attention_inputs(layer, R1):
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        _bake(W.weight, R1, "right")


def rotate_attention_output(layer, R1):
    W = layer.self_attn.o_proj
    _bake(W.weight, R1, "left")
    if W.bias is not None:
        _bake(W.bias, R1, "left")


def rotate_mlp_input(layer, R1):
    for W in [layer.mlp.up_proj, layer.mlp.gate_proj]:
        _bake(W.weight, R1, "right")


def rotate_mlp_output(layer, R1):
    W = layer.mlp.down_proj
    # Down-proj receives the rotated residual stream (output side) ...
    _bake(W.weight, R1, "left")
    # ... and the R4 Hadamard is baked into its input-side weights.
    hadamard_utils.apply_exact_had_to_linear(W, had_dim=-1, output=False)
    if W.bias is not None:
        _bake(W.bias, R1, "left")


def rotate_head(model, R1):
    W = model.lm_head
    _bake(W.weight, R1, "right")


def rotate_ov_proj(layer, head_num, head_dim, R2=None):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    if R2 is not None and not isinstance(R2, jt.Var):
        R2 = jt.array(np.asarray(R2).astype(np.float64))
    # R2 acts on the head_dim groups of the value/output projections.
    hadamard_utils.apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)
    hadamard_utils.apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2)


def rotate_model(model, rotate_mode="random", optimized_rotation_path=None):
    """Bake R1 and (optionally learned) R2 / R4 into the model weights.

    After this call the model should be run with R1=R2=None (the rotations are
    already absorbed).  *optimized_rotation_path* points to a `.npy` checkpoint
    saved by `save_rotations` (keys "R1" and "model.layers.{i}.self_attn.R2").
    """
    started = time.perf_counter()
    log.info("Rotation fusion started; R1 uses CPU FP64 matrix products. GPU utilization may be low during this stage.")
    R1 = get_orthogonal_matrix(model.config.hidden_size, rotate_mode)
    R_cpk = None
    if optimized_rotation_path is not None:
        log.info("Loading learned R1/R2 from %s.", optimized_rotation_path)
        R_cpk = np.load(optimized_rotation_path, allow_pickle=True).item()
        R1 = R_cpk["R1"]

    config = model.config
    num_heads = config.num_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    log.info("Rotation fusion: embedding weights.")
    rotate_embeddings(model, R1)
    log.info("Rotation fusion: output-head weights.")
    rotate_head(model, R1)

    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(layers):
        log.info("Rotation fusion: decoder layer %d/%d (elapsed %.1fs).", idx + 1, len(layers), time.perf_counter() - started)
        if R_cpk is not None:
            key = f"model.layers.{idx}.self_attn.R2"
            R2 = R_cpk[key]
        else:
            R2 = get_orthogonal_matrix(head_dim, rotate_mode)
        rotate_attention_inputs(layer, R1)
        rotate_attention_output(layer, R1)
        rotate_mlp_input(layer, R1)
        rotate_mlp_output(layer, R1)
        rotate_ov_proj(layer, num_heads, head_dim, R2=R2)
    jt.sync_all()
    log.info("Rotation fusion complete in %.1fs.", time.perf_counter() - started)


# --------------------------------------------------------------------------- #
# K-cache (KV) quantization wrapper
# --------------------------------------------------------------------------- #
class QKRotationWrapper(nn.Module):
    """Hadamard-transform q/k and per-token / per-head quantize k.

    Mirrors eval_utils/rotation_utils.QKRotationWrapper.  The original installs
    this as a monkeypatch around the RoPE call; here it is attached to
    `self_attn.qk_rotation` and invoked inside LlamaAttention.execute.
    """

    def __init__(self, config, k_bits=16, k_groupsize=-1, k_sym=True, k_clip_ratio=1.0):
        super().__init__()
        self.config = config
        num_heads = config.num_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert hadamard_utils.is_pow2(head_dim), \
            "Only power of 2 head_dim is supported for K-cache Quantization!"
        self.k_quantizer = ActQuantizer()
        self.k_bits = k_bits
        self.k_groupsize = k_groupsize
        self.k_sym = k_sym
        self.k_clip_ratio = k_clip_ratio
        self.head_dim = head_dim
        if k_bits < 16:
            self.k_quantizer.configure(
                bits=k_bits,
                groupsize=-1,  # token-wise; head-wise handled in execute()
                sym=k_sym,
                clip_ratio=k_clip_ratio,
            )

    def execute(self, q, k):
        dtype = q.dtype
        # R3: Hadamard-transform q and k after RoPE, then fake-quantize k.
        q = hadamard_utils.matmul_hadU(q.float()).to(dtype)
        k = hadamard_utils.matmul_hadU(k.float()).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            with jt.no_grad():  # scale is a STE constant; keep it out of the graph
                self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .cast(str(q.dtype))
            )
        else:  # head-wise quantization
            per_head_k = k.reshape(-1, head_dim)
            with jt.no_grad():
                self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .cast(str(q.dtype))
            )

        self.k_quantizer.free()
        return q, k


def add_qk_rotation_wrapper(model, k_bits=16, k_groupsize=-1, k_sym=True, k_clip_ratio=1.0):
    """Jittor equivalent of the original (monkeypatch-based) QK rotation hook.

    Attaches a QKRotationWrapper to every attention module.  LlamaAttention.execute
    applies it to (q, k) immediately after RoPE -- functionally identical to the
    original `add_qk_rotation_wrapper_after_function_call_in_forward`.
    """
    config = model.config
    for layer in model.model.layers:
        # NB: LlamaAttention.__init__ always defines `self.qk_rotation = None`,
        # so a plain `hasattr` check would always fail.  Guard against a real
        # double-attachment by checking the value is still None.
        assert getattr(layer.self_attn, "qk_rotation", None) is None, \
            "qk_rotation is already attached to an attention module"
        layer.self_attn.qk_rotation = QKRotationWrapper(
            config,
            k_bits=k_bits,
            k_groupsize=k_groupsize,
            k_sym=k_sym,
            k_clip_ratio=k_clip_ratio,
        )


# Backwards-compatible alias for the original name.
add_qk_rotation_wrapper_after_function_call_in_forward = add_qk_rotation_wrapper


# --------------------------------------------------------------------------- #
# Training-side rotation wiring
# --------------------------------------------------------------------------- #
class RotateModule(nn.Module):
    """Holds a learnable orthogonal matrix (R1 or R2) as a trainable parameter."""

    def __init__(self, R_init):
        super().__init__()
        if isinstance(R_init, jt.Var):
            R_init = R_init.numpy()
        self.weight = nn.Parameter(jt.array(R_init.astype(np.float32)))

    def execute(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        return self.weight @ x


def add_rotate_module(model, rotate_mode="hadamard"):
    """Attach learnable R1 / R2 and return the trainable parameter list.

    Mirrors optimize_rotation.py: all base weights are frozen and only the R1 / R2
    rotations are optimized (with SGDG, see optimizer.py).  Returns the list of
    trainable Vars to hand to the optimizer.
    """
    for param in model.parameters():
        try:
            param.requires_grad_(False)
        except Exception:
            param.requires_grad = False

    R1 = get_orthogonal_matrix(model.config.hidden_size, rotate_mode)
    model.R1 = RotateModule(R1)
    for i in range(model.config.num_hidden_layers):
        R2 = get_orthogonal_matrix(
            model.config.hidden_size // model.config.num_heads, rotate_mode
        )
        model.model.layers[i].self_attn.R2 = RotateModule(R2)

    trainable_parameters = [model.R1.weight] + [
        model.model.layers[i].self_attn.R2.weight
        for i in range(model.config.num_hidden_layers)
    ]
    return trainable_parameters
