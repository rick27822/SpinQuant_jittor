# coding=utf-8
# Jittor port of the Llama model used by SpinQuant.
#
# A *single* unified LlamaForCausalLM is provided here. It supports both modes
# that SpinQuant needs:
#   * Training (rotation optimization): R1 / R2 are jt.nn.Parameter matrices and
#     are applied ONLINE inside the projections (via QuantizeLinear). The base
#     weights are frozen; only R1 / R2 are trained with the SGDG (Cayley) update.
#   * Evaluation / PTQ: after `rotate_model` bakes R1 / R2 (and the R4 Hadamard)
#     into the weights, the model is run with R1=R2=None. Activation / weight /
#     KV quantization wrappers are then attached around the (already rotated)
#     weights.
#
# Only the eager attention implementation is kept (flash-attn / sdpa / dynamic
# RoPE / gradient-checkpointing / KV-cache generation are dropped, per scope).
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import jittor as jt
import jittor.nn as nn


# NOTE: Jittor 1.3.8.5's `Var.to(dtype)` is unreliable -- passing a dtype string
# or dtype object frequently does NOT cast (it returns the input unchanged).  The
# robust cast path is `jt.cast(x, "float16")`.  We centralise it here so the
# model stays correct in both fp32 and fp16 (real fp16 checkpoints otherwise hit
# "type of two inputs should be the same" GEMM errors).
def _cast(x, dtype_like):
    """Cast *x* to *dtype_like*.

    *dtype_like* may be a Var (use its dtype), a jittor NanoString, or a dtype
    string such as "float16"/"float32".  We route everything through
    ``jt.cast(x, str(...))`` because Jittor 1.3.8.5's ``Var.to("float32")``
    (string form) silently fails to cast.
    """
    if hasattr(dtype_like, "dtype"):  # it's a Var -> use its dtype string
        return jt.cast(x, str(dtype_like.dtype))
    return jt.cast(x, str(dtype_like))  # NanoString or plain string

try:
    from .quant_linear import QuantizeLinear
    from ..math import hadamard as hadamard_utils
except ImportError:
    from jittor_spinquant.core.quant_linear import QuantizeLinear
    from jittor_spinquant.math import hadamard as hadamard_utils


# --------------------------------------------------------------------------- #
# RoPE helpers
# --------------------------------------------------------------------------- #
def rotate_half(x):
    # x is 4D (batch, heads, seq, head_dim); slice the last (head) dimension.
    half = x.shape[-1] // 2
    x1 = x[:, :, :, :half]
    x2 = x[:, :, :, half:]
    return jt.concat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    b, num_kv, slen, hd = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2)
    hidden_states = hidden_states.expand(b, num_kv, n_rep, slen, hd)
    return hidden_states.reshape(b, num_kv * n_rep, slen, hd)


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(jt.ones((hidden_size,)))
        self.variance_epsilon = eps

    def execute(self, hidden_states):
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        variance = x.pow(2).mean(dim=-1, keepdims=True)
        x = x * (1.0 / jt.sqrt(variance + self.variance_epsilon))
        return self.weight * _cast(x, input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    """Pre-computes inv_freq (default RoPE, no scaling) and produces cos/sin."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
        )
        # (dim/2,); stored as a constant (non-trainable). Excluded from state_dict.
        self.inv_freq = jt.array(inv_freq.astype("float32"))
        if hasattr(self.inv_freq, "persistent"):
            self.inv_freq.persistent = False

    def execute(self, x, position_ids):
        # position_ids: (batch, seq). Produces cos/sin of shape (batch, seq, dim)
        # exactly like the HF LlamaRotaryEmbedding reference.
        inv_freq = self.inv_freq  # (dim/2,)
        inv_freq_expanded = inv_freq.reshape((1, -1, 1))  # (1, dim/2, 1)
        pos = position_ids.float()  # (b, seq)
        pos_expanded = pos.reshape((position_ids.shape[0], 1, -1))  # (b, 1, seq)
        freqs = (inv_freq_expanded * pos_expanded).transpose(1, 2)  # (b, seq, dim/2)
        emb = jt.concat((freqs, freqs), dim=-1)  # (b, seq, dim)
        cos = emb.cos()
        sin = emb.sin()
        return _cast(cos, x.dtype), _cast(sin, x.dtype)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = QuantizeLinear(
            self.hidden_size, self.intermediate_size, bias=getattr(config, "mlp_bias", False)
        )
        self.up_proj = QuantizeLinear(
            self.hidden_size, self.intermediate_size, bias=getattr(config, "mlp_bias", False)
        )
        self.down_proj = QuantizeLinear(
            self.intermediate_size, self.hidden_size, bias=getattr(config, "mlp_bias", False)
        )
        self.act_fn = _get_act_fn(config.hidden_act)

    def execute(self, x, R1=None):
        # The MLP intermediate (down_proj input) is Hadamard-transformed (R4).
        # This mirrors rotate_mlp_output baking the exact Hadamard into down_proj;
        # it is applied ONLINE only while rotations are active (R1 is not None),
        # During PTQ (R1=None), ActQuantWrapper applies the online activation
        # transform, paired with the transform baked into down_proj weights.
        h = self.act_fn(self.gate_proj(x, R1)) * self.up_proj(x, R1)
        if R1 is not None:
            h = hadamard_utils.matmul_hadU(h)
        down = self.down_proj(h, R1, transpose=True)
        return down


class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        self.q_proj = QuantizeLinear(
            self.hidden_size, self.num_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.k_proj = QuantizeLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.v_proj = QuantizeLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = QuantizeLinear(
            self.num_heads * self.head_dim, self.hidden_size,
            bias=getattr(config, "attention_bias", False),
        )
        # Per-layer R2 (head_dim x head_dim) rotation, set by add_rotate_module
        # during training; None for the eval / PTQ phase (R2 is baked offline).
        self.R2 = None
        # K-cache (KV) quantization hook, attached by
        # rotation_utils.add_qk_rotation_wrapper during PTQ when k_bits < 16.
        # Applied to (q, k) right after RoPE.  None otherwise.
        self.qk_rotation = None
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def execute(self, hidden_states, attention_mask=None, position_ids=None,
                position_embeddings=None, R1=None, **kwargs):
        bsz, q_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states, R1)
        key = self.k_proj(hidden_states, R1)
        value = self.v_proj(hidden_states, R1, R2=self.R2.weight if self.R2 is not None else None)

        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value, position_ids)
        else:
            cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # K-cache (KV) quantization: Hadamard-transform q/k and quantize k.
        # Attached by rotation_utils.add_qk_rotation_wrapper during PTQ.
        if self.qk_rotation is not None:
            query, key = self.qk_rotation(query, key)

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attn_weights = jt.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + _cast(attention_mask, attn_weights.dtype)
        # upcast to fp32 for the softmax (numerical stability)
        attn_probs = nn.softmax(attn_weights.float(), dim=-1)
        # value may still be in the model's working dtype (e.g. fp16); cast it to
        # match the fp32 probs, then cast the result back so the residual add is
        # dtype-consistent.
        attn_output = jt.matmul(attn_probs, _cast(value, "float32"))
        attn_output = _cast(attn_output, value.dtype)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(
            attn_output, R1, R2=self.R2.weight if self.R2 is not None else None,
            transpose=True,
        )
        return attn_output, None, None


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def execute(self, hidden_states, attention_mask=None, position_ids=None,
                position_embeddings=None, R1=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            R1=R1,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, R1=R1)
        hidden_states = residual + hidden_states
        return (hidden_states,)


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim_of(config))

    @staticmethod
    def head_dim_of(config):
        return getattr(config, "head_dim", config.hidden_size // config.num_heads)

    def execute(self, input_ids=None, attention_mask=None, position_ids=None,
                inputs_embeds=None, R1=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "Exactly one of input_ids or inputs_embeds must be provided."
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Online R1 applied to the embedding stream (training only).
        # NB: rotations are kept in the working dtype (fp32) -- Jittor's CUDA
        # fp64 GEMM is unreliable on some builds, and SpinQuant rotates in
        # fp32/bf16 anyway.
        if R1 is not None:
            # Keep the fp32 rotation parameter but match the base-weight dtype
            # for the GEMM (Jittor rejects mixed-dtype matmul operands).
            R1_emb = (R1 if R1.dtype == inputs_embeds.dtype
                      else R1.to(inputs_embeds.dtype))
            inputs_embeds = inputs_embeds @ R1_emb

        bsz, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            if self.padding_idx is not None:
                # simple monotonic positions (fine for PPL on unpadded batches)
                position_ids = jt.array(np.arange(seq_len)[None, :].astype("int64"))
            else:
                position_ids = jt.array(np.arange(seq_len)[None, :].astype("int64"))
        if attention_mask is None:
            attention_mask = self._causal_mask(seq_len)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                R1=R1,
            )[0]
        hidden_states = self.norm(hidden_states)
        return hidden_states

    @staticmethod
    def _causal_mask(seq_len):
        # (1, 1, seq_len, seq_len) float mask; strictly upper-triangular = -inf.
        # np.triu(np.ones(...), k=1) yields 1.0 strictly above the diagonal; we
        # then set exactly those entries to -inf.  (A previous version used
        # np.full(..., 0.0), whose triu is all-zero -> no masking at all, which
        # silently turned the causal LM bidirectional and wrecked the PPL.)
        m = np.triu(np.ones((seq_len, seq_len), dtype="float32"), k=1)
        m[m == 1.0] = float("-inf")
        m = m.reshape(1, 1, seq_len, seq_len)
        return jt.array(m)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Top-level R1 (hidden x hidden) rotation, set by add_rotate_module.
        self.R1 = None

    def execute(self, input_ids=None, attention_mask=None, position_ids=None,
                inputs_embeds=None, labels=None, R1=None, **kwargs):
        # During training R1 is supplied explicitly; fall back to self.R1 so the
        # training entry point can simply call model(input_ids).
        if R1 is None:
            R1 = self.R1.weight if self.R1 is not None else None

        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            R1=R1,
        )
        # Online R1 applied to the output stream before the head (training only).
        if R1 is not None:
            R1_out = (R1 if R1.dtype == hidden_states.dtype
                      else R1.to(hidden_states.dtype))
            hidden_states = hidden_states @ R1_out.t()

        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].reshape(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].reshape(-1)
            loss = jt.nn.cross_entropy_loss(shift_logits, shift_labels)
        return {"logits": logits, "loss": loss}


# --------------------------------------------------------------------------- #
# Activation helpers
# --------------------------------------------------------------------------- #
_ACT_FNS = {
    "silu": nn.silu,
    "gelu": nn.gelu,
    "relu": nn.relu,
    "gelu_new": nn.gelu,
}


def _get_act_fn(name):
    if name in _ACT_FNS:
        return _ACT_FNS[name]
    # Fallback: identity-ish; TinyLlama uses silu.
    return nn.silu


# --------------------------------------------------------------------------- #
# HF-style checkpoint loading (single-card, mixed loading)
# --------------------------------------------------------------------------- #
@classmethod
def _from_pretrained(cls, pretrained_model_name_or_path, token=None,
                     dtype="float16"):
    """Load a HF Llama checkpoint into this Jittor model.

    Mirrors the original `LlamaForCausalLM.from_pretrained` entry point.  The
    heavy lifting (config + raw weights) lives in loader.py; importing it lazily
    avoids a circular import (loader imports this module).
    """
    from ..io.loader import load_llama_model

    return load_llama_model(
        pretrained_model_name_or_path, token=token, dtype=dtype
    )


LlamaForCausalLM.from_pretrained = _from_pretrained
