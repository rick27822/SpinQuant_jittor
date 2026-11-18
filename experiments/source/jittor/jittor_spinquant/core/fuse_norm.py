# coding=utf-8
# Jittor port of utils/fuse_norm_utils.py
#
# Fuses the affine parameters of RMSNorm / LayerNorm into the adjacent linear
# weights (and biases), then resets the norm weights to ones. After this, the
# norms retain RMS normalization; only their learnable scale becomes one.
# The normalization itself must not be removed.
#
# NB: kept in the model's working dtype (fp32) -- Jittor's CUDA fp64 GEMM is
# unreliable on some builds, and SpinQuant fuses norms in fp32/bf16 anyway.
import jittor as jt


def fuse_ln_linear(layernorm, linear_layers):
    """Fuse the affine parameters of RMSNorm / LayerNorm into the adjacent linears."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # W <- W * ln.weight
        W_ = linear.weight
        linear.weight.assign((W_ * layernorm.weight).to(linear_dtype))

        if getattr(layernorm, "bias", None) is not None:
            if linear.bias is None:
                linear.bias = jt.nn.Parameter(
                    jt.zeros((linear.out_features,), dtype=linear_dtype)
                )
            linear.bias.assign(
                (linear.bias + jt.matmul(W_, layernorm.bias)).to(linear_dtype)
            )


def fuse_layer_norms(model):
    # Embedding centering: subtract the mean over the hidden dim.
    W = model.model.embed_tokens
    W_ = W.weight
    W.weight.assign((W_ - W_.mean(dim=-1, keepdims=True)).to(W.weight.dtype))

    layers = [layer for layer in model.model.layers]

    for layer in layers:
        # fuse input_layernorm into q/k/v projections
        fuse_ln_linear(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        # fuse post_attention_layernorm into gate/up projections
        fuse_ln_linear(
            layer.post_attention_layernorm,
            [layer.mlp.up_proj, layer.mlp.gate_proj],
        )

        W_norm = layer.input_layernorm.weight
        layer.input_layernorm.weight.assign(jt.ones_like(W_norm))
        W_norm = layer.post_attention_layernorm.weight
        layer.post_attention_layernorm.weight.assign(jt.ones_like(W_norm))

    # fuse final norm into the lm_head
    fuse_ln_linear(model.model.norm, [model.lm_head])
    W_norm = model.model.norm.weight
    model.model.norm.weight.assign(jt.ones_like(W_norm))
