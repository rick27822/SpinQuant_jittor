# coding=utf-8
# Jittor port of utils/quant_utils.py
#
# Contains:
#   * get_minq_maxq, (a)sym quant / dequant helpers
#   * STEQuantize / AsymSTEQuantize  -> implemented as plain functions using the
#     straight-through estimator (x + stop_grad(quantized - x))
#   * ActQuantizer         (per-token activation quantization)
#   * ActQuantWrapper      (wraps a Linear, applies optional online Hadamard +
#                           activation quantization before/after the linear)
#   * WeightQuantizer      (RTN weight fake quantization)
#   * add_actquant / find_qlayers
#
# Notes on the Jittor port:
#   * torch.nn.Module.forward  -> jt.nn.Module.execute
#   * register_buffer(...)     -> plain jt.Var attributes (Jittor has no buffers;
#                                non-Parameter Vars are simply stored on the module)
#   * torch.autograd.Function  -> stop_grad-based STE (no custom Function needed)
#   * reduce ops use dim=/keepdims= (e.g. jt.max(x, dim=1, keepdims=True));
#     jt.max / jt.min return the *values* only (not a (values, indices) tuple).
#   * Device handling is dropped (Jittor is single-device per process).
try:
    from ..core.quant_linear import QuantizeLinear
    from ..math import hadamard as hadamard_utils
except ImportError:  # allow running as a standalone module
    from jittor_spinquant.core.quant_linear import QuantizeLinear
    from jittor_spinquant.math import hadamard as hadamard_utils

import math
import jittor as jt
import jittor.nn as nn


# --------------------------------------------------------------------------- #
# min / max q helpers
# --------------------------------------------------------------------------- #
def get_minq_maxq(bits, sym):
    if sym:
        maxq = jt.array(2 ** (bits - 1) - 1)
        minq = -(maxq + 1)
    else:
        maxq = jt.array(2**bits - 1)
        minq = jt.array(0)
    return minq, maxq


def _align_quantizer_dtype(q, x):
    """Keep a quantizer's maxq / zero / scale constants in x's working dtype.

    Jittor's CUDA ops reject mixed fp16/fp32 operands.  Historically this was
    handled by materialising an fp32 copy of every activation and weight just
    for the quantizer, but on a 7B model that adds ~28GB of fp32 copies (one
    full copy of every online-rotated weight) and OOMs a 96GB card.  Aligning
    the small constant buffers instead keeps the whole fake-quant path in the
    base dtype with zero extra copies.  For the normal fp32 path this is a
    no-op, so TinyLlama results are bit-for-bit unaffected.
    """
    if str(x.dtype) == "float32":
        return
    if isinstance(q.maxq, jt.Var) and q.maxq.dtype != x.dtype:
        q.maxq = q.maxq.to(x.dtype)


def asym_quant(x, scale, zero, maxq):
    q = jt.clamp(jt.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    q, scale, zero = asym_quant(x, scale, zero, maxq)
    return asym_dequant(q, scale, zero)


def sym_quant(x, scale, maxq):
    q = jt.clamp(jt.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    q, scale = sym_quant(x, scale, maxq)
    return sym_dequant(q, scale)


# --------------------------------------------------------------------------- #
# Straight-through estimators
# --------------------------------------------------------------------------- #
def sym_quant_ste(x, scale, maxq):
    """Symmetric fake-quantization with a straight-through gradient to x."""
    q = jt.clamp(jt.round(x / scale), -(maxq + 1), maxq)
    # Jittor has no module-level jt.stop_grad; it is a Var method.
    return x + (scale * q - x).stop_grad()


def asym_quant_ste(x, scale, zero, maxq):
    """Asymmetric fake-quantization with a straight-through gradient to x."""
    q = jt.clamp(jt.round(x / scale) + zero, 0, maxq)
    # Jittor has no module-level jt.stop_grad; it is a Var method.
    return x + (scale * (q - zero) - x).stop_grad()


# --------------------------------------------------------------------------- #
# Activation quantizer (per-token)
# --------------------------------------------------------------------------- #
class ActQuantizer(nn.Module):
    """
    Per-token activation quantizer. Supports both symmetric and asymmetric
    quantization, optionally group-wise.
    """

    def __init__(self):
        super().__init__()
        self.maxq = jt.array(0)
        self.scale = jt.array([0.0])
        self.zero = jt.array([0.0])
        self.bits = 16
        self.configured = False

    def free(self):
        self.zero = None
        self.scale = None

    def execute(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.sym:
            return sym_quant_ste(x, self.scale, self.maxq).to(x_dtype)
        return asym_quant_ste(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Returns quantized integers + scales (+ zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(self, bits: int, groupsize: int = -1, sym: bool = False,
                  clip_ratio: float = 1.0):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.configured = True
        assert 0 < self.clip_ratio <= 1, "Clip ratio should be in (0, 1]"

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(
            -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
        )

        # jt.min/jt.max reduce fp16 inputs to fp32 in Jittor 1.3.8.5, so cast
        # the statistics back to the working dtype before combining them with
        # the (dtype-aligned) maxq / scale constants.
        xmax = jt.max(reshaped_x, dim=3, keepdims=True).to(reshaped_x.dtype) * self.clip_ratio
        xmin = jt.min(reshaped_x, dim=3, keepdims=True).to(reshaped_x.dtype) * self.clip_ratio
        if self.sym:
            xmax = jt.maximum(jt.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = jt.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = jt.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return
        _align_quantizer_dtype(self, x)

        init_shape = x.shape

        if self.groupsize > 0:
            self.find_params_per_token_groupwise(x)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        # jt.min/jt.max promote fp16 -> fp32; keep tmp in fp32 for that reduce,
        # then cast the statistic back to the working dtype.
        tmp = jt.zeros((reshaped_x.shape[0],))
        xmin = jt.minimum(jt.min(reshaped_x, dim=1), tmp).to(reshaped_x.dtype) * self.clip_ratio
        xmax = jt.maximum(jt.max(reshaped_x, dim=1), tmp).to(reshaped_x.dtype) * self.clip_ratio
        if self.sym:
            xmax = jt.maximum(jt.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = jt.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = jt.round(-xmin / self.scale)

            self.scale = (
                self.scale.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )
            self.zero = (
                self.zero.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )


# --------------------------------------------------------------------------- #
# Activation quantization wrapper
# --------------------------------------------------------------------------- #
class ActQuantWrapper(nn.Module):
    """
    Wraps a Linear (or QuantizeLinear) to apply (optional) online Hadamard
    rotation and per-token activation quantization before / after the linear.

    PTQ enables online_full_had on down_proj: its input transform cancels the
    matching transform baked into that layer's weights. During rotation
    training LlamaMLP applies this transform instead, so the wrapper stays off.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        # Alias the wrapped module's weight / bias. RTN accesses
        # `layer.weight.data = ...` where `layer` is this wrapper, so the alias
        # must point at the *same* Var object (the Jittor equivalent of the
        # original `self.weight = module.weight`).
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.had_K = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False

    def extra_repr(self):
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
        if self.quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else f" (Symmetric Per-Token)"
            )
        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
        if self.out_quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else f" (Symmetric Per-Token)"
            )
        return str_

    def execute(self, x, R1=None, R2=None, transpose=False):
        x_dtype = x.dtype

        # --- optional online Hadamard rotation ------------------------------
        if self.online_full_had:
            if self.fp32_had:
                x = hadamard_utils.matmul_hadU_cuda(
                    x.float(), self.had_K, self.K
                ).to(x_dtype)
            else:
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            if self.fp32_had:
                x = x.float()
            init_shape = x.shape
            if self.K == 1:
                x = (
                    hadamard_utils.matmul_hadU(
                        x.reshape(
                            -1, init_shape[-1] // self.had_dim, self.had_dim
                        ).transpose(1, 2)
                    )
                ).transpose(1, 2) / math.sqrt(init_shape[-1] // self.had_dim)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)
            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        # --- quantize the input activation, if needed -----------------------
        # Detach find_params: the scale is a constant for the STE, so it must
        # not build / bloat the backward graph.  The quantizer aligns its own
        # constant buffers to x.dtype (see _align_quantizer_dtype), so no fp32
        # copy of x is needed even in the fp16 7B single-card mode.
        if self.quantizer.bits < 16:
            with jt.no_grad():
                self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        # --- the underlying linear (with optional R1/R2 weight rotation) ----
        if R1 is not None:
            x = self.module(x, R1, R2, transpose).to(x_dtype)
        else:
            x = self.module(x).to(x_dtype)

        # --- quantize the output activation, if needed ----------------------
        if self.out_quantizer.bits < 16:
            with jt.no_grad():
                self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


# --------------------------------------------------------------------------- #
# Weight quantizer (RTN)
# --------------------------------------------------------------------------- #
class WeightQuantizer(nn.Module):
    """Per-channel/group weight quantizer used by RTN and rotation training."""

    def __init__(self, shape: int = 1):
        super().__init__()
        self.maxq = jt.array(0)
        self.scale = jt.zeros((shape,))
        self.zero = jt.zeros((shape,))
        self.configured = False

    def configure(self, bits, perchannel: bool = False, sym: bool = True,
                  mse: bool = False, norm: float = 2.4, grid: int = 100,
                  maxshrink: float = 0.8, weight_groupsize: int = -1):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.weight_groupsize = weight_groupsize
        self.configured = True
        if sym:
            self.maxq = jt.array(2 ** (bits - 1) - 1)
        else:
            self.maxq = jt.array(2**bits - 1)

    def find_params_weight_groupwise(self, x):
        _align_quantizer_dtype(self, x)
        init_shape = x.shape
        x = x.reshape(
            x.shape[-2], x.shape[-1] // self.weight_groupsize, self.weight_groupsize
        )

        xmax = jt.max(x, dim=-1, keepdims=True).to(x.dtype)
        xmin = jt.min(x, dim=-1, keepdims=True).to(x.dtype)

        if self.sym:
            xmax = jt.maximum(jt.maximum(jt.abs(xmin), xmax), 1e-5)
            self.scale = xmax / self.maxq
            self.zero = jt.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = jt.maximum(xmax - xmin, 1e-5) / self.maxq
            self.zero = jt.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, self.weight_groupsize)
        self.zero = self.zero.repeat(1, 1, self.weight_groupsize)

        if self.mse:
            # err is fp32 even for fp16 x (jt.pow promotes to fp32), so best
            # must stay fp32 for the err < best comparison.
            best = jt.full((x.shape[0], x.shape[1]), float("inf"))
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = jt.zeros_like(scale1)
                    scale1 = scale1.repeat(1, 1, self.weight_groupsize)
                    zero1 = zero1.repeat(1, 1, self.weight_groupsize)
                    q = sym_quant_dequant(x, scale1, self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = jt.round(-xmin1 / scale1)
                    scale1 = scale1.repeat(1, 1, self.weight_groupsize)
                    zero1 = zero1.repeat(1, 1, self.weight_groupsize)
                    q = asym_quant_dequant(x, scale1, zero1, self.maxq)

                q = q - x
                q = jt.abs(q)
                q = jt.pow(q, self.norm)
                err = jt.sum(q, dim=-1)
                tmp = err < best
                # Update `best` so the search keeps the *error-minimising* scale.
                # (A previous version omitted this, so `best` stayed inf and every
                # iteration overwrote the scale -- ending on the smallest, most
                # aggressive clip and catastrophically clipping the weights.)
                best[tmp] = err[tmp]
                # masked assignment (empty mask -> no-op)
                self.scale[tmp] = scale1[tmp]
                self.zero[tmp] = zero1[tmp]

        self.scale = self.scale.reshape(init_shape)
        self.zero = self.zero.reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return
        _align_quantizer_dtype(self, x)
        orig_shape = x.shape

        if self.weight_groupsize > 0:
            self.find_params_weight_groupwise(x)
            return
        elif self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = jt.zeros((x.shape[0],))
        xmin = jt.minimum(jt.min(x, dim=1), tmp).to(x.dtype)
        xmax = jt.maximum(jt.max(x, dim=1), tmp).to(x.dtype)

        if self.sym:
            xmax = jt.maximum(jt.maximum(jt.abs(xmin), xmax), 1e-5)
            self.scale = xmax / self.maxq
            self.zero = jt.zeros_like(self.scale)
        else:
            tmp2 = (xmin == 0) & (xmax == 0)
            xmin[tmp2] = -1
            xmax[tmp2] = +1
            self.scale = jt.maximum(xmax - xmin, 1e-5) / self.maxq
            self.zero = jt.round(-xmin / self.scale)

        if self.mse:
            # err is fp32 even for fp16 x (jt.pow promotes to fp32).
            best = jt.full((x.shape[0],), float("inf"))
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = jt.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = jt.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                q = q - x
                q = jt.abs(q)
                q = jt.pow(q, self.norm)
                err = jt.sum(q, dim=1)
                tmp3 = err < best
                best[tmp3] = err[tmp3]  # keep the error-minimising scale (see above)
                self.scale[tmp3] = scale1[tmp3]
                self.zero[tmp3] = zero1[tmp3]

        if not self.perchannel:
            self.scale = self.scale.repeat(orig_shape[0])
            self.zero = self.zero.repeat(orig_shape[0])

        shape = [-1] + [1] * (len(orig_shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    def quantize(self, x):
        x_dtype = x.dtype
        if self.configured and self.bits < 16:
            if self.sym:
                return sym_quant_ste(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_ste(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    # Returns (fake-quantized weight, integer-valued floating q, scale).
    def fake_quantize(self, x):
        x_dtype = x.dtype
        if self.configured and self.bits < 16:
            scale = self.scale
            if self.sym:
                q = jt.clamp(jt.round(x / scale), -(self.maxq + 1), self.maxq)
                value = scale * q
            else:
                q = jt.clamp(jt.round(x / scale) + self.zero, 0, self.maxq)
                value = scale * (q - self.zero)
            return value.to(x_dtype), q, scale
        return None, None, None

    def enabled(self):
        return self.bits < 16

# --------------------------------------------------------------------------- #
# Module-tree helpers
# --------------------------------------------------------------------------- #
def _named_children(module):
    """Jittor has no ``named_children``; emulate it from ``named_modules``.

    ``named_modules()`` returns names relative to *module*, so a *direct* child
    is precisely any entry whose (relative) name contains no '.'.
    """
    for n, child in module.named_modules():
        if n == "" or "." in n:
            continue
        yield n, child


def add_actquant(module, name="", layers=None):
    if layers is None:
        layers = [nn.Linear, QuantizeLinear, ActQuantWrapper]

    if isinstance(module, ActQuantWrapper):
        return

    layer_types = tuple(layers)
    for attr in dir(module):
        tmp = getattr(module, attr)
        if isinstance(tmp, layer_types):
            setattr(module, attr, ActQuantWrapper(tmp))
        elif isinstance(tmp, nn.Sequential):
            replaced = []
            for child in tmp.children():
                if isinstance(child, layer_types):
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, nn.Sequential(*replaced))
        elif isinstance(tmp, nn.ModuleList):
            replaced = []
            for child in tmp.children():
                if isinstance(child, layer_types):
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, nn.ModuleList(replaced))

    for name1, child in _named_children(module):
        add_actquant(
            child, name + "." + name1 if name != "" else name1, layers
        )


def find_qlayers(module, layers=None, name=""):
    if layers is None:
        layers = [nn.Linear, ActQuantWrapper, QuantizeLinear]
    layer_types = tuple(layers)

    if nn.Embedding in layers and isinstance(module, nn.Embedding):
        return {"embed_tokens": module}
    if isinstance(module, layer_types):
        return {name: module}

    res = {}
    for name1, child in _named_children(module):
        res.update(
            find_qlayers(
                child, layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res
