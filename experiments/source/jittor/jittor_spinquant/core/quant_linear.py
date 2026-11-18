# coding=utf-8
# Jittor port of train_utils/quant_linear.py
#
# QuantizeLinear: a drop-in replacement for torch.nn.Linear that, in its forward
# (execute) pass, optionally (1) multiplies its weight by the learned rotation
# matrices R1 / R2 (SpinQuant online rotation during the rotation-optimization
# phase) and (2) applies a weight quantizer (during PTQ). When neither R1/R2 nor
# a quantizer is present, it behaves exactly like a plain linear layer.
import jittor as jt
import jittor.nn as nn


class QuantizeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        # Calibration callback consumes every input immediately; no tensor is
        # retained as last_input. GPTQ installs and removes it in try/finally.
        self.input_observer = None

    def execute(self, input, R1=None, R2=None, transpose=False):
        if self.input_observer is not None:
            self.input_observer(input)
        # --- (1) online rotation of the weight by R1 / R2 -------------------
        # Kept in the working dtype (fp32); Jittor's CUDA fp64 GEMM is
        # unreliable on some builds, and SpinQuant rotates in fp32/bf16.
        if R1 is not None:
            weight = self.weight
            # R1/R2 parameters stay fp32 (the Cayley/NumPy optimizer needs that
            # precision), but a Jittor CUDA GEMM requires both operands to have
            # the same dtype.  When the base weights are fp16 (7B single-card
            # mode) cast the rotations for this multiplication only; gradients
            # still flow back to the fp32 parameters through the cast op.
            wdtype = weight.dtype
            R1w = R1 if R1.dtype == wdtype else R1.to(wdtype)
            if not transpose:
                weight = weight @ R1w
            else:
                weight = R1w.t() @ weight
            if R2 is not None:
                R2w = R2 if R2.dtype == wdtype else R2.to(wdtype)
                # Each head dim = R2.shape[0] (head_dim for the value projection)
                had_dim = R2w.shape[0]
                if transpose:
                    W_ = weight
                    init_shape = W_.shape
                    temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
                    temp = (temp @ R2w).reshape(init_shape)
                    weight = temp
                else:
                    W_ = weight.t()
                    transposed_shape = W_.shape
                    temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                    temp = (temp @ R2w).reshape(transposed_shape).t()
                    weight = temp
        else:
            weight = self.weight

        # --- (2) weight quantization (PTQ) ----------------------------------
        if hasattr(self, "quantizer"):
            # find_params must NOT build gradient graph (its 80-iter MSE loop
            # and the scale computation would otherwise OOM the backward and
            # give the scale a spurious gradient).  The original calls
            # find_params on `weight.data` (detached); jt.no_grad() is the
            # equivalent.  The subsequent quantize() applies the STE to the
            # *live* weight.  The quantizer aligns its own constant buffers to
            # the weight dtype (see _align_quantizer_dtype), so no fp32 copy of
            # the rotated weight is needed -- that copy alone cost ~28GB on
            # Mistral-7B and pushed the fp16 run back into a 94GB OOM.
            with jt.no_grad():
                self.quantizer.find_params(weight)
            weight = self.quantizer.quantize(weight)

        # --- linear map ----------------------------------------------------
        y = jt.matmul(input, weight.t())
        if self.bias is not None:
            y = y + self.bias
        return y
