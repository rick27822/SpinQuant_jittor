# coding=utf-8
"""Single-device GPTQ fake quantization for the Jittor Llama/Mistral port.

Implements the GPTQ inverse-Hessian error-feedback algorithm (Frantar et al.,
https://arxiv.org/abs/2210.17323), following the sequential projection ordering
used by SpinQuant/QuaRot. Calibration and weight updates use Jittor FP32;
Cholesky solves use NumPy FP64 on CPU for Jittor linalg compatibility.
Outputs are floating-point dequantized weights, not packed INT4 kernels.
"""
import logging

import numpy as np
import jittor as jt
import jittor.nn as nn

from .gptq_math import HessianAccumulator, inverse_hessian_factor, GPTQQuantizationParameters
from .quantization import WeightQuantizer, find_qlayers, QuantizeLinear

log = logging.getLogger("spinquant")


class GPTQ:
    def __init__(self, layer):
        if len(layer.weight.shape) != 2:
            raise ValueError("GPTQ currently supports 2-D linear weights only.")
        self.layer = layer
        self.rows, self.columns = layer.weight.shape
        self.statistics = HessianAccumulator(self.columns)
        self.quantizer = None

    def add_batch(self, inp):
        if len(inp.shape) < 2 or inp.shape[-1] != self.columns:
            raise ValueError("GPTQ calibration input width does not match its linear layer.")
        with jt.no_grad():
            values = inp.float().reshape(-1, self.columns)
            # Synchronize just the Gram matrix; never retain the input graph.
            gram = (values.t() @ values).numpy()
        self.statistics.add_gram(gram, values.shape[0])

    def fasterquant(self, blocksize=128, percdamp=0.01, groupsize=-1,
                    actorder=False, static_groups=False):
        if blocksize <= 0 or (groupsize != -1 and groupsize <= 0):
            raise ValueError("GPTQ blocksize must be positive; groupsize must be -1 or positive.")
        if static_groups and groupsize == -1:
            raise ValueError("Static GPTQ groups require a positive groupsize.")
        quantizer = self.quantizer
        if quantizer is None or not quantizer.configured or not 2 <= quantizer.bits < 16:
            raise ValueError("Configure a low-bit GPTQ WeightQuantizer before quantizing.")
        if not quantizer.perchannel or quantizer.weight_groupsize != -1:
            raise ValueError("GPTQ fits per-channel parameters on each group itself; "
                             "configure WeightQuantizer with weight_groupsize=-1.")
        upper, permutation, dead = inverse_hessian_factor(
            self.statistics.hessian(), percdamp, actorder)
        with jt.no_grad():
            stored_weight = self.layer.weight.numpy()
            original = stored_weight.astype(np.float32)
            if not np.all(np.isfinite(original)):
                raise ValueError("GPTQ received nonfinite weights.")
            original[:, dead] = 0
            n_groups = 1 if groupsize == -1 else (self.columns + groupsize - 1) // groupsize
            scales = np.empty((self.rows, n_groups), dtype=np.float32)
            zeros = np.empty_like(scales)
            column_groups = np.empty(self.columns, dtype=np.int64)
            static_params = []
            if static_groups:
                for start in range(0, self.columns, groupsize):
                    quantizer.find_params(jt.array(original[:, start:start + groupsize]))
                    static_params.append((quantizer.scale.clone(), quantizer.zero.clone()))
                    scales[:, start // groupsize] = quantizer.scale.numpy().reshape(-1)
                    zeros[:, start // groupsize] = quantizer.zero.numpy().reshape(-1)
            weight = jt.array(original[:, permutation])
            inverse = jt.array(upper.astype(np.float32))
            if groupsize == -1:
                quantizer.find_params(weight)
                scales[:, 0] = quantizer.scale.numpy().reshape(-1)
                zeros[:, 0] = quantizer.zero.numpy().reshape(-1)

            for start in range(0, self.columns, blocksize):
                end = min(start + blocksize, self.columns)
                block = weight[:, start:end].clone()
                quantized = jt.zeros_like(block)
                errors = jt.zeros_like(block)
                for local in range(end - start):
                    column = start + local
                    if static_groups:
                        quantizer.scale, quantizer.zero = static_params[permutation[column] // groupsize]
                    elif groupsize > 0 and column % groupsize == 0:
                        group_end = min(column + groupsize, self.columns)
                        current = weight[:, column:group_end].clone()
                        inside = min(group_end, end)
                        current[:, :inside - column] = block[:, local:inside - start]
                        if group_end > end and local:
                            # Earlier errors in this block have not reached the
                            # global tail yet. Include them before fitting a
                            # group that crosses the current block boundary.
                            current[:, end - column:] = (
                                current[:, end - column:]
                                - errors[:, :local] @ inverse[start:column, end:group_end])
                        quantizer.find_params(current)
                        scales[:, column // groupsize] = quantizer.scale.numpy().reshape(-1)
                        zeros[:, column // groupsize] = quantizer.zero.numpy().reshape(-1)
                    vector = block[:, local:local + 1].clone()
                    value, _, _ = quantizer.fake_quantize(vector)
                    quantized[:, local:local + 1] = value
                    column_groups[column] = (0 if groupsize == -1 else
                                             (permutation[column] if static_groups else column) // groupsize)
                    error = (vector - value) / inverse[column, column]
                    block[:, local:] = (block[:, local:]
                                        - error @ inverse[column:column + 1, column:end])
                    errors[:, local:local + 1] = error
                weight[:, start:end] = quantized
                if end < self.columns:
                    weight[:, end:] = weight[:, end:] - errors @ inverse[start:end, end:]

            undo = np.argsort(permutation)
            output = weight.numpy()[:, undo].copy()
            with np.errstate(over="ignore", invalid="ignore"):
                committed = output.astype(stored_weight.dtype)
            if not (np.all(np.isfinite(committed)) and np.all(np.isfinite(scales))
                    and np.all(np.isfinite(zeros)) and np.all(scales > 0)):
                raise ValueError("GPTQ produced nonfinite weights or invalid scales.")
            # Commit only after all numeric checks. Restore original column
            # order for both weights and their saved quantization parameters.
            self.layer.weight.assign(jt.array(committed))
            info = dict(
                tokens=self.statistics.tokens, batches=self.statistics.batches,
                percdamp=percdamp, groupsize=groupsize, blocksize=blocksize,
                act_order=actorder, static_groups=static_groups,
                hessian="2 * sum(X.T @ X) / calibration_tokens",
            )
        return GPTQQuantizationParameters(quantizer.bits, quantizer.sym, scales, zeros,
                                          column_groups[undo].copy(), info)

    def free(self):
        self.statistics = None


class _CapturedInput(Exception):
    """Private sentinel: abort the model forward after capturing layer 0 input."""


def _to_numpy(value):
    if isinstance(value, jt.Var):
        return value.numpy().copy()
    if isinstance(value, tuple):
        return tuple(_to_numpy(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    return value


def _to_jittor(value):
    if isinstance(value, np.ndarray):
        return jt.array(value)
    if isinstance(value, tuple):
        return tuple(_to_jittor(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_jittor(item) for key, item in value.items()}
    return value


def _capture_inputs(model, dataloader, nsamples):
    captured = []

    class Catcher(nn.Module):
        def execute(self, hidden_states, **kwargs):
            captured.append((_to_numpy(hidden_states), _to_numpy(kwargs)))
            raise _CapturedInput()

    layers = model.model.layers
    key = next(iter(layers.keys()))
    first_layer = layers[0]
    layers.add_module(key, Catcher())
    try:
        for index, batch in enumerate(dataloader):
            if index >= nsamples:
                break
            if isinstance(batch, dict):
                inputs = {key: value for key, value in batch.items()
                          if key in ("input_ids", "attention_mask", "position_ids")}
            else:
                ids = batch[0] if isinstance(batch, (tuple, list)) else batch
                inputs = {"input_ids": ids}
            if "input_ids" not in inputs:
                raise ValueError("A GPTQ calibration batch must contain input_ids.")
            ids = inputs["input_ids"]
            if len(ids.shape) == 1:
                ids = ids.reshape(1, -1)
            if len(ids.shape) != 2 or not ids.shape[1]:
                raise ValueError("GPTQ input_ids must be a nonempty (batch, sequence) array.")
            inputs["input_ids"] = ids
            try:
                model(**_to_jittor(inputs))
            except _CapturedInput:
                continue
            raise RuntimeError("The model did not invoke its first decoder layer during calibration.")
    finally:
        layers.add_module(key, first_layer)
    if not captured:
        raise ValueError("GPTQ requires nonempty calibration data; no batches were captured.")
    return captured


def _groups(subset):
    """Replay dependent projections after their predecessors have been quantized."""
    normalized = {name.replace(".module", ""): name for name in subset}
    order = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
             ("self_attn.o_proj",), ("mlp.up_proj", "mlp.gate_proj"), ("mlp.down_proj",))
    used = set()
    for names in order:
        group = [normalized[name] for name in names if name in normalized]
        if group:
            used.update(group)
            yield group
    for name in subset:
        if name not in used:
            yield [name]


def gptq_fwrd(model, dataloader, args):
    """Quantize decoder linears sequentially; preserve masks, RoPE inputs and GQA.

    The first-layer inputs come from the actual model forward. Cached arrays
    live on CPU; only the current projection group's Hessians are retained.
    Supplied random-token data is for code tests, not a formal calibration set.
    """
    if model.R1 is not None:
        raise ValueError("Bake rotations before GPTQ; online trainable R1 is unsupported.")
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False
    quantizers = {}
    try:
        with jt.no_grad():
            cached = _capture_inputs(model, dataloader, args.nsamples)
            for index, layer in enumerate(model.model.layers):
                subset = find_qlayers(layer, layers=[QuantizeLinear])
                if not subset:
                    raise ValueError("No QuantizeLinear projections found in GPTQ decoder layer.")
                for group in _groups(subset):
                    collectors = {}
                    try:
                        for name in group:
                            linear = subset[name]
                            if linear.input_observer is not None:
                                raise ValueError("A GPTQ input observer is already installed.")
                            collector = GPTQ(linear)
                            bits = (8 if args.int8_down_proj and "down_proj" in name else args.w_bits)
                            collector.quantizer = WeightQuantizer()
                            collector.quantizer.configure(bits, perchannel=True,
                                                           sym=not args.w_asym, mse=args.w_clip)
                            collectors[name] = collector
                            linear.input_observer = collector.add_batch
                        for hidden, kwargs in cached:
                            layer(jt.array(hidden), **_to_jittor(kwargs))[0].sync()
                    finally:
                        for name in collectors:
                            subset[name].input_observer = None
                    for name, collector in collectors.items():
                        try:
                            quantizers[f"model.layers.{index}.{name}"] = collector.fasterquant(
                                blocksize=args.gptq_blocksize, percdamp=args.percdamp,
                                groupsize=args.w_groupsize, actorder=args.act_order,
                                static_groups=args.static_groups)
                            log.info("GPTQ layer %d %s: %d batches / %d tokens", index, name,
                                     collector.statistics.batches, collector.statistics.tokens)
                        finally:
                            collector.free()
                    del collectors
                next_inputs = []
                for hidden, kwargs in cached:
                    output = layer(jt.array(hidden), **_to_jittor(kwargs))[0]
                    next_inputs.append((output.numpy(), kwargs))
                cached = next_inputs
    finally:
        model.config.use_cache = use_cache
    return quantizers
