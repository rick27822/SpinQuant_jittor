# coding=utf-8
"""RTN weight fake quantization. Low-bit values remain floating-point tensors.

Historical runs used this RTN algorithm with symmetric weights; asymmetric
weights are handled explicitly. GPTQ is implemented separately in gptq.py.
"""
import logging
import jittor.nn as nn
from .quantization import WeightQuantizer, find_qlayers, QuantizeLinear

log = logging.getLogger("spinquant")


def rtn_fwrd(model, args):
    """RTN (round-to-nearest) weight quantization.

    Quantize decoder-layer linear weights; the language-model head is excluded.
    """
    layers = model.model.layers

    quantizers = {}
    for i in range(len(layers)):
        log.info("RTN weight quantization: decoder layer %d/%d.", i + 1, len(layers))
        layer = layers[i]
        subset = find_qlayers(
            layer, layers=[nn.Linear, nn.Embedding, QuantizeLinear]
        )
        for name in subset:
            layer_weight_bits = args.w_bits
            w_groupsize = args.w_groupsize
            if "lm_head" in name:
                layer_weight_bits = 16
                continue
            if getattr(args, "int8_down_proj", False) and "down_proj" in name:
                layer_weight_bits = 8
            quantizer = WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not (args.w_asym),
                mse=args.w_clip, weight_groupsize=w_groupsize,
            )
            W = subset[name].weight
            quantizer.find_params(W)
            q, int_weight, scale = quantizer.fake_quantize(W)
            subset[name].weight.assign(q.cast(W.dtype))
            quantizers["model.layers.%d.%s" % (i, name)] = quantizer
    return quantizers
