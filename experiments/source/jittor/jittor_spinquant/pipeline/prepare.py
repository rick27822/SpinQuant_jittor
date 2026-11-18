# coding=utf-8
# Jittor port of eval_utils/main.py -- the PTQ preparation pipeline.
#
# `ptq_model` bakes the rotations (if requested), attaches the activation /
# weight quantizers, runs RTN/GPTQ weight quantization and (optionally) K/V
# quantization.  Distributed / FSDP / ExecuTorch code paths are excluded.

import logging

import jittor as jt

from ..quant import quantization as quant_utils
from ..rotation import rotation_utils
from ..math import hadamard as hadamard_utils
from ..quant.rtn import rtn_fwrd
from ..arguments import resolve_quantization_groups
from ..core import fuse_norm as fuse_norm_utils
from ..misc import set_seed, cleanup_memory, get_logger, is_cuda_available

log = get_logger("spinquant")


def ptq_model(args, model, model_args=None, calibration_data=None):
    set_seed(args.seed)
    model.eval()
    resolve_quantization_groups(args, model.config)
    if args.rotate and not args.optimized_rotation_path:
        log.info("Using untrained %s rotations (no learned R.npy supplied).", args.rotate_mode)

    # ------------------------------------------------------------------ #
    # 1. Rotate the weights (SpinQuant offline baking)
    # ------------------------------------------------------------------ #
    if args.rotate:
        log.info("Fusing normalization scales before rotation.")
        fuse_norm_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(
            model, args.rotate_mode, args.optimized_rotation_path
        )
        cleanup_memory()

        quant_utils.add_actquant(model)  # wrap each QuantizeLinear
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        # Even without rotation we still wrap linears so the rest of the code
        # can assume the activation quantizer is present.
        quant_utils.add_actquant(model)

    # ------------------------------------------------------------------ #
    # 2. Weight fake quantization (RTN by default; explicit GPTQ selection)
    # ------------------------------------------------------------------ #
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # load a previously quantized+rotated model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, \
                "Cannot save a quantized model if it is already loaded!"
            log.info("Load quantized model from %s", args.load_qmodel_path)
            save_dict = jt.load(args.load_qmodel_path)
            model.load_parameters(save_dict["model"])

        else:
            if args.w_quant_method == "gptq":
                log.info("Starting GPTQ weight quantization.")
                from ..quant.gptq import gptq_fwrd
                if calibration_data is None:
                    calibration_data = _get_gptq_calibration(args)
                quantizers = gptq_fwrd(model, calibration_data, args)
            else:
                log.info("Starting RTN weight quantization.")
                quantizers = rtn_fwrd(model, args)
            jt.sync_all()
            log.info("Weight quantization complete.")
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            jt.save(save_dict, args.save_qmodel_path)

    # ------------------------------------------------------------------ #
    # 3. Activation / V-cache quantization
    # ------------------------------------------------------------------ #
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(
            model, layers=[quant_utils.ActQuantWrapper]
        )
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = _llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads

            if "v_proj" in name and args.v_bits < 16:  # V-cache precision
                v_groupsize = args.v_groupsize
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name:
                layer_groupsize = head_dim

            if "lm_head" in name:  # skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name:  # down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    # ------------------------------------------------------------------ #
    # 4. K-cache quantization (needs an extra q/k rotation)
    # ------------------------------------------------------------------ #
    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE K-cache quantization is not supported.")
        k_quant_config = {
            "k_bits": args.k_bits,
            "k_groupsize": args.k_groupsize,
            "k_sym": not (args.k_asym),
            "k_clip_ratio": args.k_clip_ratio,
        }
        rotation_utils.add_qk_rotation_wrapper(model, **k_quant_config)

    return model


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _get_gptq_calibration(args):
    """Real WikiText-2 calibration for explicitly requested GPTQ PTQ runs.

    Code tests inject their own tiny arrays into ptq_model and never call this.
    There is no automatic random-token fallback for a real GPTQ invocation.
    """
    from ..io.loader import load_tokenizer
    from ..io.data_utils import get_wikitext2
    tokenizer = load_tokenizer(args.input_model, token=args.access_token,
                               model_max_length=args.model_max_length)
    return get_wikitext2(nsamples=args.nsamples, seed=args.seed,
                        seqlen=args.model_max_length, tokenizer=tokenizer)


def _llama_down_proj_groupsize(model, groupsize):
    assert groupsize > 1, "groupsize should be greater than 1!"
    if model.config.intermediate_size % groupsize == 0:
        log.info("(Act.) Groupsize = Down_proj Groupsize: %d", groupsize)
        return groupsize
    group_num = int(model.config.hidden_size / groupsize)
    assert groupsize * group_num == model.config.hidden_size, \
        "Invalid groupsize for llama!"
    down_proj_groupsize = model.config.intermediate_size // group_num
    assert down_proj_groupsize * group_num == model.config.intermediate_size, \
        "Invalid groupsize for down_proj!"
    log.info("(Act.) Groupsize: %d, Down_proj Groupsize: %d",
             groupsize, down_proj_groupsize)
    return down_proj_groupsize
