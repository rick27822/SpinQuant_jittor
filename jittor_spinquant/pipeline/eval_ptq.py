# coding=utf-8
# Jittor port of ptq.py -- post-training quantization + WikiText-2 PPL.
#
# Single-card only: distributed / FSDP / ExecuTorch are excluded.  The flow is
# load model -> ptq_model (rotate + quantize) -> evaluate perplexity.

from ..arguments import process_args_ptq


def main():
    model_args, training_args, ptq_args = process_args_ptq()
    import jittor as jt
    from ..io.loader import load_llama_model, load_tokenizer
    from .prepare import ptq_model
    from .evaluator import evaluator
    from ..misc import get_logger, set_seed
    log = get_logger("spinquant")
    set_seed(ptq_args.seed)
    jt.flags.use_cuda = 1

    # Historical Jittor evaluation expanded BF16 checkpoints into FP32.
    # Keep that working dtype explicit; this does not enable native BF16 GEMM.
    dtype = "float16" if ptq_args.fp16 else "float32"
    log.info("Evaluation base-weight dtype = %s; weight fake quantization = %s.",
             dtype, ptq_args.w_quant_method)
    log.info("[1/3] Loading model from %s.", model_args.input_model)
    model = load_llama_model(
        model_args.input_model, token=model_args.access_token, dtype=dtype,
    )

    log.info("[1/3] Model loaded. [2/3] Preparing rotation and quantization.")
    with jt.no_grad():
        model = ptq_model(ptq_args, model, model_args)
    jt.sync_all()
    log.info("[2/3] Preparation complete. [3/3] Starting WikiText-2 PPL evaluation.")

    model.seqlen = training_args.model_max_length
    tokenizer = load_tokenizer(
        model_args.input_model, token=model_args.access_token,
        model_max_length=training_args.model_max_length,
    )
    ppl = evaluator(model, tokenizer, ptq_args)
    log.info("Final WikiText2 PPL: %.3f", ppl)


if __name__ == "__main__":
    main()
