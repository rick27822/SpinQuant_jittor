# coding=utf-8
# Jittor port of utils/eval_utils.py -- WikiText-2 perplexity evaluator.
#
# A simplified, memory-friendly port.  Because the A800 has ample memory we run
# the whole model in one pass (no layer-by-layer catcher), which is equivalent
# for PPL but far simpler.  Distributed / FSDP / ExecuTorch paths are excluded.

import logging
import time

import numpy as np
import jittor as jt

from ..io.data_utils import get_wikitext2
from ..misc import get_logger

log = get_logger("spinquant")


@jt.no_grad()
def evaluator(model, tokenizer, args):
    """Compute WikiText-2 perplexity for *model* (already PTQ-prepared)."""
    model.eval()
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    seqlen = args.model_max_length
    testenc = get_wikitext2(
        seed=args.seed, seqlen=seqlen, model=args.input_model,
        tokenizer=tokenizer, eval_mode=True,
    )
    input_ids = jt.array(testenc["input_ids"])           # (1, text_len)
    nsamples = input_ids.shape[1] // seqlen
    if nsamples == 0:
        raise ValueError("Evaluation text is shorter than model_max_length; no complete block to evaluate.")
    input_ids = input_ids[:, : nsamples * seqlen].reshape(nsamples, seqlen)

    batch_size = getattr(args, "bsz", 1)
    input_ids = [input_ids[i:i + batch_size]
                 for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)
    log.info("PPL evaluation: %d blocks, %d batches, sequence length %d.", nsamples, nbatches, seqlen)
    started = time.perf_counter()

    nlls = []
    for batch_index, batch in enumerate(input_ids, 1):
        out = model(batch)
        logits = out["logits"]                              # (b, seqlen, vocab)
        # Unambiguous per-token NLL: log-softmax in fp32 (stable) then gather the
        # ground-truth token's log-prob.  This avoids any ambiguity in Jittor's
        # cross_entropy_loss `reduction=` behaviour and matches the canonical PPL
        # definition exactly (mean negative log-likelihood over all tokens).
        log_probs = jt.nn.log_softmax(logits[:, :-1, :].float(), dim=-1)
        labels = batch[:, 1:]                               # (b, seqlen-1)
        # Per-token NLL: pick the ground-truth token's log-prob via gather
        # (Jittor 1.3.8.5 has no one_hot; gather is the exact equivalent).
        nll = -jt.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)  # (b, seqlen-1)
        # average per-token NLL per sample
        per_sample = nll.mean(dim=1)
        nlls.append(per_sample)
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == nbatches:
            jt.sync_all()
            log.info("PPL progress: %d/%d batches (elapsed %.1fs).", batch_index, nbatches, time.perf_counter() - started)

    nlls = jt.concat(nlls)
    ppl = jt.exp(nlls.mean()).item()
    model.config.use_cache = use_cache
    log.info("WikiText2 PPL: %.3f", ppl)
    return ppl
