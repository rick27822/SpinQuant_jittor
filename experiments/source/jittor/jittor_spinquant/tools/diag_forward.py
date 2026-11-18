# coding=utf-8
# Diagnostic: is the Jittor Llama forward sane?  Loads TinyLlama in fp32, runs a
# greedy-style read of the top-5 next-token predictions for a known prompt, and
# prints the causal-mask corner to check the mask bug.
import numpy as np
import jittor as jt

jt.flags.use_cuda = 1

from jittor_spinquant.io.loader import load_llama_model, load_tokenizer

MP = "/root/autodl-tmp/SpinQuant_Project/data/models/TinyLlama-1.1B"

print("== loading model (fp32) ==")
model = load_llama_model(MP, dtype="float32")
model.eval()
tok = load_tokenizer(MP)

# 1) causal mask sanity
m = model.model._causal_mask(6).numpy()[0, 0]
print("== causal mask (6x6) ==")
print(m)
print("mask upper-tri should be -inf; got unique:", np.unique(m))

# 2) forward top-5
prompt = "The capital of France is"
ids = jt.array(tok(prompt, return_tensors="np")["input_ids"].astype("int64"))
print("== prompt tokens ==", ids.numpy().tolist())
with jt.no_grad():
    out = model(ids)
logits = out["logits"][0, -1].float().numpy()  # (vocab,)
top = np.argsort(-logits)[:8]
print("== top-8 next-token predictions ==")
for t in top:
    print(f"   {tok.decode([int(t)])!r:20s} logit={float(logits[int(t)]):.3f}")
print("logits stats: mean=%.4f std=%.4f min=%.3f max=%.3f"
      % (logits.mean(), logits.std(), logits.min(), logits.max()))

# 3) quick PPL via two methods on a short held-out-ish slice to cross-check gather
with jt.no_grad():
    text = "The capital of France is Paris. The capital of Germany is Berlin."
    enc = tok(text, return_tensors="np")["input_ids"].astype("int64")
    ids2 = jt.array(enc)
    out2 = model(ids2)
    lg = out2["logits"].float()
    lp = jt.nn.log_softmax(lg[:, :-1, :], dim=-1)
    lab = ids2[:, 1:]
    nll_gather = -jt.gather(lp, 2, lab.unsqueeze(-1)).squeeze(-1)
    ppl_gather = float(jt.exp(nll_gather.mean()).item())
    print("PPL(gather) on short text = %.3f" % ppl_gather)
