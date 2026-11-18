# coding=utf-8
# Numerical / smoke verification for the Jittor SpinQuant port.
#
# Run on the GPU box, e.g.:
#   python -m jittor_spinquant.tools.verify_math
#
# Checks (no network needed):
#   1. Hadamard transform is orthonormal (H @ H^T == I) for power-of-2 and
#      non-power-of-2 (K-factorized) sizes -> validates hadamard.py.
#   2. Online-rotation forward == offline-baked forward (the core SpinQuant
#      invariant) on a tiny Llama -> validates model.py + rotation_utils.py.
#   3. A single forward/backward through the (online) model runs without error.

import tempfile
from pathlib import Path
import numpy as np

import jittor as jt

jt.flags.use_cuda = 1

from ..math.hadamard import matmul_hadU, get_hadK, random_hadamard_matrix
from ..rotation.rotation_utils import (
    add_rotate_module, rotate_model, get_orthogonal_matrix,
)
from ..core.model import LlamaForCausalLM
from ..arguments import process_args_ptq, resolve_quantization_groups
from ..train.train_rotation import _prepare_for_rotation
from ..pipeline.prepare import ptq_model


def _check_hadamard_orthogonal():
    print("[1] Hadamard orthonormality ...")
    for n in (8, 32, 128, 44 * 128):  # 44*128 = 5632 (TinyLlama non-pow2 path)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, n)).astype(np.float64)
        Hx = matmul_hadU(jt.array(x)).numpy()
        # H normalized -> H @ H^T should be I
        H = matmul_hadU(jt.array(np.eye(n))).numpy()
        err = np.max(np.abs(H @ H.T - np.eye(n)))
        # and the transform must preserve the per-row norm up to float error
        norm_err = np.max(np.abs(np.linalg.norm(Hx, axis=1) -
                                 np.linalg.norm(x, axis=1)))
        print(f"    n={n:5d}  ||H H^T - I||_max = {err:.2e}  "
              f"norm-preserve err = {norm_err:.2e}")
        # The Jittor matmul runs in fp32 on CUDA, so we allow fp32-level error.
        assert err < 1e-4, f"Hadamard not orthonormal for n={n}"
        assert norm_err < 1e-4, f"Hadamard not norm-preserving for n={n}"
    print("    OK")


def _tiny_config():
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        hidden_size=64, num_hidden_layers=2, num_heads=4, num_attention_heads=4,
        num_key_value_heads=4, intermediate_size=128, vocab_size=200,
        rms_norm_eps=1e-6, max_position_embeddings=128, hidden_act="silu",
        head_dim=16, pad_token_id=None, tie_word_embeddings=False,
        attention_bias=False, mlp_bias=False, bos_token_id=1, eos_token_id=2,
    )
    return cfg


def _check_rotation_equivalence():
    print("[2] Online vs offline-baked rotation equivalence ...")
    cfg = _tiny_config()

    # model_a: online rotation (R1/R2 applied inside the forward)
    model_a = LlamaForCausalLM(cfg)
    # model_b: same base weights, then bake R1/R2/R4 into the weights.
    # Copy the *base* weights explicitly (named_parameters -> assign) so the two
    # models are byte-identical apart from the (learnable) R1/R2 modules.  This
    # isolates the rotation-baking invariant from any state_dict/load_parameters
    # quirks.
    base = {
        name: param.numpy()
        for name, param in model_a.named_parameters()
        if "R1" not in name and "R2" not in name
    }
    model_b = LlamaForCausalLM(cfg)
    model_b.load_parameters(base)
    # sanity: confirm the copy actually landed
    copied = {
        name: param.numpy()
        for name, param in model_b.named_parameters()
        if "R1" not in name and "R2" not in name
    }
    max_copy_err = max(
        float(np.max(np.abs(base[k] - copied[k]))) for k in base
    )
    print(f"    base-weight copy max diff = {max_copy_err:.2e}")
    assert max_copy_err < 1e-6, "Base-weight copy failed (models not identical)!"

    # Use the real training/PTQ preparation paths. In particular R4 needs
    # both the baked weight transform and the matching online activation
    # transform; rotate_model alone does not install the latter.
    _, _, args = process_args_ptq(["--input_model", "unused-tiny-model"])
    resolve_quantization_groups(args, cfg)
    _prepare_for_rotation(model_a, args)
    np.random.seed(0)
    add_rotate_module(model_a, rotate_mode="hadamard")
    rotations = {"R1": model_a.R1.weight.numpy()}
    for i, layer in enumerate(model_a.model.layers):
        rotations[f"model.layers.{i}.self_attn.R2"] = layer.self_attn.R2.weight.numpy()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "R.npy"
        np.save(checkpoint, rotations)
        args.rotate = True
        args.optimized_rotation_path = str(checkpoint)
        ptq_model(args, model_b)
    model_a.eval()
    model_b.eval()

    rng = np.random.default_rng(1)
    input_ids = jt.array(rng.integers(0, cfg.vocab_size, size=(2, 32)).astype("int64"))

    out_a = model_a(input_ids)
    out_b = model_b(input_ids)
    la = out_a["logits"].numpy()
    lb = out_b["logits"].numpy()
    err = np.max(np.abs(la - lb))
    print(f"    max |logits_online - logits_baked| = {err:.3e}")
    assert err < 1e-3, "Baked rotation does not match online rotation!"
    print("    OK")


def _check_forward_backward():
    print("[3] Forward + backward (online R1/R2) runs ...")
    cfg = _tiny_config()
    model = LlamaForCausalLM(cfg)
    _, _, args = process_args_ptq([
        "--input_model", "unused-tiny-model", "--w_bits", "4",
        "--a_bits", "4", "--k_bits", "4", "--v_bits", "4",
    ])
    resolve_quantization_groups(args, cfg)
    _prepare_for_rotation(model, args)
    add_rotate_module(model, rotate_mode="hadamard")
    model.train()
    rng = np.random.default_rng(2)
    input_ids = jt.array(rng.integers(0, cfg.vocab_size, size=(2, 24)).astype("int64"))
    out = model(input_ids, labels=input_ids)
    loss = out["loss"]
    grads = jt.grad(loss, [model.R1.weight] +
                    [model.model.layers[i].self_attn.R2.weight
                     for i in range(cfg.num_hidden_layers)], retain_graph=False)
    print(f"    loss = {float(loss.item()):.4f}  #grads = {len(grads)}  "
          f"grad[0] finite = {bool(np.all(np.isfinite(grads[0].numpy())))})")
    gradient_arrays = [gradient.numpy() for gradient in grads]
    assert all(np.all(np.isfinite(gradient)) for gradient in gradient_arrays)
    assert any(np.any(np.abs(gradient) > 1e-10) for gradient in gradient_arrays), \
        "Quantized training produced no rotation gradient"
    print("    OK")


if __name__ == "__main__":
    _check_hadamard_orthogonal()
    _check_rotation_equivalence()
    _check_forward_backward()
    print("\nALL CHECKS PASSED")
