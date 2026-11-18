# coding=utf-8
# Jittor port of SpinQuant -- exact & randomized Hadamard transforms.
#
# Ported from utils/hadamard_utils.py. The fast CUDA kernel (fast_hadamard_transform)
# is excluded; instead we implement the *exact* dense/butterfly Hadamard directly in
# Jittor, which is numerically identical (just slower). The 12 precomputed non-power-of-2
# Hadamard matrices (had12/20/28/36/40/44/52/60/108/140/156/172) are stored in
# hadamard_data.npz (extracted from the original repository). For TinyLlama the only
# non-power-of-2 dimension that appears is intermediate_size=5632 -> K=44.

import os
import math
import numpy as np
import jittor as jt

_HAD_DATA = None


def _load_had_data():
    global _HAD_DATA
    if _HAD_DATA is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "hadamard_data.npz"
        )
        with np.load(path) as f:
            _HAD_DATA = {k: f[k] for k in f.files}
    return _HAD_DATA


def is_pow2(n: int) -> bool:
    return (n & (n - 1) == 0) and (n > 0)


def get_hadK(n: int, transpose: bool = False):
    """Return (hadK, K) for the K-factorized Hadamard of order *n*.

    Mirrors the original get_hadK branch ordering exactly.
    """
    data = _load_had_data()
    hadK, K = None, None
    if n % 172 == 0:
        assert is_pow2(n // 172)
        K = 172
        hadK = data["had172"].T if transpose else data["had172"]
    elif n % 156 == 0:
        assert is_pow2(n // 156)
        K = 156
        hadK = data["had156"].T if transpose else data["had156"]
    elif n % 140 == 0:
        assert is_pow2(n // 140)
        K = 140
        hadK = data["had140"].T if transpose else data["had140"]
    elif n % 108 == 0:
        assert is_pow2(n // 108)
        K = 108
        hadK = data["had108"].T if transpose else data["had108"]
    elif n % 60 == 0:
        assert is_pow2(n // 60)
        K = 60
        hadK = data["had60"].T if transpose else data["had60"]
    elif n % 52 == 0:
        assert is_pow2(n // 52)
        K = 52
        hadK = data["had52"].T if transpose else data["had52"]
    elif n % 36 == 0:
        assert is_pow2(n // 36)
        K = 36
        hadK = data["had36"].T if transpose else data["had36"]
    elif n % 28 == 0:
        assert is_pow2(n // 28)
        K = 28
        hadK = data["had28"].T if transpose else data["had28"]
    elif n % 44 == 0:
    # TinyLlama intermediate_size=5632 falls here.
        assert is_pow2(n // 44)
        K = 44
        hadK = data["had44"].T if transpose else data["had44"]
    elif n % 40 == 0:
        assert is_pow2(n // 40)
        K = 40
        hadK = data["had40"].T if transpose else data["had40"]
    elif n % 20 == 0:
        assert is_pow2(n // 20)
        K = 20
        hadK = data["had20"].T if transpose else data["had20"]
    elif n % 12 == 0:
        assert is_pow2(n // 12)
        K = 12
        hadK = data["had12"].T if transpose else data["had12"]
    else:
        assert is_pow2(n), f"Only power-of-2 (or K-factorizable) sizes are supported, got {n}"
        K = 1
    return hadK, K


def matmul_hadU(X, transpose: bool = False):
    """Exact orthonormal Hadamard transform over the last dimension.

    Returns H_n @ X (or H_n^T @ X) divided by sqrt(n), i.e. an orthonormal transform
    (rows/columns have unit norm) -- this matches the original PyTorch implementation
    exactly. Implemented with the butterfly recursion; for non-power-of-2 *n* the
    K-factorized Hadamard is used so that any dimension supported by get_hadK works.
    """
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    # Flatten everything except the last dim into the batch axis.
    B = 1
    for s in X.shape[:-1]:
        B *= s
    x = X.reshape(B, n, 1)
    m = n
    d = 1
    while m > K:
        m = m // 2
        # x is always 3D: (B, m, 2*d) -- use explicit indexing (no Ellipsis).
        x = x.reshape(B, m, 2 * d)
        a = x[:, :, 0:d]
        b = x[:, :, d : 2 * d]
        x = jt.concat([a + b, a - b], dim=2)
        d = 2 * d
    if K > 1:
        # x is (B, K, m) with m = n // K.  Apply hadK (K x K) to every block.
        # We avoid relying on broadcast-batch matmul (Jittor's cublas batched
        # op requires matching batch dims), doing a single 2D matmul instead.
        x = x.transpose(1, 0).reshape(K, B * (n // K))          # (K, B*m)
        hadK_jt = jt.array(hadK).cast(x.dtype)                  # (K, K)
        x = hadK_jt @ x                                         # (K, B*m)
        x = x.reshape(K, B, n // K).transpose(1, 0).reshape(B, n)
    return x.reshape(X.shape) / math.sqrt(n)


def matmul_hadUt(X):
    return matmul_hadU(X, transpose=True)


def matmul_hadU_cuda(X, hadK=None, K=None, transpose: bool = False):
    """Drop-in replacement for the original matmul_hadU_cuda (no custom CUDA kernel)."""
    return matmul_hadU(X, transpose=transpose)


def random_hadamard_matrix(size: int):
    """Randomized Hadamard matrix of order *size* (orthonormal); see QuIP# paper."""
    rng = np.random
    q = (rng.randint(0, 2, size=size).astype(np.float64) * 2 - 1)
    Q = np.diag(q)
    return matmul_hadU(jt.array(Q))


def hadamard_matrix(size: int):
    I = np.eye(size)
    return matmul_hadU(jt.array(I))


def apply_exact_had_to_linear(module, had_dim: int = -1, output: bool = False, R2=None):
    """Bake an exact Hadamard transform into a linear layer's weight.

    Mirrors utils/hadamard_utils.apply_exact_had_to_linear. *module* must expose a
    `.weight` jt.Var of shape (out_features, in_features).
    """
    out_features, in_features = module.weight.shape
    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    # Bake in the model's working dtype (fp32).  Jittor's CUDA fp64 GEMM is
    # unreliable on some builds, and SpinQuant bakes in fp32/bf16.
    W_ = module.weight
    dtype = module.weight.dtype

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.transpose(0, 1), had_K, K).transpose(0, 1)
        else:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K)
    else:
        hadK = hadamard_matrix(had_dim)
        if R2 is not None:
            hadK = R2
        hadK_jt = jt.cast(hadK, dtype)                     # (had_dim, had_dim)
        if output:
            W_ = W_.transpose(0, 1)
            transposed_shape = W_.shape
            temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
            n_blocks = temp.shape[0] * temp.shape[1]
            temp = temp.reshape(n_blocks, had_dim) @ hadK_jt    # 2D matmul
            W_ = temp.reshape(transposed_shape).transpose(0, 1)
        else:
            init_shape = W_.shape
            temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
            n_blocks = temp.shape[0] * temp.shape[1]
            temp = temp.reshape(n_blocks, had_dim) @ hadK_jt    # 2D matmul
            W_ = temp.reshape(init_shape)

    module.weight.assign(W_.cast(dtype))
