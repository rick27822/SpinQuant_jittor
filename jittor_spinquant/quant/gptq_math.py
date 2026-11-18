"""Dependency-light calibration statistics and stable GPTQ factorization.

Hessian accumulation and Cholesky solves run in NumPy/FP64 on CPU. Gram
products, GPTQ weight updates and quantizer operations use Jittor in gptq.py.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class GPTQQuantizationParameters:
    """Compact CPU metadata for already-baked weights, including act-order maps.

    scale/zero have shape (output_channels, quantization_groups), rather than
    allocating two full weight-sized tensors on the GPU.
    """
    bits: int
    sym: bool
    scale: np.ndarray
    zero: np.ndarray
    column_groups: np.ndarray
    gptq_info: dict

    def expanded_parameters(self):
        """Expand only on explicit request, e.g. small-tensor verification."""
        return self.scale[:, self.column_groups], self.zero[:, self.column_groups]


class HessianAccumulator:
    """Accumulate every calibration token, rather than retaining the last input."""

    def __init__(self, columns):
        if columns <= 0:
            raise ValueError("GPTQ input width must be positive.")
        self.columns = columns
        self.gram = np.zeros((columns, columns), dtype=np.float64)
        self.tokens = 0
        self.batches = 0

    def add_gram(self, gram, tokens):
        gram = np.asarray(gram, dtype=np.float64)
        if gram.shape != self.gram.shape or tokens <= 0 or not np.all(np.isfinite(gram)):
            raise ValueError("Invalid GPTQ calibration Gram matrix or token count.")
        self.gram += gram
        self.tokens += int(tokens)
        self.batches += 1

    def add_batch(self, values):
        """NumPy entry for CPU regression tests; Jittor computes its own Gram."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim < 2 or values.shape[-1] != self.columns:
            raise ValueError("GPTQ calibration input has the wrong shape.")
        values = values.reshape(-1, self.columns)
        self.add_gram(values.T @ values, len(values))

    def hessian(self):
        if not self.tokens:
            raise ValueError("GPTQ requires nonempty calibration data.")
        return 2.0 * self.gram / self.tokens


def inverse_hessian_factor(hessian, percdamp=0.01, actorder=False):
    """Return U, permutation and dead columns, with U.T @ U = inv(H_damped).

    If L @ L.T = H, inv(H) is inv(L.T) @ inv(L), in that order. Solving two
    triangular systems avoids the reversed-product bug in the previous port.
    Dead columns are identified in original column order and regularized before
    damping. The permutation describes the processing order, not an output order.
    """
    hessian = np.asarray(hessian, dtype=np.float64).copy()
    if (hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]
            or not hessian.shape[0] or not np.all(np.isfinite(hessian))):
        raise ValueError("GPTQ Hessian must be a finite square matrix.")
    if not np.isfinite(percdamp) or percdamp < 0:
        raise ValueError("GPTQ damping must be finite and nonnegative.")
    hessian = (hessian + hessian.T) * 0.5
    diagonal = np.diag(hessian).copy()
    if np.any(diagonal < 0):
        raise ValueError("A calibration Gram matrix cannot have negative diagonal entries.")
    dead = diagonal == 0
    hessian[dead, :] = 0
    hessian[:, dead] = 0
    hessian[np.diag_indices_from(hessian)] = np.where(dead, 1.0, diagonal)
    permutation = (np.argsort(-np.diag(hessian), kind="stable") if actorder
                   else np.arange(len(hessian)))
    hessian = hessian[np.ix_(permutation, permutation)]
    hessian[np.diag_indices_from(hessian)] += percdamp * np.mean(np.diag(hessian))
    try:
        lower = np.linalg.cholesky(hessian)
        # LAPACK implementations can accept a rank-deficient matrix after a
        # tiny positive roundoff pivot. Reject numerically singular factors
        # consistently instead of propagating enormous inverse entries.
        pivot_floor = np.finfo(np.float64).eps * len(hessian) * np.max(np.diag(hessian))
        if np.any(np.diag(lower) ** 2 <= pivot_floor):
            raise np.linalg.LinAlgError("Numerically singular Cholesky pivot")
        identity = np.eye(len(hessian))
        inverse = np.linalg.solve(lower.T, np.linalg.solve(lower, identity))
        upper = np.linalg.cholesky((inverse + inverse.T) * 0.5).T
    except np.linalg.LinAlgError as error:
        raise ValueError("GPTQ Hessian is not positive definite; use more calibration "
                         "data or increase --percdamp.") from error
    if not np.all(np.isfinite(upper)):
        raise ValueError("GPTQ inverse-Hessian factor contains nonfinite values.")
    return upper, permutation, dead
