# coding=utf-8
# Jittor port of train_utils/optimizer.py (SGDG on the Stiefel manifold).
#
# SGD-G keeps the rotation matrices R1 / R2 orthogonal during rotation training.
# The original implementation lives in torch.optim.Optimizer and operates on
# torch tensors; here we replicate the *algorithm* exactly but perform every
# matrix operation (column normalization, Cayley loop, QR retraction) in NumPy
# (float64, CPU) and write the result back into the Jittor parameter with
# `Var.assign`. This avoids dependence on Jittor's optim API. Different working
# precision and update implementations still require numerical validation;
# algorithm correspondence alone does not establish bitwise equivalence.
#
# Adapted from:
#   https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform

import random

import numpy as np
import jittor as jt

_EPS = 1e-8


def _unit(v, dim=1, eps=_EPS):
    """Column-normalize a 2D matrix (each column to unit L2 norm)."""
    vnorm = np.linalg.norm(v, axis=dim, keepdims=True)
    return v / (vnorm + eps), vnorm


def _matrix_norm_one(W):
    """max over columns of the L1 norm of the columns."""
    return np.max(np.sum(np.abs(W), axis=0))


def _cayley_loop(X, W, tan_vec, t):
    # X: (p, n),  W: (p, p),  tan_vec: (p, n)
    Y = X + t * tan_vec
    for _ in range(5):
        Y = X + t * (W @ (0.5 * (X + Y)))
    return Y.T  # (n, p)


def _qr_retraction(tan_vec):
    # tan_vec: (p, n) with p <= n -> orthonormal (p, n)
    tan_vec = tan_vec.T  # (n, p)
    q, r = np.linalg.qr(tan_vec)
    d = np.diag(r)
    ph = np.sign(d)
    q = q * ph[None, :]
    return q.T  # (p, n)


class SGDG:
    """Stiefel-manifold SGD (SGD-G).

    Mirrors the original `SGDG` optimizer.  `params` is the list of Jittor
    Vars to optimize (R1.weight and the per-layer R2.weight).  Call `step()`
    after populating each parameter's `.grad` (see train_rotation.py).
    """

    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0,
                 nesterov=False, stiefel=False, omega=0, grad_clip=None):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.stiefel = stiefel
        self.omega = omega
        self.grad_clip = grad_clip
        # momentum buffers keyed by id(param): numpy array of shape (p, n)
        self.state = {}

    @staticmethod
    def _as_np(v):
        if isinstance(v, jt.Var):
            return v.numpy()
        return np.asarray(v)

    def step(self, grads=None):
        """Advance one optimization step.

        *grads* (optional) is a list of gradient arrays / jt.Vars aligned with
        `self.params` -- this is how the Jittor training loop feeds gradients
        produced by `jt.grad(loss, params)`.  If *grads* is None we fall back to
        each parameter's `.grad` attribute (torch-style).
        """
        for idx, p in enumerate(self.params):
            if grads is not None:
                g = grads[idx]
            elif p.grad is not None:
                g = p.grad
            else:
                continue
            n = p.shape[0]
            data = self._as_np(p).reshape(n, -1)        # (n, p)
            grad = self._as_np(g).reshape(n, -1)        # (n, p)
            unity, _ = _unit(data)                      # (n, p), column-normalized
            p_dim, n_dim = unity.shape

            if self.stiefel and p_dim <= n_dim:
                # Match upstream: retract before constructing the tangent
                # update, and multiply QR signs along Q's columns.
                if random.randint(1, 101) == 1:
                    unity = _qr_retraction(unity)
                gT = grad.T                              # (p, n)
                key = id(p)
                if key not in self.state:
                    self.state[key] = {"momentum_buffer": np.zeros((n_dim, n))}
                V = self.state[key]["momentum_buffer"]   # (p, n)
                V = self.momentum * V - gT
                MX = V @ unity                           # (p, n) @ (n, p) -> (p, p)
                XMX = unity @ MX                         # (n, p) @ (p, p) -> (n, p)
                XXMX = unity.T @ XMX                     # (p, n) @ (n, p) -> (p, p)
                W_hat = MX - 0.5 * XXMX                  # (p, p)
                W = W_hat - W_hat.T                      # (p, p) skew-symmetric
                t = 1.0 / (_matrix_norm_one(W) + _EPS)
                alpha = min(t, self.lr)

                p_new = _cayley_loop(unity.T, W, V, alpha)   # (n, p)
                V_new = W @ unity.T                         # (p, p) @ (p, n) -> (p, n)
                self.state[key]["momentum_buffer"] = V_new
                p.assign(jt.array(p_new.reshape(p.shape)).cast(p.dtype))
            else:
                d_p = grad.copy()
                if self.weight_decay != 0:
                    d_p = d_p + self.weight_decay * data
                key = id(p)
                if self.momentum != 0:
                    if key not in self.state:
                        self.state[key] = {"momentum_buffer": d_p.copy()}
                    else:
                        buf = self.state[key]["momentum_buffer"]
                        buf = self.momentum * buf + (1 - self.dampening) * d_p
                        self.state[key]["momentum_buffer"] = buf
                        d_p = buf
                    if self.nesterov:
                        d_p = d_p + self.momentum * buf
                p.assign(
                    jt.array((data - self.lr * d_p).reshape(p.shape)).cast(p.dtype)
                )
        return None
