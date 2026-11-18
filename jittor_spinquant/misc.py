# coding=utf-8
# Jittor port of SpinQuant -- misc utilities (logger / seed / memory).
#
# Ported from utils/utils.py (minus all distributed / FSDP helpers, which are
# excluded per the task scope). Jittor uses a global device selected via
# jt.flags.use_cuda, so there is no per-tensor .cuda()/.to(device) here.

import logging
import random
import numpy as np
import jittor as jt

log = logging.getLogger("spinquant")


def get_logger(name: str) -> logging.Logger:
    """Return a console logger named *name* (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        jt.set_seed(seed)
    except Exception:
        # Older / different Jittor builds expose seed differently; ignore.
        pass


def cleanup_memory(verbose: bool = True) -> None:
    """Best-effort memory cleanup. Jittor manages its own memory; we just GC."""
    import gc

    gc.collect()


def is_cuda_available() -> bool:
    try:
        return bool(getattr(jt.flags, "use_cuda", 0))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reductions that behave like torch.min/max(dim=..., keepdims=...) but are safe
# across Jittor versions (some return a tuple, some a bare tensor).
# ---------------------------------------------------------------------------
def _reduce_min(x, dim, keepdims=True):
    r = jt.min(x, dim=dim, keepdims=keepdims)
    return r[0] if isinstance(r, tuple) else r


def _reduce_max(x, dim, keepdims=True):
    r = jt.max(x, dim=dim, keepdims=keepdims)
    return r[0] if isinstance(r, tuple) else r
