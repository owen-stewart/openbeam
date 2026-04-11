"""
backend.py — Transparent NumPy / CuPy backend for openbeam
-----------------------------------------------------------
Usage
-----
    from openbeam.backend import set_backend, get_array_module as xp

    set_backend("cuda")   # or "cpu"

    arr = xp().array([1, 2, 3])   # works on whichever backend is active
"""

from __future__ import annotations
import importlib
import warnings
from typing import Literal

_BACKEND: str = "cpu"
_xp = None  # cached module reference


def set_backend(backend: Literal["cpu", "cuda"] = "cpu") -> None:
    """
    Select the compute backend.

    Parameters
    ----------
    backend : {"cpu", "cuda"}
        * ``"cpu"``  — use NumPy (always available, no GPU required).
        * ``"cuda"`` — use CuPy (requires an NVIDIA GPU + CuPy install).
                       Falls back to NumPy with a warning if CuPy is not found.
    """
    global _BACKEND, _xp

    if backend == "cuda":
        try:
            import cupy  # noqa: F401  (import check only)
            _xp = importlib.import_module("cupy")
            _BACKEND = "cuda"
            print(f"[openbeam] Backend: CuPy  (GPU: {_xp.cuda.runtime.getDeviceProperties(0)['name'].decode()})")
        except (ImportError, Exception) as exc:
            warnings.warn(
                f"[openbeam] CuPy not available ({exc}). Falling back to NumPy (CPU).",
                RuntimeWarning,
                stacklevel=2,
            )
            _xp = importlib.import_module("numpy")
            _BACKEND = "cpu"
    else:
        _xp = importlib.import_module("numpy")
        _BACKEND = "cpu"
        print("[openbeam] Backend: NumPy (CPU)")


def get_array_module():
    """Return the active array module (numpy or cupy)."""
    global _xp
    if _xp is None:
        _xp = importlib.import_module("numpy")  # default: numpy
    return _xp


def to_numpy(arr):
    """
    Move an array to CPU NumPy (no-op if already on CPU).
    Useful for plotting / saving results.
    """
    xp = get_array_module()
    if _BACKEND == "cuda":
        return xp.asnumpy(arr)
    return arr


def current_backend() -> str:
    """Return the name of the active backend: ``'cpu'`` or ``'cuda'``."""
    return _BACKEND
