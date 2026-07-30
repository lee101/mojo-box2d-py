from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_BOX2D_LIB", os.path.join(ROOT, "dist", "libmojo-box2d-py.so"))

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mb2_distance": ([I, I, F, F, F, F, F, I, I, F, F, F, F, F, I, I], I),
    "mb2_distance_packed": ([I, I, I, I, I], I),
    "mb2_batch_distance": ([I, I, F, I, I, I, F, I, I, I, I], None),
    "mb2_test_point": ([I, I, I, F, F, F, F, F, F], I),
    "mb2_compute_aabb": ([I, I, F, F, F, F, I], None),
    "mb2_batch_aabb_overlap": ([I, I, I, I], None),
    "mb2_mass": ([I, I, I, F, F, I], None),
    "mb2_ray_cast": ([I, I, I, F, F, F, F, F, F, F, F, F, I], I),
}


class BuildError(RuntimeError):
    pass


def build() -> str:
    if os.path.exists(LIB):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library = None


def _configure(library):
    for name, (argtypes, restype) in _SIGNATURES.items():
        fn = getattr(library, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return library


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = _configure(ctypes.CDLL(build()))
    return _library




def f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if not array.flags.c_contiguous:
        raise ValueError("native buffers must be C-contiguous")
    if array.dtype not in (np.dtype(np.float64), np.dtype(np.int64)):
        raise TypeError("native buffers must contain float64 or int64 values")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise ValueError("native buffer has a null data pointer")
    return address
