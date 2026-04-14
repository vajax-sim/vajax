"""Dense CUDA LU solver for JAX - GPU-fusible via XLA FFI.

This module provides a dense linear solver (LU with partial pivoting) that runs
entirely in GPU shared memory. It is registered as kCmdBufferCompatible so XLA
can embed it in command buffers inside lax.while_loop without host sync.

Example:
    from vajax.dense_cuda import dense_cuda_jax

    if dense_cuda_jax.is_available():
        x = dense_cuda_jax.solve(J, f)  # GPU-fusible solve
"""

from __future__ import annotations

import jax
import jax.extend.core
from jax.core import ShapedArray
from jax.interpreters import mlir
from jaxtyping import Array

__all__ = ["is_available", "solve"]

# Maximum matrix dimension supported by the CUDA kernel
MAX_N = 64

# =============================================================================
# Check if the CUDA FFI extension is available
# =============================================================================

_DENSE_CUDA_AVAILABLE = False
_dense_cuda_jax_cpp = None

try:
    import dense_cuda_jax_cpp as _dense_cuda_jax_cpp

    # Only available on CUDA backend
    if jax.default_backend() == "gpu":
        _DENSE_CUDA_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if the dense CUDA LU solver is available.

    Returns True only when:
    - The dense_cuda_jax_cpp extension is importable (built with CUDA)
    - JAX's default backend is GPU (CUDA)
    """
    return _DENSE_CUDA_AVAILABLE


# =============================================================================
# JAX Primitive
# =============================================================================

_dense_lu_solve_f64 = jax.extend.core.Primitive("dense_lu_solve_f64")


def solve(A: Array, f: Array) -> Array:
    """Solve A @ x = -f using GPU-fusible dense LU factorization.

    The sign convention matches the NR solver: returns x such that A @ x = -f
    (i.e., delta = -J^{-1} f).

    Args:
        A: Square matrix (n, n), float64
        f: Right-hand side vector (n,), float64

    Returns:
        x: Solution vector (n,), float64, such that A @ x = -f
    """
    n = A.shape[0]
    assert A.shape == (n, n), f"A must be square, got {A.shape}"
    assert f.shape == (n,), f"f must have shape ({n},), got {f.shape}"
    assert n <= MAX_N, f"Matrix too large for shared memory LU: n={n} > {MAX_N}"

    # Negate f so the kernel solves A @ x = (-f), matching the NR convention
    # where linear_solve(J, f) returns delta = -J^{-1} f
    neg_f = -f

    return _dense_lu_solve_f64.bind(A.ravel(), neg_f, n=n)


# =============================================================================
# Primitive implementation
# =============================================================================


@_dense_lu_solve_f64.def_impl
def _dense_lu_solve_f64_impl(A_flat: Array, neg_f: Array, *, n: int) -> Array:
    call = jax.ffi.ffi_call(
        "dense_lu_solve_f64",
        jax.ShapeDtypeStruct((n,), neg_f.dtype),
    )
    return call(A_flat, neg_f, n=n)


# =============================================================================
# Abstract evaluation
# =============================================================================


@_dense_lu_solve_f64.def_abstract_eval
def _dense_lu_solve_f64_abstract(A_flat: ShapedArray, neg_f: ShapedArray, *, n: int) -> ShapedArray:
    return ShapedArray((n,), neg_f.dtype)


# =============================================================================
# FFI target registration + MLIR lowering
# =============================================================================

if _DENSE_CUDA_AVAILABLE:
    jax.ffi.register_ffi_target(
        "dense_lu_solve_f64",
        _dense_cuda_jax_cpp.dense_lu_solve_f64(),
        platform="CUDA",
    )

    _lowering = mlir.lower_fun(_dense_lu_solve_f64_impl, multiple_results=False)
    mlir.register_lowering(_dense_lu_solve_f64, _lowering)
