"""Dense CUDA LU solver for JAX - GPU-fusible dense linear solve via XLA FFI.

This package provides a shared-memory LU factorization + solve kernel that runs
entirely on the GPU without host synchronization. It is registered as
kCmdBufferCompatible so XLA can embed it in command buffers alongside other
operations inside lax.while_loop, eliminating the ~90ms/iteration host-sync
overhead from cusolver_getrf_ffi + cublas$triangularSolve.

Supports matrices up to n=64 (limited by GPU shared memory).
"""

from vajax.dense_cuda.dense_cuda_jax import is_available, solve

__all__ = ["is_available", "solve"]
