"""Tests for the dense CUDA LU solver (GPU-fusible via XLA FFI).

All tests skip if the CUDA extension is not available (e.g., CPU-only env).
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

# Import availability check; actual solve tested only when extension is built
try:
    from vajax.dense_cuda import dense_cuda_jax

    CUDA_SOLVE_AVAILABLE = dense_cuda_jax.is_available()
except ImportError:
    CUDA_SOLVE_AVAILABLE = False

requires_dense_cuda = pytest.mark.skipif(
    not CUDA_SOLVE_AVAILABLE,
    reason="dense-cuda-jax extension not available (requires CUDA GPU)",
)


@requires_dense_cuda
@pytest.mark.parametrize("n", [4, 8, 16, 32, 46, 64])
def test_random_system(n):
    """Solve random well-conditioned system, compare to NumPy."""
    rng = np.random.default_rng(seed=42 + n)
    A_np = rng.standard_normal((n, n))
    # Make well-conditioned: add n to diagonal
    A_np += n * np.eye(n)
    f_np = rng.standard_normal(n)

    A = jnp.array(A_np)
    f = jnp.array(f_np)

    # dense_cuda_jax.solve returns x such that A @ x = -f
    x = dense_cuda_jax.solve(A, f)
    x_np = np.linalg.solve(A_np, -f_np)

    npt.assert_allclose(np.array(x), x_np, rtol=1e-10, atol=1e-12)


@requires_dense_cuda
def test_identity():
    """Solve with identity matrix: x = -f."""
    n = 10
    A = jnp.eye(n)
    f = jnp.arange(n, dtype=jnp.float64)

    x = dense_cuda_jax.solve(A, f)
    npt.assert_allclose(np.array(x), -np.arange(n, dtype=np.float64), atol=1e-14)


@requires_dense_cuda
def test_near_singular():
    """Solve near-singular system — regularization should prevent NaN."""
    n = 8
    rng = np.random.default_rng(seed=99)
    A_np = rng.standard_normal((n, n))
    # Make nearly singular: set last row to first row * epsilon
    A_np[-1, :] = A_np[0, :] * 1e-15
    f_np = rng.standard_normal(n)

    A = jnp.array(A_np)
    f = jnp.array(f_np)

    x = dense_cuda_jax.solve(A, f)
    # Just check that we get finite values (not NaN/inf)
    assert jnp.all(jnp.isfinite(x)), f"Got non-finite values: {x}"


@requires_dense_cuda
def test_jit_compatible():
    """Verify the solve works inside jax.jit."""
    n = 16
    rng = np.random.default_rng(seed=123)
    A_np = rng.standard_normal((n, n)) + n * np.eye(n)
    f_np = rng.standard_normal(n)

    @jax.jit
    def solve_jit(A, f):
        return dense_cuda_jax.solve(A, f)

    x = solve_jit(jnp.array(A_np), jnp.array(f_np))
    x_np = np.linalg.solve(A_np, -f_np)

    npt.assert_allclose(np.array(x), x_np, rtol=1e-10, atol=1e-12)


@requires_dense_cuda
def test_residual_small():
    """Check that ||A @ x + f|| is small (the NR residual convention)."""
    n = 46  # Ring circuit size
    rng = np.random.default_rng(seed=456)
    A_np = rng.standard_normal((n, n)) + n * np.eye(n)
    f_np = rng.standard_normal(n)

    A = jnp.array(A_np)
    f = jnp.array(f_np)

    x = dense_cuda_jax.solve(A, f)
    residual = A @ x + f  # Should be ~0 since A @ x = -f
    assert jnp.max(jnp.abs(residual)) < 1e-8, f"Residual too large: {jnp.max(jnp.abs(residual))}"


def test_is_available_returns_bool():
    """is_available() should return a bool regardless of environment."""
    from vajax.dense_cuda import dense_cuda_jax

    result = dense_cuda_jax.is_available()
    assert isinstance(result, bool)


def test_max_n_constant():
    """MAX_N should be 64."""
    from vajax.dense_cuda import dense_cuda_jax

    assert dense_cuda_jax.MAX_N == 64
