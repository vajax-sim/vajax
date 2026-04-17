#!/usr/bin/env python3
"""Spike: Nested while-inside-while vs flattened single while.

Validates that flattening two nested lax.while_loops into a single while
with jnp.where state-machine logic eliminates the inner kWhile thunk and
kConditional thunk from the XLA thunk tree.

Usage:
    # CPU — check HLO structure
    JAX_PLATFORMS=cpu uv run python scripts/spike_flatten_while.py --dump-hlo

    # GPU — benchmark both styles
    uv run python scripts/spike_flatten_while.py --benchmark --steps 50

    # GPU with HLO dump
    XLA_FLAGS="--xla_dump_to=/tmp/hlo_spike --xla_dump_hlo_as_text=true" \
        uv run python scripts/spike_flatten_while.py --dump-hlo --benchmark
"""

from __future__ import annotations

import argparse
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

# ---------------------------------------------------------------------------
# Ring-like dimensions
# ---------------------------------------------------------------------------
N = 47  # unknowns (ring circuit node count)
MAX_NR = 8  # max NR iterations per step


# ---------------------------------------------------------------------------
# Fake build_system: mimics ring's ~45 kernel ops per NR iteration
# ---------------------------------------------------------------------------
def fake_build_system(X: jax.Array, Q_prev: jax.Array, integ_c0: float) -> tuple:
    """Synthetic build_system producing J, f, Q with ring-like op count.

    Uses scatter, exp, multiply, broadcast, reduce — similar op mix to real
    PSP103 device eval + MNA assembly. NOT numerically meaningful.
    """
    n = X.shape[0]

    # Simulate device evaluation (exp, multiply, select — ~12 fused kernels)
    v1 = jnp.exp(X * 0.01) - 1.0  # diode-like
    v2 = X * 0.5 + v1 * 0.1
    v3 = jnp.where(X > 0, v1, v2)
    v4 = jnp.sqrt(jnp.abs(X) + 1e-12) * 0.3
    v5 = v3 * v4 + jnp.roll(X, 1) * 0.1

    # Simulate Jacobian assembly (scatter, broadcast — ~15 kernels)
    J_diag = 1.0 + jnp.abs(v3) * 0.1
    J = jnp.diag(J_diag)
    # Off-diagonal: tridiagonal coupling
    offdiag = jnp.roll(v5, 1) * 0.01
    J = J + jnp.diag(offdiag[1:], 1) + jnp.diag(offdiag[1:], -1)

    # Simulate residual computation (~8 kernels)
    Q = v2 + v4
    f_cap = integ_c0 * Q + (-integ_c0) * Q_prev
    f = J @ X - v5 + f_cap

    # Simulate reduce operations for convergence metrics (~5 kernels)
    _max_res = jnp.max(jnp.abs(f))
    _sum_q = jnp.sum(jnp.abs(Q))

    return J, f, Q


def fake_linear_solve(J: jax.Array, f: jax.Array) -> jax.Array:
    """Proxy linear solve: J @ f (wrong answer, right shape).

    Avoids cusolver host-sync. For the spike we only care about the
    loop structure in XLA, not numerical correctness.
    """
    # Matrix-vector multiply generates a kKernel, not a kCustomCall
    return J @ (-f) * 0.01  # scale down to keep values stable


# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------
class NestedState(NamedTuple):
    X: jax.Array
    Q_prev: jax.Array
    step: jax.Array
    t_out: jax.Array
    v_out: jax.Array
    total_nr: jax.Array


class NRState(NamedTuple):
    X: jax.Array
    nr_iter: jax.Array
    converged: jax.Array
    Q: jax.Array
    max_f: jax.Array


class FlatState(NamedTuple):
    X: jax.Array
    Q: jax.Array
    Q_prev: jax.Array
    step: jax.Array
    nr_iter: jax.Array
    nr_converged: jax.Array
    t_out: jax.Array
    v_out: jax.Array
    total_nr: jax.Array


# ---------------------------------------------------------------------------
# A) Nested: outer while (steps) containing inner while (NR)
# ---------------------------------------------------------------------------
def make_nested(n_steps: int, dt: float):
    integ_c0 = 1.0 / dt

    def run(init: NestedState) -> NestedState:
        def outer_cond(state: NestedState) -> jax.Array:
            return state.step < n_steps

        def outer_body(state: NestedState) -> NestedState:
            t_next = (jnp.float64(state.step) + 1.0) * dt

            # --- Inner NR loop ---
            nr_init = NRState(
                X=state.X,
                nr_iter=jnp.array(0, jnp.int32),
                converged=jnp.array(False),
                Q=jnp.zeros(N, dtype=state.X.dtype),
                max_f=jnp.array(jnp.inf),
            )

            def nr_cond(nr: NRState) -> jax.Array:
                return ~nr.converged & (nr.nr_iter < MAX_NR)

            def nr_body(nr: NRState) -> NRState:
                J, f, Q = fake_build_system(nr.X, state.Q_prev, integ_c0)
                delta = fake_linear_solve(J, f)
                X_new = nr.X + delta
                max_f = jnp.max(jnp.abs(f))
                converged = max_f < 1e-6
                return NRState(
                    X=X_new,
                    nr_iter=nr.nr_iter + 1,
                    converged=converged,
                    Q=Q,
                    max_f=max_f,
                )

            nr_result = lax.while_loop(nr_cond, nr_body, nr_init)

            # --- Conditional recompute (mimics iter_zero_converged lax.cond) ---
            iter_zero = nr_result.nr_iter == jnp.int32(1)

            def _recompute(_):
                _, _, Q_r = fake_build_system(nr_result.X, state.Q_prev, integ_c0)
                return Q_r

            def _use_existing(_):
                return nr_result.Q

            Q_final = lax.cond(iter_zero, _recompute, _use_existing, operand=None)

            # --- Step advance ---
            new_step = state.step + 1
            new_t_out = state.t_out.at[new_step].set(t_next)
            new_v_out = state.v_out.at[new_step].set(nr_result.X[:N])

            return NestedState(
                X=nr_result.X,
                Q_prev=Q_final,
                step=new_step,
                t_out=new_t_out,
                v_out=new_v_out,
                total_nr=state.total_nr + nr_result.nr_iter,
            )

        return lax.while_loop(outer_cond, outer_body, init)

    return run


# ---------------------------------------------------------------------------
# B) Flattened: single while with jnp.where state machine
# ---------------------------------------------------------------------------
def make_flattened(n_steps: int, dt: float):
    integ_c0 = 1.0 / dt

    def run(init: FlatState) -> FlatState:
        def cond(state: FlatState) -> jax.Array:
            return state.step < n_steps

        def body(state: FlatState) -> FlatState:
            # Phase 1: If NR converged, accept step and advance
            # (nr_iter > 0 avoids triggering on the initial state)
            step_done = state.nr_converged & (state.nr_iter > 0)

            new_step = jnp.where(step_done, state.step + 1, state.step)
            new_nr_iter = jnp.where(step_done, jnp.int32(0), state.nr_iter)
            new_Q_prev = jnp.where(step_done, state.Q, state.Q_prev)
            new_total_nr = jnp.where(step_done, state.total_nr + state.nr_iter, state.total_nr)

            # Store output on step acceptance
            out_idx = new_step  # 1-based: step 0 stored at init, step 1 at idx 1
            t_val = jnp.float64(new_step) * dt
            new_t_out = jnp.where(
                step_done,
                state.t_out.at[out_idx].set(t_val),
                state.t_out,
            )
            new_v_out = jnp.where(
                step_done,
                state.v_out.at[out_idx].set(state.X[:N]),
                state.v_out,
            )

            # Phase 2: One NR iteration (always)
            J, f, Q = fake_build_system(state.X, new_Q_prev, integ_c0)
            delta = fake_linear_solve(J, f)
            X_new = state.X + delta
            max_f = jnp.max(jnp.abs(f))
            converged = max_f < 1e-6

            return FlatState(
                X=X_new,
                Q=Q,
                Q_prev=new_Q_prev,
                step=new_step,
                nr_iter=new_nr_iter + 1,
                nr_converged=converged,
                t_out=new_t_out,
                v_out=new_v_out,
                total_nr=new_total_nr,
            )

        return lax.while_loop(cond, body, init)

    return run


# ---------------------------------------------------------------------------
# HLO analysis helpers
# ---------------------------------------------------------------------------
def count_thunk_kinds(hlo_text: str) -> dict:
    """Count occurrences of key thunk kinds in HLO text."""
    import re

    counts = {}
    for kind in ["while", "conditional", "custom-call"]:
        # Match HLO ops like %while.42, %conditional.25, etc.
        pattern = rf"\b{kind}\b"
        counts[kind] = len(re.findall(pattern, hlo_text, re.IGNORECASE))
    return counts


def analyze_hlo(jitted_fn, *args, label: str = ""):
    """Lower a JIT function and analyze its HLO."""
    lowered = jax.jit(jitted_fn).lower(*args)
    hlo_text = lowered.as_text()

    counts = count_thunk_kinds(hlo_text)
    print(f"\n{'=' * 60}")
    print(f"HLO Analysis: {label}")
    print(f"{'=' * 60}")
    print(f"  while ops:       {counts['while']}")
    print(f"  conditional ops: {counts['conditional']}")
    print(f"  custom-call ops: {counts['custom-call']}")
    print(f"  HLO text size:   {len(hlo_text):,} chars")

    # Check for the key pattern
    if counts["while"] == 1 and counts["conditional"] == 0:
        print(f"  => SINGLE WHILE, NO CONDITIONAL")
    elif counts["while"] >= 2:
        print(f"  => NESTED WHILES ({counts['while']} total)")
    if counts["conditional"] > 0:
        print(f"  => HAS {counts['conditional']} CONDITIONAL(S)")

    return hlo_text


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
def benchmark(fn, init_state, n_warmup: int = 3, n_runs: int = 10, label: str = ""):
    """Benchmark a JIT-compiled function."""
    jitted = jax.jit(fn)

    # Warmup (includes JIT compilation)
    t0 = time.perf_counter()
    result = jitted(init_state)
    # Block until done
    jax.tree.map(lambda x: x.block_until_ready(), result)
    warmup_time = time.perf_counter() - t0
    print(f"\n{label}: warmup (incl JIT) = {warmup_time:.3f}s")

    for _ in range(n_warmup - 1):
        result = jitted(init_state)
        jax.tree.map(lambda x: x.block_until_ready(), result)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = jitted(init_state)
        jax.tree.map(lambda x: x.block_until_ready(), result)
        times.append(time.perf_counter() - t0)

    n_steps = int(result.step if hasattr(result, "step") else result[2])
    avg = sum(times) / len(times)
    ms_per_step = avg / max(n_steps, 1) * 1000

    print(f"{label}: {avg:.3f}s avg over {n_runs} runs")
    print(f"{label}: {ms_per_step:.2f} ms/step ({n_steps} steps)")
    print(f"{label}: total_nr = {int(result.total_nr)}")
    return avg, ms_per_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Spike: nested vs flattened while loop")
    parser.add_argument("--steps", type=int, default=50, help="Number of timesteps")
    parser.add_argument("--dump-hlo", action="store_true", help="Dump and analyze HLO")
    parser.add_argument("--benchmark", action="store_true", help="Run timing benchmark")
    parser.add_argument("--save-hlo", type=str, default=None, help="Save HLO text to files")
    args = parser.parse_args()

    if not args.dump_hlo and not args.benchmark:
        args.dump_hlo = True
        args.benchmark = True

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"N={N}, steps={args.steps}, max_nr={MAX_NR}")

    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    dt = 1e-9
    n_steps = args.steps
    total_out = n_steps + 1

    # --- Build functions ---
    nested_fn = make_nested(n_steps, dt)
    flat_fn = make_flattened(n_steps, dt)

    # --- Initial states ---
    X0 = jnp.ones(N, dtype=dtype) * 0.7

    nested_init = NestedState(
        X=X0,
        Q_prev=jnp.zeros(N, dtype=dtype),
        step=jnp.array(0, jnp.int32),
        t_out=jnp.zeros(total_out, dtype=dtype),
        v_out=jnp.zeros((total_out, N), dtype=dtype),
        total_nr=jnp.array(0, jnp.int32),
    )

    flat_init = FlatState(
        X=X0,
        Q=jnp.zeros(N, dtype=dtype),
        Q_prev=jnp.zeros(N, dtype=dtype),
        step=jnp.array(0, jnp.int32),
        nr_iter=jnp.array(0, jnp.int32),
        nr_converged=jnp.array(False),
        t_out=jnp.zeros(total_out, dtype=dtype),
        v_out=jnp.zeros((total_out, N), dtype=dtype),
        total_nr=jnp.array(0, jnp.int32),
    )

    # --- HLO analysis ---
    if args.dump_hlo:
        hlo_nested = analyze_hlo(nested_fn, nested_init, label="NESTED")
        hlo_flat = analyze_hlo(flat_fn, flat_init, label="FLATTENED")

        if args.save_hlo:
            with open(f"{args.save_hlo}_nested.txt", "w") as f:
                f.write(hlo_nested)
            with open(f"{args.save_hlo}_flat.txt", "w") as f:
                f.write(hlo_flat)
            print(f"\nHLO saved to {args.save_hlo}_nested.txt and {args.save_hlo}_flat.txt")

    # --- Benchmark ---
    if args.benchmark:
        print(f"\n{'=' * 60}")
        print("Benchmark")
        print(f"{'=' * 60}")

        nested_avg, nested_ms = benchmark(
            nested_fn, nested_init, label="NESTED", n_warmup=3, n_runs=5
        )
        flat_avg, flat_ms = benchmark(
            flat_fn, flat_init, label="FLATTENED", n_warmup=3, n_runs=5
        )

        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(f"  NESTED:    {nested_ms:.2f} ms/step")
        print(f"  FLATTENED: {flat_ms:.2f} ms/step")
        if nested_ms > 0:
            speedup = nested_ms / flat_ms
            print(f"  Speedup:   {speedup:.2f}x")


if __name__ == "__main__":
    main()
