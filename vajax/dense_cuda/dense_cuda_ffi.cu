/**
 * GPU-fusible dense LU solve via XLA FFI
 *
 * Single thread-block kernel that performs LU factorization with partial
 * pivoting + forward/backward substitution entirely in shared memory.
 * Registered as kCmdBufferCompatible so XLA can embed it in command buffers
 * inside lax.while_loop without host synchronization.
 *
 * Supports n up to 64 (limited by shared memory: n*n*8 + 3*n*8 + n*4 bytes).
 * For n=46 (ring circuit): ~17KB, well within T4's 48KB default.
 */

#include <cstdint>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Maximum matrix dimension. 64x64 uses ~34KB shared memory.
static constexpr int kMaxN = 64;

//==============================================================================
// CUDA Kernel: shared-memory LU solve with partial pivoting
//==============================================================================

/**
 * Dense LU solve: solves A*x = b in shared memory.
 *
 * Layout: single thread block with n threads (one per row).
 * All data lives in shared memory during factorization and solve.
 *
 * Args:
 *   A_global: n*n matrix (row-major), read-only
 *   b_global: n vector, read-only
 *   x_global: n vector, output (solution)
 *   n: matrix dimension
 */
__global__ void dense_lu_solve_kernel(
    const double* __restrict__ A_global,
    const double* __restrict__ b_global,
    double* __restrict__ x_global,
    int n
) {
    // Shared memory layout:
    //   A[n][n]  (double) - working copy of matrix
    //   b[n]     (double) - working copy of RHS (becomes y after forward sub)
    //   x[n]     (double) - solution vector
    //   piv[n]   (int)    - pivot permutation
    //   max_val  (double) - pivot search scratch (single value)
    //   max_idx  (int)    - pivot search scratch (single value)
    extern __shared__ char smem[];

    double* A = reinterpret_cast<double*>(smem);
    double* b = A + n * n;
    double* x = b + n;
    int* piv = reinterpret_cast<int*>(x + n);

    const int tid = threadIdx.x;

    // Load A and b into shared memory (each thread loads its row)
    if (tid < n) {
        for (int j = 0; j < n; j++) {
            A[tid * n + j] = A_global[tid * n + j];
        }
        // Add Tikhonov regularization to diagonal
        A[tid * n + tid] += 1e-14;
        b[tid] = b_global[tid];
        piv[tid] = tid;
    }
    __syncthreads();

    // === LU factorization with partial pivoting ===
    for (int k = 0; k < n; k++) {
        // Step 1: Find pivot (max |A[i][k]| for i >= k)
        // Thread k does the sequential pivot search. For n<=64 this is fast
        // enough — a parallel reduction would add complexity for no benefit.
        if (tid == 0) {
            double best_val = 0.0;
            int best_idx = k;
            for (int i = k; i < n; i++) {
                double v = A[i * n + k];
                if (v < 0.0) v = -v;
                if (v > best_val) {
                    best_val = v;
                    best_idx = i;
                }
            }
            // Store pivot row index in piv[k] (reuse as scratch)
            // We use piv[n-1] as temporary for the swap index
            // Actually, just use a simple convention: store swap target
            // in A[k*n + k] position temporarily? No, simpler:
            // We'll broadcast via shared memory.
            piv[k] = best_idx;
        }
        __syncthreads();

        // Step 2: Swap rows k and piv[k]
        int pivot_row = piv[k];
        if (pivot_row != k && tid < n) {
            // Each thread swaps one element of rows k and pivot_row
            double tmp = A[k * n + tid];
            A[k * n + tid] = A[pivot_row * n + tid];
            A[pivot_row * n + tid] = tmp;
        }
        // Also swap b entries (single thread)
        if (tid == 0 && pivot_row != k) {
            double tmp = b[k];
            b[k] = b[pivot_row];
            b[pivot_row] = tmp;
        }
        __syncthreads();

        // Step 3: Compute multipliers and eliminate
        // Each thread handles one row below the pivot
        if (tid > k && tid < n) {
            double pivot_val = A[k * n + k];
            // Guard against zero pivot (shouldn't happen with regularization)
            if (pivot_val != 0.0) {
                double mult = A[tid * n + k] / pivot_val;
                A[tid * n + k] = mult;  // Store L factor in-place
                for (int j = k + 1; j < n; j++) {
                    A[tid * n + j] -= mult * A[k * n + j];
                }
                b[tid] -= mult * b[k];
            }
        }
        __syncthreads();
    }

    // === Backward substitution (Ux = b, where b has been modified in-place) ===
    // Sequential from thread 0 (n<=64 iterations, negligible cost)
    if (tid == 0) {
        for (int i = n - 1; i >= 0; i--) {
            double sum = b[i];
            for (int j = i + 1; j < n; j++) {
                sum -= A[i * n + j] * x[j];
            }
            double diag = A[i * n + i];
            x[i] = (diag != 0.0) ? (sum / diag) : 0.0;
        }
    }
    __syncthreads();

    // Write solution to global memory
    if (tid < n) {
        x_global[tid] = x[tid];
    }
}

//==============================================================================
// XLA FFI Handler
//==============================================================================

static constexpr char kAttrN[] = "n";

static ffi::Error DenseLuSolveF64Impl(
    cudaStream_t stream,
    ffi::Attr<int32_t, kAttrN> n_attr,
    ffi::Buffer<ffi::DataType::F64> A,
    ffi::Buffer<ffi::DataType::F64> f,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x
) {
    int32_t n = *n_attr;

    if (n <= 0 || n > kMaxN) {
        return ffi::Error::InvalidArgument(
            "dense_lu_solve: n must be in [1, 64], got " + std::to_string(n));
    }

    const double* A_ptr = A.typed_data();
    const double* f_ptr = f.typed_data();
    double* x_ptr = x->typed_data();

    // Shared memory: A[n*n] + b[n] + x[n] (all double) + piv[n] (int)
    size_t smem_bytes = (n * n + 2 * n) * sizeof(double) + n * sizeof(int);

    // Launch: 1 block, n threads
    dense_lu_solve_kernel<<<1, n, smem_bytes, stream>>>(
        A_ptr, f_ptr, x_ptr, n
    );

    // No cudaStreamSynchronize — the kernel runs on XLA's stream and
    // completes before any dependent thunk executes.

    return ffi::Error::Success();
}

// Register as kCmdBufferCompatible so XLA can capture this into command
// buffers / CUDA graphs without host synchronization.
//
// We expand XLA_FFI_DEFINE_HANDLER_SYMBOL manually because the
// .Attr<int32_t, kAttrN>() template comma confuses the C preprocessor.
extern "C" XLA_FFI_Error* dense_lu_solve_f64(XLA_FFI_CallFrame* call_frame) {
    static auto* handler = ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t, kAttrN>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // A (n*n, row-major)
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // f (n,)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // x (n,)
        .To(DenseLuSolveF64Impl, {ffi::Traits::kCmdBufferCompatible})
        .release();
    return (*handler)(call_frame);
}
