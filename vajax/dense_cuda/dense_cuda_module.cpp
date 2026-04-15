/**
 * Nanobind module + XLA FFI handler for dense_cuda_jax_cpp
 *
 * The XLA FFI handler and binding live here (compiled by the host C++ compiler)
 * rather than in the .cu file, because nvcc struggles with the complex template
 * metaprogramming in the XLA FFI headers.
 *
 * The actual CUDA kernel launch is in dense_cuda_ffi.cu, exposed via a plain
 * C function (launch_dense_lu_solve).
 */

#include <cstdint>
#include <string>

// cuda_runtime_api.h for cudaStream_t, cudaError_t, cudaSuccess
#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>

#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

// CUDA kernel launcher defined in dense_cuda_ffi.cu
extern "C" int launch_dense_lu_solve(
    void* stream, int32_t n,
    const double* A, const double* f, double* x
);

//==============================================================================
// XLA FFI Handler
//==============================================================================

static ffi::Error DenseLuSolveF64Impl(
    cudaStream_t stream,
    int32_t n,
    ffi::Buffer<ffi::DataType::F64> A,
    ffi::Buffer<ffi::DataType::F64> f,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x
) {

    int err = launch_dense_lu_solve(
        static_cast<void*>(stream), n,
        A.typed_data(), f.typed_data(), x->typed_data()
    );

    if (err != 0) {
        return ffi::Error::Internal(
            "dense_lu_solve CUDA kernel launch failed (error=" +
            std::to_string(err) + ", n=" + std::to_string(n) + ")");
    }

    return ffi::Error::Success();
}

// Register as kCmdBufferCompatible so XLA can capture this into command
// buffers / CUDA graphs without host synchronization.
extern "C" XLA_FFI_Error* dense_lu_solve_f64(XLA_FFI_CallFrame* call_frame) {
    static auto* handler = ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("n")
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // A (n*n, row-major)
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // f (n,)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // x (n,)
        .To(DenseLuSolveF64Impl, {ffi::Traits::kCmdBufferCompatible})
        .release();
    return handler->Call(call_frame);
}

//==============================================================================
// Python module
//==============================================================================

NB_MODULE(dense_cuda_jax_cpp, m) {
    m.doc() = "Dense CUDA LU solver FFI for JAX - GPU-fusible kCmdBufferCompatible";

    m.def("dense_lu_solve_f64", []() {
        return nb::capsule(reinterpret_cast<void*>(dense_lu_solve_f64));
    }, "Get FFI capsule for dense LU solve (float64, CUDA)");
}
