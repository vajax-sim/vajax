/**
 * Nanobind module for dense_cuda_jax_cpp
 *
 * Exposes the XLA FFI handler symbol as a PyCapsule for JAX registration.
 * Separated from the CUDA kernel (.cu) because nanobind headers and
 * nvcc don't mix cleanly.
 */

#include <nanobind/nanobind.h>

#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;

// Declared in dense_cuda_ffi.cu, linked via static library
extern "C" XLA_FFI_DECLARE_HANDLER_SYMBOL(dense_lu_solve_f64);

NB_MODULE(dense_cuda_jax_cpp, m) {
    m.doc() = "Dense CUDA LU solver FFI for JAX - GPU-fusible kCmdBufferCompatible";

    m.def("dense_lu_solve_f64", []() {
        return nb::capsule(reinterpret_cast<void*>(dense_lu_solve_f64));
    }, "Get FFI capsule for dense LU solve (float64, CUDA)");
}
