#ifndef AMIGO_CUDA_VECTOR_BACKEND_H
#define AMIGO_CUDA_VECTOR_BACKEND_H

#include <cuda_runtime.h>

#include <limits>
#include <type_traits>

#include "amigo.h"
#include "cuda/csr_matrix_backend.cuh"

namespace amigo {

namespace detail {

// Threads-per-block for vector backend kernels.
static constexpr int VEC_TPB = 256;

// Branch-free absolute value usable for any signed numeric type.
template <typename T>
AMIGO_DEVICE inline T abs_value(T x) {
  return x < T(0) ? -x : x;
}

// -------------------------------------------------------------------------
// cuBLAS adapters
// -------------------------------------------------------------------------

template <typename T>
struct CublasVecOps;  // primary template left undefined on purpose

template <>
struct CublasVecOps<float> {
  static cublasStatus_t dot(cublasHandle_t h, int n, const float* x, int incx,
                            const float* y, int incy, float* result) {
    return cublasSdot(h, n, x, incx, y, incy, result);
  }

  static cublasStatus_t axpy(cublasHandle_t h, int n, const float* alpha,
                             const float* x, int incx, float* y, int incy) {
    return cublasSaxpy(h, n, alpha, x, incx, y, incy);
  }

  static cublasStatus_t scal(cublasHandle_t h, int n, const float* alpha,
                             float* x, int incx) {
    return cublasSscal(h, n, alpha, x, incx);
  }
};

template <>
struct CublasVecOps<double> {
  static cublasStatus_t dot(cublasHandle_t h, int n, const double* x, int incx,
                            const double* y, int incy, double* result) {
    return cublasDdot(h, n, x, incx, y, incy, result);
  }

  static cublasStatus_t axpy(cublasHandle_t h, int n, const double* alpha,
                             const double* x, int incx, double* y, int incy) {
    return cublasDaxpy(h, n, alpha, x, incx, y, incy);
  }

  static cublasStatus_t scal(cublasHandle_t h, int n, const double* alpha,
                             double* x, int incx) {
    return cublasDscal(h, n, alpha, x, incx);
  }
};

// -------------------------------------------------------------------------
// Element-wise kernels
// -------------------------------------------------------------------------

template <typename T>
AMIGO_KERNEL void vec_fill_kernel(int n, T value, T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_ptr[i] = value;
  }
}

template <typename T>
AMIGO_KERNEL void vec_add_scalar_kernel(int n, T value,
                                        T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_ptr[i] += value;
  }
}

// d_ptr[d_idx[i]] = d_src[d_idx[i]] -- full-size source.
template <typename T>
AMIGO_KERNEL void vec_copy_at_kernel(int n, const int* __restrict__ d_idx,
                                     const T* __restrict__ d_src,
                                     T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int idx = d_idx[i];
    d_ptr[idx] = d_src[idx];
  }
}

template <typename T>
AMIGO_KERNEL void vec_fill_at_kernel(int n, const int* __restrict__ d_idx,
                                     T value, T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_ptr[d_idx[i]] = value;
  }
}

template <typename T>
AMIGO_KERNEL void vec_add_scalar_at_kernel(int n,
                                           const int* __restrict__ d_idx,
                                           T scalar, T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_ptr[d_idx[i]] += scalar;
  }
}

template <typename T>
AMIGO_KERNEL void vec_scale_at_kernel(int n, const int* __restrict__ d_idx,
                                      T alpha, T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_ptr[d_idx[i]] *= alpha;
  }
}

// d_ptr[d_idx[i]] += alpha * d_x[d_idx[i]] -- full-size source.
template <typename T>
AMIGO_KERNEL void vec_axpy_at_kernel(int n, const int* __restrict__ d_idx,
                                     T alpha, const T* __restrict__ d_x,
                                     T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int idx = d_idx[i];
    d_ptr[idx] += alpha * d_x[idx];
  }
}

// d_vals[i] = d_ptr[d_idx[i]] -- compact destination.
template <typename T>
AMIGO_KERNEL void vec_get_values_at_kernel(int n,
                                           const int* __restrict__ d_idx,
                                           const T* __restrict__ d_ptr,
                                           T* __restrict__ d_vals) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_vals[i] = d_ptr[d_idx[i]];
  }
}

// d_ptr[d_idx[i]] = d_vals[i] -- compact source.
template <typename T>
AMIGO_KERNEL void vec_set_values_at_kernel(int n,
                                           const int* __restrict__ d_idx,
                                           const T* __restrict__ d_vals,
                                           T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_ptr[d_idx[i]] = d_vals[i];
  }
}

// -------------------------------------------------------------------------
// Reduction kernels  (single-block, strided loop, shared-memory tree)
// -------------------------------------------------------------------------

// Argmax of |d_ptr[i]| over i in [0, n).  Writes max value and the
// corresponding index into the device-side scalars d_max / d_idx_out.
template <typename T>
AMIGO_KERNEL void vec_maxabs_kernel(int n, const T* __restrict__ d_ptr,
                                    T* d_max, int* d_idx_out) {
  extern __shared__ unsigned char smem[];
  T* s_val = reinterpret_cast<T*>(smem);
  int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

  int tid = threadIdx.x;
  int stride = blockDim.x;

  T local_val = T(0);
  int local_idx = -1;

  for (int i = tid; i < n; i += stride) {
    T v = abs_value(d_ptr[i]);
    if (v > local_val) {
      local_val = v;
      local_idx = i;
    }
  }

  s_val[tid] = local_val;
  s_idx[tid] = local_idx;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_val[tid + offset] > s_val[tid]) {
        s_val[tid] = s_val[tid + offset];
        s_idx[tid] = s_idx[tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *d_max = s_val[0];
    *d_idx_out = s_idx[0];
  }
}

// Sum of |d_ptr[i]| over i in [0, n).
template <typename T>
AMIGO_KERNEL void vec_abssum_kernel(int n, const T* __restrict__ d_ptr,
                                    T* d_out) {
  extern __shared__ unsigned char smem[];
  T* s = reinterpret_cast<T*>(smem);

  int tid = threadIdx.x;
  int stride = blockDim.x;

  T local = T(0);
  for (int i = tid; i < n; i += stride) {
    local += abs_value(d_ptr[i]);
  }

  s[tid] = local;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s[tid] += s[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *d_out = s[0];
  }
}

}  // namespace detail

// =========================================================================
// CudaVecBackend
// =========================================================================

template <typename T>
class CudaVecBackend {
 public:
  CudaVecBackend() : size(0), d_ptr(nullptr), handle(nullptr) {
    AMIGO_CHECK_CUBLAS(cublasCreate(&handle));
  }
  ~CudaVecBackend() {
    if (d_ptr) {
      cudaFree(d_ptr);
    }
    if (handle) {
      cublasDestroy(handle);
    }
  }

  void allocate(int size_) {
    if (d_ptr) {
      cudaFree(d_ptr);
    }
    size = size_;
    AMIGO_CHECK_CUDA(cudaMalloc(&d_ptr, size * sizeof(T)));
  }

  void copy_host_to_device(const T* h_ptr) {
    AMIGO_CHECK_CUDA(
        cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copy_device_to_host(T* h_ptr) {
    AMIGO_CHECK_CUDA(
        cudaMemcpy(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  void copy(const T* d_src) {
    AMIGO_CHECK_CUDA(
        cudaMemcpy(d_ptr, d_src, size * sizeof(T), cudaMemcpyDeviceToDevice));
  }

  void zero() { AMIGO_CHECK_CUDA(cudaMemset(d_ptr, 0, size * sizeof(T))); }

  // d_ptr[i] = scalar  for i in [0, size)
  void fill(T scalar) {
    if (size <= 0) {
      return;
    }
    int grid = (size + detail::VEC_TPB - 1) / detail::VEC_TPB;
    detail::vec_fill_kernel<T><<<grid, detail::VEC_TPB>>>(size, scalar, d_ptr);
  }

  // d_ptr[i] += scalar  for i in [0, size)
  void add_scalar(T scalar) {
    if (size <= 0) {
      return;
    }
    int grid = (size + detail::VEC_TPB - 1) / detail::VEC_TPB;
    detail::vec_add_scalar_kernel<T>
        <<<grid, detail::VEC_TPB>>>(size, scalar, d_ptr);
  }

  void scale(T alpha) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
      AMIGO_CHECK_CUBLAS(
          detail::CublasVecOps<T>::scal(handle, size, &alpha, d_ptr, 1));
    }
  }

  void axpy(T alpha, const T* d_x) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
      AMIGO_CHECK_CUBLAS(detail::CublasVecOps<T>::axpy(handle, size, &alpha,
                                                       d_x, 1, d_ptr, 1));
    }
  }

  T dot(const T* d_src) const {
    T result{};
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
      AMIGO_CHECK_CUBLAS(detail::CublasVecOps<T>::dot(handle, size, d_ptr, 1,
                                                      d_src, 1, &result));
    }
    return result;  // host scalar
  }

  // Single-block argmax reduction over |d_ptr[i]|.
  T maxabs(int& index) {
    index = -1;
    if (size <= 0) {
      return T(0);
    }

    T* d_max;
    int* d_idx_out;
    AMIGO_CHECK_CUDA(cudaMalloc(&d_max, sizeof(T)));
    AMIGO_CHECK_CUDA(cudaMalloc(&d_idx_out, sizeof(int)));

    constexpr int block_size = detail::VEC_TPB;
    int grid_size = 1;
    size_t shmem = block_size * (sizeof(T) + sizeof(int));
    detail::vec_maxabs_kernel<T><<<grid_size, block_size, shmem>>>(
        size, d_ptr, d_max, d_idx_out);

    T host_max = T(0);
    AMIGO_CHECK_CUDA(
        cudaMemcpy(&host_max, d_max, sizeof(T), cudaMemcpyDeviceToHost));
    AMIGO_CHECK_CUDA(
        cudaMemcpy(&index, d_idx_out, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_max);
    cudaFree(d_idx_out);
    return host_max;
  }

  // Single-block sum reduction over |d_ptr[i]|.
  T abssum() {
    if (size <= 0) {
      return T(0);
    }

    T* d_out;
    AMIGO_CHECK_CUDA(cudaMalloc(&d_out, sizeof(T)));

    constexpr int block_size = detail::VEC_TPB;
    int grid_size = 1;
    size_t shmem = block_size * sizeof(T);
    detail::vec_abssum_kernel<T>
        <<<grid_size, block_size, shmem>>>(size, d_ptr, d_out);

    T host_out = T(0);
    AMIGO_CHECK_CUDA(
        cudaMemcpy(&host_out, d_out, sizeof(T), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    return host_out;
  }

  // d_ptr[d_idx[i]] = d_src[d_idx[i]]  for i in [0, n)
  void copy_at(int n, const int d_idx[], const T d_src[]) {
    if (n <= 0) {
      return;
    }
    int grid = (n + detail::VEC_TPB - 1) / detail::VEC_TPB;
    detail::vec_copy_at_kernel<T>
        <<<grid, detail::VEC_TPB>>>(n, d_idx, d_src, d_ptr);
  }

  // d_ptr[d_idx[i]] = value
  void fill_at(int n, const int d_idx[], T value) {
    if (n <= 0) {
      return;
    }
    int grid = (n + detail::VEC_TPB - 1) / detail::VEC_TPB;
    detail::vec_fill_at_kernel<T>
        <<<grid, detail::VEC_TPB>>>(n, d_idx, value, d_ptr);
  }

  // d_ptr[d_idx[i]] += scalar
  void add_scalar_at(int n, const int d_idx[], T scalar) {
    if (n <= 0) {
      return;
    }
    int grid = (n + detail::VEC_TPB - 1) / detail::VEC_TPB;
    detail::vec_add_scalar_at_kernel<T>
        <<<grid, detail::VEC_TPB>>>(n, d_idx, scalar, d_ptr);
  }

  // d_ptr[d_idx[i]] *= scalar
  void scale_at(int n, const int d_idx[], T scalar) {
    if (n <= 0) {
      return;
    }
    int grid = (n + detail::VEC_TPB - 1) / detail::VEC_TPB;
    detail::vec_scale_at_kernel<T>
        <<<grid, detail::VEC_TPB>>>(n, d_idx, scalar, d_ptr);
  }

  // d_ptr[d_idx[i]] += alpha * d_x[d_idx[i]]  -- full-size source.
  void axpy_at(int n, const int d_idx[], T alpha, const T d_x[]) {
    if (n <= 0) {
      return;
    }
    int grid = (n + detail::VEC_TPB - 1) / detail::VEC_TPB;
    detail::vec_axpy_at_kernel<T>
        <<<grid, detail::VEC_TPB>>>(n, d_idx, alpha, d_x, d_ptr);
  }

  // d_vals[i] = d_ptr[d_idx[i]]  -- compact destination.
  void get_values_at(int n, const int d_idx[], T d_vals[]) {
    if (n <= 0) {
      return;
    }
    int grid = (n + detail::VEC_TPB - 1) / detail::VEC_TPB;
    detail::vec_get_values_at_kernel<T>
        <<<grid, detail::VEC_TPB>>>(n, d_idx, d_ptr, d_vals);
  }

  // d_ptr[d_idx[i]] = d_vals[i]  -- compact source.
  void set_values_at(int n, const int d_idx[], const T d_vals[]) {
    if (n <= 0) {
      return;
    }
    int grid = (n + detail::VEC_TPB - 1) / detail::VEC_TPB;
    detail::vec_set_values_at_kernel<T>
        <<<grid, detail::VEC_TPB>>>(n, d_idx, d_vals, d_ptr);
  }

  T* get_device_ptr() { return d_ptr; }
  const T* get_device_ptr() const { return d_ptr; }

 private:
  int size;
  T* d_ptr;
  cublasHandle_t handle;
};

}  // namespace amigo

#endif  // AMIGO_CUDA_VECTOR_BACKEND_H
