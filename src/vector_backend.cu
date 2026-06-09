#include <cuda_runtime.h>

#include "amigo.h"
#include "cuda/vector_backend.cuh"

namespace amigo {

namespace detail {

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

template <typename T>
AMIGO_DEVICE inline T abs_value(T x) {
  return x < T(0) ? -x : x;
}

// -------------------------------------------------------------------------
// Element-wise kernels
// -------------------------------------------------------------------------

template <typename T>
AMIGO_KERNEL void vec_fill_kernel(int n, T value, T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d_ptr[i] = value;
}

template <typename T>
AMIGO_KERNEL void vec_add_scalar_kernel(int n, T value, T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d_ptr[i] += value;
}

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
  if (i < n) d_ptr[d_idx[i]] = value;
}

template <typename T>
AMIGO_KERNEL void vec_add_scalar_at_kernel(int n, const int* __restrict__ d_idx,
                                           T scalar, T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d_ptr[d_idx[i]] += scalar;
}

template <typename T>
AMIGO_KERNEL void vec_scale_at_kernel(int n, const int* __restrict__ d_idx,
                                      T alpha, T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d_ptr[d_idx[i]] *= alpha;
}

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

template <typename T>
AMIGO_KERNEL void vec_get_values_at_kernel(int n, const int* __restrict__ d_idx,
                                           const T* __restrict__ d_ptr,
                                           T* __restrict__ d_vals) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d_vals[i] = d_ptr[d_idx[i]];
}

template <typename T>
AMIGO_KERNEL void vec_set_values_at_kernel(int n, const int* __restrict__ d_idx,
                                           const T* __restrict__ d_vals,
                                           T* __restrict__ d_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d_ptr[d_idx[i]] = d_vals[i];
}

// -------------------------------------------------------------------------
// Reduction kernels
// -------------------------------------------------------------------------

template <typename T>
AMIGO_KERNEL void vec_maxabs_kernel(int n, const T* __restrict__ d_ptr,
                                    T* d_max, int* d_idx_out) {
  extern __shared__ unsigned char smem[];
  T* s_val = reinterpret_cast<T*>(smem);
  int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

  int tid = threadIdx.x;
  T local_val = T(0);
  int local_idx = -1;

  for (int i = tid; i < n; i += blockDim.x) {
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
    if (tid < offset && s_val[tid + offset] > s_val[tid]) {
      s_val[tid] = s_val[tid + offset];
      s_idx[tid] = s_idx[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *d_max = s_val[0];
    *d_idx_out = s_idx[0];
  }
}

template <typename T>
AMIGO_KERNEL void vec_abssum_kernel(int n, const T* __restrict__ d_ptr,
                                    T* d_out) {
  extern __shared__ unsigned char smem[];
  T* s = reinterpret_cast<T*>(smem);

  int tid = threadIdx.x;
  T local = T(0);
  for (int i = tid; i < n; i += blockDim.x) local += abs_value(d_ptr[i]);

  s[tid] = local;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) s[tid] += s[tid + offset];
    __syncthreads();
  }

  if (tid == 0) *d_out = s[0];
}

// -------------------------------------------------------------------------
// Host launchers (definitions matching declarations in vector_backend.cuh)
// -------------------------------------------------------------------------

template <typename T>
void vec_fill(int n, T value, T* d_ptr) {
  int grid = (n + VEC_TPB - 1) / VEC_TPB;
  vec_fill_kernel<T><<<grid, VEC_TPB>>>(n, value, d_ptr);
}

template <typename T>
void vec_add_scalar(int n, T value, T* d_ptr) {
  int grid = (n + VEC_TPB - 1) / VEC_TPB;
  vec_add_scalar_kernel<T><<<grid, VEC_TPB>>>(n, value, d_ptr);
}

template <typename T>
void vec_copy_at(int n, const int* d_idx, const T* d_src, T* d_ptr) {
  int grid = (n + VEC_TPB - 1) / VEC_TPB;
  vec_copy_at_kernel<T><<<grid, VEC_TPB>>>(n, d_idx, d_src, d_ptr);
}

template <typename T>
void vec_fill_at(int n, const int* d_idx, T value, T* d_ptr) {
  int grid = (n + VEC_TPB - 1) / VEC_TPB;
  vec_fill_at_kernel<T><<<grid, VEC_TPB>>>(n, d_idx, value, d_ptr);
}

template <typename T>
void vec_add_scalar_at(int n, const int* d_idx, T scalar, T* d_ptr) {
  int grid = (n + VEC_TPB - 1) / VEC_TPB;
  vec_add_scalar_at_kernel<T><<<grid, VEC_TPB>>>(n, d_idx, scalar, d_ptr);
}

template <typename T>
void vec_scale_at(int n, const int* d_idx, T alpha, T* d_ptr) {
  int grid = (n + VEC_TPB - 1) / VEC_TPB;
  vec_scale_at_kernel<T><<<grid, VEC_TPB>>>(n, d_idx, alpha, d_ptr);
}

template <typename T>
void vec_axpy_at(int n, const int* d_idx, T alpha, const T* d_x, T* d_ptr) {
  int grid = (n + VEC_TPB - 1) / VEC_TPB;
  vec_axpy_at_kernel<T><<<grid, VEC_TPB>>>(n, d_idx, alpha, d_x, d_ptr);
}

template <typename T>
void vec_get_values_at(int n, const int* d_idx, const T* d_ptr, T* d_vals) {
  int grid = (n + VEC_TPB - 1) / VEC_TPB;
  vec_get_values_at_kernel<T><<<grid, VEC_TPB>>>(n, d_idx, d_ptr, d_vals);
}

template <typename T>
void vec_set_values_at(int n, const int* d_idx, const T* d_vals, T* d_ptr) {
  int grid = (n + VEC_TPB - 1) / VEC_TPB;
  vec_set_values_at_kernel<T><<<grid, VEC_TPB>>>(n, d_idx, d_vals, d_ptr);
}

template <typename T>
T vec_maxabs(int n, const T* d_ptr, int& index) {
  T* d_max;
  int* d_idx_out;
  AMIGO_CHECK_CUDA(cudaMalloc(&d_max, sizeof(T)));
  AMIGO_CHECK_CUDA(cudaMalloc(&d_idx_out, sizeof(int)));

  size_t shmem = VEC_TPB * (sizeof(T) + sizeof(int));
  vec_maxabs_kernel<T><<<1, VEC_TPB, shmem>>>(n, d_ptr, d_max, d_idx_out);

  T host_max = T(0);
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&host_max, d_max, sizeof(T), cudaMemcpyDeviceToHost));
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&index, d_idx_out, sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d_max);
  cudaFree(d_idx_out);
  return host_max;
}

template <typename T>
T vec_abssum(int n, const T* d_ptr) {
  T* d_out;
  AMIGO_CHECK_CUDA(cudaMalloc(&d_out, sizeof(T)));

  size_t shmem = VEC_TPB * sizeof(T);
  vec_abssum_kernel<T><<<1, VEC_TPB, shmem>>>(n, d_ptr, d_out);

  T host_out = T(0);
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&host_out, d_out, sizeof(T), cudaMemcpyDeviceToHost));
  cudaFree(d_out);
  return host_out;
}

// -------------------------------------------------------------------------
// Explicit instantiations for float and double
// -------------------------------------------------------------------------

#define INSTANTIATE(T)                                               \
  template void vec_fill<T>(int, T, T*);                             \
  template void vec_add_scalar<T>(int, T, T*);                       \
  template void vec_copy_at<T>(int, const int*, const T*, T*);       \
  template void vec_fill_at<T>(int, const int*, T, T*);              \
  template void vec_add_scalar_at<T>(int, const int*, T, T*);        \
  template void vec_scale_at<T>(int, const int*, T, T*);             \
  template void vec_axpy_at<T>(int, const int*, T, const T*, T*);    \
  template void vec_get_values_at<T>(int, const int*, const T*, T*); \
  template void vec_set_values_at<T>(int, const int*, const T*, T*); \
  template T vec_maxabs<T>(int, const T*, int&);                     \
  template T vec_abssum<T>(int, const T*);

INSTANTIATE(int)
INSTANTIATE(float)
INSTANTIATE(double)

#undef INSTANTIATE

}  // namespace detail

}  // namespace amigo
