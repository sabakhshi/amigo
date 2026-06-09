#ifndef AMIGO_CUDA_VECTOR_BACKEND_H
#define AMIGO_CUDA_VECTOR_BACKEND_H

#include <cuda_runtime.h>

#include <type_traits>

#include "amigo.h"
#include "cuda/csr_matrix_backend.cuh"

namespace amigo {

namespace detail {

static constexpr int VEC_TPB = 256;

// -------------------------------------------------------------------------
// cuBLAS adapters
// -------------------------------------------------------------------------

template <typename T>
struct CublasVecOps;

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
// Host-callable launcher declarations (defined / instantiated in
// src/vector_backend.cu, compiled only by nvcc).
// -------------------------------------------------------------------------

template <typename T>
void vec_fill(int n, T value, T* d_ptr);

template <typename T>
void vec_add_scalar(int n, T value, T* d_ptr);

template <typename T>
void vec_copy_at(int n, const int* d_idx, const T* d_src, T* d_ptr);

template <typename T>
void vec_fill_at(int n, const int* d_idx, T value, T* d_ptr);

template <typename T>
void vec_add_scalar_at(int n, const int* d_idx, T scalar, T* d_ptr);

template <typename T>
void vec_scale_at(int n, const int* d_idx, T alpha, T* d_ptr);

template <typename T>
void vec_axpy_at(int n, const int* d_idx, T alpha, const T* d_x, T* d_ptr);

template <typename T>
void vec_get_values_at(int n, const int* d_idx, const T* d_ptr, T* d_vals);

template <typename T>
void vec_set_values_at(int n, const int* d_idx, const T* d_vals, T* d_ptr);

template <typename T>
T vec_maxabs(int n, const T* d_ptr, int& index);

template <typename T>
T vec_abssum(int n, const T* d_ptr);

}  // namespace detail

// =========================================================================
// CudaVecBackend — inline methods, no __global__ code here.
// =========================================================================

template <typename T>
class CudaVecBackend {
 public:
  CudaVecBackend() : size(0), d_ptr(nullptr), handle(nullptr) {
    AMIGO_CHECK_CUBLAS(cublasCreate(&handle));
  }
  ~CudaVecBackend() {
    if (d_ptr) cudaFree(d_ptr);
    if (handle) cublasDestroy(handle);
  }

  void allocate(int size_) {
    if (d_ptr) cudaFree(d_ptr);
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

  void fill(T scalar) {
    if (size > 0) {
      detail::vec_fill(size, scalar, d_ptr);
    }
  }
  void add_scalar(T scalar) {
    if (size > 0) {
      detail::vec_add_scalar(size, scalar, d_ptr);
    }
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
    return result;
  }

  T maxabs(int& index) {
    index = -1;
    if (size <= 0) {
      return T(0);
    }
    return detail::vec_maxabs(size, d_ptr, index);
  }
  T abssum() {
    if (size <= 0) {
      return T(0);
    }
    return detail::vec_abssum(size, d_ptr);
  }

  void copy_at(int n, const int d_idx[], const T d_src[]) {
    if (n > 0) {
      detail::vec_copy_at(n, d_idx, d_src, d_ptr);
    }
  }
  void fill_at(int n, const int d_idx[], T value) {
    if (n > 0) {
      detail::vec_fill_at(n, d_idx, value, d_ptr);
    }
  }
  void add_scalar_at(int n, const int d_idx[], T scalar) {
    if (n > 0) {
      detail::vec_add_scalar_at(n, d_idx, scalar, d_ptr);
    }
  }
  void scale_at(int n, const int d_idx[], T scalar) {
    if (n > 0) {
      detail::vec_scale_at(n, d_idx, scalar, d_ptr);
    }
  }
  void axpy_at(int n, const int d_idx[], T alpha, const T d_x[]) {
    if (n > 0) {
      detail::vec_axpy_at(n, d_idx, alpha, d_x, d_ptr);
    }
  }
  void get_values_at(int n, const int d_idx[], T d_vals[]) {
    if (n > 0) {
      detail::vec_get_values_at(n, d_idx, d_ptr, d_vals);
    }
  }
  void set_values_at(int n, const int d_idx[], const T d_vals[]) {
    if (n > 0) {
      detail::vec_set_values_at(n, d_idx, d_vals, d_ptr);
    }
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
