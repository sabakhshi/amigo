#ifndef AMIGO_CUDA_COMPONENT_GROUP_H
#define AMIGO_CUDA_COMPONENT_GROUP_H

#include "amigo.h"
#include "csr_matrix.h"
#include "layout.h"
#include "vector.h"

namespace amigo {

namespace detail {

// All __device__ / __global__ code below is only compiled by nvcc.
// mpicxx sees only the CudaGroupBackend class declaration (which is
// guarded by AMIGO_USE_CUDA at the include site).
#ifdef __CUDACC__

template <typename T, class Data, class Input, class Component, class... Remain>
AMIGO_DEVICE T add_lagrangian(T alpha, Data& data, Input& input) {
  T value = 0.0;

  if constexpr (!Component::is_compute_empty) {
    value = Component::lagrange(alpha, data, input);
  }

  if constexpr (sizeof...(Remain) > 0) {
    return value +
           add_lagrangian<T, Data, Input, Remain...>(alpha, data, input);
  } else {
    return value;
  }
}

template <typename T, class Input, class Data, class Component, class... Remain>
AMIGO_DEVICE void add_gradient(T alpha, Data& data, Input& input, Input& grad) {
  if constexpr (!Component::is_compute_empty) {
    Component::gradient(alpha, data, input, grad);
  }

  if constexpr (sizeof...(Remain) > 0) {
    add_gradient<T, Input, Data, Remain...>(alpha, data, input, grad);
  }
}

template <typename T, class Input, class Data, class Component, class... Remain>
AMIGO_DEVICE void add_hessian_product(T alpha, Data& data, Input& input,
                                      Input& dir, Input& h) {
  if constexpr (!Component::is_compute_empty) {
    Input grad;
    Component::hessian(alpha, data, input, dir, grad, h);
  }

  if constexpr (sizeof...(Remain) > 0) {
    add_hessian_product<T, Input, Data, Remain...>(alpha, data, input, dir, h);
  }
}

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
AMIGO_KERNEL void gradient_kernel_atomic(int num_elements, T alpha,
                                         const int* data_indices,
                                         const int* vec_indices,
                                         const T* data_values,
                                         const T* vec_values, T* grad_values) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_elements) {
    return;
  }

  Input input, grad;
  Data data;
  IndexLayout<ndata>::get_values_device(index, data_indices, data_values, data);
  IndexLayout<ncomp>::get_values_device(index, vec_indices, vec_values, input);

  // Compute the gradient
  grad.zero();
  add_gradient<T, Input, Data, Components...>(alpha, data, input, grad);

  // Add the values to grad_values
  IndexLayout<ncomp>::add_values_atomic(index, vec_indices, grad, grad_values);
}

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
AMIGO_KERNEL void add_lagrangian_kernel_atomic(int num_elements, T alpha,
                                               const int* data_indices,
                                               const int* vec_indices,
                                               const T* data_values,
                                               const T* vec_values, T* value) {
  extern __shared__ unsigned char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);

  int tid = threadIdx.x;
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  T local_value = 0.0;

  if constexpr (ncomp > 0) {
    for (int index = global_id; index < num_elements; index += stride) {
      Input input;
      Data data;
      IndexLayout<ndata>::get_values_device(index, data_indices, data_values,
                                            data);
      IndexLayout<ncomp>::get_values_device(index, vec_indices, vec_values,
                                            input);

      local_value +=
          add_lagrangian<T, Data, Input, Components...>(alpha, data, input);
    }
  }

  smem[tid] = local_value;
  __syncthreads();

  // Block reduction
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      smem[tid] += smem[tid + offset];
    }
    __syncthreads();
  }

  // One atomic add per block
  if (tid == 0) {
    atomicAdd(value, smem[0]);
  }
}

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
AMIGO_KERNEL void hessian_product_kernel_atomic(
    int num_elements, T alpha, const int* data_indices, const int* vec_indices,
    const T* data_values, const T* vec_values, const T* dir_values,
    T* grad_values) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > num_elements) {
    return;
  }

  Input input, dir, h;
  Data data;
  IndexLayout<ndata>::get_values_device(index, data_indices, data_values, data);
  IndexLayout<ncomp>::get_values_device(index, vec_indices, vec_values, input);
  IndexLayout<ncomp>::get_values_device(index, vec_indices, dir_values, dir);

  // Compute the gradient
  h.zero();
  add_hessian_product<T, Input, Data, Components...>(alpha, data, input, dir,
                                                     h);

  // Add the values to grad_values
  IndexLayout<ncomp>::add_values_atomic(index, vec_indices, h, grad_values);
}

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
AMIGO_KERNEL void hessian_kernel_atomic(int num_elements, T alpha,
                                        const int* data_indices,
                                        const int* vec_indices,
                                        const int* csr_pos,
                                        const T* data_values,
                                        const T* vec_values, T* csr_data) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (index >= num_elements) {
    return;
  }

  Input input, dir, h;
  Data data;
  IndexLayout<ndata>::get_values_device(index, data_indices, data_values, data);
  IndexLayout<ncomp>::get_values_device(index, vec_indices, vec_values, input);

  h.zero();
  dir.zero();
  dir[row] = 1.0;
  add_hessian_product<T, Input, Data, Components...>(alpha, data, input, dir,
                                                     h);

  // Now add the Hessian contribution
  for (int i = 0; i < ncomp; i++) {
    int pos = csr_pos[ncomp * (ncomp * index + row) + i];
    if (pos >= 0) {
      atomicAdd(&csr_data[pos], h[i]);
    }
  }
}

#endif  // __CUDACC__

}  // namespace detail

template <typename T, int ncomp, class Input, int ndata, class Data,
          class... Components>
class CudaGroupBackend {
 public:
  template <int noutputs>
  CudaGroupBackend(IndexLayout<ndata>& data_layout, IndexLayout<ncomp>& layout,
                   IndexLayout<noutputs>& outputs) {
    data_layout.copy_host_to_device();
    layout.copy_host_to_device();
    outputs.copy_host_to_device();

    d_hess_pos = nullptr;
  }
  ~CudaGroupBackend() {
    if (d_hess_pos) {
      cudaFree(d_hess_pos);
    }
  }

  void initialize_hessian_pattern(const IndexLayout<ncomp>& layout,
                                  const NodeOwners& owners, CSRMat<T>& mat) {
    int num_elements;
    const int* vec_indices;
    layout.get_data(&num_elements, nullptr, &vec_indices);

    const int *rowp, *cols;
    mat.get_data(nullptr, nullptr, nullptr, &rowp, &cols, nullptr);

    // Allocate space for the positions of the Hessian entries
    int size = num_elements * ncomp * ncomp;
    int* hess_pos = new int[size];

    // Locate the positions within the CSRMatrix
    for (int i = 0; i < num_elements; i++) {
      int rows[ncomp], columns[ncomp];
      for (int j = 0; j < ncomp; j++) {
        rows[j] = vec_indices[ncomp * i + j];
      }

      owners.local_to_global(ncomp, rows, columns);

      for (int j = 0; j < ncomp; j++) {
        int row = rows[j];
        int row_size = rowp[row + 1] - rowp[row];
        const int* start = &cols[rowp[row]];
        const int* end = start + row_size;

        for (int k = 0; k < ncomp; k++) {
          auto* it = std::lower_bound(start, end, columns[k]);

          if (it != end && *it == columns[k]) {
            hess_pos[ncomp * (ncomp * i + j) + k] = it - cols;
          } else {
            hess_pos[ncomp * (ncomp * i + j) + k] = -1;
          }
        }
      }
    }

    // Copy the result to the device
    cudaMalloc(&d_hess_pos, size * sizeof(int));
    cudaMemcpy(d_hess_pos, hess_pos, size * sizeof(int),
               cudaMemcpyHostToDevice);

    // This is not needed on the host
    delete[] hess_pos;
  }

  // This isn't handled properly yet...
  T lagrangian_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                      const IndexLayout<ncomp>& layout,
                      const Vector<T>& data_vec, const Vector<T>& vec) const {
    T host_value = 0.0;

#ifdef __CUDACC__
    const int TPB = 32;
    int num_elements;
    const int* data_indices;
    const int* vec_indices;
    data_layout.get_device_data(&num_elements, nullptr, &data_indices);
    layout.get_device_data(nullptr, nullptr, &vec_indices);

    const T* data_values = data_vec.get_device_array();
    const T* vec_values = vec.get_device_array();

    dim3 grid((num_elements + TPB - 1) / TPB);
    dim3 block(TPB);

    T* device_value = nullptr;
    cudaMalloc(&device_value, sizeof(T));
    cudaMemset(device_value, 0, sizeof(T));

    std::size_t shared_bytes = TPB * sizeof(T);

    detail::add_lagrangian_kernel_atomic<T, ncomp, Input, ndata, Data,
                                         Components...>
        <<<grid, block, shared_bytes>>>(num_elements, alpha, data_indices,
                                        vec_indices, data_values, vec_values,
                                        device_value);

    cudaMemcpy(&host_value, device_value, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(device_value);
#endif
    return host_value;
  }

  void add_gradient_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                           const IndexLayout<ncomp>& layout,
                           const Vector<T>& data_vec, const Vector<T>& vec,
                           Vector<T>& res) const {
#ifdef __CUDACC__
    const int TPB = 32;
    int num_elements;
    const int* data_indices;
    const int* vec_indices;
    data_layout.get_device_data(&num_elements, nullptr, &data_indices);
    layout.get_device_data(nullptr, nullptr, &vec_indices);

    const T* data_values = data_vec.get_device_array();
    const T* vec_values = vec.get_device_array();
    T* res_values = res.get_device_array();

    dim3 grid((num_elements + TPB - 1) / TPB);
    dim3 block(TPB);

    detail::gradient_kernel_atomic<T, ncomp, Input, ndata, Data, Components...>
        <<<grid, block>>>(num_elements, alpha, data_indices, vec_indices,
                          data_values, vec_values, res_values);
#endif
  }

  void add_hessian_product_kernel(T alpha,
                                  const IndexLayout<ndata>& data_layout,
                                  const IndexLayout<ncomp>& layout,
                                  const Vector<T>& data_vec,
                                  const Vector<T>& vec, const Vector<T>& dir,
                                  Vector<T>& res) const {
#ifdef __CUDACC__
    const int TPB = 32;
    int num_elements;
    const int* data_indices;
    const int* vec_indices;
    data_layout.get_device_data(&num_elements, nullptr, &data_indices);
    layout.get_device_data(nullptr, nullptr, &vec_indices);

    dim3 grid((num_elements + TPB - 1) / TPB);
    dim3 block(TPB);

    const T* data_values = data_vec.get_device_array();
    const T* vec_values = vec.get_device_array();
    const T* dir_values = dir.get_device_array();
    T* res_values = res.get_device_array();

    detail::hessian_product_kernel_atomic<T, ncomp, Input, ndata, Data,
                                          Components...>
        <<<grid, block>>>(num_elements, alpha, data_indices, vec_indices,
                          data_values, vec_values, dir_values, res_values);
#endif
  }

  // Need to add the hessian...
  void add_hessian_kernel(T alpha, const IndexLayout<ndata>& data_layout,
                          const IndexLayout<ncomp>& layout,
                          const Vector<T>& data_vec, const Vector<T>& vec,
                          const NodeOwners& owners, CSRMat<T>& mat) const {
#ifdef __CUDACC__
    const int TPB = 32;
    int num_elements;
    const int* data_indices;
    const int* vec_indices;
    data_layout.get_device_data(&num_elements, nullptr, &data_indices);
    layout.get_device_data(nullptr, nullptr, &vec_indices);

    dim3 grid((num_elements + TPB - 1) / TPB, ncomp);
    dim3 block(TPB);

    const T* data_values = data_vec.get_device_array();
    const T* vec_values = vec.get_device_array();

    T* csr_data;
    mat.get_device_data(nullptr, nullptr, &csr_data);

    detail::hessian_kernel_atomic<T, ncomp, Input, ndata, Data, Components...>
        <<<grid, block>>>(num_elements, alpha, data_indices, vec_indices,
                          d_hess_pos, data_values, vec_values, csr_data);
#endif
  }

  void add_grad_jac_product_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const Vector<T>& data_vec, const Vector<T>& vec, const Vector<T>& dir,
      Vector<T>& res) const {}

  void add_grad_jac_tproduct_wrt_data_kernel(
      const IndexLayout<ndata>& data_layout, const IndexLayout<ncomp>& layout,
      const Vector<T>& data_vec, const Vector<T>& vec, const Vector<T>& dir,
      Vector<T>& res) const {}

  void add_grad_jac_wrt_data_kernel(const IndexLayout<ndata>& data_layout,
                                    const IndexLayout<ncomp>& layout,
                                    const Vector<T>& data_vec,
                                    const Vector<T>& vec,
                                    const NodeOwners& owners,
                                    CSRMat<T>& jac) const {}

 private:
  int* d_hess_pos;
};

}  // namespace amigo

#endif  // AMIGO_CUDA_COMPONENT_GROUP_H