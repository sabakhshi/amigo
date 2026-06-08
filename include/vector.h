#ifndef AMIGO_VECTOR_H
#define AMIGO_VECTOR_H

#include <algorithm>
#include <random>

#include "amigo.h"

#ifdef AMIGO_USE_CUDA
#include "cuda/vector_backend.cuh"
#endif

namespace amigo {

template <typename T>
class SerialVecBackend {
 public:
  SerialVecBackend() {}
  ~SerialVecBackend() {}

  void allocate(int size_) {}
  void copy_host_to_device(T* h_dest) {}
  void copy_device_to_host(T* h_src) {}

  void copy(const T* d_src) {}
  void zero() {}
  void fill(T scalar) {}
  void add_scalar(T scalar) {}
  void scale(T alpha) {}
  void axpy(T alpha, const T* d_x) {}

  T dot(const T* d_src) const { return T(0); }
  T maxabs(int& index) { return T(0); }
  T abssum() { return T(0); }

  void copy_at(int n, const int d_idx[], const T d_src[]) {}
  void fill_at(int n, const int d_idx[], T value) {}
  void add_scalar_at(int n, const int d_idx[], T scalar) {}
  void scale_at(int n, const int d_idx[], T scalar) {}
  void axpy_at(int n, const int d_idx[], T alpha, const T d_x[]) {}
  void get_values_at(int n, const int d_idx[], T d_vals[]) {}
  void set_values_at(int n, const int d_idx[], const T d_vals[]) {}

  T* get_device_ptr() { return nullptr; }
  const T* get_device_ptr() const { return nullptr; }
};

#ifdef AMIGO_USE_CUDA
template <typename T>
using DefaultVecBackend = CudaVecBackend<T>;
#else
template <typename T>
using DefaultVecBackend = SerialVecBackend<T>;
#endif  // AMIGO_USE_CUDA

template <typename T, class Backend = DefaultVecBackend<T>>
class Vector {
 public:
  Vector(int local_size, int ext_size = 0,
         MemoryLocation mem_loc = MemoryLocation::HOST_AND_DEVICE)
      : local_size(local_size),
        ext_size(ext_size),
        size(local_size + ext_size),
        mem_loc(mem_loc) {
    if (mem_loc == MemoryLocation::HOST_AND_DEVICE ||
        mem_loc == MemoryLocation::HOST_ONLY) {
      array = new T[size];
      std::fill(array, array + size, T(0.0));
    }
    if (mem_loc == MemoryLocation::HOST_AND_DEVICE ||
        mem_loc == MemoryLocation::DEVICE_ONLY) {
      backend.allocate(size);
    }
  }
  Vector(int local_size, int ext_size, T** array_)
      : local_size(local_size),
        ext_size(ext_size),
        size(local_size + ext_size),
        mem_loc(MemoryLocation::HOST_ONLY) {
    array = *array_;
    *array_ = nullptr;
  }

  ~Vector() {
    if (array) {
      delete[] array;
    }
  }

  MemoryLocation get_memory_location() const { return mem_loc; }

  std::shared_ptr<Vector<T>> duplicate() const {
    return std::make_shared<Vector<T>>(local_size, ext_size, mem_loc);
  }

  void copy_host_to_device() {
    if (mem_loc == MemoryLocation::HOST_AND_DEVICE) {
      backend.copy_host_to_device(array);
    }
  }

  void copy_device_to_host() {
    if (mem_loc == MemoryLocation::HOST_AND_DEVICE) {
      backend.copy_device_to_host(array);
    }
  }

  void copy(const T* src) {
    if (array) {
      std::copy(src, src + size, array);
    }
  }

  void copy(const std::shared_ptr<Vector<T>> src) {
    if (array && src->array) {
      std::copy(src->array, src->array + size, array);
    }
    backend.copy(src->get_device_array());
  }

  void zero() {
    if (array) {
      std::fill(array, array + size, T(0));
    }
    backend.zero();
  }

  template <ExecPolicy policy>
  void fill(T value) {
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      for (int i = 0; i < local_size; i++) {
        array[i] = value;
      }
    } else {
      backend.fill(value);
    }
  }

  template <ExecPolicy policy>
  void add_scalar(T value) {
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      for (int i = 0; i < local_size; i++) {
        array[i] += value;
      }
    } else {
      backend.add_scalar(value);
    }
  }

  template <ExecPolicy policy>
  void scale(T alpha) {
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      for (int i = 0; i < size; i++) {
        array[i] *= alpha;
      }
    } else {
      backend.scale(alpha);
    }
  }

  template <ExecPolicy policy>
  void axpy(T alpha, const std::shared_ptr<Vector<T>> x) {
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      for (int i = 0; i < local_size; i++) {
        array[i] += alpha * x->array[i];
      }
    } else {
      backend.axpy(alpha, x->get_device_array());
    }
  }

  template <ExecPolicy policy>
  T dot(const std::shared_ptr<Vector<T>> x) const {
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      T value = 0.0;
      if (array && x->array) {
        for (int i = 0; i < local_size; i++) {
          value += array[i] * x->array[i];
        }
      }
      return value;
    } else {
      return backend.dot(x->get_device_array());
    }
  }

  template <ExecPolicy policy>
  T maxabs(int& index) {
    index = -1;
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      T value = 0.0;
      for (int i = 0; i < local_size; i++) {
        if (std::fabs(array[i]) > value) {
          value = std::fabs(array[i]);
          index = i;
        }
      }
      return value;
    } else {
      return backend.maxabs(index);
    }
  }

  template <ExecPolicy policy>
  T abssum() {
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      T value = 0.0;
      for (int i = 0; i < local_size; i++) {
        value += std::fabs(array[i]);
      }
      return value;
    } else {
      return backend.abssum();
    }
  }

  template <ExecPolicy policy>
  void copy_at(std::shared_ptr<Vector<int>> indices,
               std::shared_ptr<Vector<T>> src) {
    int nentries = indices->get_local_size();
    const int* idx = indices->template get_array<policy>();
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      for (int i = 0; i < nentries; i++) {
        array[idx[i]] = src->array[idx[i]];
      }
    } else {
      backend.copy_at(nentries, idx, src->get_device_array());
    }
  }

  template <ExecPolicy policy>
  void fill_at(std::shared_ptr<Vector<int>> indices, T value) {
    int nentries = indices->get_local_size();
    const int* idx = indices->template get_array<policy>();
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      for (int i = 0; i < nentries; i++) {
        array[idx[i]] = value;
      }
    } else {
      backend.fill_at(nentries, idx, value);
    }
  }

  template <ExecPolicy policy>
  void add_scalar_at(std::shared_ptr<Vector<int>> indices, T value) {
    int nentries = indices->get_local_size();
    const int* idx = indices->template get_array<policy>();
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      for (int i = 0; i < nentries; i++) {
        array[idx[i]] += value;
      }
    } else {
      backend.add_scalar_at(nentries, idx, value);
    }
  }

  template <ExecPolicy policy>
  void scale_at(std::shared_ptr<Vector<int>> indices, T alpha) {
    int nentries = indices->get_local_size();
    const int* idx = indices->template get_array<policy>();
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      for (int i = 0; i < size; i++) {
        array[idx[i]] *= alpha;
      }
    } else {
      backend.scale_at(nentries, idx, alpha);
    }
  }

  template <ExecPolicy policy>
  void axpy_at(std::shared_ptr<Vector<int>> indices, T alpha,
               const std::shared_ptr<Vector<T>> x) {
    int nentries = indices->get_local_size();
    const int* idx = indices->template get_array<policy>();
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      for (int i = 0; i < nentries; i++) {
        array[idx[i]] += alpha * x.array[idx[i]];
      }
    } else {
      backend.axpy_at(nentries, idx, alpha, x->get_device_array());
    }
  }

  template <ExecPolicy policy>
  void get_values_at(std::shared_ptr<Vector<int>> indices,
                     std::shared_ptr<Vector<T>> values) {
    int nentries = indices->get_local_size();
    const int* idx = indices->template get_array<policy>();
    T* v = values->template get_array<policy>();
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      for (int i = 0; i < nentries; i++) {
        v[i] = array[idx[i]];
      }
    } else {
      backend.get_values_at(nentries, idx, v);
    }
  }

  template <ExecPolicy policy>
  void set_values_at(std::shared_ptr<Vector<int>> indices,
                     std::shared_ptr<Vector<T>> values) {
    int nentries = indices->get_local_size();
    const int* idx = indices->template get_array<policy>();
    const T* v = values->template get_array<policy>();
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      for (int i = 0; i < nentries; i++) {
        array[idx[i]] = v[i];
      }
    } else {
      backend.set_values_at(nentries, idx, v);
    }
  }

  template <ExecPolicy policy>
  T* get_array() {
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      return array;
    } else {
      return backend.get_device_ptr();
    }
  }
  template <ExecPolicy policy>
  const T* get_array() const {
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      return array;
    } else {
      return backend.get_device_ptr();
    }
  }

  T& operator[](int i) { return array[i]; }
  const T& operator[](int i) const { return array[i]; }

  int get_size() const { return size; }
  int get_local_size() const { return local_size; }

  T* get_array() { return array; }
  const T* get_array() const { return array; }

  T* get_device_array() { return backend.get_device_ptr(); }
  const T* get_device_array() const { return backend.get_device_ptr(); }

 private:
  int local_size;  // The locally owned nodes
  int ext_size;    // Size of externally owned nodes referenced on this proc
  int size;        // Total size of the vector
  MemoryLocation mem_loc;  // Location of the data
  T* array;                // Host array

  // Backend for the GPU implementation
  Backend backend;
};

}  // namespace amigo

#endif  // AMIGO_VECTOR_H