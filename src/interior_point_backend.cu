/*
  CUDA backend for the primal-dual interior-point optimizer.

  This file mirrors the host-side functions in
  include/interior_point_backend.h.  Each *_cuda function performs the
  same computation as its host counterpart on the device.

  Implementation notes:

    * Element-wise kernels (project, initialize, residual, diagonal,
      back-substitution, apply step, dual residual) are launched with
      a multi-block grid sized to cover num_primals or num_constraints.

    * Reduction kernels (max-step, complementarity, KKT error, log
      barrier and its derivative, sum-of-squared-complementarity,
      infeasibility) use a single block of TPB threads with a strided
      grid-stride loop and a shared-memory reduction.  This handles
      arbitrary problem sizes with a single launch and a single
      device->host copy of the scalar result(s).

  The OptState<T> and OptProblemInfo<T> structs are small (a handful of
  pointers and ints) and are passed to kernels by value so the device
  receives the contained pointers directly.
*/

#include <cuda_runtime.h>

#include <cmath>
#include <limits>

#include "a2dcore.h"
#include "amigo.h"
#include "interior_point_backend.h"

namespace amigo {

namespace detail {

// Threads-per-block for all kernels.
static constexpr int IPM_TPB = 256;

// =========================================================================
// project_primals_into_interior_cuda
// =========================================================================

template <typename T>
AMIGO_KERNEL void project_primals_into_interior_kernel(
    int num_primals, const int* __restrict__ primal_indices,
    const T* __restrict__ lbx, const T* __restrict__ ubx, T kappa1, T kappa2,
    T* __restrict__ xlam) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_primals) {
    return;
  }

  int idx = primal_indices[i];
  T x = xlam[idx];
  T lb = lbx[i];
  T ub = ubx[i];
  bool has_lb = !::isinf(lb);
  bool has_ub = !::isinf(ub);

  if (has_lb && has_ub) {
    T range = ub - lb;
    T pl = A2D::min2(kappa1 * A2D::max2(T(1), A2D::fabs(lb)), kappa2 * range);
    T pu = A2D::min2(kappa1 * A2D::max2(T(1), A2D::fabs(ub)), kappa2 * range);
    xlam[idx] = A2D::max2(A2D::min2(x, ub - pu), lb + pl);
  } else if (has_lb) {
    xlam[idx] = A2D::max2(x, lb + kappa1 * A2D::max2(T(1), A2D::fabs(lb)));
  } else if (has_ub) {
    xlam[idx] = A2D::min2(x, ub - kappa1 * A2D::max2(T(1), A2D::fabs(ub)));
  }
}

template <typename T>
void project_primals_into_interior_cuda(const OptProblemInfo<T>& info, T* xlam,
                                        T kappa1, T kappa2,
                                        cudaStream_t stream) {
  if (info.num_primals <= 0) {
    return;
  }
  int grid = (info.num_primals + IPM_TPB - 1) / IPM_TPB;
  project_primals_into_interior_kernel<T><<<grid, IPM_TPB, 0, stream>>>(
      info.num_primals, info.primal_indices, info.lbx, info.ubx, kappa1, kappa2,
      xlam);
}

// =========================================================================
// initialize_bound_duals_cuda
// =========================================================================

template <typename T>
AMIGO_KERNEL void initialize_bound_duals_kernel(int num_primals, T mu,
                                                const T* __restrict__ lbx,
                                                const T* __restrict__ ubx,
                                                T* __restrict__ zl,
                                                T* __restrict__ zu) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_primals) {
    return;
  }
  zl[i] = ::isinf(lbx[i]) ? T(0) : mu;
  zu[i] = ::isinf(ubx[i]) ? T(0) : mu;
}

template <typename T>
void initialize_bound_duals_cuda(T mu, const OptProblemInfo<T>& info,
                                 const T* /*xlam*/, T* zl, T* zu,
                                 cudaStream_t stream) {
  if (info.num_primals <= 0) {
    return;
  }
  int grid = (info.num_primals + IPM_TPB - 1) / IPM_TPB;
  initialize_bound_duals_kernel<T>
      <<<grid, IPM_TPB, 0, stream>>>(info.num_primals, mu, info.lbx, info.ubx,
                                     zl, zu);
}

// =========================================================================
// compute_residual_cuda
// =========================================================================

template <typename T>
AMIGO_KERNEL void compute_residual_primal_kernel(T mu, OptProblemInfo<T> info,
                                                 OptState<const T> current,
                                                 const T* __restrict__ grad,
                                                 T* __restrict__ res) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= info.num_primals) {
    return;
  }

  int idx = info.primal_indices[i];
  T x = current.x[idx];

  // Stationarity residual (Blocks 1-2)
  T r = grad[idx] - current.zl[i] + current.zu[i];

  // Condense complementarity into stationarity
  if (!::isinf(info.lbx[i])) {
    T gap = x - info.lbx[i];
    r += (gap * current.zl[i] - mu) / gap;
  }
  if (!::isinf(info.ubx[i])) {
    T gap = info.ubx[i] - x;
    r -= (gap * current.zu[i] - mu) / gap;
  }
  res[idx] = -r;
}

template <typename T>
AMIGO_KERNEL void compute_residual_constraint_kernel(
    OptProblemInfo<T> info, const T* __restrict__ grad, T* __restrict__ res) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j >= info.num_constraints) {
    return;
  }
  int idx = info.constraint_indices[j];
  res[idx] = -(grad[idx] - info.lbh[j]);
}

template <typename T>
void compute_residual_cuda(T mu, const OptProblemInfo<T>& info,
                           OptState<const T>& current, const T* grad, T* res,
                           cudaStream_t stream) {
  if (info.num_primals > 0) {
    int gp = (info.num_primals + IPM_TPB - 1) / IPM_TPB;
    compute_residual_primal_kernel<T>
        <<<gp, IPM_TPB, 0, stream>>>(mu, info, current, grad, res);
  }
  if (info.num_constraints > 0) {
    int gc = (info.num_constraints + IPM_TPB - 1) / IPM_TPB;
    compute_residual_constraint_kernel<T>
        <<<gc, IPM_TPB, 0, stream>>>(info, grad, res);
  }
}

// =========================================================================
// compute_diagonal_cuda
// =========================================================================

template <typename T>
AMIGO_KERNEL void compute_diagonal_kernel(OptProblemInfo<T> info,
                                          OptState<const T> current,
                                          T* __restrict__ diag) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= info.num_primals) {
    return;
  }

  int idx = info.primal_indices[i];
  T x = current.x[idx];

  T sigma = T(0);
  if (!::isinf(info.lbx[i])) {
    T gap = x - info.lbx[i];
    sigma += current.zl[i] / gap;
  }
  if (!::isinf(info.ubx[i])) {
    T gap = info.ubx[i] - x;
    sigma += current.zu[i] / gap;
  }
  diag[idx] = sigma;
}

template <typename T>
void compute_diagonal_cuda(const OptProblemInfo<T>& info,
                           OptState<const T>& current, T* diag,
                           cudaStream_t stream) {
  if (info.num_primals <= 0) {
    return;
  }
  int grid = (info.num_primals + IPM_TPB - 1) / IPM_TPB;
  compute_diagonal_kernel<T>
      <<<grid, IPM_TPB, 0, stream>>>(info, current, diag);
}

// =========================================================================
// compute_bound_dual_step_cuda
// =========================================================================

template <typename T>
AMIGO_KERNEL void compute_bound_dual_step_kernel(T mu, OptProblemInfo<T> info,
                                                 OptState<const T> current,
                                                 const T* __restrict__ px,
                                                 T* __restrict__ dzl,
                                                 T* __restrict__ dzu) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= info.num_primals) {
    return;
  }

  int idx = info.primal_indices[i];
  T x = current.x[idx];
  T dx = px[idx];

  T dzl_i = T(0);
  T dzu_i = T(0);

  if (!::isinf(info.lbx[i])) {
    T gap = x - info.lbx[i];
    T rhs = gap * current.zl[i] - mu;
    dzl_i = -(rhs + current.zl[i] * dx) / gap;
  }
  if (!::isinf(info.ubx[i])) {
    T gap = info.ubx[i] - x;
    T rhs = gap * current.zu[i] - mu;
    dzu_i = -(rhs - current.zu[i] * dx) / gap;
  }

  dzl[i] = dzl_i;
  dzu[i] = dzu_i;
}

template <typename T>
void compute_bound_dual_step_cuda(T mu, const OptProblemInfo<T>& info,
                                  OptState<const T>& current, const T* px,
                                  T* dzl, T* dzu, cudaStream_t stream) {
  if (info.num_primals <= 0) {
    return;
  }
  int grid = (info.num_primals + IPM_TPB - 1) / IPM_TPB;
  compute_bound_dual_step_kernel<T>
      <<<grid, IPM_TPB, 0, stream>>>(mu, info, current, px, dzl, dzu);
}

// =========================================================================
// compute_max_step_cuda  (fraction-to-the-boundary, with argmin)
// =========================================================================
//
// Single-block reduction.  Each thread keeps a per-thread best
// (alpha, idx) for both x and z, then a shared-memory tree reduction
// selects the block-wide minimum.

template <typename T>
AMIGO_KERNEL void compute_max_step_kernel(T tau, OptProblemInfo<T> info,
                                          OptState<const T> current,
                                          OptState<const T> step,
                                          T init_alpha_x, T init_alpha_z,
                                          T* d_alpha_x_out, int* d_xi_out,
                                          T* d_alpha_z_out, int* d_zi_out) {
  extern __shared__ unsigned char smem[];
  T* s_alpha_x = reinterpret_cast<T*>(smem);
  int* s_xi = reinterpret_cast<int*>(s_alpha_x + blockDim.x);
  T* s_alpha_z = reinterpret_cast<T*>(s_xi + blockDim.x);
  int* s_zi = reinterpret_cast<int*>(s_alpha_z + blockDim.x);

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  T local_alpha_x = init_alpha_x;
  int local_xi = -1;
  T local_alpha_z = init_alpha_z;
  int local_zi = -1;

  for (int i = tid; i < info.num_primals; i += stride) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];
    T dx = step.x[idx];
    T dzl = step.zl[i];
    T dzu = step.zu[i];

    if (!::isinf(info.lbx[i])) {
      if (dx < T(0)) {
        T gap = x - info.lbx[i];
        T a = -tau * gap / dx;
        if (a < local_alpha_x) {
          local_alpha_x = a;
          local_xi = idx;
        }
      }
      if (dzl < T(0)) {
        T a = -tau * current.zl[i] / dzl;
        if (a < local_alpha_z) {
          local_alpha_z = a;
          local_zi = idx;
        }
      }
    }
    if (!::isinf(info.ubx[i])) {
      if (dx > T(0)) {
        T gap = info.ubx[i] - x;
        T a = tau * gap / dx;
        if (a < local_alpha_x) {
          local_alpha_x = a;
          local_xi = idx;
        }
      }
      if (dzu < T(0)) {
        T a = -tau * current.zu[i] / dzu;
        if (a < local_alpha_z) {
          local_alpha_z = a;
          local_zi = idx;
        }
      }
    }
  }

  s_alpha_x[tid] = local_alpha_x;
  s_xi[tid] = local_xi;
  s_alpha_z[tid] = local_alpha_z;
  s_zi[tid] = local_zi;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_alpha_x[tid + offset] < s_alpha_x[tid]) {
        s_alpha_x[tid] = s_alpha_x[tid + offset];
        s_xi[tid] = s_xi[tid + offset];
      }
      if (s_alpha_z[tid + offset] < s_alpha_z[tid]) {
        s_alpha_z[tid] = s_alpha_z[tid + offset];
        s_zi[tid] = s_zi[tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *d_alpha_x_out = s_alpha_x[0];
    *d_xi_out = s_xi[0];
    *d_alpha_z_out = s_alpha_z[0];
    *d_zi_out = s_zi[0];
  }
}

template <typename T>
void compute_max_step_cuda(T tau, const OptProblemInfo<T>& info,
                           OptState<const T>& current, OptState<const T>& step,
                           T& ax, int& xi, T& az, int& zi,
                           cudaStream_t stream) {
  if (info.num_primals <= 0) {
    return;
  }

  T* d_alpha_x;
  T* d_alpha_z;
  int* d_xi;
  int* d_zi;
  AMIGO_CHECK_CUDA(cudaMalloc(&d_alpha_x, sizeof(T)));
  AMIGO_CHECK_CUDA(cudaMalloc(&d_alpha_z, sizeof(T)));
  AMIGO_CHECK_CUDA(cudaMalloc(&d_xi, sizeof(int)));
  AMIGO_CHECK_CUDA(cudaMalloc(&d_zi, sizeof(int)));

  T init_ax = ax;
  T init_az = az;

  int block_size = IPM_TPB;
  int grid_size = 1;
  size_t shmem_size = 2 * block_size * (sizeof(T) + sizeof(int));

  compute_max_step_kernel<T><<<grid_size, block_size, shmem_size, stream>>>(
      tau, info, current, step, init_ax, init_az, d_alpha_x, d_xi, d_alpha_z,
      d_zi);

  AMIGO_CHECK_CUDA(cudaStreamSynchronize(stream));

  AMIGO_CHECK_CUDA(
      cudaMemcpy(&ax, d_alpha_x, sizeof(T), cudaMemcpyDeviceToHost));
  AMIGO_CHECK_CUDA(cudaMemcpy(&xi, d_xi, sizeof(int), cudaMemcpyDeviceToHost));
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&az, d_alpha_z, sizeof(T), cudaMemcpyDeviceToHost));
  AMIGO_CHECK_CUDA(cudaMemcpy(&zi, d_zi, sizeof(int), cudaMemcpyDeviceToHost));

  cudaFree(d_alpha_x);
  cudaFree(d_alpha_z);
  cudaFree(d_xi);
  cudaFree(d_zi);
}

// =========================================================================
// apply_step_cuda
// =========================================================================

template <typename T>
AMIGO_KERNEL void apply_step_primal_kernel(T ax, OptProblemInfo<T> info,
                                           OptState<const T> current,
                                           OptState<const T> step,
                                           OptState<T> result) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= info.num_primals) {
    return;
  }
  int idx = info.primal_indices[i];
  result.x[idx] = current.x[idx] + ax * step.x[idx];
}

template <typename T>
AMIGO_KERNEL void apply_step_constraint_kernel(T az, OptProblemInfo<T> info,
                                               OptState<const T> current,
                                               OptState<const T> step,
                                               OptState<T> result) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j >= info.num_constraints) {
    return;
  }
  int idx = info.constraint_indices[j];
  result.x[idx] = current.x[idx] + az * step.x[idx];
}

template <typename T>
AMIGO_KERNEL void apply_step_dual_kernel(T az, OptProblemInfo<T> info,
                                         OptState<const T> current,
                                         OptState<const T> step,
                                         OptState<T> result) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= info.num_primals) {
    return;
  }
  if (!::isinf(info.lbx[i])) {
    result.zl[i] = current.zl[i] + az * step.zl[i];
  }
  if (!::isinf(info.ubx[i])) {
    result.zu[i] = current.zu[i] + az * step.zu[i];
  }
}

template <typename T>
void apply_step_cuda(T ax, T az, const OptProblemInfo<T>& info,
                     OptState<const T>& current, OptState<const T>& step,
                     OptState<T>& result, cudaStream_t stream) {
  if (info.num_primals > 0) {
    int gp = (info.num_primals + IPM_TPB - 1) / IPM_TPB;
    apply_step_primal_kernel<T>
        <<<gp, IPM_TPB, 0, stream>>>(ax, info, current, step, result);
    apply_step_dual_kernel<T>
        <<<gp, IPM_TPB, 0, stream>>>(az, info, current, step, result);
  }
  if (info.num_constraints > 0) {
    int gc = (info.num_constraints + IPM_TPB - 1) / IPM_TPB;
    apply_step_constraint_kernel<T>
        <<<gc, IPM_TPB, 0, stream>>>(az, info, current, step, result);
  }
}

// =========================================================================
// compute_complementarity_cuda
// =========================================================================
//
// Single-block reduction that produces:
//   partial_sum[0] += sum_i gap_i * z_i
//   partial_sum[1] += count of finite bounds
//   local_min       = min over comp products
//
// The kernel adds to partial_sum (not assigns) to match the host
// version's behavior of accumulating into pre-initialized values.

template <typename T>
AMIGO_KERNEL void compute_complementarity_kernel(OptProblemInfo<T> info,
                                                 OptState<const T> current,
                                                 T init_min,
                                                 T* d_partial_sum, T* d_min) {
  extern __shared__ unsigned char smem[];
  T* s_sum0 = reinterpret_cast<T*>(smem);
  T* s_sum1 = s_sum0 + blockDim.x;
  T* s_min = s_sum1 + blockDim.x;

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  T sum0 = T(0);
  T sum1 = T(0);
  T lmin = init_min;

  for (int i = tid; i < info.num_primals; i += stride) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];
    if (!::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      T comp = gap * current.zl[i];
      sum0 += comp;
      sum1 += T(1);
      lmin = A2D::min2(lmin, comp);
    }
    if (!::isinf(info.ubx[i])) {
      T gap = info.ubx[i] - x;
      T comp = gap * current.zu[i];
      sum0 += comp;
      sum1 += T(1);
      lmin = A2D::min2(lmin, comp);
    }
  }

  s_sum0[tid] = sum0;
  s_sum1[tid] = sum1;
  s_min[tid] = lmin;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum0[tid] += s_sum0[tid + offset];
      s_sum1[tid] += s_sum1[tid + offset];
      s_min[tid] = A2D::min2(s_min[tid], s_min[tid + offset]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_partial_sum[0] += s_sum0[0];
    d_partial_sum[1] += s_sum1[0];
    *d_min = A2D::min2(*d_min, s_min[0]);
  }
}

template <typename T>
void compute_complementarity_cuda(const OptProblemInfo<T>& info,
                                  OptState<const T>& current, T partial_sum[],
                                  T& local_min, cudaStream_t stream) {
  T* d_partial_sum;
  T* d_min;
  AMIGO_CHECK_CUDA(cudaMalloc(&d_partial_sum, 2 * sizeof(T)));
  AMIGO_CHECK_CUDA(cudaMalloc(&d_min, sizeof(T)));

  // Seed device scalars with the host's running totals so the kernel
  // can accumulate into them.
  AMIGO_CHECK_CUDA(cudaMemcpy(d_partial_sum, partial_sum, 2 * sizeof(T),
                              cudaMemcpyHostToDevice));
  AMIGO_CHECK_CUDA(
      cudaMemcpy(d_min, &local_min, sizeof(T), cudaMemcpyHostToDevice));

  if (info.num_primals > 0) {
    int block_size = IPM_TPB;
    int grid_size = 1;
    size_t shmem_size = 3 * block_size * sizeof(T);
    compute_complementarity_kernel<T>
        <<<grid_size, block_size, shmem_size, stream>>>(
            info, current, std::numeric_limits<T>::max(), d_partial_sum, d_min);
    AMIGO_CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  AMIGO_CHECK_CUDA(cudaMemcpy(partial_sum, d_partial_sum, 2 * sizeof(T),
                              cudaMemcpyDeviceToHost));
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&local_min, d_min, sizeof(T), cudaMemcpyDeviceToHost));

  cudaFree(d_partial_sum);
  cudaFree(d_min);
}

// =========================================================================
// compute_kkt_error_cuda
// =========================================================================
//
// Two single-block max reductions:
//   Kernel A (over primals)     -> dual, comp
//   Kernel B (over constraints) -> primal
//
// Results are packed into a single device buffer of length 3 so we
// only need one device->host copy.

template <typename T>
AMIGO_KERNEL void compute_kkt_error_primal_kernel(T mu, OptProblemInfo<T> info,
                                                  OptState<const T> current,
                                                  const T* __restrict__ grad,
                                                  T* d_out) {
  extern __shared__ unsigned char smem[];
  T* s_dual = reinterpret_cast<T*>(smem);
  T* s_comp = s_dual + blockDim.x;

  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  T local_dual = T(0);
  T local_comp = T(0);

  for (int i = tid; i < info.num_primals; i += stride) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];

    local_dual = A2D::max2(
        local_dual, A2D::fabs(grad[idx] - current.zl[i] + current.zu[i]));
    if (!::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      local_comp = A2D::max2(local_comp, A2D::fabs(gap * current.zl[i] - mu));
    }
    if (!::isinf(info.ubx[i])) {
      T gap = info.ubx[i] - x;
      local_comp = A2D::max2(local_comp, A2D::fabs(gap * current.zu[i] - mu));
    }
  }

  s_dual[tid] = local_dual;
  s_comp[tid] = local_comp;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_dual[tid] = A2D::max2(s_dual[tid], s_dual[tid + offset]);
      s_comp[tid] = A2D::max2(s_comp[tid], s_comp[tid + offset]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[0] = s_dual[0];
    d_out[2] = s_comp[0];
  }
}

template <typename T>
AMIGO_KERNEL void compute_kkt_error_constraint_kernel(
    OptProblemInfo<T> info, const T* __restrict__ grad, T* d_out) {
  extern __shared__ unsigned char smem[];
  T* s_primal = reinterpret_cast<T*>(smem);

  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  T local_primal = T(0);

  for (int j = tid; j < info.num_constraints; j += stride) {
    int idx = info.constraint_indices[j];
    local_primal = A2D::max2(local_primal, A2D::fabs(grad[idx] - info.lbh[j]));
  }

  s_primal[tid] = local_primal;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_primal[tid] = A2D::max2(s_primal[tid], s_primal[tid + offset]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[1] = s_primal[0];
  }
}

template <typename T>
void compute_kkt_error_cuda(T mu, const OptProblemInfo<T>& info,
                            OptState<const T>& current, const T* grad, T& dual,
                            T& primal, T& comp, cudaStream_t stream) {
  T* d_out;
  AMIGO_CHECK_CUDA(cudaMalloc(&d_out, 3 * sizeof(T)));
  AMIGO_CHECK_CUDA(cudaMemsetAsync(d_out, 0, 3 * sizeof(T), stream));

  int block_size = IPM_TPB;
  int grid_size = 1;
  if (info.num_primals > 0) {
    size_t shmem = 2 * block_size * sizeof(T);
    compute_kkt_error_primal_kernel<T>
        <<<grid_size, block_size, shmem, stream>>>(mu, info, current, grad,
                                                   d_out);
  }
  if (info.num_constraints > 0) {
    size_t shmem = block_size * sizeof(T);
    compute_kkt_error_constraint_kernel<T>
        <<<grid_size, block_size, shmem, stream>>>(info, grad, d_out);
  }

  AMIGO_CHECK_CUDA(cudaStreamSynchronize(stream));

  T host_out[3];
  AMIGO_CHECK_CUDA(
      cudaMemcpy(host_out, d_out, 3 * sizeof(T), cudaMemcpyDeviceToHost));
  dual = host_out[0];
  primal = host_out[1];
  comp = host_out[2];

  cudaFree(d_out);
}

// =========================================================================
// compute_log_barrier_cuda  (sum reduction)
// =========================================================================

template <typename T>
AMIGO_KERNEL void compute_log_barrier_kernel(T mu, OptProblemInfo<T> info,
                                             OptState<const T> current,
                                             T* d_out) {
  extern __shared__ unsigned char smem[];
  T* s = reinterpret_cast<T*>(smem);

  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  T local = T(0);

  for (int i = tid; i < info.num_primals; i += stride) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];
    if (!::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      if (gap > T(0)) {
        local -= mu * log(gap);
      }
    }
    if (!::isinf(info.ubx[i])) {
      T gap = info.ubx[i] - x;
      if (gap > T(0)) {
        local -= mu * log(gap);
      }
    }
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

template <typename T>
T compute_log_barrier_cuda(T mu, const OptProblemInfo<T>& info,
                           OptState<const T>& current, cudaStream_t stream) {
  T* d_out;
  AMIGO_CHECK_CUDA(cudaMalloc(&d_out, sizeof(T)));
  AMIGO_CHECK_CUDA(cudaMemsetAsync(d_out, 0, sizeof(T), stream));

  if (info.num_primals > 0) {
    int block_size = IPM_TPB;
    int grid_size = 1;
    size_t shmem = block_size * sizeof(T);
    compute_log_barrier_kernel<T><<<grid_size, block_size, shmem, stream>>>(
        mu, info, current, d_out);
  }

  AMIGO_CHECK_CUDA(cudaStreamSynchronize(stream));

  T result = T(0);
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&result, d_out, sizeof(T), cudaMemcpyDeviceToHost));
  cudaFree(d_out);
  return result;
}

// =========================================================================
// compute_log_barrier_derivative_cuda  (sum reduction)
// =========================================================================

template <typename T>
AMIGO_KERNEL void compute_log_barrier_derivative_kernel(
    T mu, OptProblemInfo<T> info, OptState<const T> current,
    OptState<const T> step, T* d_out) {
  extern __shared__ unsigned char smem[];
  T* s = reinterpret_cast<T*>(smem);

  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  T local = T(0);

  for (int i = tid; i < info.num_primals; i += stride) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];
    T dx = step.x[idx];
    if (!::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      local -= mu * dx / gap;
    }
    if (!::isinf(info.ubx[i])) {
      T gap = info.ubx[i] - x;
      local += mu * dx / gap;
    }
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

template <typename T>
T compute_log_barrier_derivative_cuda(T mu, const OptProblemInfo<T>& info,
                                      OptState<const T>& current,
                                      OptState<const T>& step,
                                      cudaStream_t stream) {
  T* d_out;
  AMIGO_CHECK_CUDA(cudaMalloc(&d_out, sizeof(T)));
  AMIGO_CHECK_CUDA(cudaMemsetAsync(d_out, 0, sizeof(T), stream));

  if (info.num_primals > 0) {
    int block_size = IPM_TPB;
    int grid_size = 1;
    size_t shmem = block_size * sizeof(T);
    compute_log_barrier_derivative_kernel<T>
        <<<grid_size, block_size, shmem, stream>>>(mu, info, current, step,
                                                   d_out);
  }

  AMIGO_CHECK_CUDA(cudaStreamSynchronize(stream));

  T result = T(0);
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&result, d_out, sizeof(T), cudaMemcpyDeviceToHost));
  cudaFree(d_out);
  return result;
}

// =========================================================================
// compute_sum_squared_complementarity_cuda  (sum reduction)
// =========================================================================

template <typename T>
AMIGO_KERNEL void compute_sum_squared_complementarity_kernel(
    T mu, OptProblemInfo<T> info, OptState<const T> current, T* d_out) {
  extern __shared__ unsigned char smem[];
  T* s = reinterpret_cast<T*>(smem);

  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  T local = T(0);

  for (int i = tid; i < info.num_primals; i += stride) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];
    if (!::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      T r = gap * current.zl[i] - mu;
      local += r * r;
    }
    if (!::isinf(info.ubx[i])) {
      // NOTE: matches the serial version, which uses (x - lbx) for the
      // upper-bound term as well.
      T gap = x - info.lbx[i];
      T r = gap * current.zu[i] - mu;
      local += r * r;
    }
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

template <typename T>
T compute_sum_squared_complementarity_cuda(T mu, const OptProblemInfo<T>& info,
                                           OptState<const T>& current,
                                           cudaStream_t stream) {
  T* d_out;
  AMIGO_CHECK_CUDA(cudaMalloc(&d_out, sizeof(T)));
  AMIGO_CHECK_CUDA(cudaMemsetAsync(d_out, 0, sizeof(T), stream));

  if (info.num_primals > 0) {
    int block_size = IPM_TPB;
    int grid_size = 1;
    size_t shmem = block_size * sizeof(T);
    compute_sum_squared_complementarity_kernel<T>
        <<<grid_size, block_size, shmem, stream>>>(mu, info, current, d_out);
  }

  AMIGO_CHECK_CUDA(cudaStreamSynchronize(stream));

  T result = T(0);
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&result, d_out, sizeof(T), cudaMemcpyDeviceToHost));
  cudaFree(d_out);
  return result;
}

// =========================================================================
// compute_infeasibility_cuda  (sum reduction over constraints, l1 norm)
// =========================================================================

template <typename T>
AMIGO_KERNEL void compute_infeasibility_kernel(OptProblemInfo<T> info,
                                               const T* __restrict__ grad,
                                               T* d_out) {
  extern __shared__ unsigned char smem[];
  T* s = reinterpret_cast<T*>(smem);

  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  T local = T(0);

  for (int j = tid; j < info.num_constraints; j += stride) {
    int idx = info.constraint_indices[j];
    local += A2D::fabs(grad[idx] - info.lbh[j]);
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

template <typename T>
T compute_infeasibility_cuda(const OptProblemInfo<T>& info, const T* grad,
                             cudaStream_t stream) {
  T* d_out;
  AMIGO_CHECK_CUDA(cudaMalloc(&d_out, sizeof(T)));
  AMIGO_CHECK_CUDA(cudaMemsetAsync(d_out, 0, sizeof(T), stream));

  if (info.num_constraints > 0) {
    int block_size = IPM_TPB;
    int grid_size = 1;
    size_t shmem = block_size * sizeof(T);
    compute_infeasibility_kernel<T>
        <<<grid_size, block_size, shmem, stream>>>(info, grad, d_out);
  }

  AMIGO_CHECK_CUDA(cudaStreamSynchronize(stream));

  T result = T(0);
  AMIGO_CHECK_CUDA(
      cudaMemcpy(&result, d_out, sizeof(T), cudaMemcpyDeviceToHost));
  cudaFree(d_out);
  return result;
}

// =========================================================================
// compute_dual_residual_cuda
// =========================================================================

template <typename T>
AMIGO_KERNEL void compute_dual_residual_kernel(OptProblemInfo<T> info,
                                               OptState<const T> current,
                                               const T* __restrict__ grad,
                                               T* __restrict__ out) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= info.num_primals) {
    return;
  }
  int idx = info.primal_indices[i];
  out[idx] = grad[idx] - current.zl[i] + current.zu[i];
}

template <typename T>
void compute_dual_residual_cuda(const OptProblemInfo<T>& info,
                                OptState<const T>& current, const T* grad,
                                T* out, int /*size*/, cudaStream_t stream) {
  if (info.num_primals <= 0) {
    return;
  }
  int grid = (info.num_primals + IPM_TPB - 1) / IPM_TPB;
  compute_dual_residual_kernel<T>
      <<<grid, IPM_TPB, 0, stream>>>(info, current, grad, out);
}

// =========================================================================
// Explicit instantiations for T = double
// =========================================================================

template void project_primals_into_interior_cuda<double>(
    const OptProblemInfo<double>& info, double* xlam, double kappa1,
    double kappa2, cudaStream_t stream);

template void initialize_bound_duals_cuda<double>(
    double mu, const OptProblemInfo<double>& info, const double* xlam,
    double* zl, double* zu, cudaStream_t stream);

template void compute_residual_cuda<double>(double mu,
                                            const OptProblemInfo<double>& info,
                                            OptState<const double>& current,
                                            const double* grad, double* res,
                                            cudaStream_t stream);

template void compute_diagonal_cuda<double>(const OptProblemInfo<double>& info,
                                            OptState<const double>& current,
                                            double* diag, cudaStream_t stream);

template void compute_bound_dual_step_cuda<double>(
    double mu, const OptProblemInfo<double>& info,
    OptState<const double>& current, const double* px, double* dzl, double* dzu,
    cudaStream_t stream);

template void compute_max_step_cuda<double>(
    double tau, const OptProblemInfo<double>& info,
    OptState<const double>& current, OptState<const double>& step, double& ax,
    int& xi, double& az, int& zi, cudaStream_t stream);

template void apply_step_cuda<double>(double ax, double az,
                                      const OptProblemInfo<double>& info,
                                      OptState<const double>& current,
                                      OptState<const double>& step,
                                      OptState<double>& result,
                                      cudaStream_t stream);

template void compute_complementarity_cuda<double>(
    const OptProblemInfo<double>& info, OptState<const double>& current,
    double partial_sum[], double& local_min, cudaStream_t stream);

template void compute_kkt_error_cuda<double>(
    double mu, const OptProblemInfo<double>& info,
    OptState<const double>& current, const double* grad, double& dual,
    double& primal, double& comp, cudaStream_t stream);

template double compute_log_barrier_cuda<double>(
    double mu, const OptProblemInfo<double>& info,
    OptState<const double>& current, cudaStream_t stream);

template double compute_log_barrier_derivative_cuda<double>(
    double mu, const OptProblemInfo<double>& info,
    OptState<const double>& current, OptState<const double>& step,
    cudaStream_t stream);

template double compute_sum_squared_complementarity_cuda<double>(
    double mu, const OptProblemInfo<double>& info,
    OptState<const double>& current, cudaStream_t stream);

template double compute_infeasibility_cuda<double>(
    const OptProblemInfo<double>& info, const double* grad,
    cudaStream_t stream);

template void compute_dual_residual_cuda<double>(
    const OptProblemInfo<double>& info, OptState<const double>& current,
    const double* grad, double* out, int size, cudaStream_t stream);

// =========================================================================
// Explicit instantiations for T = float
// =========================================================================

template void project_primals_into_interior_cuda<float>(
    const OptProblemInfo<float>& info, float* xlam, float kappa1, float kappa2,
    cudaStream_t stream);

template void initialize_bound_duals_cuda<float>(
    float mu, const OptProblemInfo<float>& info, const float* xlam, float* zl,
    float* zu, cudaStream_t stream);

template void compute_residual_cuda<float>(float mu,
                                           const OptProblemInfo<float>& info,
                                           OptState<const float>& current,
                                           const float* grad, float* res,
                                           cudaStream_t stream);

template void compute_diagonal_cuda<float>(const OptProblemInfo<float>& info,
                                           OptState<const float>& current,
                                           float* diag, cudaStream_t stream);

template void compute_bound_dual_step_cuda<float>(
    float mu, const OptProblemInfo<float>& info,
    OptState<const float>& current, const float* px, float* dzl, float* dzu,
    cudaStream_t stream);

template void compute_max_step_cuda<float>(float tau,
                                           const OptProblemInfo<float>& info,
                                           OptState<const float>& current,
                                           OptState<const float>& step,
                                           float& ax, int& xi, float& az,
                                           int& zi, cudaStream_t stream);

template void apply_step_cuda<float>(float ax, float az,
                                     const OptProblemInfo<float>& info,
                                     OptState<const float>& current,
                                     OptState<const float>& step,
                                     OptState<float>& result,
                                     cudaStream_t stream);

template void compute_complementarity_cuda<float>(
    const OptProblemInfo<float>& info, OptState<const float>& current,
    float partial_sum[], float& local_min, cudaStream_t stream);

template void compute_kkt_error_cuda<float>(float mu,
                                            const OptProblemInfo<float>& info,
                                            OptState<const float>& current,
                                            const float* grad, float& dual,
                                            float& primal, float& comp,
                                            cudaStream_t stream);

template float compute_log_barrier_cuda<float>(
    float mu, const OptProblemInfo<float>& info,
    OptState<const float>& current, cudaStream_t stream);

template float compute_log_barrier_derivative_cuda<float>(
    float mu, const OptProblemInfo<float>& info,
    OptState<const float>& current, OptState<const float>& step,
    cudaStream_t stream);

template float compute_sum_squared_complementarity_cuda<float>(
    float mu, const OptProblemInfo<float>& info,
    OptState<const float>& current, cudaStream_t stream);

template float compute_infeasibility_cuda<float>(
    const OptProblemInfo<float>& info, const float* grad, cudaStream_t stream);

template void compute_dual_residual_cuda<float>(
    const OptProblemInfo<float>& info, OptState<const float>& current,
    const float* grad, float* out, int size, cudaStream_t stream);

}  // namespace detail

}  // namespace amigo
