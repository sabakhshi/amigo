#ifndef AMIGO_INTERIOR_POINT_OPTIMIZER_H
#define AMIGO_INTERIOR_POINT_OPTIMIZER_H

#include <mpi.h>

#include <memory>

#include "amigo.h"
#include "interior_point_backend.h"
#include "optimization_problem.h"

namespace amigo {

/**
 * Optimization state: solution vector + bound duals.
 *
 * Storage:
 *   x_     = xlam vector (primals + multipliers)
 *   duals = [zl(n_primal) | zu(n_primal)]
 */
template <typename T>
class OptVector {
 public:
  OptVector(int num_primals, int num_constraints, std::shared_ptr<Vector<T>> x)
      : num_primals(num_primals), num_constraints(num_constraints), x(x) {
    // Create the slack and dual variable vectors for the upper and lower bounds
    MemoryLocation loc = x->get_memory_location();

    zl = std::make_shared<Vector<T>>(num_primals, 0, loc);
    zu = std::make_shared<Vector<T>>(num_primals, 0, loc);
  }

  void zero() {
    zl->zero();
    zu->zero();
  }

  void copy(std::shared_ptr<OptVector<T>> src) {
    x->copy(src->x);
    zl->copy(src->zl);
    zu->copy(src->zu);
  }

  int get_num_primals() const { return num_primals; }
  int get_num_constraints() const { return num_constraints; }

  std::shared_ptr<Vector<T>> get_solution() { return x; }
  std::shared_ptr<Vector<T>> get_zl() { return zl; }
  std::shared_ptr<Vector<T>> get_zu() { return zu; }

  const std::shared_ptr<Vector<T>> get_solution() const { return x; }

  template <ExecPolicy policy>
  void get_bound_duals(T** zl_, T** zu_) {
    if (zl_) {
      *zl_ = zl->template get_array<policy>();
    }
    if (zu_) {
      *zu_ = zu->template get_array<policy>();
    }
  }
  template <ExecPolicy policy>
  void get_bound_duals(const T** zl_, const T** zu_) const {
    if (zl_) {
      *zl_ = zl->template get_array<policy>();
    }
    if (zu_) {
      *zu_ = zu->template get_array<policy>();
    }
  }

  template <ExecPolicy policy>
  T* get_solution_array() {
    return x->template get_array<policy>();
  }
  template <ExecPolicy policy>
  const T* get_solution_array() const {
    return x->template get_array<policy>();
  }

 private:
  int num_primals, num_constraints;

  // The primal/dual vector
  std::shared_ptr<Vector<T>> x;

  // Duals associated with the lower/upper bounds
  std::shared_ptr<Vector<T>> zl, zu;
};

/**
 * Interior-point optimizer for the 2x2 augmented system.
 *
 * Every variable is either a bounded primal or an equality constraint.
 * Wraps detail:: backend functions for the Python/pybind interface.
 */
template <typename T, ExecPolicy policy>
class InteriorPointOptimizer {
 public:
  InteriorPointOptimizer(
      std::shared_ptr<OptimizationProblem<T, policy>> problem)
      : problem(problem) {
    comm = problem->get_mpi_comm();

    int size = problem->get_num_variables();
    const Vector<int>& vtypes = *problem->get_var_types();
    const Vector<T>& lb = *problem->get_lower();
    const Vector<T>& ub = *problem->get_upper();

    primal_indices = problem->get_primal_indices();
    num_primals = primal_indices->get_local_size();

    constraint_indices = problem->get_constraint_indices();
    num_constraints = constraint_indices->get_local_size();

    // Set the memory location depending on the execution policy
    MemoryLocation loc = MemoryLocation::HOST_ONLY;
    if (policy == ExecPolicy::CUDA) {
      loc = MemoryLocation::HOST_AND_DEVICE;
    }

    lbx = std::make_shared<Vector<T>>(num_primals, 0, loc);
    ubx = std::make_shared<Vector<T>>(num_primals, 0, loc);
    for (int i = 0; i < num_primals; i++) {
      int idx = (*primal_indices)[i];
      (*lbx)[i] = lb[idx];
      (*ubx)[i] = ub[idx];
    }
    lbx->copy_host_to_device();
    ubx->copy_host_to_device();

    // Set the constraint bounds
    lbh = std::make_shared<Vector<T>>(num_constraints, 0, loc);
    for (int i = 0; i < num_constraints; i++) {
      int idx = (*constraint_indices)[i];
      (*lbh)[i] = lb[idx];
    }
    lbh->copy_host_to_device();

    // Set the host/device pointers into the info
    info.num_primals = num_primals;
    info.num_constraints = num_constraints;
    info.primal_indices = primal_indices->template get_array<policy>();
    info.constraint_indices = constraint_indices->template get_array<policy>();
    info.lbx = lbx->template get_array<policy>();
    info.ubx = ubx->template get_array<policy>();
    info.lbh = lbh->template get_array<policy>();

    // Add up the total number of constraints
    MPI_Allreduce(&num_primals, &num_global_primals, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&num_constraints, &num_global_constraints, 1, MPI_INT,
                  MPI_SUM, comm);
  }

  /**
   * @brief Get the number of primal variables
   *
   * @return int
   */
  int get_num_primals() const { return num_global_primals; }

  /**
   * @brief Get the number of duals/constraints
   *
   * @return int
   */
  int get_num_constraints() const { return num_global_constraints; }

  /**
   * @brief Create an instance of the optimization state vector
   *
   * @return std::shared_ptr<OptVector<T>>
   */
  std::shared_ptr<OptVector<T>> create_opt_vector() const {
    return std::make_shared<OptVector<T>>(num_primals, num_constraints,
                                          problem->create_vector());
  }

  /**
   * @brief Create an instance of an optimization state vector with the provided
   * design vector
   *
   * @return std::shared_ptr<OptVector<T>>
   */
  std::shared_ptr<OptVector<T>> create_opt_vector(
      std::shared_ptr<Vector<T>> x) const {
    return std::make_shared<OptVector<T>>(num_primals, num_constraints, x);
  }

  /**
   * @brief Initialize the dual and slack variables in the problem
   *
   * @param vars All of the optimization variables
   */
  void initialize_duals(T mu, std::shared_ptr<OptVector<T>> vars) const {
    // Project all primals into strict interior of bounds (Section 3.6),
    // then initialize bound duals and slacks from the projected values.
    T* xlam = vars->template get_solution_array<policy>();
    T *zl, *zu;
    vars->template get_bound_duals<policy>(&zl, &zu);

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::project_primals_into_interior(info, xlam);
      detail::initialize_bound_duals(mu, info, xlam, zl, zu);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::project_primals_into_interior_cuda(info, xlam);
      detail::initialize_bound_duals_cuda(mu, info, xlam, zl, zu);
    }
#endif
  }

  /**
   * @brief Compute the negative of the primal-dual residuals based on the value
   * of the gradient and the optimizer state variables
   *
   * This function computes the condensed augmented system RHS (8-block to
   * 4-block).
   *
   * @param mu The barrier parameter for the residual
   * @param vars The optimization variables
   * @param grad The gradient computed from the problem
   * @param res The full KKT residual
   * @return T Returns L2 norm of the condensed residual.
   */
  T compute_residual(T mu, const std::shared_ptr<OptVector<T>> vars,
                     const std::shared_ptr<Vector<T>> grad,
                     std::shared_ptr<Vector<T>> res) const {
    detail::OptState<const T> pt =
        detail::OptState<const T>::template make<policy>(vars);
    T* g = grad->template get_array<policy>();
    T* r = res->template get_array<policy>();

    res->zero();

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_residual(mu, info, pt, g, r);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_residual_cuda(mu, info, pt, g, r);
    }
#endif

    // Compute the local contributions to the residual norm
    T local = res->template dot<policy>(res);
    T total;
    MPI_Allreduce(&local, &total, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return std::sqrt(total);
  }

  /**
   * @brief Compute the diagonal contribution to the KKT matrix
   *
   * @param vars The values of the optimization variables
   * @param diag The vector containing the diagonal components of the matrix
   */
  void compute_diagonal(const std::shared_ptr<OptVector<T>> vars,
                        std::shared_ptr<Vector<T>> diagonal) const {
    // Zero the diagonal
    diagonal->zero();

    detail::OptState<const T> pt =
        detail::OptState<const T>::template make<policy>(vars);
    T* diag = diagonal->template get_array<policy>();

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_diagonal(info, pt, diag);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_diagonal_cuda(info, pt, diag);
    }
#endif
  }

  /**
   * @brief Compute the step in the primal, dual and bound dual variables based
   * on the computed primal dual step. This is equvalent to a block
   * back-substitution for the KKT system.
   *
   * @param mu The barrier parameter
   * @param vars The values of the optimization variables
   * @param px The primal dual update
   * @param update The full primal/dual step
   */
  void compute_update(T mu, const std::shared_ptr<OptVector<T>> vars,
                      const std::shared_ptr<Vector<T>> px,
                      std::shared_ptr<OptVector<T>> update) const {
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    std::shared_ptr<Vector<T>> px_update = update->get_solution();
    if (px_update != px) {
      px_update->copy(px);
    }

    // Set the updates for the dual variables
    T *dzl, *dzu;
    update->template get_bound_duals<policy>(&dzl, &dzu);

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_bound_dual_step(
          mu, info, current, px->template get_array<policy>(), dzl, dzu);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_bound_dual_step_cuda(
          mu, info, current, px->template get_array<policy>(), dzl, dzu);
    }
#endif
  }

  /**
   * @brief Compute the max primal/dual step lengths based on the
   * fraction-to-the-boundary rule.
   *
   * @param tau The fraction to the boundary
   * @param vars The values of the optimization variables
   * @param update The full step (often the result of compute_update)
   * @param ax The max primal step
   * @param xi The index of the constraining variable
   * @param az The max dual step
   * @param zi The index of the constraining dual variable
   */
  void compute_max_step(T tau, const std::shared_ptr<OptVector<T>> vars,
                        const std::shared_ptr<OptVector<T>> update, T& ax,
                        int& xi, T& az, int& zi) const {
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    detail::OptState<const T> step =
        detail::OptState<const T>::template make<policy>(update);

    const T *dzl, *dzu;
    update->template get_bound_duals<policy>(&dzl, &dzu);

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_max_step(tau, info, current, step, ax, xi, az, zi);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_max_step_cuda(tau, info, current, step, ax, xi, az, zi);
    }
#endif
  }

  /**
   * @brief Apply the primal dual step to the variables
   *
   * @param ax The max primal step
   * @param az The max dual step
   * @param vars The values of the optimization variables
   * @param update The full step
   * @param result The variables at the full step
   */
  void apply_step_update(T ax, T az, const std::shared_ptr<OptVector<T>> vars,
                         const std::shared_ptr<OptVector<T>> update,
                         std::shared_ptr<OptVector<T>> result) const {
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    detail::OptState<const T> step =
        detail::OptState<const T>::template make<policy>(update);
    detail::OptState<T> result_state =
        detail::OptState<T>::template make<policy>(result);

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::apply_step(ax, az, info, current, step, result_state);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::apply_step_cuda(ax, az, info, current, step, result_state);
    }
#endif
  }

  /**
   * @brief Compute the average complementarity and the xi parameter from LOQO
   *
   * @param vars The values of the optimization variables
   * @param avg The average complementarity
   * @param xi The parameter xi
   */
  void compute_complementarity(const std::shared_ptr<OptVector<T>> vars, T& avg,
                               T& xi) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T ps[2] = {0, 0};
    T lm = std::numeric_limits<T>::max();

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_complementarity(info, s, ps, lm);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_complementarity_cuda(info, s, ps, lm);
    }
#endif

    T gps[2];
    T gm;
    MPI_Allreduce(ps, gps, 2, get_mpi_type<T>(), MPI_SUM, comm);
    MPI_Allreduce(&lm, &gm, 1, get_mpi_type<T>(), MPI_MIN, comm);

    // Compute the average complementarity
    avg = 0.0;
    if (gps[1] > 0.0) {
      avg = gps[0] / gps[1];
    }

    // Compute the xi vactor
    xi = 1.0;
    if (avg > 0.0) {
      xi = A2D::max2(0.0, A2D::min2(T(1.0), gm / avg));
    }
  }

  /**
   * @brief Compute the KKT errors for the dual, primal and complementarity
   * equations in the infinity norm
   *
   * @param mu The barrier parameter
   * @param vars The values of the optimization variables
   * @param grad The gradient
   * @param d_inf The output dual error
   * @param p_inf The output primal error
   * @param c_inf The output complementarity error
   */
  void compute_kkt_error(T mu, const std::shared_ptr<OptVector<T>> vars,
                         const std::shared_ptr<Vector<T>> grad, T& d_inf,
                         T& p_inf, T& c_inf) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T ld = 0, lp = 0, lc = 0;

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_kkt_error(mu, info, s, grad->template get_array<policy>(),
                                ld, lp, lc);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_kkt_error_cuda(
          mu, info, s, grad->template get_array<policy>(), ld, lp, lc);
    }
#endif

    T lv[3] = {ld, lp, lc}, gv[3];
    MPI_Allreduce(lv, gv, 3, get_mpi_type<T>(), MPI_MAX, comm);
    d_inf = gv[0];
    p_inf = gv[1];
    c_inf = gv[2];
  }

  /**
   * @brief Compute the log-barrier term
   *
   * Returns the value of
   *
   * - mu * log(x - lb) - mu * log(ub - x)
   *
   * @param mu The barrier parameter
   * @param vars The values of the optimization variables
   * @return The value of the log-barrier term
   */
  T compute_log_barrier(T mu, const std::shared_ptr<OptVector<T>> vars) const {
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    T local = 0.0;

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      local = detail::compute_log_barrier(mu, info, current);
    }
#ifdef AMIGO_USE_CUDA
    else {
      local = detail::compute_log_barrier_cuda(mu, info, current);
    }
#endif

    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return result;
  }

  /**
   * @brief Compute the directional derivative of the log-barrier term
   *
   * Returns the value of
   *
   * - mu * e^{T} * (X - LB)^{-1} * px + mu * e^{T} * (UB - X)^{-1} * px
   *
   * @param mu The barrier parameter
   * @param vars The values of the optimization variables
   * @param update The update for the optimization variables
   * @return The value of the directional derivative of the log-barrier term
   */
  T compute_log_barrier_derivative(
      T mu, const std::shared_ptr<OptVector<T>> vars,
      const std::shared_ptr<OptVector<T>> update) const {
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    detail::OptState<const T> step =
        detail::OptState<const T>::template make<policy>(update);

    T local = 0.0;
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      local = detail::compute_log_barrier_derivative(mu, info, current, step);
    }
#ifdef AMIGO_USE_CUDA
    else {
      local =
          detail::compute_log_barrier_derivative_cuda(mu, info, current, step);
    }
#endif

    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return result;
  }

  /**
   * @brief Compute the squared sum of the complementarity
   *
   * @param mu The barrier parameter
   * @param vars The values of the optimization variables
   */
  T compute_sum_squared_complementarity(
      T mu, const std::shared_ptr<OptVector<T>> vars) {
    detail::OptState<const T> current =
        detail::OptState<const T>::template make<policy>(vars);
    T local = 0.0;

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      local = detail::compute_sum_squared_complementarity(mu, info, current);
    }
#ifdef AMIGO_USE_CUDA
    else {
      local =
          detail::compute_sum_squared_complementarity_cuda(mu, info, current);
    }
#endif

    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return result;
  }

  /**
   * @brief Compute the infeasibility in the l1 norm based on the gradient
   *
   * @param gradient The gradient at a point
   */
  T compute_infeasibility(const std::shared_ptr<Vector<T>> gradient) {
    const T* grad = gradient->template get_array<policy>();
    T local = 0.0;

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      local = detail::compute_infeasibility(info, grad);
    }
#ifdef AMIGO_USE_CUDA
    else {
      local = detail::compute_infeasibility_cuda(info, grad);
    }
#endif

    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return result;
  }

  /**
   * @brief Compute the dual residual = grad(f) - zl + zu
   *
   * This is used to find a least-squares estiamte of the initial multipliers
   *
   * @param vars The values of the optimization variables
   * @param gradient The gradient at a point
   * @param out The result = grad(f) - zl + zu
   */
  void compute_dual_residual(const std::shared_ptr<OptVector<T>> vars,
                             const std::shared_ptr<Vector<T>> grad,
                             std::shared_ptr<Vector<T>> out) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);

    out->zero();

    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::compute_dual_residual(info, s, grad->template get_array<policy>(),
                                    out->template get_array<policy>(),
                                    out->get_size());
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::compute_dual_residual_cuda(
          info, s, grad->template get_array<policy>(),
          out->template get_array<policy>(), out->get_size());
    }
#endif
  }

  void get_kkt_element_counts(int& n_d, int& n_p, int& n_c) const {
    n_d = num_primals;      // dual stationarity has n_primal components
    n_p = num_constraints;  // primal feasibility has n_constraints components
    n_c = 0;                // complementarity count: sum of finite bounds
    for (int i = 0; i < num_primals; i++) {
      if (!std::isinf((*lbx)[i])) n_c++;
      if (!std::isinf((*ubx)[i])) n_c++;
    }
  }

  std::shared_ptr<Vector<T>> get_lbx() const { return lbx; }
  std::shared_ptr<Vector<T>> get_ubx() const { return ubx; }

  // Return relaxed bounds if available, otherwise original bounds.
  // These are the bounds actually used by the IPM backend (info.lbx/ubx).
  std::shared_ptr<Vector<T>> get_lbx_relaxed() const {
    if (lbx_relaxed) {
      return lbx_relaxed;
    }
    return lbx;
  }
  std::shared_ptr<Vector<T>> get_ubx_relaxed() const {
    if (ubx_relaxed) {
      return ubx_relaxed;
    }
    return ubx;
  }

  // Relax bounds by bound_relax_factor (default 1e-8).
  // Must be called before initialize_multipliers_and_slacks.
  void relax_bounds(T factor = 1e-8, T constr_viol_tol = 1e-4) {
    if (factor <= 0) return;
    lbx_relaxed =
        std::make_shared<Vector<T>>(num_primals, 0, lbx->get_memory_location());
    ubx_relaxed =
        std::make_shared<Vector<T>>(num_primals, 0, ubx->get_memory_location());

    T* lb_buf = lbx_relaxed->template get_array<policy>();
    T* ub_buf = ubx_relaxed->template get_array<policy>();
    detail::relax_bounds(info, lb_buf, ub_buf, factor, constr_viol_tol);

    lbx_relaxed->copy_host_to_device();
    ubx_relaxed->copy_host_to_device();
  }

 private:
  std::shared_ptr<OptimizationProblem<T, policy>> problem;
  MPI_Comm comm;

  int num_primals;
  int num_constraints;
  std::shared_ptr<Vector<int>> primal_indices;
  std::shared_ptr<Vector<int>> constraint_indices;
  std::shared_ptr<Vector<T>> lbx, ubx;
  std::shared_ptr<Vector<T>> lbx_relaxed, ubx_relaxed;
  std::shared_ptr<Vector<T>> lbh;

  // Object for storing informabout about the problem
  detail::OptProblemInfo<T> info;

  // Number of global constraints
  int num_global_primals;
  int num_global_constraints;
};

}  // namespace amigo

#endif  // AMIGO_INTERIOR_POINT_OPTIMIZER_H
