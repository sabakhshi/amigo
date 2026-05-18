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
  OptVector(int num_primal, int num_constraints, std::shared_ptr<Vector<T>> x)
      : num_primal(num_primal), num_constraints(num_constraints), x(x) {
    // Create the slack and dual variable vectors for the upper and lower bounds
    sl = std::make_shared<Vector<T>>(num_primal, 0, x->get_memory_location());
    su = std::make_shared<Vector<T>>(num_primal, 0, x->get_memory_location());
    zl = std::make_shared<Vector<T>>(num_primal, 0, x->get_memory_location());
    zu = std::make_shared<Vector<T>>(num_primal, 0, x->get_memory_location());
  }

  void zero() {
    x->zero();
    sl->zero();
    su->zero();
    zl->zero();
    zu->zero();
  }

  void copy(std::shared_ptr<OptVector<T>> src) {
    x->copy(*src->x);
    sl->copy(*src->sl);
    su->copy(*src->su);
    zl->copy(*src->zl);
    zu->copy(*src->zu);
  }

  std::shared_ptr<Vector<T>> get_zl() { return zl; }
  std::shared_ptr<Vector<T>> get_zu() { return zu; }
  std::shared_ptr<Vector<T>> get_sl() { return sl; }
  std::shared_ptr<Vector<T>> get_su() { return su; }

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
  void get_bound_slacks(T** sl_, T** su_) {
    if (sl_) {
      *sl_ = sl->template get_array<policy>();
    }
    if (su_) {
      *su_ = su->template get_array<policy>();
    }
  }
  template <ExecPolicy policy>
  void get_bound_slacks(const T** sl_, const T** su_) const {
    if (sl_) {
      *sl_ = sl->template get_array<policy>();
    }
    if (su_) {
      *su_ = su->template get_array<policy>();
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

  std::shared_ptr<Vector<T>> get_solution() { return x; }
  const std::shared_ptr<Vector<T>> get_solution() const { return x; }

  int get_num_primal() const { return num_primal; }
  int get_num_constraints() const { return num_constraints; }

 private:
  int num_primal, num_constraints;

  // The primal/dual vector
  std::shared_ptr<Vector<T>> x;

  // Duals and primals associated with the lower/upper bounds
  std::shared_ptr<Vector<T>> sl, su;
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

    num_primal = 0;
    num_constraints = 0;

    // Count up the number of primal variables and number of constraints/dual
    // variables. Note that the number of primal and constraints will not sum
    // up to the number of variables since variables may also be FIXED!
    for (int i = 0; i < size; i++) {
      if (vtypes[i] == static_cast<int>(OptVarType::PRIMAL) ||
          vtypes[i] == static_cast<int>(OptVarType::SLACK)) {
        num_primal++;
      } else if (vtypes[i] == static_cast<int>(OptVarType::DUAL_EQUALITY) ||
                 vtypes[i] == static_cast<int>(OptVarType::DUAL_INEQUALITY)) {
        num_constraints++;
      }
    }

    // Set the memory location depending on the execution policy
    MemoryLocation loc = MemoryLocation::HOST_ONLY;
    if (policy == ExecPolicy::CUDA) {
      loc = MemoryLocation::HOST_AND_DEVICE;
    }

    // Make the vectors that contain the indices of the primal values and
    // constraints
    primal_indices = std::make_shared<Vector<int>>(num_primal, 0, loc);
    constraint_indices = std::make_shared<Vector<int>>(num_constraints, 0, loc);

    lbx = std::make_shared<Vector<T>>(num_primal, 0, loc);
    ubx = std::make_shared<Vector<T>>(num_primal, 0, loc);
    lbh = std::make_shared<Vector<T>>(num_constraints, 0, loc);

    for (int i = 0, primal = 0, con = 0; i < size; i++) {
      if (vtypes[i] == static_cast<int>(OptVarType::PRIMAL) ||
          vtypes[i] == static_cast<int>(OptVarType::SLACK)) {
        (*constraint_indices)[con] = i;
        (*lbh)[con] = lb[i];
        con++;
      } else if (vtypes[i] == static_cast<int>(OptVarType::DUAL_EQUALITY) ||
                 vtypes[i] == static_cast<int>(OptVarType::DUAL_INEQUALITY)) {
        (*primal_indices)[primal] = i;
        (*lbx)[primal] = lb[i];
        (*ubx)[primal] = ub[i];
        primal++;
      }
    }

    // Copy the variable information to the device
    primal_indices->copy_host_to_device();
    constraint_indices->copy_host_to_device();
    lbx->copy_host_to_device();
    ubx->copy_host_to_device();
    lbh->copy_host_to_device();

    // Set the host/device pointers into the info
    info.num_primal = num_primal;
    info.num_constraints = num_constraints;
    info.primal_indices = primal_indices->template get_array<policy>();
    info.constraint_indices = constraint_indices->template get_array<policy>();
    info.lbx = lbx->template get_array<policy>();
    info.ubx = ubx->template get_array<policy>();
    info.lbh = lbh->template get_array<policy>();
  }

  /**
   * @brief Create an instance of the optimization state vector
   *
   * @return std::shared_ptr<OptVector<T>>
   */
  std::shared_ptr<OptVector<T>> create_opt_vector() const {
    return std::make_shared<OptVector<T>>(num_primal, num_constraints,
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
    return std::make_shared<OptVector<T>>(num_primal, num_constraints, x);
  }

  /**
   * @brief Set the multiplier/dual varaibles to the specified value
   *
   * @param value Value to place into the multiplier components
   * @param x Vector
   */
  void set_dual_values(T value, std::shared_ptr<Vector<T>> x) const {
    T* x_array = x->template get_array<policy>();
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::set_dual_values(info, value, x_array);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::set_dual_values(info, value, x_array);
    }
#endif
  }

  /**
   * @brief Set the design variables to the specified value
   *
   * @param value Value to place into the design variable components
   * @param x Vector
   */
  void set_primal_values(T value, std::shared_ptr<Vector<T>> x) const {
    T* x_array = x->template get_array<policy>();
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::set_primal_values(info, value, x_array);
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::set_primal_values_cuda(info, value, x_array);
    }
#endif
  }

  /**
   * @brief Copy only the duals/multipliers from the src to the dest vector
   *
   * @param dest Destination vector
   * @param src Source vector
   */
  void copy_duals(std::shared_ptr<Vector<T>> dest,
                  std::shared_ptr<Vector<T>> src) const {
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::copy_duals(info, src->template get_array<policy>(),
                         dest->template get_array<policy>());
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::copy_duals_cuda(info, src->template get_array<policy>(),
                              dest->template get_array<policy>());
    }
#endif
  }

  /**
   * @brief Copy only the design variables from the src to the dest vector
   *
   * @param dest Destination vector
   * @param src Source vector
   */
  void copy_primals(std::shared_ptr<Vector<T>> dest,
                    std::shared_ptr<Vector<T>> src) const {
    if constexpr (policy == ExecPolicy::SERIAL ||
                  policy == ExecPolicy::OPENMP) {
      detail::copy_primals(info, src->template get_array<policy>(),
                           dest->template get_array<policy>());
    }
#ifdef AMIGO_USE_CUDA
    else {
      detail::copy_primals_cuda(info, src->template get_array<policy>(),
                                dest->template get_array<policy>());
    }
#endif
  }

  /**
   * @brief Initialize the dual and slack variables in the problem
   *
   * @param vars All of the optimization variables
   */
  void initialize_duals_and_slacks(T mu,
                                   std::shared_ptr<OptVector<T>> vars) const {
    // Project all primals into strict interior of bounds (Section 3.6),
    // then initialize bound duals and slacks from the projected values.
    T* xlam = vars->template get_solution_array<policy>();
    detail::project_primals_into_interior(info, xlam);

    T *zl, *zu;
    vars->template get_bound_duals<policy>(&zl, &zu);
    detail::initialize_bound_duals(mu, info, xlam, zl, zu);

    T *sl, *su;
    vars->template get_bound_slacks<policy>(&sl, &su);
    detail::initialize_slacks(info, xlam, sl, su);
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
    T local = res->template dot<policy>(*res);
    T total;
    MPI_Allreduce(&local, &total, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return std::sqrt(total);
  }

  void compute_residual_and_infeasibility(
      T mu, const std::shared_ptr<OptVector<T>> vars,
      const std::shared_ptr<Vector<T>> grad, std::shared_ptr<Vector<T>> res,
      T& dsq, T& psq) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    res->zero();
    detail::compute_residual_and_infeasibility(
        mu, info, s, grad->template get_array<policy>(),
        res->template get_array<policy>(), dsq, psq);
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

  // Copy augmented solution into update, then back-substitute for bound
  // duals.
  void compute_update(T mu, const std::shared_ptr<OptVector<T>> vars,
                      const std::shared_ptr<Vector<T>> px,
                      std::shared_ptr<OptVector<T>> upd) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    upd->get_solution()->copy(*px);
    T *dzl, *dzu;
    upd->template get_bound_duals<policy>(&dzl, &dzu);
    detail::compute_bound_dual_step(mu, info, s,
                                    px->template get_array<policy>(), dzl, dzu);
  }

  void compute_max_step(T tau, const std::shared_ptr<OptVector<T>> vars,
                        const std::shared_ptr<OptVector<T>> upd, T& ax, int& xi,
                        T& az, int& zi) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    const T *dzl, *dzu;
    upd->template get_bound_duals<policy>(&dzl, &dzu);
    detail::compute_max_step(tau, info, s,
                             upd->template get_solution_array<policy>(), dzl,
                             dzu, ax, xi, az, zi);
  }

  void apply_step_update(T ax, T az, const std::shared_ptr<OptVector<T>> vars,
                         const std::shared_ptr<OptVector<T>> upd,
                         std::shared_ptr<OptVector<T>> tmp) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    const T* dxlam = upd->template get_solution_array<policy>();
    const T *dzl, *dzu;
    upd->template get_bound_duals<policy>(&dzl, &dzu);
    T* xlam_n = tmp->template get_solution_array<policy>();
    T *zl_n, *zu_n;
    tmp->template get_bound_duals<policy>(&zl_n, &zu_n);
    T *sl_n, *su_n;
    tmp->template get_bound_slacks<policy>(&sl_n, &su_n);
    int n = info.num_primal + info.num_constraints;
    detail::apply_step(ax, az, info, s, dxlam, dzl, dzu, xlam_n, n, zl_n, zu_n,
                       sl_n, su_n);
  }

  // Python: avg_comp, xi = optimizer.compute_complementarity(vars)
  void compute_complementarity(const std::shared_ptr<OptVector<T>> vars, T& avg,
                               T& xi) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T ps[2] = {0, 0};
    T lm = std::numeric_limits<T>::max();
    detail::compute_complementarity(info, s, ps, lm);
    T gps[2];
    T gm;
    MPI_Allreduce(ps, gps, 2, get_mpi_type<T>(), MPI_SUM, comm);
    MPI_Allreduce(&lm, &gm, 1, get_mpi_type<T>(), MPI_MIN, comm);
    avg = (gps[1] > 0) ? gps[0] / gps[1] : 0.0;
    xi = (avg > 0) ? A2D::max2(T(0), A2D::min2(T(1), gm / avg)) : T(1);
  }

  // Python: scalar = optimizer.compute_complementarity_sq(vars)
  void compute_complementarity_sq(const std::shared_ptr<OptVector<T>> vars,
                                  T& sq) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T local = 0;
    detail::compute_complementarity_sq(info, s, T(0), local);
    MPI_Allreduce(&local, &sq, 1, get_mpi_type<T>(), MPI_SUM, comm);
  }

  // Python: dev = optimizer.compute_max_comp_deviation(vars, mu)
  void compute_max_comp_deviation(const std::shared_ptr<OptVector<T>> vars,
                                  T mu, T& md) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T local;
    detail::compute_max_comp_deviation(info, s, mu, local);
    MPI_Allreduce(&local, &md, 1, get_mpi_type<T>(), MPI_MAX, comm);
  }

  // Python: d_sq, p_sq, c_sq = optimizer.compute_kkt_error(vars, grad)
  void compute_kkt_error(const std::shared_ptr<OptVector<T>> vars,
                         const std::shared_ptr<Vector<T>> grad, T& d_sq,
                         T& p_sq, T& c_sq) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T ld = 0, lp = 0, lc = 0;
    detail::compute_kkt_error_sq(info, s, grad->template get_array<policy>(),
                                 ld, lp, lc);
    T lv[3] = {ld, lp, lc}, gv[3];
    MPI_Allreduce(lv, gv, 3, get_mpi_type<T>(), MPI_SUM, comm);
    d_sq = gv[0];
    p_sq = gv[1];
    c_sq = gv[2];
  }

  // Python: theta = optimizer.compute_constraint_violation_1norm(vars, grad)
  // Constraint violation 1-norm for filter line search.
  T compute_constraint_violation_1norm(
      const std::shared_ptr<OptVector<T>> vars,
      const std::shared_ptr<Vector<T>> grad) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T local = detail::compute_constraint_violation_1norm(
        info, s, grad->template get_array<policy>());
    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return result;
  }

  // Python: d_inf, p_inf, c_inf = optimizer.compute_kkt_error_mu(mu, vars,
  // grad) Eq. 5: infinity-norm KKT error with barrier complementarity.
  void compute_kkt_error_mu(T mu, const std::shared_ptr<OptVector<T>> vars,
                            const std::shared_ptr<Vector<T>> grad, T& d_inf,
                            T& p_inf, T& c_inf) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T ld = 0, lp = 0, lc = 0;
    detail::compute_kkt_error(mu, info, s, grad->template get_array<policy>(),
                              ld, lp, lc);
    T lv[3] = {ld, lp, lc}, gv[3];
    MPI_Allreduce(lv, gv, 3, get_mpi_type<T>(), MPI_MAX, comm);
    d_inf = gv[0];
    p_inf = gv[1];
    c_inf = gv[2];
  }

  // Python: dphi = optimizer.compute_barrier_dphi(mu, vars, update, res, px,
  // diag) KKT residual form (used by QF oracle for quality function
  // evaluation).
  T compute_barrier_dphi(T mu, const std::shared_ptr<OptVector<T>> vars,
                         const std::shared_ptr<OptVector<T>> update,
                         const std::shared_ptr<Vector<T>> res,
                         const std::shared_ptr<Vector<T>> px,
                         const std::shared_ptr<Vector<T>> diag) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T local = detail::compute_barrier_dphi_from_kkt(
        info, s, res->template get_array<policy>(),
        px->template get_array<policy>(), diag->template get_array<policy>());
    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return result;
  }

  // Python: dphi = optimizer.compute_barrier_dphi_direct(mu, vars, grad, px)
  // Direct form: grad_barrier^T * dx = sum(grad_f*dx - mu*dx/sl + mu*dx/su)
  // Direct barrier directional derivative for the filter line search.
  T compute_barrier_dphi_direct(T mu, const std::shared_ptr<OptVector<T>> vars,
                                const std::shared_ptr<Vector<T>> grad,
                                const std::shared_ptr<Vector<T>> px) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T local = detail::compute_barrier_dphi(mu, info, s,
                                           grad->template get_array<policy>(),
                                           px->template get_array<policy>());
    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return result;
  }

  T compute_barrier_log_sum(T mu,
                            const std::shared_ptr<OptVector<T>> vars) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T local = detail::compute_barrier_log_sum(mu, info, s);
    T result;
    MPI_Allreduce(&local, &result, 1, get_mpi_type<T>(), MPI_SUM, comm);
    return result;
  }

  // Python: optimizer.reset_bound_multipliers(mu, kappa, vars) -- in-place
  void reset_bound_multipliers(T mu, T kappa,
                               std::shared_ptr<OptVector<T>> vars) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    T *zl, *zu;
    vars->template get_bound_duals<policy>(&zl, &zu);
    detail::reset_bound_multipliers(mu, kappa, info, s, zl, zu);
  }

  void compute_affine_start_point(T bm,
                                  const std::shared_ptr<OptVector<T>> vars,
                                  const std::shared_ptr<OptVector<T>> upd,
                                  std::shared_ptr<OptVector<T>> dst) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    const T *dzl, *dzu;
    upd->template get_bound_duals<policy>(&dzl, &dzu);
    T *zl_o, *zu_o;
    dst->template get_bound_duals<policy>(&zl_o, &zu_o);
    detail::compute_affine_start_point(bm, info, s, dzl, dzu, zl_o, zu_o);
  }

  void compute_dual_residual_vector(const std::shared_ptr<OptVector<T>> vars,
                                    const std::shared_ptr<Vector<T>> grad,
                                    std::shared_ptr<Vector<T>> out) const {
    detail::OptState<const T> s =
        detail::OptState<const T>::template make<policy>(vars);
    detail::compute_dual_residual_vector(
        info, s, grad->template get_array<policy>(),
        out->template get_array<policy>(), out->get_size());
  }

  void get_kkt_element_counts(int& n_d, int& n_p, int& n_c) const {
    n_d = num_primal;       // dual stationarity has n_primal components
    n_p = num_constraints;  // primal feasibility has n_constraints components
    n_c = 0;                // complementarity count: sum of finite bounds
    for (int i = 0; i < num_primal; i++) {
      if (!std::isinf((*lbx)[i])) n_c++;
      if (!std::isinf((*ubx)[i])) n_c++;
    }
  }

  // Debug: verify KKT system solve by forming full unreduced residual
  void check_update(T mu, const std::shared_ptr<Vector<T>> grad,
                    const std::shared_ptr<OptVector<T>> vars,
                    const std::shared_ptr<OptVector<T>> update,
                    const std::shared_ptr<CSRMat<T>> hess) const {
    // TODO: implement debug verification for new 2x2 system
  }

  // Slack mapping
  // Register which primals are slacks and which constraints they correspond
  // to. After this call, initialize_slacks() becomes available.
  //   slack_global:  global variable indices of slack variables
  //   constr_global: global variable indices of the inequality constraints
  // Both arrays must have the same length (n_slacks).
  void set_slack_mapping(int n_slacks, const int* slack_global,
                         const int* constr_global) {
    auto ml = (policy == ExecPolicy::CUDA) ? MemoryLocation::HOST_AND_DEVICE
                                           : MemoryLocation::HOST_ONLY;
    n_slacks_ = n_slacks;
    slack_global_ = std::make_shared<Vector<int>>(n_slacks, 0, ml);
    constr_global_ = std::make_shared<Vector<int>>(n_slacks, 0, ml);
    for (int k = 0; k < n_slacks; k++) {
      (*slack_global_)[k] = slack_global[k];
      (*constr_global_)[k] = constr_global[k];
    }
    slack_global_->copy_host_to_device();
    constr_global_->copy_host_to_device();
  }

  // Initialize slacks to s = d(x).
  //
  // Each inequality is reformulated as c_k(x) - s_k = 0.  The gradient
  // at constraint index ci holds the primal residual c_k(x) - s_k.
  // Recovering the constraint body: d_k(x) = (c_k(x) - s_k) + s_k.
  //
  // Requires gradient to have been evaluated at the current (x, s).
  // After this call, use initialize_multipliers_and_slacks() to push
  // the new slacks into bounds and reset bound duals.
  void initialize_slacks(const std::shared_ptr<Vector<T>> grad,
                         std::shared_ptr<OptVector<T>> vars) const {
    if (n_slacks_ == 0) return;
    T* xlam = vars->template get_solution_array<policy>();
    const T* g = grad->template get_array<policy>();
    for (int k = 0; k < n_slacks_; k++) {
      int si = (*slack_global_)[k];
      int ci = (*constr_global_)[k];
      T residual = g[ci];           // c_k(x) - s_k
      T s_old = xlam[si];           // current slack
      xlam[si] = residual + s_old;  // d_k(x) = c_k(x)
    }
  }

  bool has_slacks() const { return n_slacks_ > 0; }
  int get_num_slacks() const { return n_slacks_; }

  // NLP scaling: gradient-based.  Computed once at the initial point.
  // obj_scale_ scales the objective via alpha; constr_scale_ scales
  // Jacobian rows via D*H*D and gradient post-processing.

  /// Compute scaling factors from initial gradient and Jacobian row norms.
  void compute_nlp_scaling(const std::shared_ptr<Vector<T>> x,
                           const std::shared_ptr<Vector<T>> grad,
                           T max_gradient = T(100), T min_value = T(1e-8)) {
    const T* g = grad->template get_array<policy>();

    // Objective: df = min(1, max_gradient / ||grad_f||_inf)
    T obj_max = T(0);
    for (int i = 0; i < num_primal; i++) {
      T v = std::abs(g[(*primal_indices)[i]]);
      if (v > obj_max) obj_max = v;
    }
    T global_obj_max;
    MPI_Allreduce(&obj_max, &global_obj_max, 1, get_mpi_type<T>(), MPI_MAX,
                  comm);

    obj_scale_ =
        (global_obj_max > max_gradient) ? max_gradient / global_obj_max : T(1);
    if (obj_scale_ < min_value) obj_scale_ = min_value;

    // Constraint: dc[j] = min(1, max_gradient / ||J_row_j||_inf)
    // Assemble KKT at (x, lam=0) to read Jacobian rows.
    auto hess = problem->create_matrix();
    problem->hessian(T(1), x, hess);
    hess->copy_data_device_to_host();

    int nr, num_constraintsmat, nnz_mat;
    const int *rp, *cl;
    T* dt;
    hess->get_data(&nr, &num_constraintsmat, &nnz_mat, &rp, &cl, &dt);

    auto ml = (policy == ExecPolicy::CUDA) ? MemoryLocation::HOST_AND_DEVICE
                                           : MemoryLocation::HOST_ONLY;
    constr_scale_ = std::make_shared<Vector<T>>(num_constraints, 0, ml);

    // Scan each constraint row for its inf-norm
    for (int j = 0; j < num_constraints; j++) {
      int row = (*constraint_indices)[j];
      T row_max = T(0);
      for (int k = rp[row]; k < rp[row + 1]; k++) {
        T v = std::abs(dt[k]);
        if (v > row_max) row_max = v;
      }
      T dc = (row_max > max_gradient) ? max_gradient / row_max : T(1);
      if (dc < min_value) dc = min_value;
      (*constr_scale_)[j] = dc;
    }

    // Scale constraint targets for consistency
    for (int j = 0; j < num_constraints; j++) {
      T dc = (*constr_scale_)[j];
      if (dc != T(1)) (*lbh)[j] *= dc;
    }
    lbh->copy_host_to_device();
    info.lbh = lbh->template get_array<policy>();

    // Per-variable scaling: 1.0 for primals, dc[j] for constraints.
    scale_vec_.resize(nr, T(1));
    for (int j = 0; j < num_constraints; j++)
      scale_vec_[(*constraint_indices)[j]] = (*constr_scale_)[j];

    constr_scale_->copy_host_to_device();
    scaling_active_ = true;
  }

  /// Scale constraint rows of the gradient in-place.
  void apply_gradient_scaling(std::shared_ptr<Vector<T>> grad) const {
    if (!scaling_active_) return;
    T* g = grad->template get_array<policy>();
    for (int j = 0; j < num_constraints; j++)
      g[(*constraint_indices)[j]] *= (*constr_scale_)[j];
  }

  /// D*H*D similarity transform on the KKT matrix (scales Jacobian blocks).
  void apply_hessian_scaling(std::shared_ptr<CSRMat<T>> hess) const {
    if (!scaling_active_) return;
    int nr, nc, nnz_mat;
    const int *rp, *cl;
    T* dt;
    hess->get_data(&nr, &nc, &nnz_mat, &rp, &cl, &dt);
    const T* sv = scale_vec_.data();
    for (int i = 0; i < nr; i++) {
      T di = sv[i];
      for (int k = rp[i]; k < rp[i + 1]; k++) {
        dt[k] *= di * sv[cl[k]];
      }
    }
  }

  /// Scale constraint multipliers into scaled space: y[j] *= dc[j].
  void scale_multipliers(std::shared_ptr<Vector<T>> x) const {
    if (!scaling_active_) return;
    T* xlam = x->template get_array<policy>();
    for (int j = 0; j < num_constraints; j++)
      xlam[(*constraint_indices)[j]] *= (*constr_scale_)[j];
  }

  /// Unscale constraint multipliers: y[j] /= dc[j].
  void unscale_multipliers(std::shared_ptr<Vector<T>> x) const {
    if (!scaling_active_) return;
    T* xlam = x->template get_array<policy>();
    for (int j = 0; j < num_constraints; j++)
      xlam[(*constraint_indices)[j]] /= (*constr_scale_)[j];
  }

  T get_obj_scale() const { return obj_scale_; }
  bool has_scaling() const { return scaling_active_; }

  // Accessors
  int get_num_design_variables() const { return num_primal; }
  int get_num_constraints() const { return num_constraints; }
  int get_num_equalities() const { return num_constraints; }
  int get_num_inequalities() const { return 0; }

  std::shared_ptr<Vector<T>> get_lbx() const { return lbx; }
  std::shared_ptr<Vector<T>> get_ubx() const { return ubx; }

  // Return relaxed bounds if available, otherwise original bounds.
  // These are the bounds actually used by the IPM backend (info.lbx/ubx).
  std::shared_ptr<Vector<T>> get_lbx_relaxed() const {
    return lbx_relaxed ? lbx_relaxed : lbx;
  }
  std::shared_ptr<Vector<T>> get_ubx_relaxed() const {
    return ubx_relaxed ? ubx_relaxed : ubx;
  }

  // Relax bounds by bound_relax_factor (default 1e-8).
  // Must be called before initialize_multipliers_and_slacks.
  void relax_bounds(T factor = 1e-8, T constr_viol_tol = 1e-4) {
    if (factor <= 0) return;
    lbx_relaxed =
        std::make_shared<Vector<T>>(num_primal, 0, lbx->get_memory_location());
    ubx_relaxed =
        std::make_shared<Vector<T>>(num_primal, 0, ubx->get_memory_location());

    T* lb_buf = lbx_relaxed->template get_array<policy>();
    T* ub_buf = ubx_relaxed->template get_array<policy>();
    detail::relax_bounds(info, lb_buf, ub_buf, factor, constr_viol_tol);

    lbx_relaxed->copy_host_to_device();
    ubx_relaxed->copy_host_to_device();
  }

 private:
  std::shared_ptr<OptimizationProblem<T, policy>> problem;
  MPI_Comm comm;

  int num_primal, num_constraints;
  std::shared_ptr<Vector<int>> primal_indices;
  std::shared_ptr<Vector<int>> constraint_indices;
  std::shared_ptr<Vector<T>> lbx, ubx;
  std::shared_ptr<Vector<T>> lbx_relaxed, ubx_relaxed;
  std::shared_ptr<Vector<T>> lbh;
  detail::OptProblemInfo<T> info;

  // Slack-to-constraint mapping (set via set_slack_mapping)
  int n_slacks_ = 0;
  std::shared_ptr<Vector<int>> slack_global_;
  std::shared_ptr<Vector<int>> constr_global_;

  // NLP scaling state
  T obj_scale_ = T(1);
  std::shared_ptr<Vector<T>> constr_scale_;
  std::vector<T> scale_vec_;  // per-variable: 1.0 primals, dc[j] constraints
  bool scaling_active_ = false;
};

}  // namespace amigo

#endif  // AMIGO_INTERIOR_POINT_OPTIMIZER_H
