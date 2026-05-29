"""Per-iteration state carried through the optimization loop.

IpmState holds the scalar counters, step-size history, and
filter/rejection counts that change every iteration.  StepContext
is a lightweight bag of per-iteration scratch data handed to the
barrier strategy each step.
"""

import amigo as am
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional
from .solvers import LinearSolver


class InteriorPointState:
    # Communicator rank
    comm_rank: int

    # Iteration counter
    iter: int

    # Barrier parameter
    mu: float

    # Fraction to the boundary parameter
    tau: float

    # Objective scaling factor
    obj_scale: float

    # Objective function value (scaled) and log-barrier term
    objective_value: float
    log_barrier_value: float
    con_infeasibility: float

    # Current primal-dual vector
    current: am.OptVector

    # Gradient information
    gradient: am.Vector
    gradient_current: bool

    # Second-order information
    diagonal: am.Vector
    hessian: am.CSRMat
    hessian_current: bool

    # Max primal and max dual step lengths
    max_alpha_primal: float
    max_alpha_dual: float
    step: am.OptVector
    step_current: bool

    # Residual and step information
    residual_norm: float
    residual: am.Vector

    # Current error measures
    primal_infeas: float
    dual_infeas: float
    complementarity: float
    kkt_error: float
    residual_current: bool

    def __init__(self, x, options, problem, optimizer):
        self.comm_rank = 0
        self.iter = 0
        self.mu = options["initial_barrier_param"]
        self.tau = options["fraction_to_boundary"]
        self.obj_scale = 1.0
        self.objective_value = 0.0
        self.log_barrier_value = 0.0
        self.con_infeasibility = 0.0

        self.current = optimizer.create_opt_vector(x)
        self.gradient = problem.create_vector()
        self.gradient_current = False

        self.residual = problem.create_vector()
        self.diagonal = problem.create_vector()
        self.hessian = problem.create_matrix()
        self.hessian_current = False

        self.max_alpha_primal = 1.0
        self.max_alpha_dual = 1.0
        self.step = optimizer.create_opt_vector()
        self.step_current = False

        self.residual_norm = 0.0
        self.residual = problem.create_vector()
        self.primal_infeas = 0.0
        self.dual_infeas = 0.0
        self.complementarity = 0.0
        self.kkt_error = 0.0
        self.residual_current = False

    def get_current_point(self):
        """Get the current primal-dual vector"""
        return self.current.get_solution()

    def get_trial_point(self):
        """Get the trial primal-dual vector"""
        return self.trial.get_solution()

    def invalidate(self):
        self.gradient_current = False
        self.hessian_current = False
        self.residual_current = False
        self.step_current = False


class Evaluator:
    def __init__(self, problem, optimizer):
        self.problem = problem
        self.optimizer = optimizer

        # Create a temporary constraint vector for later usage
        self.temp_con = self.problem.create_constraint_vector()

    def evaluate_gradient(self, state):
        """Evaluate the gradient at the current point and store in the state"""
        if not state.gradient_current:
            self.evaluate_gradient_from_point(
                state.obj_scale, state.current, state.gradient
            )
            state.gradient_current = True

    def evaluate_gradient_from_point(self, obj_scale, vars, gradient):
        """Evaluate the gradient at a trial point and store in an external vector"""
        x = vars.get_solution()
        self.problem.update(x)
        self.problem.gradient(obj_scale, x, gradient)

    def evaluate_hessian(self, state):
        """Evaluate the hessian at the current point and store in the state"""
        if not state.hessian_current:
            x = state.current.get_solution()
            self.problem.hessian(state.obj_scale, x, state.hessian)
            state.hessian_current = True

    def evaluate_objective_and_barrier_from_point(self, mu, obj_scale, vars):
        # Zero dual, evaluate L(x,0) = f(x), restore
        con_indices = self.problem.get_constraint_indices()

        # Get the values of the design variables
        x = vars.get_solution()

        # Save the dual values and then zero them
        x.get_values_at(con_indices, self.temp_con)
        x.fill_at(con_indices, 0.0)
        fobj = self.problem.lagrangian(obj_scale, x)

        # Restore the dual variable values
        x.set_values_at(con_indices, self.temp_con)

        barrier = self.optimizer.compute_barrier_log_sum(mu, vars)

        return fobj, barrier

    def evaluate_objective_and_barrier(self, state):
        """Evaluate the objective and log-barrier terms at the current point"""
        fobj, barrier = self.evaluate_objective_and_barrier_from_point(
            state.mu, state.obj_scale, state.current
        )
        state.objective_value = fobj
        state.log_barrier_value = barrier
        return

    def evaluate_directional_derivative(self, state):
        if not state.gradient_current:
            self.evaluate_gradient(state)
        if not state.step_current:
            raise ValueError("Step not current at this point")

        update = state.step.get_solution()
        return self.optimizer.compute_barrier_dphi_direct(
            state.mu, state.current, state.gradient, update
        )

    def evalate_infeasibility_from_gradient(self, grad):
        con_indices = self.problem.get_constraint_indices()
        grad.get_values_at(con_indices, self.temp_con)
        return self.problem.abssum(self.temp_con)

    def evaluate_infeasibility(self, state):
        if not state.gradient_current:
            self.evaluate_gradient(state)

        infeas = self.evalate_infeasibility_from_gradient(state.gradient)
        state.con_infeasibility = infeas
        return

    def evaluate_residual(self, state):
        if not state.gradient_current:
            self.evaluate_gradient(state)

        # Evaluate the residual and update the residual norm
        state.residual_norm = self.optimizer.compute_residual(
            state.mu, state.current, state.gradient, state.residual
        )

        # Now compute the infeasibilities
        state.dual_infeas, state.primal_infeas, state.complementarity = (
            self.optimizer.compute_kkt_error_mu(0.0, state.current, state.gradient)
        )

        # Compute the scaling
        s_d_conv, s_c_conv = self._compute_optimality_scaling(state)
        state.kkt_error = max(
            state.dual_infeas / s_d_conv,
            state.primal_infeas,
            state.complementarity / s_c_conv,
        )

    def _compute_optimality_scaling(self, state):
        """Compute optimality error scaling factors (s_d, s_c)."""
        # TODO: move to backend: backend.optimality_scaling() that returns
        # (s_d, s_c) without reading zl/zu/bounds from Python.
        s_max = 100.0

        # Get the vectors
        zl = state.current.get_zl()
        zu = state.current.get_zu()
        z_asum = self.problem.abssum(zl) + self.problem.abssum(zu)

        # Get the lower/upper bounds, copy them to the host. Need to fix this
        lbx = self.optimizer.get_lbx()
        ubx = self.optimizer.get_ubx()
        lbx_array = lbx.get_array()
        ubx_array = ubx.get_array()

        n_bounds = int(np.sum(np.isfinite(lbx_array)) + np.sum(np.isfinite(ubx_array)))

        # Get the sum of the absolute values of the multipliers
        con_indices = self.problem.get_constraint_indices()
        y = self.problem.create_constraint_vector()
        x = state.current.get_solution()
        x.get_values_at(con_indices, y)
        y_asum = self.problem.abssum(y)

        n_constraints = self.optimizer.get_num_constraints()

        # s_c: bound multiplier scaling
        if n_bounds == 0:
            s_c = 1.0
        else:
            s_c = max(s_max, z_asum / n_bounds) / s_max

        # s_d: dual (stationarity) scaling
        n_all = n_constraints + n_bounds
        if n_all == 0:
            s_d = 1.0
        else:
            s_d = max(s_max, (y_asum + z_asum) / n_all) / s_max

        s_c = 1.0
        s_d = 1.0

        return s_d, s_c

    def evaluate_diagonal(self, state):
        """Evaluate the diagonal entries"""
        self.optimizer.compute_diagonal(state.current, state.diagonal)


@dataclass
class IpmState:
    """Mutable per-iteration state of the interior-point loop."""

    # Step tracking (updated after each accepted step)
    line_iters: int = 0
    alpha_x_prev: float = 0.0
    alpha_z_prev: float = 0.0
    x_index_prev: int = -1
    z_index_prev: int = -1

    # Convergence tracking
    prev_res_norm: float = float("inf")
    acceptable_counter: int = 0
    precision_floor_count: int = 0

    # Rejection tracking
    consecutive_rejections: int = 0
    zero_step_count: int = 0

    # Classical-barrier filter monotone fallback
    filter_monotone_mode: bool = False
    filter_monotone_mu: Optional[float] = None

    # Filter reset heuristic
    count_successive_filter_rejections: int = 0
    filter_reset_count: int = 0

    # Barrier parameter at the start of the most recent residual eval
    res_norm_mu: float = 0.0


@dataclass
class StepContext:
    """Per-iteration inputs passed to BarrierStrategy.step()."""

    i: int = 0
    comm_rank: int = 0
    res_norm: float = 0.0
    tol: float = 0.0
    compl_inf_tol: float = 0.0

    # Problem structure
    x: Any = None
    diag_base: Any = None

    # Inertia correction + zero-Hessian handling
    inertia_corrector: Any = None
    zero_hessian_indices: Any = None
    zero_hessian_eps: float = 0.0

    # Classical-barrier filter monotone fallback (strategy may update
    # filter_monotone_mu in place; driver reads it back)
    filter_monotone_mode: bool = False
    filter_monotone_mu: Optional[float] = None
