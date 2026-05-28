"""Per-iteration state carried through the optimization loop.

IpmState holds the scalar counters, step-size history, and
filter/rejection counts that change every iteration.  StepContext
is a lightweight bag of per-iteration scratch data handed to the
barrier strategy each step.
"""

import amigo as am
from dataclasses import dataclass
from typing import Any, Optional
from .solvers import LinearSolver


class InteriorPointState:
    # Iteration counters
    iter: int
    restoration_iter: int

    # Barrier parameter
    mu: float

    # Fraction to the boundary parameter
    tau: float

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
    step: am.Vector

    # Reduced residual and step information
    residual_norm: float
    residual: am.Vector

    # Current error measures
    primal_infeas: float
    dual_infeas: float
    complementarity: float
    kkt_error: float

    def __init__(self, x, options, problem, optimizer):
        self.iter = 0
        self.restoration_iter
        self.mu = options["initial_barrier_param"]
        self.tau = options["fraction_to_boundary"]

        self.current = optimizer.create_opt_vector(x)
        self.gradient = problem.create_vector()
        self.gradient_current = False

        self.residual = self.create_vector()
        self.diagonal = problem.create_vector()
        self.hessian = problem.create_matrix()
        self.hessian_current = False

        self.max_alpha_primal = 1.0
        self.max_alpha_dual = 1.0
        self.residual = problem.create_vector()
        self.step = problem.create_vector()

        self.full_residual = self.create_opt_vector()
        self.full_step = self.create_opt_vector()

    def get_current_point(self):
        """Get the current primal-dual vector"""
        return self.current.get_solution()

    def get_trial_point(self):
        """Get the trial primal-dual vector"""
        return self.trial.get_solution()

    def invalidate(self):
        self.gradient_current = False
        self.hessian_current = False


class Evaluator:
    def __init__(self, problem, optimizer):
        self.problem = problem
        self.optimizer = optimizer

        # Create a temporary constraint vector for later usage
        self.temp_con = self.problem.create_constraint_vector()

    def evaluate_gradient(self, state):
        if not state.gradient_current:
            x = state.current.get_solution()
            self.problem.gradient(state.obj_scale, x, state.gradient)
            state.gradient_current = True

    def evaluate_hessian(self, state):
        if not state.hessian_current:
            x = state.current.get_solution()
            self.problem.hessian(state.obj_scale, x, state.hessian)
            state.hessian_current = True

    def evaluate_barrier_objective(self, state):
        """Evaluate the log-barrier objective at the current point"""

        return

    def _evaluate_barrier_objective(self, vars):
        """Evaluate the log-barrier objective at the current point"""
        x = vars.get_solution().get_current_point()

        # Zero dual, evaluate L(x,0) = f(x), restore
        con_indices = self.problem.get_constraint_indices()

        # Save the dual values and then zero them
        x.get_values_at(con_indices, self.temp_con)
        x.fill_at(con_indices, 0.0)
        f_obj = self.problem.lagrangian(self.obj_scale, x)

        # Restore the dual values
        x.set_values_at(con_indices, self.temp_con)

        barrier_log = self.optimizer.compute_barrier_log_sum(self.barrier_param, vars)

        return f_obj + barrier_log

    def compute_infeasibility(self, vars=None):
        if vars is None:
            vars = self.vars
        return self.optimizer.compute_constraint_violation_1norm(vars, self.grad)

    def evaluate_residual(self, state):
        if not state.gradient_current:
            self.evaluate_gradient(state)

        # Evaluate the residual and update the residual norm
        self.residual_norm = self.optimizer.compute_residual(
            state.mu, self.current, self.gradient, self.residual
        )

        d_inf_nlp, p_inf_nlp, c_inf_nlp = evaluator.evaluate_kkt_error()
        s_d_conv, s_c_conv = iterate.compute_optimality_scaling()
        overall_error = max(d_inf_nlp / s_d_conv, p_inf_nlp, c_inf_nlp / s_c_conv)


@dataclass
class IpmData:
    options: dict
    problem: am.OptimizationProblem
    optimizer: am.InteriorPointOptimizer
    solver: LinearSolver  # Linear solver class instance
    vars: am.OptVector  # Variables
    grad: am.Vector  # Gradient of the problem at the current design point
    diag: am.Vector  # Diagonal contributions
    hess: am.CSRMat  # CSR matrix that stores the Hessian
    obj_scale: float = 1.0

    def zero_multipliers(self, x):
        con_indices = self.problem.get_constraint_indices()
        x.fill_at(con_indices, 0.0)

    def compute_gradient(self):
        x = self.vars.get_solution()
        alpha = self.obj_scale
        self.problem.update(x)
        self.problem.gradient(alpha, x, self.grad)
        self.optimizer.apply_gradient_scaling(self.grad)

    def compute_hessian(self):
        x = self.vars.get_solution()
        alpha = self.obj_scale
        self.problem.hessian(alpha, x, self.hess)


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
