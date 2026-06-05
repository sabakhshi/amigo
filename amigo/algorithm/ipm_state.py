"""Per-iteration state carried through the optimization loop.

IpmState holds the scalar counters, step-size history, and
filter/rejection counts that change every iteration.  StepContext
is a lightweight bag of per-iteration scratch data handed to the
barrier strategy each step.
"""

import amigo as am


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
    objective_current: bool

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
        self.objective_current = False

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

    def invalidate(self, obj=True, grad=True, hess=True, res=True, step=True):
        if obj:
            self.objective_current = False
        if grad:
            self.gradient_current = False
        if hess:
            self.hessian_current = False
        if res:
            self.residual_current = False
        if step:
            self.step_current = False
