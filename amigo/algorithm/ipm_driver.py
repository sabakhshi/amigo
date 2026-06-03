"""Primal-dual interior-point optimizer.

The Optimizer class composes the algorithmic pieces from the sibling modules
and runs the main iteration loop.  Each iteration evaluates the KKT
residual, checks convergence, updates the barrier parameter, computes
a Newton direction, runs a line search, and handles step acceptance
or feasibility restoration.
"""

import warnings

# Raw pybind11 classes
from ..amigo import InteriorPointOptimizer, Vector

# Import from the model
from ..model import ModelVector

# Optimizer imports from algorithm classes
from .barrier_strategy import make_barrier_strategy
from .convergence_check import ConvergenceCheck, CONTINUE
from .default_options import get_default_options
from .evaluator import Evaluator
from .feasibility_restoration import FeasibilityRestoration
from .iterate_initialization import SlackInitializer
from .iteration_logger import OptimizationLogger
from .ipm_state import InteriorPointState
from .line_search import make_line_search
from .multiplier_initialization import MultiplierInitializer
from .newton_direction import NewtonStep
from .solvers import InertiaCorrector, make_solver


class Optimizer:
    """Primal-dual interior-point optimizer."""

    def __init__(
        self,
        model=None,
        x=None,
        problem=None,
        comm=None,
        **kwargs,
    ):
        """Initialize the optimizer.

        Parameters
        ----------
        model : Model
            The amigo model to optimize
        x : array-like, optional
            Initial point
        comm : MPI communicator, optional
            For distributed optimization
        """

        if "solver" in kwargs:
            warnings.warn("Set the solver through the options")

        # Set the model and problem
        if model is not None:
            self.model = model
            self.problem = self.model.get_problem()
        elif problem is not None:
            self.model = None
            self.problem = problem
        else:
            raise ValueError("Must provide a model or a problem instance")

        # Set the design vector
        if isinstance(x, ModelVector):
            self.x = x.get_vector()
        elif isinstance(x, Vector):
            self.x = x
        else:
            self.x = self.problem.create_vector()

        x_init = self.problem.get_initial_point()
        self.x.copy(x_init)
        self.lower = self.problem.get_lower()
        self.upper = self.problem.get_upper()

        self.comm = comm

        # Set up the vectors
        self._create_interior_point_backend()

        return

    def _create_interior_point_backend(self):
        """Create the C++ InteriorPointOptimizer backend and slack mapping."""
        data_vec = self.problem.get_data_vector()
        self.x.copy_host_to_device()
        self.lower.copy_host_to_device()
        self.upper.copy_host_to_device()
        data_vec.copy_host_to_device()

        self.optimizer = InteriorPointOptimizer(self.problem)
        self.vars = self.optimizer.create_opt_vector(self.x)
        self.update = self.optimizer.create_opt_vector()
        self.temp = self.optimizer.create_opt_vector()

    def get_options(self, options={}):
        return get_default_options(options)

    def get_optimized_point(self):
        return ModelVector(self.model, x=self.x)

    def optimize(self, options={}):
        """
        The set up of the new class structure:

        All data about the current state of the optimizer (scalars, vectors, Hessian etc.) are
        stored in the InteriorPointState object. This contains all info about the current design
        point.

        Evaluator is responsible for evaluating the quantities of interest (gradient, Hessian etc.)
        for the current state and trial points that may become the current state. Each algorithm
        is responsible for updating the state object so that its internal state remains consistent.

        FilterLineSearch performs a filter line search
        """

        # Check and normalize the options dictionary for internal use
        options = self.get_options(options=options)

        # TODO: Where should this go?
        self.optimizer.relax_bounds(1e-8, options["constr_viol_tol"])

        # Continuation control object, if any
        continuation_control = options["continuation_control"]

        # Class for evaluating problem-specific quantities
        evaluator = Evaluator(self.problem, self.optimizer)

        # The interior point state object contains information about the design point, the
        # gradient and the Hessian of the Lagrangian.
        state = InteriorPointState(self.x, options, self.problem, self.optimizer)

        # Create the solver depending on options
        solver = make_solver(options, state)

        # The inertia correction
        inertia_corrector = InertiaCorrector(options, self.problem, self.optimizer)

        # Initialize the line search algorithm
        line_search = make_line_search(options, self.problem, self.optimizer)

        # Allocate the Newton step
        newton_step = NewtonStep(options, self.problem, self.optimizer)

        # Feasibility restoration phase algorithm
        feasible_resto = FeasibilityRestoration(options, self.problem, self.optimizer)

        # Initialize the barrier strategy correction algorithm
        barrier_strategy = make_barrier_strategy(options, self.problem, self.optimizer)

        # Initialize the convergence check
        check = ConvergenceCheck(options, self.problem, self.optimizer)

        # Initialize the logger. The logger takes in additional objects that may
        # provide logging info via "obj.get_log_info()"
        objs = [line_search, inertia_corrector]
        logger = OptimizationLogger(objs, options, self.problem, self.optimizer)

        # Initialize the dual and slack variable values. This utilizes the solver object
        # to find initial values of the dual variables.
        slack_init = SlackInitializer(options, self.model, self.problem, self.optimizer)
        slack_init.initialize_slacks(evaluator, state)

        # Initialize the multipliers
        multiplier_init = MultiplierInitializer(options, self.problem, self.optimizer)
        multiplier_init.initialize_multipliers(evaluator, solver, state)

        # Set the initial status
        status = CONTINUE

        # Initialize the barrier strategy prior to optimization
        barrier_strategy.initialize(evaluator, state)

        max_iters = options["max_iterations"]
        for counter in range(max_iters):
            # Update the iteration counter
            state.iter = counter

            # Evaluate the objective and barrier function
            evaluator.evaluate_objective_and_barrier(state)

            # Evaluate the residuals for the convergence check
            evaluator.evaluate_residual(state)

            # Check for convergence based on the initial point
            status = check.test_convergence(evaluator, state)

            # Log the information about the iteration and the status of all the
            # internal objects within the optimizer
            logger.log_iteration(status, state)

            # Break if the check indicates we shouldn't continue
            if status != CONTINUE:
                break

            # Callback for the continuation control.
            if continuation_control is not None:
                continuation_control(state)

            # Perform an update of the barrier parameter prior to any factorization or step
            barrier_info = barrier_strategy.update_barrier(evaluator, state)

            # Let the line search object determine if a reset is appropriate based on the barrier parameter
            # update. For instance, this call may reset the filter.
            line_search.reset_on_new_barrier(state, barrier_info)

            # Factor the KKT system considering the inertia.
            # TODO: Implement a inertia info class
            factor_ok = inertia_corrector.factor_for_inertia(solver, evaluator, state)

            do_feasible_resto = True
            if factor_ok:
                # Compute the direction and store in state.step. This should be a descent direction
                # because of the inertia check
                newton_step.compute_step(solver, evaluator, state)

                # Using the same factorization and solver, assess whether a correction step is required
                # and compute it.
                barrier_strategy.add_step_correction(solver, evaluator, state)

                # Perform a line search along the step direction
                line_search_info = line_search.line_search(solver, evaluator, state)

                # Assess what happened after the line search
                barrier_strategy.update_after_line_search(
                    line_search_info, evaluator, state
                )

                if line_search_info.success:
                    do_feasible_resto = False

            # If the line search was not successful, perform feasibility restoration
            if do_feasible_resto:
                warnings.warn("Feasibility restoration phase not implemented")
                # feasible_resto.restoration_phase(solver, evaluator, state)

        else:
            # The optimization for loop completed normally, so we did not converge
            # Check the convergence status
            state.iter = max_iters
            status = check.test_convergence(evaluator, state)

            # Log the iteration
            logger.log_iteration(status, state)

        return logger.get_data()
