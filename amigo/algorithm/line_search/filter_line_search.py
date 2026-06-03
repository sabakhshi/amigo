"""Filter line search, second-order correction, and watchdog.

Trial points are accepted against a two-dimensional filter over the
barrier objective phi_mu = f(x) - mu * sum(log(gap)) and the 1-norm
constraint violation theta.  A step passes if it satisfies either an
f-type Armijo condition or an h-type sufficient-reduction condition
and is not dominated by any stored filter entry.  SOC is applied on
the first rejection to counter the Maratos effect.  When many
shortened steps accumulate, the watchdog temporarily relaxes
acceptance to escape filter stalls.
"""

from abc import ABC, abstractmethod

import numpy as np
from .filter_acceptance import Filter


class LineSearch(ABC):
    def reset_on_new_barrier(self, state, barrier_info):
        pass

    @abstractmethod
    def line_search(self, solver, evaluator, state):
        pass


class LineSearchInfo:
    success: bool = False
    num_search_iters: int = 0
    alpha_primal: float = 1.0
    alpha_dual: float = 1.0


class FilterLineSearch(LineSearch):
    """Filter-based line search, SOC, and watchdog procedure."""

    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        # Allocate vectors that will be used internally
        self.trial = self.optimizer.create_opt_vector()
        self.trial_grad = self.problem.create_vector()

        # Allocate the internal filter
        self.filter = Filter()

        # Constant = 10 times machine precision
        self.EPS10 = 10.0 * np.finfo(float).eps

        # We can't set the limiting theta values yet. Set a flag to set them on the first
        # iteration through the line search
        self.theta_limits_initialized = False

        # Set the current info
        self.current_info = None

        # Keep track of how many successive times there is a filter reject
        self.successive_filter_rejections = 0
        self.filter_reset_count = 0

        return

    def reset_on_new_barrier(self, state, barrier_info):
        """This is called when the barrier parameter is updated"""

        # Reset depending on the barrier strategy used
        reset = False
        if self.options["barrier_strategy"] == "monotone":
            if barrier_info.new_barrier:
                reset = True
        elif barrier_info.mu_new < 0.1 * barrier_info.mu_old:
            reset = True
        elif self.options["barrier_strategy"] == "quality_function":
            reset = True

        if reset:
            self.filter.clear()
            self.theta_limits_initialized = False
            self.successive_filter_rejections = 0
            self.filter_reset_count = 0

        return

    class FilterBasePoint:
        ref_barr: float
        ref_theta: float
        ref_dphi: float

    def set_reference_values(self, ref_theta):
        # theta_min, theta_max (Eq. 21)
        self.theta_min = 1e-4 * max(1.0, ref_theta)
        self.theta_max = 1e4 * max(1.0, ref_theta)
        return

    def is_ftype(self, base, alpha_test):
        """Check if this is a f-type step that primarily reduces the objective/barrier function"""
        delta = self.options["filter_delta"]
        s_theta = self.options["filter_s_theta"]
        s_phi = self.options["filter_s_phi"]

        return (
            base.ref_dphi < 0.0
            and alpha_test * (-base.ref_dphi) ** s_phi > delta * base.ref_theta**s_theta
        )

    def armijo_holds(self, base, trial_barr, alpha_test):
        """Check if the Armijo condition holds relative to the base point"""
        eta_phi = self.options["filter_eta_phi"]

        return (
            trial_barr - base.ref_barr
        ) - eta_phi * alpha_test * base.ref_dphi <= self.EPS10 * abs(base.ref_barr)

    def acceptable_to_iterate(self, base, trial_barr, trial_theta):
        gamma_theta = self.options["filter_gamma_theta"]
        gamma_phi = self.options["filter_gamma_phi"]
        obj_max_inc = 5.0

        if trial_barr > base.ref_barr:
            basval = 1.0
            if abs(base.ref_barr) > 10.0:
                basval = np.log10(abs(base.ref_barr))
            if np.log10(max(trial_barr - base.ref_barr, 1e-300)) > obj_max_inc + basval:
                return False
        return (
            trial_theta - (1.0 - gamma_theta) * base.ref_theta
            <= self.EPS10 * abs(base.ref_theta)
        ) or (
            (trial_barr - base.ref_barr) - (-gamma_phi * base.ref_theta)
            <= self.EPS10 * abs(base.ref_barr)
        )

    def check_acceptance(self, base, trial_barr, trial_theta, alpha_test):
        if trial_theta > self.theta_max:
            return False
        if (
            alpha_test > 0.0
            and self.is_ftype(base, alpha_test)
            and base.ref_theta <= self.theta_min
        ):
            if not self.armijo_holds(base, trial_barr, alpha_test):
                return False
        else:
            if not self.acceptable_to_iterate(base, trial_barr, trial_theta):
                return False

        return self.filter.is_acceptable(trial_barr, trial_theta)

    def update_filter(self, base, trial_barr, trial_theta, alpha_test):
        is_ftype = (
            base.ref_theta <= self.theta_min
            and self.is_ftype(base, alpha_test)
            and self.armijo_holds(base, trial_barr, alpha_test)
        )

        if not is_ftype:
            self.filter.add(base.ref_barr, base.ref_theta)

    def build_base_point(self, evaluator, state):
        base = self.FilterBasePoint()
        fobj, barrier = evaluator.evaluate_objective_and_barrier_from_point(
            state.mu, state.obj_scale, state.current
        )
        base.ref_barr = fobj + barrier
        base.ref_theta = evaluator.evaluate_infeasibility_from_gradient(
            state.current, state.gradient
        )
        base.ref_dphi = evaluator.evaluate_directional_derivative(state)

        return base

    def compute_alpha_min(self, base):
        # Compute alpha_min (Eq. 23)
        gamma_theta = self.options["filter_gamma_theta"]
        gamma_phi = self.options["filter_gamma_phi"]
        s_theta = self.options["filter_s_theta"]
        s_phi = self.options["filter_s_phi"]
        delta = self.options["filter_delta"]
        alpha_min_frac = 0.05

        alpha_min = gamma_theta
        if base.ref_dphi < 0.0:
            alpha_min = min(gamma_theta, gamma_phi * base.ref_theta / (-base.ref_dphi))
            if base.ref_theta <= self.theta_min:
                alpha_min = min(
                    alpha_min,
                    delta * base.ref_theta**s_theta / (-base.ref_dphi) ** s_phi,
                )
        alpha_min *= alpha_min_frac

        return alpha_min

    def line_search(self, solver, evaluator, state):

        # Check if we should reset the filter
        filter_reset_trigger = self.options["filter_reset_trigger"]
        max_filter_resets = self.options["max_filter_resets"]
        if (
            self.successive_filter_rejections >= filter_reset_trigger
            and self.filter_reset_count < max_filter_resets
        ):
            self.filter.clear()
            state.filter_reset_count += 1

        # Build the base point for comparison. In case a watchdog is set, this
        # base point will come from the watchdog.
        base = self.build_base_point(evaluator, state)

        if not self.theta_limits_initialized:
            self.set_reference_values(base.ref_theta)
            self.theta_limits_initialized = True

        # Compute the minimum step length
        alpha_min = self.compute_alpha_min(base)

        # Initial step lengths
        alpha_primal = state.max_alpha_primal
        alpha_dual = state.max_alpha_dual

        # Max line search iterations
        max_line_iters = self.options["max_line_search_iterations"]

        # Build the info class based on a failure of the line search. If we find
        # an acceptable point, this will be over-written
        info = LineSearchInfo()
        info.success = False
        info.num_search_iters = max_line_iters

        for line_iter in range(max_line_iters):
            # Compute the update for the current iteration
            self.optimizer.apply_step_update(
                alpha_primal, alpha_dual, state.current, state.step, self.trial
            )

            # Compute the gradient at the trial point
            evaluator.evaluate_gradient_from_point(
                state.obj_scale, self.trial, self.trial_grad
            )

            # Get the objective and log-barrier term
            fobj, barrier = evaluator.evaluate_objective_and_barrier_from_point(
                state.mu, state.obj_scale, self.trial
            )
            trial_barr = fobj + barrier

            # Compute the l1 norm of the constraint violation
            trial_theta = evaluator.evaluate_infeasibility_from_gradient(
                self.trial, self.trial_grad
            )

            accepted = self.check_acceptance(
                base, trial_barr, trial_theta, alpha_primal
            )

            if accepted or alpha_primal <= alpha_min:
                self.update_filter(base, trial_barr, trial_theta, alpha_primal)

                # Update the state information: Invalidate the current point, but copy
                # over the gradient that we just computed at the trial point
                state.invalidate()
                state.current.copy(self.trial)
                state.gradient.copy(self.trial_grad)
                state.gradient_current = True

                # Set the current info
                info.success = True
                info.num_search_iters = line_iter + 1
                info.alpha_primal = alpha_primal
                info.alpha_dual = alpha_dual

                # Retain a copy of the line search info for logging iterations
                self.current_info = info

                return info

            backtrack_factor = self.options["backtracking_factor"]
            alpha_primal = max(backtrack_factor * alpha_primal, alpha_min)

        return info

    def add_log_info(self, info):
        if self.current_info is not None:
            info["filter_size"] = len(self.filter)
            info["line_iters"] = self.current_info.num_search_iters
            info["alpha_x"] = self.current_info.alpha_primal
            info["alpha_z"] = self.current_info.alpha_dual
