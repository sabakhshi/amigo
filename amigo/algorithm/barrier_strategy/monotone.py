"""Monotone barrier strategy"""

from .base import BarrierStrategy, BarrierInfo


class MonotoneBarrierStrategy(BarrierStrategy):
    def __init__(self, options, problem, optimizer):
        super().__init__(options)
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

    def update_barrier(self, evaluator, state):
        info = BarrierInfo()
        relative_tol = self.options["barrier_progress_tol"]

        if state.kkt_error < relative_tol * state.mu:
            opt_tol = self.options["convergence_tolerance"]
            frac = self.options["monotone_barrier_fraction"]
            mu_new = max(frac * state.mu, frac * opt_tol)

            info.new_barrier = True
            info.mu_old = state.mu
            info.mu_new = mu_new

            # Update the barrier parameter. Invalidate the residuals and the step
            # (if any) because the barrier has changed
            state.mu = mu_new

            # Only the gradient and hessian retain their status
            state.invalidate(grad=False, hess=False)

        return info
