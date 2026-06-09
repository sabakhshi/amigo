"""Base class for barrier-parameter strategies.

A BarrierStrategy owns the per-iteration barrier update:
  - decide the new mu (heuristic rule or QF oracle)
  - factorize the KKT system at the new mu
  - compute the Newton direction

Every concrete strategy reads and writes self.opt.barrier_param and uses
self.opt.{vars, grad, res, px, update, temp, optimizer, solver} plus
helpers on self.opt (_factorize_kkt, _find_direction, etc.).
"""

from abc import ABC, abstractmethod


class BarrierInfo:
    new_barrier: bool = False
    mu_new: float = 0.0
    mu_old: float = 0.0


class BarrierStrategy(ABC):
    def __init__(self, options={}):
        self.options = options

    def initialize(self, evaluator, state):
        """Initialize the barrier strategy from the initial point"""
        pass

    @abstractmethod
    def update_barrier(self, evaluator, state) -> BarrierInfo:
        """Update the barrier parameter prior to factoring the KKT matrix"""
        pass

    def add_step_correction(self, solver, evalutor, state):
        """Add the correction to the step - relevant for Mehrotra P/C steps"""
        pass

    def update_after_line_search(self, info, evaluator, state):
        """
        Update any internal state required after the results of a line search

        Default behavior is to adjust the barrier parameter if small steps are taken
        """
        pass
