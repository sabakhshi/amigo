import numpy as np


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

    def evaluate_infeasibility_from_gradient(self, vars, grad):
        return self.optimizer.compute_constraint_violation_1norm(vars, grad)
        # con_indices = self.problem.get_constraint_indices()
        # grad.get_values_at(con_indices, self.temp_con)
        # return self.problem.abssum(self.temp_con)

    def evaluate_infeasibility(self, state):
        if not state.gradient_current:
            self.evaluate_gradient(state)

        infeas = self.evaluate_infeasibility_from_gradient(
            state.current, state.gradient
        )
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

        return

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

    def evaluate_complementarity(self, state):
        comp, xi = self.optimizer.compute_complementarity(state.current)
        return comp, xi
