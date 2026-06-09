import numpy as np


class Evaluator:
    def __init__(self, problem, optimizer):
        self.problem = problem
        self.optimizer = optimizer

        # Create a temporary constraint vector for later usage
        self.temp_con = self.problem.create_constraint_vector()

        # Initialize the number of element counts
        self.num_primal, self.num_constraints, self.num_bounds = (
            self.optimizer.get_kkt_element_counts()
        )

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

    def evaluate_objective_and_infeasibility(self, state):
        """Evaluate the objective, barrier and infeasibility at the current point"""
        if not state.objective_current:
            if not state.gradient_current:
                self.evaluate_gradient(state)

            obj, barrier, infeas = self.evaluate_objective_and_infeasibility_from_point(
                state.mu, state.obj_scale, state.current, state.gradient
            )

            state.objective_value = obj
            state.log_barrier_value = barrier
            state.con_infeasibility = infeas

            # Are the objective, barrier and infeasibility current
            state.objective_current = True

    def evaluate_objective_and_infeasibility_from_point(
        self, mu, obj_scale, vars, grad
    ):
        """Evaluate the objective, barrier and infeasibility at the trial point"""
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

        # Evaluate the log barrier term
        barrier = self.optimizer.compute_log_barrier(mu, vars)

        # Compute the infeasibility from the gradient
        infeas = self.optimizer.compute_infeasibility(grad)

        return fobj, barrier, infeas

    def evaluate_directional_derivative(self, state):
        """Evaluate the directional derivative at the candidate point"""
        if not state.residual_current:
            self.evaluate_residual(state)
        if not state.step_current:
            raise ValueError("Step not current at this point")

        xtmp = self.problem.create_vector()
        gtmp = self.problem.create_vector()

        # Copy the design variable values
        xtmp.copy(state.current.get_solution())

        # Zero the multipliers
        con_indices = self.problem.get_constraint_indices()
        xtmp.fill_at(con_indices, 0.0)

        # Evaluate the gradient of the objective function alone at the current point
        self.problem.gradient(1.0, xtmp, gtmp)

        update = state.step.get_solution()
        deriv = self.problem.dot(update, gtmp)

        deriv += self.optimizer.compute_log_barrier_derivative(
            state.mu, state.current, state.step
        )

        return deriv

    def evaluate_residual_from_point(self, mu, vars, grad, res):
        return self.optimizer.compute_residual(mu, vars, grad, res)

    def evaluate_residual(self, state):
        if not state.residual_current:
            if not state.gradient_current:
                self.evaluate_gradient(state)

            # Evaluate the residual and update the residual norm
            state.residual_norm = self.evaluate_residual_from_point(
                state.mu, state.current, state.gradient, state.residual
            )

            # Now compute the infeasibilities
            state.dual_infeas, state.primal_infeas, state.complementarity = (
                self.optimizer.compute_kkt_error(0.0, state.current, state.gradient)
            )

            # Compute the scaling
            s_d_conv, s_c_conv = self.compute_optimality_scaling(state)
            state.kkt_error = max(
                state.dual_infeas / s_d_conv,
                state.primal_infeas,
                state.complementarity / s_c_conv,
            )

            state.residual_current = True

        return

    def compute_optimality_scaling(self, state, s_max=100.0):
        """Compute optimality error scaling factors (s_d, s_c)."""

        # Get the vectors
        zl = state.current.get_zl()
        zu = state.current.get_zu()
        z_asum = self.problem.abssum(zl) + self.problem.abssum(zu)

        # Get the sum of the absolute values of the multipliers
        con_indices = self.problem.get_constraint_indices()

        x = state.current.get_solution()
        x.get_values_at(con_indices, self.temp_con)
        y_asum = self.problem.abssum(self.temp_con)

        # s_c: bound multiplier scaling
        if self.num_bounds == 0:
            s_c = 1.0
        else:
            s_c = max(s_max, z_asum / self.num_bounds) / s_max

        # s_d: dual (stationarity) scaling
        num_total = self.num_constraints + self.num_bounds
        if num_total == 0:
            s_d = 1.0
        else:
            s_d = max(s_max, (y_asum + z_asum) / num_total) / s_max

        return s_d, s_c

    def evaluate_diagonal(self, state):
        """Evaluate the diagonal entries"""
        self.optimizer.compute_diagonal(state.current, state.diagonal)

    def evaluate_complementarity(self, state):
        comp, xi = self.optimizer.compute_complementarity(state.current)
        return comp, xi
