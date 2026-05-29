"""Strategies for initializing constraint and bound multipliers.

The least-squares initializer solves the normal-equation system
[I, A^T; A, 0] to obtain the y that minimizes the dual infeasibility
norm, and falls back to y = 0 if the computed multipliers exceed
lambda_max in magnitude.  The affine-scaling initializer solves the
KKT system at mu = 0 and updates y and z from the resulting affine
step.  The third option sets all multipliers to zero.
"""

import numpy as np


class MultiplierInitializer:
    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

    def initialize_multipliers(self, evaluator, solver, state):
        """
        Initialize the multipliers in state.current

        Least-squares constraint multiplier initialization (Section 3.6).

        Solves for lambda that minimizes the dual infeasibility norm:

            [ I  A^T ] [ w      ]   [ -(grad_f - zl + zu) ]
            [ A   0  ] [ lambda ] = [         0           ]

        The (1,1) block is I (no Hessian), making this a least-squares
        normal equation. w is discarded; lambda is the multiplier estimate.

        Safeguard: if ||lambda||_inf > lambda_max, discard and set to 0.
        """

        # Get the current primal-dual point
        x = state.get_current_point()

        primal_indices = self.problem.get_primal_indices()
        con_indices = self.problem.get_constraint_indices()

        # Zero the multipliers
        x.fill_at(con_indices, 0.0)

        # The gradient and Hessian information are now invalid
        state.invalidate()

        # We want to zero the contributions from the objective for the Hessian evaluation
        obj_scale_store = state.obj_scale

        # Evaluate the gradient. The multipliers are zero and the objective scaling = 1,
        # the gradient is the gradient of the objective only
        state.obj_scale = 1.0
        evaluator.evaluate_gradient(state)

        # Evaluate the Hessian. The multipliers are zero and the objective scaling = 1,
        # the Hessian only contains contributions from the constraints
        state.obj_scale = 0.0
        evaluator.evaluate_hessian(state)

        # Restore the objective scaling value
        state.obj_scale = obj_scale_store

        # Set the diagonal entries associated with primal variables = 1.0
        state.diagonal.zero()
        state.diagonal.fill_at(primal_indices, 1.0)

        # Factor the Hessian matrix in its current state
        solver.factor(state.hessian, state.diagonal)

        # Compute the residual = grad f - zl + zu
        self.optimizer.compute_dual_residual_vector(
            state.current, state.gradient, state.residual
        )
        state.residual.scale(-1.0)
        state.residual.fill_at(con_indices, 0.0)

        # Solve for the update
        update = state.step.get_solution()
        solver.solve(state.residual, update)

        # Add the updates to the indices
        x.copy_at(con_indices, update)

        # The gradient/Hessian values are invalid now since we just modified the primal-dual vector
        state.invalidate()

        return


class MultiplierInitialization:
    def _compute_least_squares_multipliers(self, lambda_max=1e3):
        """Least-squares constraint multiplier initialization (Section 3.6).

        Solves for lambda that minimizes the dual infeasibility norm:

            [ I  A^T ] [ w      ]   [ -(grad_f - zl + zu) ]
            [ A   0  ] [ lambda ] = [         0           ]

        The (1,1) block is I (no Hessian), making this a least-squares
        normal equation. w is discarded; lambda is the multiplier estimate.

        Safeguard: if ||lambda||_inf > lambda_max, discard and set to 0.
        """
        x = self.vars.get_solution()
        primal_indices = self.problem.get_primal_indices()
        con_indices = self.problem.get_constraint_indices()

        # Build RHS: -(grad_f - zl + zu) for primals, 0 for constraints
        self.optimizer.compute_dual_residual_vector(self.vars, self.grad, self.res)
        self.res.scale(-1.0)
        # self.res.get_array()[:] *= -1.0
        # self.res.copy_host_to_device()

        # Factor [I, A^T; A, 0]: W_factor=0 (no Hessian), diag=I on primals
        self.diag.zero()
        self.diag.fill_at(primal_indices, 1.0)
        # self.optimizer.set_primal_values(1.0, self.diag)
        # self.diag.copy_host_to_device()
        self.solver.factor(0.0, x, self.diag, post_hessian=self._hessian_scaling_fn)
        self.solver.solve(self.res, self.px)

        # Safeguard: discard if multipliers are too large
        # self.px.copy_device_to_host()
        # px_arr = self.px.get_array()

        # TODO: Fix max multiplier computation
        # problem_ref = self.mpi_problem if self.distribute else self.problem
        # mult_ind = np.array(problem_ref.get_multiplier_indicator(), dtype=bool)
        # lam_vals = px_arr[mult_ind]

        # Compute the max lambda value
        # if len(lam_vals) > 0 and np.max(np.abs(lam_vals)) > lambda_max:
        #     self.optimizer.set_dual_values(0.0, x)
        #     return

        x.copy_at(con_indices, self.px)

    def _compute_affine_multipliers(self, beta_min=1.0):
        """Compute the affine scaling initial point (Section 3.6).

        Solves the KKT system with mu=0 to get the affine step, then:
          - Updates the constraint multiplier: y = y + dy
          - Updates bound duals: z = max(z + dz, beta_min)
          - Primals (x, s) are NOT changed
          - Returns the initial barrier mu = avg complementarity
        """
        x = self.vars.get_solution()
        self._update_gradient(x)

        # Solve the KKT system with mu=0 (affine direction)
        mu = 0.0
        self.optimizer.compute_residual(mu, self.vars, self.grad, self.res)
        self.optimizer.compute_diagonal(self.vars, self.diag)
        self.solver.factor(
            self._obj_scale,
            x,
            self.diag,
            post_hessian=self._hessian_scaling_fn,
        )
        self.solver.solve(self.res, self.px)

        # Extract the bound dual steps via back-substitution
        self.optimizer.compute_update(mu, self.vars, self.px, self.update)

        # Update multipliers only (y = y + dy), primals unchanged
        # self.optimizer.copy_multipliers(x, self.update.get_solution())

        # Update multipliers only (y <- y + dy)
        con_indices = self.problem.get_constraint_indices()
        x.axpy_at(con_indices, 1.0, self.update)

        # Update bound duals: z = max(z + dz, beta_min)
        self.optimizer.compute_affine_start_point(
            beta_min, self.vars, self.update, self.vars
        )

        # Initial barrier = average complementarity at the new point
        barrier, _ = self.optimizer.compute_complementarity(self.vars)
        return max(barrier, beta_min)

    def _zero_multipliers(self, x):
        """Set all multipliers to zero."""
        con_indices = self.problem.get_constraint_indices()
        x.fill_at(con_indices, 0.0)
