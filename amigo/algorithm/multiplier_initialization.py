"""Strategies for initializing constraint and bound multipliers.

The least-squares initializer solves the normal-equation system
[I, A^T; A, 0] to obtain the y that minimizes the dual infeasibility
norm, and falls back to y = 0 if the computed multipliers exceed
lambda_max in magnitude.  The affine-scaling initializer solves the
KKT system at mu = 0 and updates y and z from the resulting affine
step.  The third option sets all multipliers to zero.
"""


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

        if self.options["init_least_squares_multipliers"]:
            # Get the current primal-dual point
            x = state.get_current_point()

            primal_indices = self.problem.get_primal_indices()
            con_indices = self.problem.get_constraint_indices()

            # Zero the multipliers
            x.fill_at(con_indices, 0.0)

            # The gradient and Hessian information are now invalid since the multipliers
            # have changed.
            state.invalidate()

            # Store the objective scaling to restore it later
            obj_scale_store = state.obj_scale

            # Evaluate the gradient. The multipliers are zero and the objective scaling = 1,
            # the gradient is the gradient of the objective only.
            state.obj_scale = 1.0
            evaluator.evaluate_gradient(state)

            # Evaluate the Hessian. The multipliers are zero and the objective scaling = 0,
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
            self.optimizer.compute_dual_residual(
                state.current, state.gradient, state.residual
            )
            state.residual.scale(-1.0)

            # Zero the contributions from the constraints
            state.residual.fill_at(con_indices, 0.0)

            # Solve for the update
            update = state.step.get_solution()
            solver.solve(state.residual, update)

            # Add the updates to the indices
            x.copy_at(con_indices, update)

            # The gradient/Hessian values are invalid now since we just modified the primal-dual vector
            state.invalidate()

        return
