"""Build the primal-dual starting point before the first iteration.

Relaxes variable bounds, projects the initial design vector into the
relaxed box, initializes slacks and bound multipliers, applies
gradient-based NLP scaling, and initializes constraint multipliers
(least-squares by default, affine step when requested).
"""


class SlackInitializer:
    def __init__(self, options, model, problem, optimizer):
        self.model = model
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

    def initialize_slacks(self, evaluator, state):
        x = state.get_current_point()

        mu = state.mu
        self.optimizer.initialize_duals_and_slacks(mu, state.current)

        # Evaluate the gradient at the current point
        evaluator.evaluate_gradient(state)

        if self.model is not None:
            # Get the slack and inequality indices
            slack_indices = self.model.slack_indices
            ineq_indices = self.model.ineq_constraint_indices

            # Copy the gradient and solution to the host
            x.copy_device_to_host()
            state.gradident.copy_device_to_host()
            x_array = x.get_array()
            grad_array = state.gradident.get_array()

            # Set the values
            x_array[slack_indices] += grad_array[ineq_indices]

            # Update the values
            x.copy_host_to_device()
            self.grad.copy_host_to_device()

        # The gradient is now invalid because we update the primal-dual point
        state.invalidate()

        return


class IterateInitialization:
    """Primal-dual iterate initialization sequence."""

    def _initialize_iterate(self, options, comm_rank):
        """Run the full initialization sequence before the main loop.

        Sets self._obj_scale, self._hessian_scaling_fn, and leaves
        the iterate (self.vars) and gradient (self.grad) ready for
        the first iteration.
        """
        self._obj_scale = 1.0
        self._hessian_scaling_fn = None

        x = self.vars.get_solution()

        # Step 1: Relax bounds (default: bound_relax_factor = 1e-8)
        self.optimizer.relax_bounds(1e-8, options["constr_viol_tol"])

        # Step 2: Project design variables into bounds, initialize z = 1.0
        self._zero_multipliers(x)
        self.optimizer.initialize_duals_and_slacks(self.barrier_param, self.vars)

        # Step 3: Initialize slacks to s = d(x), then push into bounds.
        # if self.optimizer.has_slacks():
        self._update_gradient(x)

        # Set the initial point - need the slack and inequality indices
        if hasattr(self, "model"):
            # Get the slack and inequality indices
            slack_indices = self.model.slack_indices
            ineq_indices = self.model.ineq_constraint_indices

            # Copy the gradient and solution to the host
            x.copy_device_to_host()
            self.grad.copy_device_to_host()
            x_array = x.get_array()
            grad_array = self.grad.get_array()

            # Set the values
            x_array[slack_indices] += grad_array[ineq_indices]

            # Update the values
            x.copy_host_to_device()
            self.grad.copy_host_to_device()

        self.optimizer.initialize_duals_and_slacks(self.barrier_param, self.vars)

        # if self.optimizer.has_slacks():
        #     self._update_gradient(x)
        #     self.optimizer.initialize_slacks(self.grad, self.vars)
        #     self.optimizer.initialize_duals_and_slacks(self.barrier_param, self.vars)

        # Step 4: Recompute gradient at the pushed x with lam=0
        self._update_gradient(x)

        # Step 4b: Gradient-based NLP scaling from initial point.
        nlp_max_grad = options["nlp_scaling_max_gradient"]
        if nlp_max_grad > 0 and not self.distribute:
            self.optimizer.compute_nlp_scaling(x, self.grad, max_gradient=nlp_max_grad)
            self._obj_scale = self.optimizer.get_obj_scale()
            if self.optimizer.has_scaling():
                self._hessian_scaling_fn = self.optimizer.apply_hessian_scaling
                self._update_gradient(x)
                if comm_rank == 0:
                    print(f"  NLP scaling: obj_scale={self._obj_scale:.4e}")

        # Step 5: Least-squares constraint multiplier initialization
        if options["init_affine_step_multipliers"]:
            self._compute_least_squares_multipliers()
            self.barrier_param = self._compute_affine_multipliers(
                beta_min=self.barrier_param
            )
        elif options["init_least_squares_multipliers"]:
            self._compute_least_squares_multipliers()

        # Step 6: Recompute gradient at final (x, lam) for the main loop
        self._update_gradient(x)
