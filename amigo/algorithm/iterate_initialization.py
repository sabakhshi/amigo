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

        # Zero the multipliers
        con_indices = self.problem.get_constraint_indices()
        x.fill_at(con_indices, 0.0)

        # Initialize the slack variables
        self.optimizer.initialize_duals(state.mu, state.current)

        # Invalidate the gradient because we just changed the slack and bound variable values
        state.invalidate()

        if self.model is not None:
            # Evaluate the gradient at the current point
            evaluator.evaluate_gradient(state)

            # Get the slack and inequality indices
            slack_indices = self.model.slack_indices
            ineq_indices = self.model.ineq_constraint_indices

            # Copy the gradient and solution to the host
            x.copy_device_to_host()
            state.gradient.copy_device_to_host()

            # Extract the arrays that are now on the host
            x_array = x.get_array()
            grad_array = state.gradient.get_array()

            # Set the values on the host
            x_array[slack_indices] += grad_array[ineq_indices]

            # Copy the values back to the host
            x.copy_host_to_device()

            # Project the slack variables back into the feasible region
            self.optimizer.initialize_duals(state.mu, state.current)

            # Everything is now invalid because we updated the primal-dual point
            state.invalidate()

        return
