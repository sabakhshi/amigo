"""Convergence tests for the interior-point loop.

Four criteria are checked each iteration.  Primary convergence
requires every KKT component below its tolerance.  Divergence halts
the solve when the iterate magnitude exceeds a safety bound.
Acceptable convergence flags a weaker solution once relaxed
tolerances hold for several iterations in a row.  Precision-floor
detection catches bit-identical residuals, the numerical limit below
which further progress is not possible.
"""

# Return codes for convergence check
CONTINUE = 0
CONVERGED = 1
CONVERGED_ACCEPTABLE = 2
DIVERGED = 3
PRECISION_FLOOR = 4
ITERATING = 5


class ConvergenceCheck:
    """Convergence checks: primary, acceptable, divergence, precision floor."""

    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        # The previous residual norm - initialize to zero
        self.prev_res_norm = 0.0

        # Set the precision floor count
        self.precision_floor_count = 0

        # Set the acceptable counter
        self.acceptable_counter = 0

    def test_convergence(self, evaluator, state):
        """
        Check for convergence
        """
        tol = self.options["convergence_tolerance"]
        dual_inf_tol = self.options["dual_inf_tol"]
        constr_viol_tol = self.options["constr_viol_tol"]
        compl_inf_tol = self.options["compl_inf_tol"]
        diverging_iterates_tol = self.options["diverging_iterates_tol"]
        acceptable_tol = self.options["acceptable_tol"]
        acceptable_iter = self.options["acceptable_iter"]
        acceptable_dual_inf_tol = self.options["acceptable_dual_inf_tol"]
        acceptable_constr_viol_tol = self.options["acceptable_constr_viol_tol"]
        acceptable_compl_inf_tol = self.options["acceptable_compl_inf_tol"]

        # Current iteration counter
        iteration = state.iter

        # Evaluate the residual to compute the KKT error metrics
        evaluator.evaluate_residual(state)

        # Compute NLP error components at mu_target=0
        d_inf_nlp = state.dual_infeas
        p_inf_nlp = state.primal_infeas
        c_inf_nlp = state.complementarity
        overall_error = state.kkt_error

        # Primary convergence: ALL 4 conditions must hold
        if (
            overall_error <= tol
            and d_inf_nlp <= dual_inf_tol
            and p_inf_nlp <= constr_viol_tol
            and c_inf_nlp <= compl_inf_tol
        ):
            return CONVERGED

        x_max = self.problem.maxabs(state.current.get_solution())
        if x_max > diverging_iterates_tol:
            if state.comm_rank == 0:
                print(f"  Diverging iterates: max |x| = {x_max:.2e}")
            return DIVERGED

        # Acceptable convergence
        is_acceptable = (
            overall_error <= acceptable_tol
            and d_inf_nlp <= acceptable_dual_inf_tol
            and p_inf_nlp <= acceptable_constr_viol_tol
            and c_inf_nlp <= acceptable_compl_inf_tol
        )
        if acceptable_iter > 0 and is_acceptable:
            self.acceptable_counter += 1
            if self.acceptable_counter >= acceptable_iter:
                return CONVERGED_ACCEPTABLE
        else:
            self.acceptable_counter = 0

        # Precision floor: bit-identical residuals
        denom = max(state.residual_norm, 1e-30)
        rel_change = abs(state.residual_norm - self.prev_res_norm) / denom

        # Update the previous residual norm
        self.prev_res_norm = state.residual_norm

        if rel_change < 1e-14 and iteration > 0:
            self.precision_floor_count += 1
        else:
            self.precision_floor_count = 0
        if self.precision_floor_count >= 3 and is_acceptable:
            if state.comm_rank == 0:
                print(
                    f"  Precision floor: residual unchanged "
                    f"for {self.precision_floor_count} iterations"
                )
            return PRECISION_FLOOR

        return CONTINUE
