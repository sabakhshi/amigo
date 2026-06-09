"""Inertia correction for the augmented KKT system.

Implements Algorithm IC from Wachter & Biegler (2006, Table 3).  Grows
primal (delta_w) and constraint (delta_c) regularization until the
factorized KKT matrix has the correct inertia (n_primal positive and
n_dual negative eigenvalues), with a state machine that detects
structural degeneracy across iterations.
"""


class InertiaCorrector:
    """Inertia correction for the KKT system (Algorithm IC, Wachter & Biegler 2006).

    Manages primal (delta_w) and constraint (delta_c) regularization to
    ensure correct inertia (n positive, m negative eigenvalues):
      - ConsiderNewSystem: save last perturbation, reset current to zero.
        If structurally degenerate, pre-apply delta_c / delta_x.
      - PerturbForSingularity: add delta_c first, then delta_x.
      - PerturbForWrongInertia: grow delta_x; on overflow, add delta_c
        and restart delta_x search.
      - finalize_test: structural degeneracy detection after consecutive
        iterations needing the same perturbation type.
      - IncreaseQuality: when too few negative eigenvalues, try improving
        pivot tolerance before treating as singular.
    """

    # Degeneracy status
    _NOT_YET = 0
    _NOT_DEGEN = 1
    _DEGENERATE = 2

    # Test status for finalize_test
    _NO_TEST = 0
    _TEST_DC0_DX0 = 1
    _TEST_DC1_DX0 = 2
    _TEST_DC0_DX1 = 3
    _TEST_DC1_DX1 = 4

    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        self.num_primal = self.optimizer.get_num_primals()
        self.num_dual = self.optimizer.get_num_constraints()

        # We're going to create our own diagonal entries. Create a vector
        # for those entries
        self.perturbed_diagonal = self.problem.create_vector()

        # Set some parameters. Should these be options?
        self.max_corrections = 10

        # Set internal values
        self.numerical_eps = 1e-12

        # Perturbation state
        self._delta_x_last = 0.0
        self._delta_c_last = 0.0
        self._delta_x_curr = 0.0
        self._delta_c_curr = 0.0

        # Exposed for iterative refinement
        self.last_delta_w = 0.0
        self.last_delta_c = 0.0

        # Algorithm IC constants (Table 3, Wachter & Biegler 2006)
        self._dw_init = 1e-4  # first_hessian_perturbation
        self._dw_min = 1e-20  # min_hessian_perturbation
        self._dw_max = 1e20  # max_hessian_perturbation
        self._kw_inc = 8.0  # perturb_inc_fact
        self._kw_first_inc = 100.0  # perturb_inc_fact_first
        self._kw_dec = 1.0 / 3  # perturb_dec_fact
        self._dc_val = 1e-8  # jacobian_regularization_value
        self._dc_exp = 0.25  # jacobian_regularization_exponent

        # Structural degeneracy detection
        self._hess_degen = self._NOT_YET
        self._jac_degen = self._NOT_YET
        self._degen_iters = 0
        self._degen_iters_max = 3
        self._test_status = self._NO_TEST

        # Adaptive pivot tolerance
        self._pivtol = 1e-6
        self._pivtolmax = 0.1

        self.verbose = self.options["verbose_barrier"]

    def _delta_cd(self, state):
        """Constraint regularization: delta_c = delta_cd_val * mu^delta_cd_exp."""
        return self._dc_val * state.mu**self._dc_exp

    def _get_deltas_for_wrong_inertia(self):
        """Grow delta_x geometrically. Returns False if delta_x exceeds max."""
        prev = self._delta_x_curr
        if self._delta_x_curr == 0.0:
            if self._delta_x_last == 0.0:
                self._delta_x_curr = self._dw_init
            else:
                self._delta_x_curr = max(
                    self._dw_min, self._delta_x_last * self._kw_dec
                )
        else:
            if (
                self._delta_x_last == 0.0
                or 1e5 * self._delta_x_last < self._delta_x_curr
            ):
                self._delta_x_curr *= self._kw_first_inc
            else:
                self._delta_x_curr *= self._kw_inc

        if self._delta_x_curr > self._dw_max:
            # Revert: the overflowed value was never used in a factorization
            self._delta_x_curr = prev
            self._delta_x_last = 0.0
            return False
        return True

    def _perturb_for_wrong_inertia(self, state):
        """Perturb for wrong inertia (too many negative eigenvalues).

        Calls finalize_test, then grows delta_x.
        On overflow with delta_c==0: add delta_c and restart delta_x.
        """
        self._finalize_test()
        if self._get_deltas_for_wrong_inertia():
            return True
        # delta_x overflow: if delta_c==0, add it and retry from scratch
        if self._delta_c_curr == 0.0:
            self._delta_c_curr = self._delta_cd(state)
            self._delta_x_curr = 0.0
            if self._hess_degen == self._DEGENERATE:
                self._hess_degen = self._NOT_YET
            self._test_status = self._NO_TEST
            return self._get_deltas_for_wrong_inertia()
        return False

    def _perturb_for_singularity(self, state):
        """Perturb for singular system (too few negative eigenvalues).

        Handles the degeneracy test state machine for singular systems.
        """
        if self._hess_degen == self._NOT_YET or self._jac_degen == self._NOT_YET:
            # Degeneracy test state machine
            ts = self._test_status

            if ts == self._TEST_DC0_DX0:
                # Haven't tried anything yet for this matrix
                if self._jac_degen == self._NOT_YET:
                    # Try adding delta_c only (test if jac is degenerate)
                    self._delta_c_curr = self._delta_cd(state)
                    self._test_status = self._TEST_DC1_DX0
                else:
                    # jac known, hess NOT_YET: try delta_x only
                    if not self._get_deltas_for_wrong_inertia():
                        return False
                    self._test_status = self._TEST_DC0_DX1

            elif ts == self._TEST_DC1_DX0:
                # Already tried delta_c>0, delta_x=0 — still singular.
                # Now try delta_x>0, delta_c=0
                self._delta_c_curr = 0.0
                if not self._get_deltas_for_wrong_inertia():
                    return False
                self._test_status = self._TEST_DC0_DX1

            elif ts == self._TEST_DC0_DX1:
                # Tried delta_x>0, delta_c=0 — still singular.
                # Now try both.
                self._delta_c_curr = self._delta_cd(state)
                if not self._get_deltas_for_wrong_inertia():
                    return False
                self._test_status = self._TEST_DC1_DX1

            elif ts == self._TEST_DC1_DX1:
                # Both active — just grow delta_x.
                if not self._get_deltas_for_wrong_inertia():
                    return False

            # else: NO_TEST should not occur here

        else:
            # Both hess/jac degeneracy resolved
            if self._delta_c_curr > 0.0:
                # Already perturbing constraints: treat like wrong inertia
                if not self._get_deltas_for_wrong_inertia():
                    return False
            else:
                # First singular encounter: add constraint regularization
                self._delta_c_curr = self._delta_cd(state)

        return True

    def _finalize_test(self):
        """Conclude degeneracy test after successful factorization.

        After degen_iters_max consecutive iterations needing the same
        perturbation type, declare structural degeneracy.
        """
        ts = self._test_status
        if ts == self._NO_TEST:
            return

        if ts == self._TEST_DC0_DX0:
            if self._hess_degen == self._NOT_YET:
                self._hess_degen = self._NOT_DEGEN
            if self._jac_degen == self._NOT_YET:
                self._jac_degen = self._NOT_DEGEN
        elif ts == self._TEST_DC1_DX0:
            if self._hess_degen == self._NOT_YET:
                self._hess_degen = self._NOT_DEGEN
            if self._jac_degen == self._NOT_YET:
                self._degen_iters += 1
                if self._degen_iters >= self._degen_iters_max:
                    self._jac_degen = self._DEGENERATE
        elif ts == self._TEST_DC0_DX1:
            if self._jac_degen == self._NOT_YET:
                self._jac_degen = self._NOT_DEGEN
            if self._hess_degen == self._NOT_YET:
                self._degen_iters += 1
                if self._degen_iters >= self._degen_iters_max:
                    self._hess_degen = self._DEGENERATE
        elif ts == self._TEST_DC1_DX1:
            self._degen_iters += 1
            if self._degen_iters >= self._degen_iters_max:
                self._hess_degen = self._DEGENERATE
                self._jac_degen = self._DEGENERATE

        self._test_status = self._NO_TEST

    def _consider_new_system(self, state):
        """Prepare for a new KKT system.

        Save last perturbation, reset current to zero. Pre-apply delta_c
        if Jacobian is structurally degenerate, and pre-populate delta_x
        if Hessian is structurally degenerate.
        """
        self._finalize_test()

        # Pivot tolerance persists across iterations.  Once IncreaseQuality
        # raises pivtol, the solver keeps the tighter setting.

        # Save last perturbation
        if self._delta_x_curr > 0.0:
            self._delta_x_last = self._delta_x_curr
        if self._delta_c_curr > 0.0:
            self._delta_c_last = self._delta_c_curr

        # Set up degeneracy test for this iteration
        if self._hess_degen == self._NOT_YET or self._jac_degen == self._NOT_YET:
            self._test_status = self._TEST_DC0_DX0
        else:
            self._test_status = self._NO_TEST

        # Pre-apply delta_c if Jacobian structurally degenerate
        if self._jac_degen == self._DEGENERATE:
            self._delta_c_curr = self._delta_cd(state)
        else:
            self._delta_c_curr = 0.0

        # Pre-apply delta_x if Hessian structurally degenerate
        if self._hess_degen == self._DEGENERATE:
            self._delta_x_curr = 0.0
            self._get_deltas_for_wrong_inertia()
        else:
            self._delta_x_curr = 0.0

    def _is_inertia_ok(self, np, nn):
        num_total = self.num_primal + self.num_dual

        itol = self.options["inertia_tolerance"]

        return (
            abs(np - self.num_primal) <= itol
            and abs(nn - self.num_dual) <= itol
            and np + nn >= num_total - itol
        )

    def _compute_perturbation(self, diagonal, perturb):
        primal_indices = self.problem.get_primal_indices()
        dual_indices = self.problem.get_constraint_indices()

        perturb.copy(diagonal)

        delta_x = self.numerical_eps + self._delta_x_curr
        if delta_x > 0:
            perturb.add_scalar_at(primal_indices, delta_x)

        delta_c = self._delta_c_curr
        if delta_c > 0:
            perturb.add_scalar_at(dual_indices, -delta_c)

        return

    def _apply_and_factor(self, solver, diagonal, perturb, hessian):
        """Apply perturbation, factorize. Returns (n_pos, n_neg, singular)."""

        self._compute_perturbation(diagonal, perturb)
        solver.factor(hessian, perturb)

        npos, nneg = solver.get_inertia()
        return npos, nneg, False

    def factor_for_inertia(self, solver, evaluator, state):
        """Assemble, regularize, and factorize the KKT matrix."""

        # Evaluate the Hessian at the current design point
        evaluator.evaluate_hessian(state)

        # Evaluate the diagonal at the current point
        evaluator.evaluate_diagonal(state)

        # Return without modifying the diagonal
        if not solver.inertia_enabled():
            self._compute_perturbation(state.diagonal, self.perturbed_diagonal)
            solver.factor(state.hessian, self.perturbed_diagonal)
            return True

        # Prepare new system: save last perturbation, reset current
        self._consider_new_system(state)

        # Sync pivot tolerance to solver
        solver.set_pivot_tolerance(self._pivtol)
        augsys_improved = False

        # Main retry loop
        for attempt in range(self.max_corrections + 1):
            n_pos, n_neg, singular = self._apply_and_factor(
                solver, state.diagonal, self.perturbed_diagonal, state.hessian
            )
            ineritia_ok = self._is_inertia_ok(n_pos, n_neg)

            if not singular and ineritia_ok:
                # Success
                self.last_delta_w = self._delta_x_curr
                self.last_delta_c = self._delta_c_curr
                if self._delta_x_curr > 0 and state.comm_rank == 0 and self.verbose:
                    print(
                        f"  Inertia correction: "
                        f"delta_w={self._delta_x_curr:.2e}, "
                        f"delta_c={self._delta_c_curr:.2e}, "
                        f"attempts={attempt + 1}"
                    )
                return True

            if state.comm_rank == 0 and not singular and self.verbose:
                print(
                    f"  Inertia: expected ({self.num_primal}+, {self.num_dual}-), "
                    f"got ({n_pos}+, {n_neg}-), "
                    f"dw={self._delta_x_curr:.1e}, pivtol={self._pivtol:.1e}"
                )

            # Dispatch based on failure type
            if singular and self.num_dual > 0:
                if not self._perturb_for_singularity(state):
                    break
            elif not singular and n_neg < self.num_dual:
                # Too few negatives: IncreaseQuality first, then singular
                assume_singular = True
                if not augsys_improved:
                    augsys_improved = self._increase_quality(solver)
                    if augsys_improved:
                        assume_singular = False
                if assume_singular:
                    if not self._perturb_for_singularity(state):
                        break
            else:
                # SYMSOLVER_WRONG_INERTIA (too many negatives) or
                # SYMSOLVER_SINGULAR with no constraints
                if not self._perturb_for_wrong_inertia(state):
                    if state.comm_rank == 0:
                        print(
                            f"  Inertia: delta_w={self._delta_x_curr:.2e} "
                            f"> max, aborting correction"
                        )
                    break

        # Inertia correction failed — store last actually-applied values
        self.last_delta_w = self._delta_x_curr
        self.last_delta_c = self._delta_c_curr

        return False

    def _increase_quality(self, solver):
        """Increase pivot tolerance: pivtol = min(pivtolmax, sqrt(pivtol))."""
        if self._pivtol >= self._pivtolmax:
            return False
        self._pivtol = min(self._pivtolmax, self._pivtol**0.5)
        solver.set_pivot_tolerance(self._pivtol)
        return True

    def add_log_info(self, info):
        """Add information to the logger"""
        info["inertia_delta"] = self.last_delta_w
