"""Filter line search, second-order correction, and watchdog.

Trial points are accepted against a two-dimensional filter over the
barrier objective phi_mu = f(x) - mu * sum(log(gap)) and the 1-norm
constraint violation theta.  A step passes if it satisfies either an
f-type Armijo condition or an h-type sufficient-reduction condition
and is not dominated by any stored filter entry.  SOC is applied on
the first rejection to counter the Maratos effect.  When many
shortened steps accumulate, the watchdog temporarily relaxes
acceptance to escape filter stalls.
"""

import numpy as np
from .filter_acceptance import Filter


class LineSearchInfo:
    success: bool
    num_search_iters: int
    alpha_primal: float
    alpha_dual: float


class FilterLineSearchNew:
    """Filter-based line search, SOC, and watchdog procedure."""

    def __init__(self, options, problem, optimizer):
        self.options = options
        self.problem = problem
        self.optimizer = optimizer

        # Allocate vectors that will be used internally
        self.trial = self.optimizer.create_opt_vector()
        self.trial_grad = self.problem.create_vector()

        # Allocate the internal filter
        self.filter = Filter()

        # Constant = 10 times machine precision
        self.EPS10 = 10.0 * np.finfo(float).eps

        # We can't set the limiting theta values yet. Set a flag to set them on the first
        # iteration through the line search
        self.theta_limits_initialized = False

        # Set the current info
        self.current_info = None

        # Keep track of how many successive times there is a filter reject
        self.successive_filter_rejections = 0
        self.filter_reset_count = 0

        return

    def reset_on_new_barrier(self, state):
        """This is called when the barrier parameter is updated"""

        self.filter.clear()
        self.theta_limits_initialized = False
        self.successive_filter_rejections = 0
        self.filter_reset_count = 0

        return

    class FilterBasePoint:
        ref_barr: float
        ref_theta: float
        ref_dphi: float

    def set_reference_values(self, ref_theta):
        # theta_min, theta_max (Eq. 21)
        self.theta_min = 1e-4 * max(1.0, ref_theta)
        self.theta_max = 1e4 * max(1.0, ref_theta)

        return

    def is_ftype(self, base, alpha_test):
        delta = self.options["filter_delta"]
        s_theta = self.options["filter_s_theta"]
        s_phi = self.options["filter_s_phi"]

        return (
            base.ref_dphi < 0.0
            and alpha_test * (-base.ref_dphi) ** s_phi > delta * base.ref_theta**s_theta
        )

    def armijo_holds(self, base, trial_barr, alpha_test):
        """Check if the Armijo condition holds relative to the base point"""
        eta_phi = self.options["filter_eta_phi"]

        return (
            trial_barr - base.ref_barr
        ) - eta_phi * alpha_test * base.ref_dphi <= self.EPS10 * abs(base.ref_barr)

    def acceptable_to_iterate(self, base, trial_barr, trial_theta):
        gamma_theta = self.options["filter_gamma_theta"]
        gamma_phi = self.options["filter_gamma_phi"]
        obj_max_inc = 5.0

        if trial_barr > base.ref_barr:
            basval = 1.0
            if abs(base.ref_barr) > 10.0:
                basval = np.log10(abs(base.ref_barr))
            if np.log10(max(trial_barr - base.ref_barr, 1e-300)) > obj_max_inc + basval:
                return False
        return (
            trial_theta - (1.0 - gamma_theta) * base.ref_theta
            <= self.EPS10 * abs(base.ref_theta)
        ) or (
            (trial_barr - base.ref_barr) - (-gamma_phi * base.ref_theta)
            <= self.EPS10 * abs(base.ref_barr)
        )

    def check_acceptance(self, base, trial_barr, trial_theta, alpha_test):
        if trial_theta > self.theta_max:
            return False, False
        if (
            alpha_test > 0.0
            and self.is_ftype(base, alpha_test)
            and base.ref_theta <= self.theta_min
        ):
            if not self.armijo_holds(base, trial_barr, alpha_test):
                return False, False
        else:
            if not self.acceptable_to_iterate(base, trial_barr, trial_theta):
                return False, False

        filt_ok = self.filter.is_acceptable(trial_barr, trial_theta)
        return filt_ok, not filt_ok

    def update_filter(self, base, trial_barr, trial_theta, alpha_test):

        ftest = self.is_ftype(base, alpha_test)
        armijo = self.armijo_holds(base, trial_barr, alpha_test)

        if not (ftest and armijo):
            self.filter.add(trial_barr, trial_theta)
            # self.filter.add(base.ref_barr, base.ref_theta)

    def build_base_point(self, evaluator, state):
        base = self.FilterBasePoint()
        fobj, barrier = evaluator.evaluate_objective_and_barrier_from_point(
            state.mu, state.obj_scale, state.current
        )
        base.ref_barr = fobj + barrier
        base.ref_theta = evaluator.evalate_infeasibility_from_gradient(state.gradient)
        base.ref_dphi = evaluator.evaluate_directional_derivative(state)

        return base

    def compute_alpha_min(self, base):
        # Compute alpha_min (Eq. 23)
        gamma_theta = self.options["filter_gamma_theta"]
        gamma_phi = self.options["filter_gamma_phi"]
        s_theta = self.options["filter_s_theta"]
        s_phi = self.options["filter_s_phi"]
        delta = self.options["filter_delta"]
        alpha_min_frac = 0.05

        alpha_min = gamma_theta
        if base.ref_dphi < 0.0:
            alpha_min = min(gamma_theta, gamma_phi * base.ref_theta / (-base.ref_dphi))
            if base.ref_theta <= self.theta_min:
                alpha_min = min(
                    alpha_min,
                    delta * base.ref_theta**s_theta / (-base.ref_dphi) ** s_phi,
                )
        alpha_min *= alpha_min_frac

        return alpha_min

    def line_search(self, solver, evaluator, state):

        # Check if we should reset the filter
        filter_reset_trigger = self.options["filter_reset_trigger"]
        max_filter_resets = self.options["max_filter_resets"]
        if (
            self.successive_filter_rejections >= filter_reset_trigger
            and self.filter_reset_count < max_filter_resets
        ):
            self.filter.clear()
            state.filter_reset_count += 1

        # Build the base point for comparison. In case a watchdog is set, this
        # base point will come from the watchdog.
        base = self.build_base_point(evaluator, state)

        if not self.theta_limits_initialized:
            self.set_reference_values(base.ref_theta)
            self.theta_limits_initialized = True

        # Compute the minimum step length
        alpha_min = self.compute_alpha_min(base)

        # Initial step lengths
        alpha_primal = state.max_alpha_primal
        alpha_dual = state.max_alpha_dual

        # Max line search iterations
        max_line_iters = self.options["max_line_search_iterations"]

        # Build the info class based on a failure of the line search. If we find
        # an acceptable point, this will be over-written
        info = LineSearchInfo()
        info.success = False
        info.num_search_iters = max_line_iters

        for line_iter in range(max_line_iters):
            # Compute the update for the current iteration
            self.optimizer.apply_step_update(
                alpha_primal, alpha_dual, state.current, state.step, self.trial
            )

            # Compute the gradient at the trial point
            evaluator.evaluate_gradient_from_point(
                state.obj_scale, self.trial, self.trial_grad
            )

            # Get the objective and log-barrier term
            fobj, barrier = evaluator.evaluate_objective_and_barrier_from_point(
                state.mu, state.obj_scale, self.trial
            )
            trial_barr = fobj + barrier

            # Compute the l1 norm of the constraint violation
            trial_theta = evaluator.evalate_infeasibility_from_gradient(self.trial_grad)

            accepted, filt_rej = self.check_acceptance(
                base, trial_barr, trial_theta, alpha_primal
            )

            if accepted or alpha_primal == alpha_min:
                self.update_filter(base, trial_barr, trial_theta, alpha_primal)

                # Update the state information: Invalidate the current point, but copy
                # over the gradient that we just computed at the trial point
                state.invalidate()
                state.current.copy(self.trial)
                state.gradient.copy(self.trial_grad)
                state.gradient_current = True

                # Check if we have repeated rejected steps here...
                if filt_rej:
                    self.successive_filter_rejections += 1
                else:
                    self.successive_filter_rejections = 0

                # Set the current info
                info.success = True
                info.num_search_iters = line_iter + 1
                info.alpha_primal = alpha_primal
                info.alpha_dual = alpha_dual

                # Retain a copy of the line search info for logging iterations
                self.current_info = info

                return info

            backtrack_factor = self.options["backtracking_factor"]
            alpha_primal = max(backtrack_factor * alpha_primal, alpha_min)

        return info

    def add_log_info(self, info):
        if self.current_info is not None:
            info["line_iters"] = self.current_info.num_search_iters
            info["alpha_x"] = self.current_info.alpha_primal
            info["alpha_z"] = self.current_info.alpha_dual


class FilterLineSearch:
    def _compute_barrier_objective(self, vars):
        """Barrier objective phi_mu = f(x) - mu * sum(ln(gaps)).

        f(x) is obtained by evaluating L(x, lam=0) = alpha * f(x).
        """
        # TODO: move to backend - the zero-multipliers-evaluate-restore
        # dance around problem.lagrangian should be a single
        # backend.barrier_objective(mu, vars) call.
        problem = self.mpi_problem if self.distribute else self.problem
        x_vec = vars.get_solution()
        # x_arr = x_vec.get_array()

        # mult_ind = np.array(
        #     (
        #         self.mpi_problem if self.distribute else self.problem
        #     ).get_multiplier_indicator(),
        #     dtype=bool,
        # )
        # lam_backup = x_arr[mult_ind].copy()
        # x_arr[mult_ind] = 0.0
        # x_vec.copy_host_to_device()
        # f_obj = problem.lagrangian(self._obj_scale, x_vec)
        # x_arr[mult_ind] = lam_backup
        # x_vec.copy_host_to_device()

        # Zero dual, evaluate L(x,0) = f(x), restore
        con_indices = self.problem.get_constraint_indices()
        temp_vec = self.problem.create_constraint_vector()

        # Save the dual values and then zero them
        x_vec.get_values_at(con_indices, temp_vec)
        x_vec.fill_at(con_indices, 0.0)
        f_obj = problem.lagrangian(self._obj_scale, x_vec)

        # Restore the dual values
        x_vec.set_values_at(con_indices, temp_vec)

        barrier_log = self.optimizer.compute_barrier_log_sum(self.barrier_param, vars)
        return f_obj + barrier_log

    def _compute_filter_theta(self, vars=None):
        """Compute theta = ||c(x)||_1 (1-norm)."""
        if vars is None:
            vars = self.vars
        return self.optimizer.compute_constraint_violation_1norm(vars, self.grad)

    def _filter_line_search(
        self,
        alpha_x,
        alpha_z,
        inner_filter,
        options,
        comm_rank,
        tau=0.995,
        phi_current=None,
        watchdog_ref=None,
        watchdog_alpha_primal_test=None,
    ):
        """Filter line search with second-order correction.

        Returns
        -------
        alpha : float
            Accepted step size (fraction of alpha_x).
        line_iters : int
            Number of backtracking iterations.
        step_accepted : bool
            False if step was rejected (triggers restoration).
        filter_rejected : bool
            True if the last rejection was due to the filter.
        """
        max_iters = options["max_line_search_iterations"]
        alpha_red = options["backtracking_factor"]
        gamma_theta = options["filter_gamma_theta"]
        gamma_phi = options["filter_gamma_phi"]
        delta = options["filter_delta"]
        s_theta = options["filter_s_theta"]
        s_phi = options["filter_s_phi"]
        eta_phi = options["filter_eta_phi"]
        alpha_min_frac = 0.05
        max_soc = options["filter_max_soc"]
        kappa_soc = options["filter_kappa_soc"]
        obj_max_inc = 5.0
        use_soc = options["second_order_correction"]
        EPS10 = 10.0 * np.finfo(float).eps

        # Reference values
        if watchdog_ref is not None:
            ref_theta, ref_barr, ref_dphi = watchdog_ref
        else:
            ref_theta = self._compute_filter_theta()
            ref_barr = phi_current
            ref_dphi = self.optimizer.compute_barrier_dphi(
                self.barrier_param,
                self.vars,
                self.update,
                self.res,
                self.px,
                self.diag,
            )

        # theta_min, theta_max (Eq. 21)
        theta_0 = getattr(self, "_filter_theta_0", ref_theta)
        theta_min = 1e-4 * max(1.0, theta_0)
        theta_max = 1e4 * max(1.0, theta_0)

        # Alpha_min (Eq. 23)
        if watchdog_ref is not None:
            alpha_min = alpha_x
        else:
            alpha_min = gamma_theta
            if ref_dphi < 0.0:
                alpha_min = min(gamma_theta, gamma_phi * ref_theta / (-ref_dphi))
                if ref_theta <= theta_min:
                    alpha_min = min(
                        alpha_min,
                        delta * ref_theta**s_theta / (-ref_dphi) ** s_phi,
                    )
            alpha_min *= alpha_min_frac

        def _is_ftype(alpha_test):
            return (
                ref_dphi < 0.0
                and alpha_test * (-ref_dphi) ** s_phi > delta * ref_theta**s_theta
            )

        def _armijo_holds(trial_barr, alpha_test):
            return (
                trial_barr - ref_barr
            ) - eta_phi * alpha_test * ref_dphi <= EPS10 * abs(ref_barr)

        def _acceptable_to_iterate(trial_barr, trial_theta):
            if trial_barr > ref_barr:
                basval = 1.0
                if abs(ref_barr) > 10.0:
                    basval = np.log10(abs(ref_barr))
                if np.log10(max(trial_barr - ref_barr, 1e-300)) > obj_max_inc + basval:
                    return False
            return (
                trial_theta - (1.0 - gamma_theta) * ref_theta <= EPS10 * abs(ref_theta)
            ) or (
                (trial_barr - ref_barr) - (-gamma_phi * ref_theta)
                <= EPS10 * abs(ref_barr)
            )

        def _check_acceptance(trial_barr, trial_theta, alpha_test):
            if trial_theta > theta_max:
                return False, False
            if alpha_test > 0.0 and _is_ftype(alpha_test) and ref_theta <= theta_min:
                if not _armijo_holds(trial_barr, alpha_test):
                    return False, False
            else:
                if not _acceptable_to_iterate(trial_barr, trial_theta):
                    return False, False
            filt_ok = inner_filter.is_acceptable(trial_barr, trial_theta)
            return filt_ok, not filt_ok

        def _update_filter(trial_barr, alpha_test):
            if not (_is_ftype(alpha_test) and _armijo_holds(trial_barr, alpha_test)):
                inner_filter.add(ref_barr, ref_theta)

        # SOC state backup
        if use_soc:
            self.optimizer.compute_residual(
                self.barrier_param, self.vars, self.grad, self.res
            )
            # res_orig = self.res.get_array().copy()
            # update_backup = self.optimizer.create_opt_vector()
            # update_backup.copy(self.update)
            # px_orig = self.px.get_array().copy()

            # Back up the original values
            res_orig = self.problem.create_vector()
            res_orig.copy(self.res)
            px_orig = self.problem.create_vector()
            px_orig.copy(self.px)
            update_backup = self.optimizer.create_opt_vector()
            update_backup.copy(self.update)

        alpha_primal = alpha_x
        n_steps = 0
        last_rejected_by_filter = False

        while alpha_primal > alpha_min or n_steps == 0:
            if n_steps >= max_iters:
                break

            self.optimizer.apply_step_update(
                alpha_primal, alpha_z, self.vars, self.update, self.temp
            )
            self._update_gradient(self.temp.get_solution())

            trial_theta = self._compute_filter_theta(self.temp)
            trial_barr = self._compute_barrier_objective(self.temp)

            alpha_primal_test = alpha_primal

            accepted, filt_rej = _check_acceptance(
                trial_barr, trial_theta, alpha_primal_test
            )
            if accepted:
                _update_filter(trial_barr, alpha_primal_test)
                self.vars.copy(self.temp)
                return (
                    alpha_primal / alpha_x,
                    n_steps + 1,
                    True,
                    last_rejected_by_filter,
                )
            last_rejected_by_filter = filt_rej

            # SOC: second-order correction for the Maratos effect
            if use_soc and n_steps == 0:
                if comm_rank == 0 and options.get("verbose_barrier"):
                    print(
                        f"  SOC check: trial_theta={trial_theta:.3e}, "
                        f"ref_theta={ref_theta:.3e}, "
                        f"trigger={'YES' if trial_theta >= ref_theta else 'no'}"
                    )
            if use_soc and n_steps == 0 and trial_theta >= ref_theta:
                # c_soc = res_orig.copy()
                c_soc = res_orig
                alpha_soc = alpha_primal
                theta_soc_old = 0.0
                soc_accepted = False

                for soc_count in range(max_soc):
                    if soc_count > 0 and trial_theta > kappa_soc * theta_soc_old:
                        break
                    theta_soc_old = trial_theta

                    self.optimizer.compute_residual(
                        self.barrier_param, self.temp, self.grad, self.res
                    )
                    # trial_res = self.res.get_array().copy()
                    # c_soc[mult_ind] = trial_res[mult_ind] + alpha_soc * c_soc[mult_ind]

                    # self.res.get_array()[:] = c_soc
                    # self.res.copy_host_to_device()
                    # self._update_gradient(self.vars.get_solution())

                    con_indices = self.problem.get_constraint_indices()
                    trial_res = self.problem.create_vector()
                    trial_res.copy(self.res)

                    trial_res.axpy_at(con_indices, alpha_soc, c_soc)
                    self._update_gradient(self.vars.get_solution())

                    try:
                        self.solver.solve(self.res, self.px)
                    except Exception:
                        break
                    self.optimizer.compute_update(
                        self.barrier_param, self.vars, self.px, self.update
                    )

                    soc_ax, _, soc_az, _ = self.optimizer.compute_max_step(
                        tau, self.vars, self.update
                    )
                    self.optimizer.apply_step_update(
                        soc_ax, soc_az, self.vars, self.update, self.temp
                    )
                    self._update_gradient(self.temp.get_solution())

                    trial_theta = self._compute_filter_theta(self.temp)
                    trial_barr = self._compute_barrier_objective(self.temp)

                    soc_acc, soc_filt_rej = _check_acceptance(
                        trial_barr, trial_theta, alpha_primal_test
                    )
                    if soc_acc:
                        _update_filter(trial_barr, alpha_primal_test)
                        self.vars.copy(self.temp)
                        soc_accepted = True
                        break

                    alpha_soc = soc_ax

                if soc_accepted:
                    if comm_rank == 0:
                        print(f"  SOC accepted (iter {soc_count+1}/{max_soc})")
                    return 1.0, n_steps + 1, True, False

                self.update.copy(update_backup)
                # self.px.get_array()[:] = px_orig
                # self.px.copy_host_to_device()
                self.px.copy(px_orig)

            alpha_primal *= alpha_red
            n_steps += 1

        # All backtracking exhausted
        self._update_gradient(self.vars.get_solution())
        return alpha_primal / alpha_x, n_steps, False, last_rejected_by_filter

    def _filter_line_search_with_watchdog(
        self,
        alpha_x,
        alpha_z,
        inner_filter,
        options,
        comm_rank,
        tau,
        watchdog,
        factorize_ok,
    ):
        """Run the filter line search with watchdog start/stop/retry logic.

        The watchdog tracks consecutive shortened steps; once too many
        accumulate, it saves the iterate, accepts a worse step, and
        checks progress over the next few iterations.  If no progress,
        it restores the saved iterate and re-runs the line search
        without the relaxation.
        """
        # Pre-line-search watchdog checks
        if watchdog.in_watchdog and (not factorize_ok or alpha_x < 1e-16):
            watchdog.restore_iterate(self.vars, self.update, self.px)
            self._update_gradient(self.vars.get_solution())
            watchdog.in_watchdog = False
            watchdog.shortened_iter = 0
            if comm_rank == 0:
                print("  Watchdog stopped (factorization/tiny step)")

        # StartWatchDog
        if (
            not watchdog.in_watchdog
            and watchdog.trigger > 0
            and watchdog.shortened_iter >= watchdog.trigger
        ):
            watchdog.save_iterate(self.vars, self.update, self.px, alpha_x)
            watchdog.theta = self._compute_filter_theta()
            watchdog.barr = self._compute_barrier_objective(self.vars)
            watchdog.dphi = self.optimizer.compute_barrier_dphi(
                self.barrier_param,
                self.vars,
                self.update,
                self.res,
                self.px,
                self.diag,
            )
            watchdog.in_watchdog = True
            watchdog.trial_iter = 0
            if comm_rank == 0:
                print("  Watchdog started")

        phi_current = self._compute_barrier_objective(self.vars)
        wd_ref = None
        wd_apt = None
        if watchdog.in_watchdog:
            wd_ref = (watchdog.theta, watchdog.barr, watchdog.dphi)
            wd_apt = watchdog.alpha_primal_test

        skip_first = False
        while True:
            alpha, line_iters, step_accepted, filter_rejected = (
                self._filter_line_search(
                    alpha_x,
                    alpha_z,
                    inner_filter,
                    options,
                    comm_rank,
                    tau=tau,
                    phi_current=phi_current,
                    watchdog_ref=wd_ref if not skip_first else None,
                    watchdog_alpha_primal_test=(wd_apt if not skip_first else None),
                )
            )

            if watchdog.in_watchdog and not skip_first:
                if step_accepted:
                    watchdog.in_watchdog = False
                    watchdog.shortened_iter = 0
                    if comm_rank == 0:
                        print("  Watchdog succeeded")
                    break
                watchdog.trial_iter += 1
                if watchdog.trial_iter > watchdog.max_trials:
                    watchdog.restore_iterate(self.vars, self.update, self.px)
                    self._update_gradient(self.vars.get_solution())
                    watchdog.in_watchdog = False
                    watchdog.shortened_iter = 0
                    alpha_x, _, alpha_z, _ = self.optimizer.compute_max_step(
                        tau, self.vars, self.update
                    )
                    phi_current = self._compute_barrier_objective(self.vars)
                    skip_first = True
                    if comm_rank == 0:
                        print("  Watchdog stopped (max trials), retrying LS")
                    continue
                # Force-accept, continue watchdog next iter
                step_accepted = True
                if comm_rank == 0:
                    print(
                        f"  Watchdog trial "
                        f"{watchdog.trial_iter}/{watchdog.max_trials} "
                        f"(force accept)"
                    )
                break
            break  # not in watchdog

        return alpha, line_iters, step_accepted, filter_rejected


class WatchdogState:
    """Watchdog procedure state (saved iterate, counters).

    Held by the Optimizer and passed to the filter line search wrapper
    each iteration.  Owns the backup vectors needed to restore the
    saved iterate when the watchdog decides to abort.
    """

    def __init__(self, backend):
        self.in_watchdog = False
        self.shortened_iter = 0
        self.trial_iter = 0
        self.iterate = None
        self.update_backup = backend.create_opt_vector()
        self.px_backup = None
        self.alpha_primal_test = 0.0
        self.theta = 0.0
        self.barr = 0.0
        self.dphi = 0.0
        self.trigger = 0
        self.max_trials = 0

    def save_iterate(self, vars_, update, px, alpha_x):
        self.iterate = vars_.get_solution().get_array().copy()
        self.update_backup.copy(update)
        self.px_backup = px.get_array().copy()
        self.alpha_primal_test = alpha_x

    def restore_iterate(self, vars_, update, px):
        vars_.get_solution().get_array()[:] = self.iterate
        vars_.get_solution().copy_host_to_device()
        update.copy(self.update_backup)
        update.copy_host_to_device()
        px.get_array()[:] = self.px_backup
        px.copy_host_to_device()

    def reset(self):
        self.in_watchdog = False
        self.shortened_iter = 0
        self.trial_iter = 0
        self.iterate = None
        self.px_backup = None
