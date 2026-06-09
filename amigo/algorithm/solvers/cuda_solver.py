from . import LinearSolver

# TODO: Current GPU solver path (CSRMatFactorCuda):
#   - No inertia query: cannot drive the IPM inertia-correction (IC) step,
#     so supports_inertia must stay False and we fall back to a heuristic
#     regularization loop instead of a real (n_pos, n_neg, n_zero) count.
#   - Only static pivoting with perturbation: no MUMPS-style delayed
#     pivots across the elimination tree, so near-singular or highly
#     indefinite KKT systems (small mu, rank-deficient constraints) can
#     get silently perturbed and lose accuracy.
#   - No tunable threshold pivoting (no analogue of MUMPS CNTL(1)) and no
#     reporting of perturbed/delayed pivot counts -> hard to diagnose
#     numerical trouble.
#   - No Schur complement, no multi-GPU, no out-of-core.
#   - GPU-resident only.


class DirectCudaSolver(LinearSolver):
    def __init__(self, options, state):
        try:
            from amigo.amigo import CSRMatFactorCuda
        except:
            raise NotImplementedError("Amigo compiled without CUDA support")

        pivot_eps = 1e-6
        self.mat_copy = state.hessian.duplicate()

        # Create the solver
        self.solver = CSRMatFactorCuda(self.mat_copy, pivot_eps)

    def factor(self, hessian, diagonal):
        self.mat_copy.copy(hessian)
        self.mat_copy.add_diagonal(diagonal)
        self.solver.factor()

    def solve(self, bx, px):
        self.solver.solve(bx, px)

    def inertia_enabled(self):
        return True

    def get_inertia(self):
        return self.solver.get_inertia()

    def set_pivot_tolerance(self, pivtol):
        pass
