from .inertia_correction import InertiaCorrector
from .linear_solver import LinearSolver

from .cuda_solver import DirectCudaSolver
from .mumps_solver import MumpsSolver
from .amigo_solver import AmigoSolver

# from .lnks_solver import LNKSInexactSolver
# from .pardiso_solver import PardisoSolver
# from .petsc_solver import DirectPetscSolver
# from .scipy_solver import DirectScipySolver

import warnings


def make_solver(options, state):
    """Make the linear solver depending on the options"""
    if isinstance(options["solver"], LinearSolver):
        return options["solver"]
    elif options["solver"] == "amigo":
        return AmigoSolver(options, state)
    elif options["solver"] == "mumps":
        try:
            return MumpsSolver(options, state)
        except:
            warnings.warn("Exception on MUMPS import, reverting to AmigoSolver")
            return AmigoSolver(options, state)
    elif options["solver"] == "cuda":
        try:
            return DirectCudaSolver(options, state)
        except:
            warnings.warn(
                "Exception on DirectCudaSolver import, reverting to AmigoSolver"
            )
            return AmigoSolver(options, state)
    else:
        solver = options["solver"]
        raise ValueError(f"Unrecognized solver {solver}")

        # if solver is None and self.distribute:
        #     self.solver = DirectPetscSolver(self.comm, self.problem)
        # elif isinstance(solver, str):
        #     solver_pref = solver.lower()
        #     if solver_pref == "scipy":
        #         self.solver = DirectScipySolver(self.problem)
        #     elif solver_pref == "pardiso":
        #         self.solver = PardisoSolver(self.problem)
        #     elif solver_pref == "mumps":
        #         try:
        #             self.solver = MumpsSolver(self.problem)
        #         except:
        #             self.solver = AmigoSolver(self.problem)
        #     elif solver_pref == "amigo":
        #         self.solver = AmigoSolver(self.problem)
        #     else:
        #         raise ValueError(
        #             f"Unknown solver string '{solver}'. "
        #             "Expected one of: 'scipy', 'pardiso', 'mumps', 'amigo'."
        #         )
