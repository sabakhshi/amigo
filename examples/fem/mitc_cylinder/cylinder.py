import argparse
import numpy as np
import amigo as am
from amigo.fem import MITCTyingStrain, MITCElement, SolutionSpace, Mesh, Problem
from amigo.fem.basis import QuadLagrangeBasis, LagrangeBasis2D
from scipy.sparse.linalg import spsolve
from utils import write_vtu, get_exact_solution
import time

from amigo.fem.basis import dot_product, eval_2d_monomials, eval_2d_monomial_grad


class NaturalShellGeoBasis(QuadLagrangeBasis):
    """Shell Geometry Basis --> Computes Jacobian"""

    def __init__(self, degree, names, kind="data"):
        super().__init__(degree, names, kind=kind)

    def compute_transform(self, geo):
        x1, x2 = geo["x"]["grad"]
        y1, y2 = geo["y"]["grad"]
        z1, z2 = geo["z"]["grad"]

        nx0 = geo["nx"]["value"]
        ny0 = geo["ny"]["value"]
        nz0 = geo["nz"]["value"]

        nx1, nx2 = geo["nx"]["grad"]
        ny1, ny2 = geo["ny"]["grad"]
        nz1, nz2 = geo["nz"]["grad"]

        # Compute the normalized direction from the interpolation of the
        # nodal normal vectors. This is used to compute the transformation
        n0_norm = am.sqrt(nx0**2 + ny0**2 + nz0**2)
        n0_inv = 1.0 / n0_norm
        nx = n0_inv * nx0
        ny = n0_inv * ny0
        nz = n0_inv * nz0

        # Project x,1 into the tangent plane of n
        x1_dot_n = x1 * nx + y1 * ny + z1 * nz

        t1x0 = x1 - x1_dot_n * nx
        t1y0 = y1 - x1_dot_n * ny
        t1z0 = z1 - x1_dot_n * nz
        t1_inv = 1.0 / am.sqrt(t1x0**2 + t1y0**2 + t1z0**2)

        t1x = t1_inv * t1x0
        t1y = t1_inv * t1y0
        t1z = t1_inv * t1z0

        # Find the second in-plane direction t2 = n x t1
        t2x = ny * t1z - nz * t1y
        t2y = nz * t1x - nx * t1z
        t2z = nx * t1y - ny * t1x

        # Form the transformation matrix
        T = [
            [t1x, t2x, nx],
            [t1y, t2y, ny],
            [t1z, t2z, nz],
        ]

        # Compute the matrix: Jinv = X,xi^{-1} * T
        # [x,1  x,2  n0x]^{-1} [t1x   t2x   nx]   [ *   *      0   ]
        # [y,1  y,2  n0y]      [t1y   t2y   ny] = [ *   *      0   ]
        # [z,1  z,2  n0z]      [t1z   t2z   nz]   [ 0   0   n0_inv ]
        # Note that since n0 is parallel with n, the resulting matrix is

        # Covariant metric tensor
        g11 = x1 * x1 + y1 * y1 + z1 * z1
        g12 = x1 * x2 + y1 * y2 + z1 * z2
        g22 = x2 * x2 + y2 * y2 + z2 * z2
        detg = g11 * g22 - g12 * g12
        detg_inv = 1.0 / detg

        # Determinant of the Jacobian transformation
        detJ = am.sqrt(detg)

        # Contravariant tensor components
        g11c = detg_inv * g22
        g12c = -detg_inv * g12
        g22c = detg_inv * g11

        d11 = x1 * t1x + y1 * t1y + z1 * t1z
        d12 = x1 * t2x + y1 * t2y + z1 * t2z
        d21 = x2 * t1x + y2 * t1y + z2 * t1z
        d22 = x2 * t2x + y2 * t2y + z2 * t2z

        # Components of the in-plane tensor
        Jinv = [
            [g11c * d11 + g12c * d21, g11c * d12 + g12c * d22],
            [g12c * d11 + g22c * d21, g12c * d12 + g22c * d22],
        ]

        # Compute zJinv = X,xi^{-1} * n0,xi * X,xi^{-1} * T
        def apply_Xinv_to_vec(qx, qy, qz):
            a1_dot_q = x1 * qx + y1 * qy + z1 * qz
            a2_dot_q = x2 * qx + y2 * qy + z2 * qz

            r1 = g11c * a1_dot_q + g12c * a2_dot_q
            r2 = g12c * a1_dot_q + g22c * a2_dot_q

            return r1, r2

        q1x = nx1 * Jinv[0][0] + nx2 * Jinv[1][0]
        q1y = ny1 * Jinv[0][0] + ny2 * Jinv[1][0]
        q1z = nz1 * Jinv[0][0] + nz2 * Jinv[1][0]

        zJ11, zJ21 = apply_Xinv_to_vec(q1x, q1y, q1z)

        q2x = nx1 * Jinv[0][1] + nx2 * Jinv[1][1]
        q2y = ny1 * Jinv[0][1] + ny2 * Jinv[1][1]
        q2z = nz1 * Jinv[0][1] + nz2 * Jinv[1][1]

        zJ12, zJ22 = apply_Xinv_to_vec(q2x, q2y, q2z)

        zJ31 = n0_inv * n0_inv * (nx0 * q1x + ny0 * q1y + nz0 * q1z)
        zJ32 = n0_inv * n0_inv * (nx0 * q2x + ny0 * q2y + nz0 * q2z)

        # Set the rate terms
        zJinv = [
            [zJ11, zJ12],
            [zJ21, zJ22],
            [zJ31, zJ32],
        ]

        # To transform derivatives into the local coordinates
        Jdict = {"Jinv": Jinv, "zJinv": zJinv, "J33": n0_inv, "T": T}

        return detJ, Jdict


class ShellSolnBasis(QuadLagrangeBasis):
    def __init__(self, degree, kind="input"):
        names = ["u", "v", "w", "rx", "ry", "rz"]
        super().__init__(degree, names, kind=kind)

    def eval(self, comp, pt):
        xi = pt[0]
        eta = pt[1]

        # Evaluate the monomials
        m = eval_2d_monomials(self.p, xi, eta, self.exps)
        N = m @ self.C

        # Evaluate the derivatives of the monomials
        mgrad = eval_2d_monomial_grad(self.p, xi, eta, self.exps)
        Nxi = mgrad[:, 0] @ self.C
        Neta = mgrad[:, 1] @ self.C

        # Get the rotations at the nodes
        ry = comp.inputs["ry"]
        rx = comp.inputs["rx"]
        rz = comp.inputs["rz"]

        # Get the normals at the nodes
        nx = comp.data["nx"]
        ny = comp.data["ny"]
        nz = comp.data["nz"]

        # Set up dx, dy, dz to interpolate the directors from the nodes
        dx = [None] * self.nnodes
        dy = [None] * self.nnodes
        dz = [None] * self.nnodes
        for i in range(self.nnodes):
            dx[i] = ry[i] * nz[i] - rz[i] * ny[i]
            dy[i] = rz[i] * nx[i] - rx[i] * nz[i]
            dz[i] = rx[i] * ny[i] - ry[i] * nx[i]

        soln = {}
        for name in self.names:
            if self.kind == "input":
                u = comp.inputs[name]
            elif self.kind == "data":
                u = comp.data[name]
            elif self.kind == "multiplier":
                u = comp.constraints.get_multipliers()[f"res_{name}"]

            soln[name] = {
                "value": dot_product(u, N, n=self.nnodes),
                "grad": [
                    dot_product(u, Nxi, n=self.nnodes),
                    dot_product(u, Neta, n=self.nnodes),
                ],
            }

        for name, u in zip(["dx", "dy", "dz"], [dx, dy, dz]):
            soln[name] = {
                "value": dot_product(u, N, n=self.nnodes),
                "grad": [
                    dot_product(u, Nxi, n=self.nnodes),
                    dot_product(u, Neta, n=self.nnodes),
                ],
            }

        return soln

    def transform(self, detJ, Jdict, orig):
        # Get the transformation information
        Jinv = Jdict["Jinv"]
        zJinv = Jdict["zJinv"]
        T = Jdict["T"]

        # Compute the the values
        soln = {}

        # The interpolated values from the solution field
        soln["u"] = {"value": orig["u"]["value"]}
        soln["v"] = {"value": orig["v"]["value"]}
        soln["w"] = {"value": orig["w"]["value"]}

        soln["rx"] = {"value": orig["rx"]["value"]}
        soln["ry"] = {"value": orig["ry"]["value"]}
        soln["rz"] = {"value": orig["rz"]["value"]}

        u1, u2 = orig["u"]["grad"]
        v1, v2 = orig["v"]["grad"]
        w1, w2 = orig["w"]["grad"]

        dx = orig["dx"]["value"]
        dy = orig["dy"]["value"]
        dz = orig["dz"]["value"]

        dx1, dx2 = orig["dx"]["grad"]
        dy1, dy2 = orig["dy"]["grad"]
        dz1, dz2 = orig["dz"]["grad"]

        # Compute c = T^{T} * [d,1  d,2  0]
        c11 = T[0][0] * dx1 + T[1][0] * dy1 + T[2][0] * dz1
        c21 = T[0][1] * dx1 + T[1][1] * dy1 + T[2][1] * dz1

        c12 = T[0][0] * dx2 + T[1][0] * dy2 + T[2][0] * dz2
        c22 = T[0][1] * dx2 + T[1][1] * dy2 + T[2][1] * dz2

        c31 = T[0][2] * dx1 + T[1][2] * dy1 + T[2][2] * dz1
        c32 = T[0][2] * dx2 + T[1][2] * dy2 + T[2][2] * dz2

        # Compute b = T^{T} * [u,1  u,2  d]
        b11 = T[0][0] * u1 + T[1][0] * v1 + T[2][0] * w1
        b12 = T[0][0] * u2 + T[1][0] * v2 + T[2][0] * w2
        b13 = T[0][0] * dx + T[1][0] * dy + T[2][0] * dz

        b21 = T[0][1] * u1 + T[1][1] * v1 + T[2][1] * w1
        b22 = T[0][1] * u2 + T[1][1] * v2 + T[2][1] * w2
        b23 = T[0][1] * dx + T[1][1] * dy + T[2][1] * dz

        b31 = T[0][2] * u1 + T[1][2] * v1 + T[2][2] * w1
        b32 = T[0][2] * u2 + T[1][2] * v2 + T[2][2] * w2
        b33 = T[0][2] * dx + T[1][2] * dy + T[2][2] * dz

        def row_transform(c1, c2, b1, b2, b3):
            r1 = (
                c1 * Jinv[0][0]
                + c2 * Jinv[1][0]
                - b1 * zJinv[0][0]
                - b2 * zJinv[1][0]
                - b3 * zJinv[2][0]
            )

            r2 = (
                c1 * Jinv[0][1]
                + c2 * Jinv[1][1]
                - b1 * zJinv[0][1]
                - b2 * zJinv[1][1]
                - b3 * zJinv[2][1]
            )

            return [r1, r2]

        # Compute c * Jinv - b * zJinv
        soln["u1"] = {"grad": row_transform(c11, c12, b11, b12, b13)}
        soln["v1"] = {"grad": row_transform(c21, c22, b21, b22, b23)}
        soln["w1"] = {"grad": row_transform(c31, c32, b31, b32, b33)}

        return soln


class MITC4ShellTying(MITCTyingStrain):
    """
    MITC4 tying strains for a 3D curved shell
    DOFS: u,v,w, rx, ry, rz

    """

    def __init__(self):
        super().__init__()

    def get_tying_points(self):
        pts = []
        for index in range(9):
            pts.append(self._get_tying_point(index))
        return pts

    def _get_tying_field_and_offset(self, idx):
        if idx == 0 or idx == 1:
            return "g11", 0
        if idx == 2 or idx == 3:
            return "g22", 2
        if idx == 4:
            return "g12", 4
        if idx == 5 or idx == 6:
            return "g23", 5
        if idx == 7 or idx == 8:
            return "g13", 7

    def _get_tying_point(self, idx):
        field, offset = self._get_tying_field_and_offset(idx)

        if field == "g11" or field == "g13":
            return [(0, -1), (0, 1)][idx - offset]
        if field == "g12":
            return (0, 0)
        if field == "g22" or field == "g23":
            return [(-1, 0), (1, 0)][idx - offset]

    def eval_tying_strain(self, idx, geo, soln):
        nx0 = geo["nx"]["value"]
        ny0 = geo["ny"]["value"]
        nz0 = geo["nz"]["value"]

        x1, x2 = geo["x"]["grad"]
        y1, y2 = geo["y"]["grad"]
        z1, z2 = geo["z"]["grad"]

        dx = soln["dx"]["value"]
        dy = soln["dy"]["value"]
        dz = soln["dz"]["value"]

        u1, u2 = soln["u"]["grad"]
        v1, v2 = soln["v"]["grad"]
        w1, w2 = soln["w"]["grad"]

        field, _ = self._get_tying_field_and_offset(idx)

        if field == "g11":
            # G11 component
            g11 = u1 * x1 + v1 * y1 + w1 * z1
            return g11
        elif field == "g22":
            # G22 component
            g22 = u2 * x2 + v2 * y2 + w2 * z2
            return g22
        elif field == "g12":
            # G12 component
            g12 = 0.5 * ((u1 * x2 + v1 * y2 + w1 * z2) + (u2 * x1 + v2 * y1 + w2 * z1))
            return g12
        elif field == "g23":
            # G23 component
            g23 = 0.5 * (
                (x2 * dx + y2 * dy + z2 * dz) + (nx0 * u2 + ny0 * v2 + nz0 * w2)
            )
            return g23
        elif field == "g13":
            # G13 component
            g13 = 0.5 * (
                (x1 * dx + y1 * dy + z1 * dz) + (nx0 * u1 + ny0 * v1 + nz0 * w1)
            )
            return g13

    def interp_and_transform(self, pt, Jdict, e):
        # Interpolate the tensorial components of the tying strains
        g11 = 0.5 * ((1.0 - pt[1]) * e[0] + (1.0 + pt[1]) * e[1])
        g22 = 0.5 * ((1.0 - pt[0]) * e[2] + (1.0 + pt[0]) * e[3])
        g12 = e[4]
        g23 = 0.5 * ((1.0 - pt[0]) * e[5] + (1.0 + pt[0]) * e[6])
        g13 = 0.5 * ((1.0 - pt[1]) * e[7] + (1.0 + pt[1]) * e[8])

        # Extract the transformation matrix
        Jinv = Jdict["Jinv"]
        J33 = Jdict.get("J33", 1.0)

        J11 = Jinv[0][0]
        J12 = Jinv[0][1]
        J21 = Jinv[1][0]
        J22 = Jinv[1][1]

        # First compute H = G * Jinv
        h11 = g11 * J11 + g12 * J21
        h12 = g11 * J12 + g12 * J22

        h21 = g12 * J11 + g22 * J21
        h22 = g12 * J12 + g22 * J22

        # Then compute Gbar = Jinv^T * H
        ex = J11 * h11 + J21 * h21
        ey = J12 * h12 + J22 * h22

        # Transform and convert to engineering shear strain
        gxy = 2.0 * (J11 * h12 + J21 * h22)
        gxz = 2.0 * J33 * (J11 * g13 + J21 * g23)
        gyz = 2.0 * J33 * (J12 * g13 + J22 * g23)

        out = {}
        out["ex"] = {"value": ex}
        out["ey"] = {"value": ey}
        out["gxy"] = {"value": gxy}

        out["gxz"] = {"value": gxz}
        out["gyz"] = {"value": gyz}

        return out


def integrand(soln, data=None, geo=None):
    x = geo["x"]["value"]
    y = geo["y"]["value"]
    z = geo["z"]["value"]

    nx0 = geo["nx"]["value"]
    ny0 = geo["ny"]["value"]
    nz0 = geo["nz"]["value"]

    rx = soln["rx"]["value"]
    ry = soln["ry"]["value"]
    rz = soln["rz"]["value"]

    u = soln["u"]["value"]
    v = soln["v"]["value"]
    w = soln["w"]["value"]

    # Gradients for the bending terms
    u1x, u1y = soln["u1"]["grad"]
    v1x, v1y = soln["v1"]["grad"]

    # In-plane strains from MITC interpolation
    ex = soln["ex"]["value"]
    ey = soln["ey"]["value"]
    gxy = soln["gxy"]["value"]

    # Shear strains from MITC interpolation
    gxz = soln["gxz"]["value"]
    gyz = soln["gyz"]["value"]

    kx = u1x
    ky = v1y
    kxy = u1y + v1x

    E = 70e9
    nu = 0.3
    ks = 5.0 / 6.0

    # Set up the cylinder parameters
    t = 0.01
    R = 1.0
    L = 2.0

    M = 4
    N = 3

    alpha = 4.0 / R
    beta = 3 * np.pi / L

    theta = am.atan2(x, y)

    # Make the pressure dependent on the coordinate
    pressure = am.sin(alpha * R * theta) * am.sin(beta * z)

    # Compute the in-plane energy
    A = E * t / (1.0 - nu**2)
    Um = 0.5 * A * ((ex**2 + ey**2 + 2.0 * nu * ex * ey) + 0.5 * (1.0 - nu) * gxy**2)

    # Compute the bending energy
    D = E * t**3 / (12 * (1.0 - nu**2))
    Ub = 0.5 * D * ((kx**2 + ky**2 + 2.0 * nu * kx * ky) + 0.5 * (1.0 - nu) * kxy**2)

    # Compute the shear energy
    G = 0.5 * E / (1.0 + nu)
    Us = 0.5 * ks * G * t * (gxz**2 + gyz**2)

    # Compute the drill penalty
    rot = rx * nx0 + ry * ny0 + rz * nz0

    k_drill = 1e-4 * E * t
    Ud = 0.5 * k_drill * rot**2

    U = Um + Ub + Us + Ud

    # Set the work term
    W = pressure * (u * nx0 + v * ny0 + w * nz0)

    return U - W


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
parser.add_argument(
    "--solver",
    dest="solver",
    choices=["cholesky", "cholesky_left", "ldl", "scipy", "cuda"],
    default="cholesky",
)
args = parser.parse_args()

# Load the mesh
mesh = Mesh("cylinder.inp")
domains = mesh.get_domains()

lateral_surfaces = ["SURFACE1"]
bottom_line = "LINE3"
lateral_line = "LINE2"
top_line = "LINE1"
print(f"Lateral: {lateral_surfaces}, bottom: {bottom_line}, top: {top_line}")

# 6 DOF/node: u, v, w translations + rx, ry, rz global rotations
soln_space = SolutionSpace(
    {"u": "H1", "v": "H1", "w": "H1", "rx": "H1", "ry": "H1", "rz": "H1"}
)
geo_space = SolutionSpace(
    {"x": "H1", "y": "H1", "z": "H1", "nx": "H1", "ny": "H1", "nz": "H1"}
)
data_space = SolutionSpace({})

etype = "CPS4"

degree = 1
soln_basis = ShellSolnBasis(degree, kind="input")
geo_basis = NaturalShellGeoBasis(degree, ["x", "y", "z", "nx", "ny", "nz"], kind="data")
quadrature = mesh.get_quadrature(etype)
data_basis = mesh.get_basis(data_space, etype, kind="data")
mitc = MITC4ShellTying()

shell_elem = MITCElement(
    "Shell", soln_basis, data_basis, geo_basis, quadrature, mitc, integrand
)

integrand_map = {
    "shell": {
        "target": lateral_surfaces,
        "integrand": integrand,
    },
}
bc_map = {
    "pinned_bottom": {
        "type": "dirichlet",
        "input": ["u", "v", "rz"],
        "target": [bottom_line],
    },
    "pinned_top": {
        "type": "dirichlet",
        "input": ["u", "v", "rz"],
        "target": [top_line],
    },
}

problem = Problem(
    mesh,
    soln_space,
    data_space,
    geo_space,
    integrand_map=integrand_map,
    bc_map=bc_map,
    element_objs={("shell", etype): shell_elem},
)

model = problem.create_model("cylinder_shell")

# Add a fixed boundary condition for the zeroth node.
model.add_fixed("soln.w[0]")

if args.build:
    model.build_module()
model.initialize()

R = 1.0
data = model.get_data_vector()
data["geo.nx"] = data["geo.x"] / R
data["geo.ny"] = data["geo.y"] / R

# Create the vectors and matrices for the model
x = model.create_vector()
g = model.create_vector()
mat = model.create_matrix()

# Copy the data over to the GPU
data = model.get_data_vector()
data.copy_host_to_device()

print("Evaluating the Hessian...")
model.eval_gradient(x, g)
model.eval_hessian(x, mat)

num_factors = 1
if args.solver == "cuda":
    from amigo.amigo import CSRMatFactorCuda

    # Duplicate the matrix
    mat_copy = mat.duplicate()
    mat_copy.copy(mat)

    pivot_eps = 1e-12
    solver = CSRMatFactorCuda(mat_copy, pivot_eps)
    solver.factor()

    start_time = time.perf_counter()
    for i in range(num_factors):
        mat_copy.copy(mat)
        solver.factor()
    end_time = time.perf_counter()
    tfactor = (end_time - start_time) / num_factors

    solver.solve(g.get_vector(), x.get_vector())

    x.copy_device_to_host()
else:
    g.copy_device_to_host()
    mat.copy_data_device_to_host()

    if args.solver == "cholesky" or args.solver == "ldl":
        stype = am.SolverType.CHOLESKY
        if args.solver == "ldl":
            stype = am.SolverType.LDL

        ldl = am.SparseLDL(mat, stype, ustab=0.05)
        flag = ldl.factor()

        start_time = time.perf_counter()
        for i in range(num_factors):
            flag = ldl.factor()
        end_time = time.perf_counter()
        if flag != 0:
            print(f"LDL factor flag {flag}")

        x[:] = g[:]
        ldl.solve(x.get_vector())
        if stype == am.SolverType.LDL:
            print("Inertia: ", ldl.get_inertia())

        tfactor = (end_time - start_time) / num_factors
    elif args.solver == "cholesky_left":
        chol = am.SparseCholesky(mat)
        start_time = time.perf_counter()
        for i in range(num_factors):
            flag = chol.factor()
        end_time = time.perf_counter()
        if flag != 0:
            print(f"Cholesky factor flag {flag}")

        x[:] = g[:]
        chol.solve(x.get_vector())

        tfactor = (end_time - start_time) / num_factors
    elif args.solver == "scipy":
        csr = am.tocsr(mat)

        # This isn't a completely fair comparison
        start_time = time.perf_counter()
        x[:] = spsolve(csr, g[:])
        end_time = time.perf_counter()

        tfactor = end_time - start_time

print(f"Factor time... {tfactor:.6f} seconds")

u = x["soln.u"]
v = x["soln.v"]
w = x["soln.w"]

u_ex, v_ex, w_ex = get_exact_solution(data["geo.x"], data["geo.y"], data["geo.z"])

w_diff = w - w_ex
print("Error = ", np.max(np.absolute(w_diff)) / np.max(w))

conn = np.vstack([mesh.get_conn(s, "CPS4") for s in lateral_surfaces])
write_vtu(mesh, conn, u, v, w, filename="cylinder_shell.vtu")
write_vtu(mesh, conn, u_ex, v_ex, w_ex, filename="cylinder_shell_exact.vtu")
