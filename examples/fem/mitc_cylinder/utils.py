import numpy as np


def compute_coefficients(E=70e9, nu=0.3, R=1.0, L=2.0, t=0.01, ks=5.0 / 6.0, M=4, N=3):
    """
    Compute the scalar coefficients for a cylindrical mesh

    Returns
    -------
    U, V, W, theta, phi : scalars
        Solution coefficients.
    """

    alpha = 4.0 / R
    beta = 3 * np.pi / L
    ainv = 1.0 / R

    G = 0.5 * E / (1.0 + nu)
    A11 = A22 = E * t / (1.0 - nu**2)
    A12 = nu * A11
    A33 = G * t
    D11 = D22 = E * t**3 / (12.0 * (1.0 - nu**2))
    D12 = nu * D11
    D33 = G * t**3 / 12.0
    bA11 = bA22 = ks * G * t

    rhs = np.zeros(5, dtype=float)
    rhs[2] = 1.0

    # Explicit matrix A, mainly for comparison/debugging
    A = np.zeros((5, 5), dtype=float)

    # The first equation for U
    A[0, 0] = (
        -(A11 * beta * beta + A33 * alpha * alpha) - D33 * ainv * ainv * alpha * alpha
    )
    A[0, 1] = -(A33 + A12) * alpha * beta
    A[0, 2] = A12 * beta * ainv
    A[0, 3] = D33 * ainv * alpha * alpha
    A[0, 4] = D33 * ainv * alpha * beta

    # The second equation for V
    A[1, 0] = -(A12 + A33) * alpha * beta
    A[1, 1] = (
        -(A33 * beta * beta + A22 * alpha * alpha)
        - ainv * ainv * bA11
        - D22 * ainv * ainv * alpha * alpha
    )
    A[1, 2] = (A22 + bA11) * ainv * alpha + D22 * alpha * ainv**3
    A[1, 3] = D12 * ainv * alpha * beta
    A[1, 4] = bA11 * ainv + D22 * ainv * alpha * alpha

    # The third equation for W
    A[2, 0] = A12 * beta * ainv
    A[2, 1] = (bA11 + A22) * alpha * ainv + D22 * alpha * ainv**3
    A[2, 2] = (
        -(bA11 * alpha * alpha + bA22 * beta * beta) - A22 * ainv * ainv - D22 * ainv**4
    )
    A[2, 3] = -bA22 * beta - D12 * beta * ainv * ainv
    A[2, 4] = -bA11 * alpha - D22 * alpha * ainv * ainv

    # Fourth equation for theta
    A[3, 0] = D33 * ainv * alpha * alpha
    A[3, 1] = D12 * ainv * alpha * beta
    A[3, 2] = -bA22 * beta - D12 * beta * ainv * ainv
    A[3, 3] = -(D11 * beta * beta + D33 * alpha * alpha) - bA22
    A[3, 4] = -(D12 + D33) * alpha * beta

    # Fifth equation for phi
    A[4, 0] = D33 * ainv * alpha * beta
    A[4, 1] = bA11 * ainv + D22 * ainv * alpha * alpha
    A[4, 2] = -bA11 * alpha - D22 * alpha * ainv * ainv
    A[4, 3] = -(D33 + D12) * alpha * beta
    A[4, 4] = -(D33 * beta * beta + D22 * alpha * alpha) - bA11

    sol = np.linalg.solve(A, rhs)

    U, V, W, theta, phi = sol
    return U, V, W, theta, phi


def get_exact_solution(x, y, z, R=1.0, L=2.0, M=4, N=3):
    theta = np.atan2(x, y)
    s = R * theta

    alpha = 4.0 / R
    beta = 3 * np.pi / L

    U, V, W, _, _ = compute_coefficients(
        E=70e9, nu=0.3, R=1.0, L=2.0, t=0.01, ks=5.0 / 6.0, M=M, N=N
    )

    # This is in the coordinate system aligned with x = axial direction
    # v = tangential, w = circumferential directions
    axial = U * np.sin(alpha * s) * np.cos(beta * z)
    tangent = V * np.cos(alpha * s) * np.sin(beta * z)
    circum = W * np.sin(alpha * s) * np.sin(beta * z)

    # Use these values to find the local coordinates
    u = circum * (x / R) + tangent * (y / R)
    v = circum * (y / R) - tangent * (x / R)
    w = axial

    return u, v, w


def write_vtu(mesh, conns, u, v, w, filename="cylinder_shell.vtu"):
    X = mesh.X
    nnodes = X.shape[0]
    conn = np.vstack(conns) if isinstance(conns, list) else conns
    nelems = conn.shape[0]

    with open(filename, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1">\n')
        f.write("<UnstructuredGrid>\n")
        f.write(f'<Piece NumberOfPoints="{nnodes}" NumberOfCells="{nelems}">\n')

        # Points
        f.write(
            '<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">\n'
        )
        for i in range(nnodes):
            f.write(f"{X[i,0]} {X[i,1]} {X[i,2]}\n")
        f.write("</DataArray></Points>\n")

        # Cells (VTK type 9 = quad)
        f.write("<Cells>\n")
        f.write('<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        for row in conn:
            f.write(" ".join(str(n) for n in row) + "\n")
        f.write("</DataArray>\n")
        f.write('<DataArray type="Int64" Name="offsets" format="ascii">\n')
        for i in range(1, nelems + 1):
            f.write(f"{i*4}\n")
        f.write("</DataArray>\n")
        f.write('<DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write(("9\n") * nelems)
        f.write("</DataArray>\n")
        f.write("</Cells>\n")

        # Point data
        f.write("<PointData>\n")
        for name, arr in [("u", u), ("v", v), ("w", w)]:
            f.write(f'<DataArray type="Float64" Name="{name}" format="ascii">\n')
            f.write("\n".join(str(val) for val in arr) + "\n")
            f.write("</DataArray>\n")
        # displacement vector for Warp by Vector filter
        f.write(
            '<DataArray type="Float64" Name="displacement" NumberOfComponents="3" format="ascii">\n'
        )
        for i in range(nnodes):
            f.write(f"{u[i]} {v[i]} {w[i]}\n")
        f.write("</DataArray>\n")
        f.write("</PointData>\n")

        f.write("</Piece></UnstructuredGrid></VTKFile>\n")
