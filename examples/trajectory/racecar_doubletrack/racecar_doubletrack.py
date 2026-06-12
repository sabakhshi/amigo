"""
Minimum lap time of a double track race car, Christ et al. (2019).

Time optimal trajectory around a closed race track. The track is described
in curvilinear coordinates with the arc length s as the independent
variable, the lap closes periodically, and the problem is discretized by
trapezoidal direct collocation with piecewise constant controls.

Model, equation numbers from the paper:
  - Extended Pacejka tire model with load degression (Eq. 5)
  - Quasi-steady-state wheel load transfer (Eq. 7-8)
  - Kamm's friction circle per wheel (Eq. 10)
  - Engine power and force limits (Eq. 11)
  - Brake/drive complementarity, relaxed (Eq. 12)
  - Actuator rate constraints (Eq. 14)
  - Track boundaries (Eq. 17)
  - Control regularization on du/ds (Eq. 24)

States x = [v, beta, omega_z, n, xi], controls u = [delta, F_drive,
F_brake, Gamma_y]. The objective is the lap time, a trapezoidal quadrature
of SF = dt/ds, plus a small control regularization; both parts are
reported separately at the end of a run.

Vehicle and tire parameters follow the reference repository
(https://github.com/TUMFTM/global_racetrajectory_optimization,
params/racecar.ini). The track preprocessing replicates its pipeline,
spline approximation smoothing and numerical curvature with preview and
review distances, implemented in tracks.py.

Options:
  --build            build the generated C++ module; required on the
                     first run and after model changes
  --track NAME       track CSV from tracks/ (default berlin_2018)
  --intervals N      number of mesh intervals (default 300)
  --stepsize DS      mesh stepsize [m]; sets the interval count from the
                     track length and overrides --intervals
  --bsafe B          track boundary safety margin [m], added to the
                     vehicle half width in the lateral bounds
  --var-friction linear
                     position dependent tire road friction: linear mue(n)
                     functions per wheel and node, fitted from the track
                     friction map frictionmaps/<track>.csv; a map is
                     shipped for berlin_2018, other tracks need their own
                     csv with x;y;mue rows
"""

import amigo as am
import numpy as np
import argparse
import json
from pathlib import Path

from frictionmap import fit_linear_friction
from tracks import load_track, intervals_for_stepsize

# Vehicle parameters
G = 9.81  # Gravity [m/s^2]
M = 1200.0  # Vehicle mass [kg]
LF = 1.6  # CG to front axle [m]
LR = 1.4  # CG to rear axle [m]
L = LF + LR  # Wheelbase [m]
TWF = 1.6  # Track width front [m]
TWR = 1.6  # Track width rear [m]
B_VEH = 3.4  # Vehicle width including safety margin [m]
H_COG = 0.38  # CG height [m]
JZZ = 1200.0  # Yaw moment of inertia [kg*m^2]
CD = 0.75  # Lumped drag coefficient [dragcoeff * v^2 convention]
CL_F = 0.45  # Lumped downforce coeff, front [liftcoeff_front * v^2]
CL_R = 0.75  # Lumped downforce coeff, rear [liftcoeff_rear * v^2]
FR = 0.013  # Rolling resistance coefficient
K_DRIVE = 0.0  # Drive force distribution (fraction at front, 0 = RWD)
K_BRAKE = 0.6  # Brake force distribution (fraction at front)
K_ROLL = 0.5  # Roll moment distribution (fraction at front)
P_MAX = 230000.0  # Maximum engine power [W]
F_DRIVE_MAX = 7000.0  # Maximum driving force [N]
F_BRAKE_MIN = -20000.0  # Maximum braking force [N] (negative)
DELTA_MAX = 0.35  # Maximum steering angle [rad]
V_MAX = 70.0  # Maximum velocity [m/s]
T_DRIVE = 0.05  # Drive actuator time constant [s]
T_BRAKE = 0.05  # Brake actuator time constant [s]
T_DELTA = 0.2  # Steering actuator time constant [s]
B_SAFE = 0.0  # Track boundary safety margin [m]

# Baseline friction coefficient
MU0 = 1.0

# Extended Pacejka tire parameters
B_F = 10.0
C_F = 2.5
E_F = 1.0
FZ0_F = 3000.0
EPS_F = -0.1

B_R = 10.0
C_R = 2.5
E_R = 1.0
FZ0_R = 3000.0
EPS_R = -0.1

# Control regularization (Eq. 24)
R_DELTA = 10.0  # Steering rate penalty
R_F = 0.01  # Force rate penalty

# Scaling factors
S_v = 50.0  # velocity [m/s]
S_beta = 0.5  # sideslip angle [rad]
S_omega = 1.0  # yaw rate [rad/s]
S_n = 5.0  # lateral offset [m]
S_xi = 1.0  # heading angle [rad]
S_delta = 0.5  # steering angle [rad]
S_Fdrive = 7500.0  # driving force [N]
S_Fbrake = 20000.0  # braking force magnitude [N]
S_Gamma = 5000.0  # lateral load transfer [N]

NS = 5  # states: [v, beta, omega_z, n, xi]
NC = 4  # controls: [delta, F_drive, F_brake, Gamma_y]


# Helper: Extended Pacejka lateral tire force (Eq. 5)
def pacejka_Fy(alpha, Fz, mu, B_pac, C_pac, E_pac, Fz0, eps):
    Ba = B_pac * alpha
    inner = Ba - E_pac * (Ba - am.atan(Ba))
    return mu * Fz * (1.0 + eps * Fz / Fz0) * am.sin(C_pac * am.atan(inner))


# Helper: normal, lateral and longitudinal tire forces per wheel
def tire_forces(v, beta, wz, delta, F_drive, F_brake_phys, Gamma_y):
    """Returns (Fz_fl, ..), (Fy_fl, ..), (Fx_fl, ..) in fl, fr, rl, rr order."""
    # Normal forces (Eq. 7, 9)
    drag = CD * v * v
    f_xroll = FR * M * G
    m_ax = F_drive + F_brake_phys - drag - f_xroll

    df_f = CL_F * v * v
    df_r = CL_R * v * v

    Fz_fl = (
        0.5 * M * G * LR / L - 0.5 * H_COG / L * m_ax - K_ROLL * Gamma_y + 0.5 * df_f
    )
    Fz_fr = (
        0.5 * M * G * LR / L - 0.5 * H_COG / L * m_ax + K_ROLL * Gamma_y + 0.5 * df_f
    )
    Fz_rl = (
        0.5 * M * G * LF / L
        + 0.5 * H_COG / L * m_ax
        - (1.0 - K_ROLL) * Gamma_y
        + 0.5 * df_r
    )
    Fz_rr = (
        0.5 * M * G * LF / L
        + 0.5 * H_COG / L * m_ax
        + (1.0 - K_ROLL) * Gamma_y
        + 0.5 * df_r
    )

    # Tire slip angles (Eq. 6)
    alpha_fl = delta - am.atan(
        (v * am.sin(beta) + LF * wz) / (v * am.cos(beta) - 0.5 * TWF * wz)
    )
    alpha_fr = delta - am.atan(
        (v * am.sin(beta) + LF * wz) / (v * am.cos(beta) + 0.5 * TWF * wz)
    )
    alpha_rl = am.atan(
        (-v * am.sin(beta) + LR * wz) / (v * am.cos(beta) - 0.5 * TWR * wz)
    )
    alpha_rr = am.atan(
        (-v * am.sin(beta) + LR * wz) / (v * am.cos(beta) + 0.5 * TWR * wz)
    )

    # Lateral tire forces (Eq. 5)
    Fy_fl = pacejka_Fy(alpha_fl, Fz_fl, MU0, B_F, C_F, E_F, FZ0_F, EPS_F)
    Fy_fr = pacejka_Fy(alpha_fr, Fz_fr, MU0, B_F, C_F, E_F, FZ0_F, EPS_F)
    Fy_rl = pacejka_Fy(alpha_rl, Fz_rl, MU0, B_R, C_R, E_R, FZ0_R, EPS_R)
    Fy_rr = pacejka_Fy(alpha_rr, Fz_rr, MU0, B_R, C_R, E_R, FZ0_R, EPS_R)

    # Longitudinal tire forces (Eq. 4)
    roll_f = 0.5 * FR * M * G * LR / L
    roll_r = 0.5 * FR * M * G * LF / L
    Fx_fl = 0.5 * K_DRIVE * F_drive + 0.5 * K_BRAKE * F_brake_phys - roll_f
    Fx_fr = Fx_fl
    Fx_rl = (
        0.5 * (1.0 - K_DRIVE) * F_drive + 0.5 * (1.0 - K_BRAKE) * F_brake_phys - roll_r
    )
    Fx_rr = Fx_rl

    return (
        (Fz_fl, Fz_fr, Fz_rl, Fz_rr),
        (Fy_fl, Fy_fr, Fy_rl, Fy_rr),
        (Fx_fl, Fx_fr, Fx_rl, Fx_rr),
    )


# Helper: vehicle dynamics d/ds (5 states)
def vehicle_dynamics(x, u, kappa):
    """Scaled d/ds rates of [v, beta, omega_z, n, xi] and the factor SF."""
    # Unscale
    v = S_v * x[0]
    beta = S_beta * x[1]
    wz = S_omega * x[2]
    n = S_n * x[3]
    xi = S_xi * x[4]
    delta = S_delta * u[0]
    F_drive = S_Fdrive * u[1]
    F_brake_phys = -S_Fbrake * u[2]
    Gamma_y = S_Gamma * u[3]

    _, (Fy_fl, Fy_fr, Fy_rl, Fy_rr), (Fx_fl, Fx_fr, Fx_rl, Fx_rr) = tire_forces(
        v, beta, wz, delta, F_drive, F_brake_phys, Gamma_y
    )
    drag = CD * v * v

    # Vehicle dynamics (Eq. 3a-3c)
    Fx_r = Fx_rl + Fx_rr
    Fy_r = Fy_rl + Fy_rr
    Fx_f = Fx_fl + Fx_fr
    Fy_f = Fy_fl + Fy_fr

    cos_b = am.cos(beta)
    sin_b = am.sin(beta)
    cos_db = am.cos(delta - beta)
    sin_db = am.sin(delta - beta)

    # Eq. 3a: longitudinal acceleration
    vdot = (1.0 / M) * (
        Fx_r * cos_b + Fx_f * cos_db + Fy_r * sin_b - Fy_f * sin_db - drag * cos_b
    )

    # Eq. 3b: lateral (side slip rate)
    betadot = -wz + (1.0 / (M * v)) * (
        -Fx_r * sin_b + Fx_f * sin_db + Fy_r * cos_b + Fy_f * cos_db + drag * sin_b
    )

    # Eq. 3c: yaw acceleration
    wzdot = (1.0 / JZZ) * (
        (Fx_rr - Fx_rl) * TWR / 2.0
        - Fy_r * LR
        + ((Fx_fr - Fx_fl) * am.cos(delta) + (Fy_fl - Fy_fr) * am.sin(delta))
        * TWF
        / 2.0
        + (Fy_f * am.cos(delta) + Fx_f * am.sin(delta)) * TWF  # TWF intentional
    )

    # Curvilinear kinematics (Eq. 1, 2)
    cos_xi_beta = am.cos(xi + beta)
    sin_xi_beta = am.sin(xi + beta)
    SF = (1.0 - n * kappa) / (v * cos_xi_beta)

    # d/ds = (d/dt) * SF, then divide by scaling factor
    dv_ds = vdot * SF / S_v
    dbeta_ds = betadot * SF / S_beta
    dwz_ds = wzdot * SF / S_omega
    dn_ds = v * sin_xi_beta * SF / S_n
    dxi_ds = (wz * SF - kappa) / S_xi

    return (dv_ds, dbeta_ds, dwz_ds, dn_ds, dxi_ds), SF


# Component: Trapezoidal dynamics + objective (combined)
class TrapDynamics(am.Component):
    """Trapezoidal collocation for one interval.

    x_{k+1} = x_k + ds/2 * (f(x_k, u_k) + f(x_{k+1}, u_k))

    Also computes trapezoidal quadrature of SF for the objective.
    """

    def __init__(self):
        super().__init__()
        self.add_data("ds", value=1.0)
        self.add_data("kappa0", value=0.0)
        self.add_data("kappa_end", value=0.0)

        self.add_input("x0", shape=NS)
        self.add_input("x_next", shape=NS)
        self.add_input("u", shape=NC)

        self.add_constraint("defect", shape=NS)
        self.add_objective("time")

    def compute(self):
        x0 = self.inputs["x0"]
        x_next = self.inputs["x_next"]
        u = self.inputs["u"]
        h = self.data["ds"]
        kappa0 = self.data["kappa0"]
        kappa_end = self.data["kappa_end"]

        f0, SF0 = vehicle_dynamics(x0, u, kappa0)
        f1, SF1 = vehicle_dynamics(x_next, u, kappa_end)

        defect = []
        for i in range(NS):
            defect.append(x_next[i] - x0[i] - 0.5 * h * (f0[i] + f1[i]))
        self.constraints["defect"] = defect

        self.objective["time"] = 0.5 * h * (SF0 + SF1)


# Component: Node constraints (friction circles, power, Gamma_y, compl.)
class NodeConstraints(am.Component):
    """Per-node inequality and equality constraints at mesh endpoints.

    Inequalities (native):
      [0-3] Kamm's friction circle per wheel (Eq. 10)
      [4]   Power limit (Eq. 11a)
      [5]   Brake/drive complementarity (Eq. 12)
    Equality:
      [0]   Gamma_y balance (Eq. 8)
    """

    def __init__(self):
        super().__init__()
        self.add_input("x", shape=NS)
        self.add_input("u", shape=NC)

        # Friction per wheel: mue = w[0] * n + w[1] (constant when w[0] = 0)
        self.add_data("w_mue_fl", shape=2, value=0.0)
        self.add_data("w_mue_fr", shape=2, value=0.0)
        self.add_data("w_mue_rl", shape=2, value=0.0)
        self.add_data("w_mue_rr", shape=2, value=0.0)

        # Kamm's circles (4) and power limit (1), upper bounded by 0
        self.add_constraint("ineq", shape=5, lower=-float("inf"), upper=0.0)
        # Relaxed complementarity: 0 <= u_drive * u_brake <= eps
        self.add_constraint(
            "compl",
            shape=1,
            lower=0.0,
            upper=-F_BRAKE_MIN / (S_Fdrive * S_Fbrake),
        )
        self.add_constraint("eq", shape=1)

    def compute(self):
        x, u = self.inputs["x"], self.inputs["u"]

        v = S_v * x[0]
        beta = S_beta * x[1]
        wz = S_omega * x[2]
        n = S_n * x[3]
        delta = S_delta * u[0]
        F_drive = S_Fdrive * u[1]
        F_brake_phys = -S_Fbrake * u[2]
        Gamma_y = S_Gamma * u[3]

        (
            (Fz_fl, Fz_fr, Fz_rl, Fz_rr),
            (Fy_fl, Fy_fr, Fy_rl, Fy_rr),
            (Fx_fl, Fx_fr, Fx_rl, Fx_rr),
        ) = tire_forces(v, beta, wz, delta, F_drive, F_brake_phys, Gamma_y)

        # Local friction coefficients (linear in n; Pacejka keeps constant MU0)
        w_fl, w_fr = self.data["w_mue_fl"], self.data["w_mue_fr"]
        w_rl, w_rr = self.data["w_mue_rl"], self.data["w_mue_rr"]
        mue_fl = w_fl[0] * n + w_fl[1]
        mue_fr = w_fr[0] * n + w_fr[1]
        mue_rl = w_rl[0] * n + w_rl[1]
        mue_rr = w_rr[0] * n + w_rr[1]

        # Kamm's friction circle (Eq. 10)
        c_fl = (Fx_fl**2 + Fy_fl**2) / (mue_fl * Fz_fl) ** 2 - 1.0
        c_fr = (Fx_fr**2 + Fy_fr**2) / (mue_fr * Fz_fr) ** 2 - 1.0
        c_rl = (Fx_rl**2 + Fy_rl**2) / (mue_rl * Fz_rl) ** 2 - 1.0
        c_rr = (Fx_rr**2 + Fy_rr**2) / (mue_rr * Fz_rr) ** 2 - 1.0

        # Power constraint (Eq. 11a)
        power_norm = v * F_drive / P_MAX - 1.0

        # Brake/drive complementarity (Eq. 12)
        compl = u[1] * u[2]

        # Gamma_y equality (Eq. 8)
        Fx_f = Fx_fl + Fx_fr
        Fy_f = Fy_fl + Fy_fr
        Gamma_y_computed = (H_COG / (0.5 * (TWF + TWR))) * (
            Fy_rl + Fy_rr + Fx_f * am.sin(delta) + Fy_f * am.cos(delta)
        )

        self.constraints["ineq"] = [c_fl, c_fr, c_rl, c_rr, power_norm]
        self.constraints["compl"] = [compl]
        self.constraints["eq"] = [(Gamma_y - Gamma_y_computed) / S_Gamma]


# Component: Actuator rate constraints (Eq. 14)
class ActuatorRates(am.Component):
    """Per-interval rate constraints on controls."""

    def __init__(self):
        super().__init__()
        self.add_data("ds", value=1.0)
        self.add_data("kappa", value=0.0)

        self.add_input("x", shape=NS)
        self.add_input("u_curr", shape=NC)
        self.add_input("u_prev", shape=NC)

        self.add_constraint("ineq", shape=4, lower=-float("inf"), upper=0.0)

    def compute(self):
        x = self.inputs["x"]
        u_curr = self.inputs["u_curr"]
        u_prev = self.inputs["u_prev"]
        ds_val = self.data["ds"]
        kappa = self.data["kappa"]

        n = S_n * x[3]
        v = S_v * x[0]
        sigma = (1.0 - n * kappa) / v
        dt_interval = sigma * ds_val

        d_Fdrive = S_Fdrive * (u_curr[1] - u_prev[1])
        d_delta = S_delta * (u_curr[0] - u_prev[0])
        d_Fbrake_phys = -S_Fbrake * (u_curr[2] - u_prev[2])

        rate_drive_max = F_DRIVE_MAX / T_DRIVE
        rate_brake_min = F_BRAKE_MIN / T_BRAKE
        rate_delta_max = DELTA_MAX / T_DELTA

        r_drv = d_Fdrive / (dt_interval * rate_drive_max)
        r_brk = -d_Fbrake_phys / (dt_interval * (-rate_brake_min))
        r_stup = d_delta / (dt_interval * rate_delta_max)

        self.constraints["ineq"] = [
            r_drv - 1.0,
            r_brk - 1.0,
            r_stup - 1.0,
            -r_stup - 1.0,
        ]


# Component: Control regularization (Eq. 24)
class ControlRegularization(am.Component):
    """Per-interval regularization on consecutive control differences."""

    def __init__(self):
        super().__init__()
        self.add_input("u_curr", shape=NC)
        self.add_input("u_next", shape=NC)
        self.add_objective("reg")

    def compute(self):
        u_curr = self.inputs["u_curr"]
        u_next = self.inputs["u_next"]

        d_delta = S_delta * (u_next[0] - u_curr[0])
        F_curr = S_Fdrive * u_curr[1] - S_Fbrake * u_curr[2]
        F_next = S_Fdrive * u_next[1] - S_Fbrake * u_next[2]
        d_F = (F_next - F_curr) / 10000.0

        self.objective["reg"] = R_DELTA * d_delta * d_delta + R_F * d_F * d_F


# Model assembly
def create_model(num_intervals, module_name="doubletrack_mod"):
    NI = num_intervals

    model = am.Model(module_name)
    model.add_component("dyn", NI, TrapDynamics())
    model.add_component("nc", NI, NodeConstraints())
    model.add_component("rates", NI - 1, ActuatorRates())
    model.add_component("reg", NI, ControlRegularization())

    # Continuity: x_next[k] = x0[k+1] for k=0..NI-2
    for i in range(NS):
        model.link(f"dyn.x_next[:{NI - 1}, {i}]", f"dyn.x0[1:, {i}]")

    # Cyclic BCs: x_next[NI-1] = x0[0]
    for i in range(NS):
        model.link(f"dyn.x_next[{NI - 1}, {i}]", f"dyn.x0[0, {i}]")

    # Node constraints: u[k] paired with the interval end state x_next[k]
    for i in range(NS):
        model.link(f"dyn.x_next[:, {i}]", f"nc.x[:, {i}]")
    for i in range(NC):
        model.link(f"dyn.u[:, {i}]", f"nc.u[:, {i}]")

    # Rate constraints: rates[j] couples u[j+1], u[j], and x_next[j+1], no wrap
    for i in range(NS):
        model.link(f"dyn.x_next[1:, {i}]", f"rates.x[:, {i}]")
    for i in range(NC):
        model.link(f"dyn.u[1:, {i}]", f"rates.u_curr[:, {i}]")
        model.link(f"dyn.u[:{NI - 1}, {i}]", f"rates.u_prev[:, {i}]")

    # Control regularization: reg[k] compares u[k] with u[k+1], cyclic
    for i in range(NC):
        model.link(f"dyn.u[:, {i}]", f"reg.u_curr[:, {i}]")
        model.link(f"dyn.u[1:, {i}]", f"reg.u_next[:{NI - 1}, {i}]")
        model.link(f"dyn.u[0, {i}]", f"reg.u_next[{NI - 1}, {i}]")

    return model


def fit_friction_map(trk, ni):
    """Per node linear fits of mue(n) per wheel from the track friction map."""
    map_path = Path(__file__).resolve().parent / "frictionmaps" / f"{trk.name}.csv"
    if not map_path.exists():
        raise FileNotFoundError(
            f"Variable friction needs frictionmaps/{map_path.name}; "
            f"a map is shipped for berlin_2018 only"
        )
    return fit_linear_friction(
        trk,
        ni,
        str(map_path),
        veh_width=B_VEH + 2.0 * B_SAFE,
        wb_front=LF,
        wb_rear=LR,
    )


# Main
parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", help="Build C++ module")
parser.add_argument("--track", default="berlin_2018", help="Track CSV name in tracks/")
parser.add_argument("--intervals", type=int, default=300, help="Mesh intervals")
parser.add_argument(
    "--stepsize",
    type=float,
    default=None,
    help="Mesh stepsize [m]; sets intervals = ceil(length/stepsize), overrides --intervals",
)
parser.add_argument(
    "--bsafe",
    type=float,
    default=None,
    help="Track boundary safety margin [m]",
)
parser.add_argument(
    "--var-friction",
    choices=["linear"],
    default=None,
    help="Position dependent friction fitted from the track friction map",
)
args = parser.parse_args()

if args.bsafe is not None:
    B_SAFE = args.bsafe

if args.stepsize is not None:
    NUM_INTERVALS = intervals_for_stepsize(args.track, args.stepsize)
else:
    NUM_INTERVALS = args.intervals

track = load_track(args.track, NUM_INTERVALS + 1)
S_FINAL = track.s_total
DS = S_FINAL / NUM_INTERVALS
kappa_nodes = track.kappa

print(f"Track: {track.name}")
print(f"Track length: {S_FINAL:.2f} m")
print(f"Intervals: {NUM_INTERVALS}, ds: {DS:.2f} m")
print(f"Curvature range: [{kappa_nodes.min():.4f}, {kappa_nodes.max():.4f}] 1/m")

print(f"Corridor margin: {B_VEH / 2 + B_SAFE:.2f} m per side")

w_mue = None
if args.var_friction == "linear":
    w_mue = fit_friction_map(track, NUM_INTERVALS)
    print(f"Friction: linear fit from frictionmaps/{track.name}.csv")
else:
    print(f"Friction: constant mue = {MU0}")

model = create_model(NUM_INTERVALS)

if args.build:
    model.build_module(source_dir=Path(__file__).resolve().parent)

# Initial guess, staged before initialize: v = 20 m/s, all other states
# and controls zero
model.set_meta("value", "dyn.x0[:, 0]", 20.0 / S_v)
model.set_meta("value", "dyn.x_next[:, 0]", 20.0 / S_v)

# State bounds
model.set_meta("lower", "dyn.x0[:, 0]", 1.0 / S_v)
model.set_meta("upper", "dyn.x0[:, 0]", V_MAX / S_v)
model.set_meta("lower", "dyn.x0[:, 1]", -0.5 * np.pi / S_beta)
model.set_meta("upper", "dyn.x0[:, 1]", 0.5 * np.pi / S_beta)
model.set_meta("lower", "dyn.x0[:, 2]", -0.5 * np.pi / S_omega)
model.set_meta("upper", "dyn.x0[:, 2]", 0.5 * np.pi / S_omega)
model.set_meta("lower", "dyn.x0[:, 4]", -0.5 * np.pi / S_xi)
model.set_meta("upper", "dyn.x0[:, 4]", 0.5 * np.pi / S_xi)

# Per-node track width bounds on n
n_lo = -(track.w_right[:NUM_INTERVALS] - B_VEH / 2 - B_SAFE) / S_n
n_hi = (track.w_left[:NUM_INTERVALS] - B_VEH / 2 - B_SAFE) / S_n
model.set_meta("lower", "dyn.x0[:, 3]", n_lo)
model.set_meta("upper", "dyn.x0[:, 3]", n_hi)

# Control bounds; Gamma_y stays unbounded
model.set_meta("lower", "dyn.u[:, 0]", -DELTA_MAX / S_delta)
model.set_meta("upper", "dyn.u[:, 0]", DELTA_MAX / S_delta)
model.set_meta("lower", "dyn.u[:, 1]", 0.0)
model.set_meta("upper", "dyn.u[:, 1]", F_DRIVE_MAX / S_Fdrive)
model.set_meta("lower", "dyn.u[:, 2]", 0.0)
model.set_meta("upper", "dyn.u[:, 2]", -F_BRAKE_MIN / S_Fbrake)

model.initialize(order_type=am.OrderingType.AMD)

print(f"Num variables:   {model.num_variables}")
print(f"Num constraints: {model.num_constraints}")

# Set curvature data
data = model.get_data_vector()
for i in range(NUM_INTERVALS):
    data[f"dyn.kappa0[{i}]"] = kappa_nodes[i]
    data[f"dyn.kappa_end[{i}]"] = kappa_nodes[i + 1]
    data[f"dyn.ds[{i}]"] = DS
for i in range(NUM_INTERVALS - 1):
    data[f"rates.kappa[{i}]"] = kappa_nodes[i + 1]
    data[f"rates.ds[{i}]"] = DS

# Friction data; nc[i] sits at node i+1
mue_names = ("w_mue_fl", "w_mue_fr", "w_mue_rl", "w_mue_rr")
if w_mue is None:
    for i in range(NUM_INTERVALS):
        for name in mue_names:
            data[f"nc.{name}[{i}, 1]"] = MU0
else:
    for i in range(NUM_INTERVALS):
        for name, w in zip(mue_names, w_mue):
            data[f"nc.{name}[{i}, 0]"] = w[i + 1, 0]
            data[f"nc.{name}[{i}, 1]"] = w[i + 1, 1]

x = model.create_vector()
opt = am.Optimizer(model, x=x)

print("\nOptimizing...")
opt_data = opt.optimize(
    {
        "max_iterations": 1000,
        "convergence_tolerance": 1e-7,
        "init_least_squares_multipliers": False,
        "filter_line_search": True,
        "second_order_correction": True,
        "barrier_strategy": "quality_function",
        "verbose_barrier": True,
    }
)
x.copy_device_to_host()

# Extract solution (physical units)
states = np.array(
    [[x[f"dyn.x0[{k}, {i}]"] for i in range(NS)] for k in range(NUM_INTERVALS)]
)
controls = np.array(
    [[x[f"dyn.u[{k}, {i}]"] for i in range(NC)] for k in range(NUM_INTERVALS)]
)

v_sol = states[:, 0] * S_v
beta_sol = states[:, 1] * S_beta
wz_sol = states[:, 2] * S_omega
n_sol = states[:, 3] * S_n
xi_sol = states[:, 4] * S_xi
delta_sol = controls[:, 0] * S_delta
Fdrive_sol = controls[:, 1] * S_Fdrive
Fbrake_sol = controls[:, 2] * S_Fbrake
gamma_sol = controls[:, 3] * S_Gamma

# Lap time: trapezoidal quadrature of SF with cyclic closure
# (equals the time part of the objective; the rest is regularization)
SF_nodes = (1.0 - n_sol * kappa_nodes[:NUM_INTERVALS]) / (
    v_sol * np.cos(xi_sol + beta_sol)
)
SF_cl = np.append(SF_nodes, SF_nodes[0])
lap_time = float(np.sum(0.5 * (SF_cl[:-1] + SF_cl[1:]) * DS))

objective = float(opt_data["iterations"][-1]["objective"])
reg_penalty = objective - lap_time

print(f"\nConverged: {opt_data['converged']}")
print(f"Iterations: {len(opt_data['iterations'])}")
print(f"\nLap time:               {lap_time:.3f} s")
print(f"Regularization penalty: {reg_penalty:.4f}")
print(f"Objective:              {objective:.4f} (lap time + regularization)")

print("\nSolution state:")
print(f"  v:     [{v_sol.min():.2f}, {v_sol.max():.2f}] m/s")
print(f"  beta:  [{beta_sol.min():.4f}, {beta_sol.max():.4f}] rad")
print(f"  wz:    [{wz_sol.min():.4f}, {wz_sol.max():.4f}] rad/s")
print(f"  n:     [{n_sol.min():.4f}, {n_sol.max():.4f}] m")
print(f"  xi:    [{xi_sol.min():.4f}, {xi_sol.max():.4f}] rad")
print(f"  delta: [{delta_sol.min():.4f}, {delta_sol.max():.4f}] rad")
print(f"  Fdrv:  [{Fdrive_sol.min():.1f}, {Fdrive_sol.max():.1f}] N")
print(f"  Fbrk:  [{Fbrake_sol.min():.1f}, {Fbrake_sol.max():.1f}] N")
print(f"  Gamma: [{gamma_sol.min():.1f}, {gamma_sol.max():.1f}] N")

# Save optimization data
with open("racecar_opt_data.json", "w") as f:
    json.dump(
        {
            "converged": opt_data["converged"],
            "iterations": [
                {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in it.items()
                }
                for it in opt_data["iterations"]
            ],
        },
        f,
        indent=2,
    )
print("\nSaved racecar_opt_data.json")

# Save trajectory (unscaled states, controls, track geometry) for downstream plotting
with open("racecar_trajectory.json", "w") as f:
    json.dump(
        {
            "track_name": track.name,
            "s_total": float(S_FINAL),
            "num_intervals": int(NUM_INTERVALS),
            "s": track.s[:NUM_INTERVALS].tolist(),
            "kappa": kappa_nodes[:NUM_INTERVALS].tolist(),
            "x_center": track.x[:NUM_INTERVALS].tolist(),
            "y_center": track.y[:NUM_INTERVALS].tolist(),
            "w_left": track.w_left[:NUM_INTERVALS].tolist(),
            "w_right": track.w_right[:NUM_INTERVALS].tolist(),
            "v": v_sol.tolist(),
            "beta": beta_sol.tolist(),
            "omega_z": wz_sol.tolist(),
            "n": n_sol.tolist(),
            "xi": xi_sol.tolist(),
            "delta": delta_sol.tolist(),
            "F_drive": Fdrive_sol.tolist(),
            "F_brake": Fbrake_sol.tolist(),
            "Gamma_y": gamma_sol.tolist(),
            "lap_time": lap_time,
            "state_names": ["v_m_s", "beta_rad", "omega_z_rad_s", "n_m", "xi_rad"],
            "control_names": ["delta_rad", "F_drive_N", "F_brake_N", "Gamma_y_N"],
            "vehicle_width": float(B_VEH),
        },
        f,
        indent=2,
    )
print("Saved racecar_trajectory.json")
