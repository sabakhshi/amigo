"""
Track loading and preprocessing for the race car lap time problem.

Tracks are CSV files with one row per centerline point, x, y, width to
the right, width to the left, in the tracks/ directory. The centerline is
smoothed by an approximating periodic B-spline and sampled at a 3.0 m
reference stepsize, the track widths are corrected for the lateral shift
of the smoothed line, and the curvature is computed numerically from
heading differences over preview and review distances. The reference grid
is then linearly resampled to the requested number of nodes, so the track
is identical at every mesh size.
"""

import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from scipy import interpolate
from scipy.spatial import cKDTree

TRACKS_DIR = Path(__file__).resolve().parent / "tracks"

# Preprocessing parameters
K_REG = 3
S_REG = 10
STEPSIZE_PREP = 1.0
STEPSIZE_REG = 3.0
D_PREVIEW_CURV = 2.0
D_REVIEW_CURV = 2.0
D_PREVIEW_HEAD = 1.0
D_REVIEW_HEAD = 1.0


@dataclass
class Track:
    name: str
    s_total: float  # total track length [m]
    s: np.ndarray  # arc length at each node [m]
    kappa: np.ndarray  # curvature at each node [1/m]
    x: np.ndarray  # centerline x [m]
    y: np.ndarray  # centerline y [m]
    w_right: np.ndarray  # half-width to the right [m]
    w_left: np.ndarray  # half-width to the left [m]
    normvec: np.ndarray  # normalized normal vectors at each node [x, y]


def _closed_length(path):
    path_cl = np.vstack((path, path[:1]))
    return float(np.sum(np.hypot(np.diff(path_cl[:, 0]), np.diff(path_cl[:, 1]))))


def _interp_closed(track, stepsize):
    """Linear interpolation of the closed track to roughly uniform spacing."""
    track_cl = np.vstack((track, track[:1]))
    el = np.hypot(np.diff(track_cl[:, 0]), np.diff(track_cl[:, 1]))
    s_cl = np.insert(np.cumsum(el), 0, 0.0)
    n_interp = math.ceil(s_cl[-1] / stepsize)
    s_new = np.linspace(0.0, s_cl[-1], n_interp + 1)[:-1]
    return np.column_stack(
        [np.interp(s_new, s_cl, track_cl[:, j]) for j in range(track.shape[1])]
    )


def _normalize_psi(psi):
    """Wrap angles to (-pi, pi]."""
    return -((-psi + np.pi) % (2.0 * np.pi) - np.pi)


def _head_curv(path, el_lengths):
    """Numerical curvature of a closed path from heading differences.

    Heading is taken over D_PREVIEW_HEAD/D_REVIEW_HEAD windows and the
    curvature over D_PREVIEW_CURV/D_REVIEW_CURV windows, expressed in
    index steps of the average element length.
    """
    n = path.shape[0]
    avg_el = float(np.average(el_lengths))
    prev_psi = max(round(D_PREVIEW_HEAD / avg_el), 1)
    rev_psi = max(round(D_REVIEW_HEAD / avg_el), 1)
    prev_curv = max(round(D_PREVIEW_CURV / avg_el), 1)
    rev_curv = max(round(D_REVIEW_CURV / avg_el), 1)

    # heading from chords over the psi window
    path_tmp = np.vstack((path[-rev_psi:], path, path[:prev_psi]))
    steps_psi = prev_psi + rev_psi
    tang = path_tmp[steps_psi:] - path_tmp[:-steps_psi]
    psi = _normalize_psi(np.arctan2(tang[:, 1], tang[:, 0]) - 0.5 * np.pi)

    # curvature from heading differences over the curv window
    steps_curv = prev_curv + rev_curv
    psi_tmp = np.concatenate((psi[-rev_curv:], psi, psi[:prev_curv]))
    delta_psi = _normalize_psi(psi_tmp[steps_curv:] - psi_tmp[:-steps_curv])

    s_cl = np.insert(np.cumsum(el_lengths), 0, 0.0)
    s_rev = np.flipud(-np.cumsum(np.flipud(el_lengths)))
    s_tmp = np.concatenate((s_rev[-rev_curv:], s_cl[:-1], s_cl[-1] + s_cl[:prev_curv]))
    return delta_psi / (s_tmp[steps_curv:] - s_tmp[:-steps_curv])


def _smooth_track(reftrack):
    """Approximating periodic B-spline smoothing of the centerline.

    Returns the smoothed track sampled at the reference stepsize
    [x, y, w_right, w_left], the right pointing unit normal vectors and
    the element lengths between the sample points. The original track
    widths are corrected by the signed lateral shift of the smoothed line
    and reattached by interpolation.
    """
    track_prep = _interp_closed(reftrack, STEPSIZE_PREP)
    prep_cl = np.vstack((track_prep, track_prep[:1]))

    tck = interpolate.splprep([prep_cl[:, 0], prep_cl[:, 1]], k=K_REG, s=S_REG, per=1)[
        0
    ]

    # dense sampling for the spline arc length
    raw_len = _closed_length(reftrack[:, :2])
    n_dense = math.ceil(raw_len) * 4
    t_dense = np.linspace(0.0, 1.0, n_dense)
    xy_dense = np.array(interpolate.splev(t_dense, tck)).T
    s_dense = np.insert(
        np.cumsum(np.hypot(np.diff(xy_dense[:, 0]), np.diff(xy_dense[:, 1]))),
        0,
        0.0,
    )

    # sample the smoothed line at the reference stepsize
    n_reg = math.ceil(s_dense[-1] / STEPSIZE_REG)
    t_reg = np.linspace(0.0, 1.0, n_reg + 1)
    xy_cl = np.array(interpolate.splev(t_reg, tck)).T
    xy = xy_cl[:-1]
    dxy = np.array(interpolate.splev(t_reg[:-1], tck, der=1)).T
    tangent = dxy / np.linalg.norm(dxy, axis=1, keepdims=True)
    normvec = np.column_stack((tangent[:, 1], -tangent[:, 0]))  # points right
    el_lengths = np.diff(np.interp(t_reg, t_dense, s_dense))

    # correct the track widths for the lateral shift of the smoothed line:
    # project every smoothed sample onto the original centerline, take the
    # original widths there, and shift them by the signed lateral offset
    orig_cl = np.vstack((reftrack[:, :2], reftrack[:1, :2]))
    seg = np.diff(orig_cl, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    s_orig = np.insert(np.cumsum(seg_len), 0, 0.0)
    w_right_cl = np.append(reftrack[:, 2], reftrack[0, 2])
    w_left_cl = np.append(reftrack[:, 3], reftrack[0, 3])

    n_orig = reftrack.shape[0]
    _, near = cKDTree(reftrack[:, :2]).query(xy)

    w_right_reg = np.empty(n_reg)
    w_left_reg = np.empty(n_reg)
    for i in range(n_reg):
        best = None
        for j in ((near[i] - 1) % n_orig, near[i]):
            u = np.dot(xy[i] - orig_cl[j], seg[j]) / seg_len[j] ** 2
            u = min(max(u, 0.0), 1.0)
            c = orig_cl[j] + u * seg[j]
            d2 = np.sum((xy[i] - c) ** 2)
            if best is None or d2 < best[0]:
                best = (d2, j, u, c)
        d2, j, u, c = best
        rel = xy[i] - c
        # positive when the smoothed line lies left of the original line
        side = np.sign(seg[j, 0] * rel[1] - seg[j, 1] * rel[0])
        dist = math.sqrt(d2)
        s_proj = s_orig[j] + u * seg_len[j]
        w_r = np.interp(s_proj, s_orig, w_right_cl)
        w_l = np.interp(s_proj, s_orig, w_left_cl)
        w_right_reg[i] = w_r + side * dist
        w_left_reg[i] = w_l - side * dist

    track_reg = np.column_stack((xy, w_right_reg, w_left_reg))
    return track_reg, normvec, el_lengths


def _load_csv(csv_path, num_nodes):
    reftrack_imp = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    track_reg, normvec, el_lengths = _smooth_track(reftrack_imp)
    kappa = _head_curv(track_reg[:, :2], el_lengths)

    # resample the reference grid to the requested nodes (linear, matching
    # the interpolation the NLP applies between its nodes)
    s_total = float(np.sum(el_lengths))
    n_ref = track_reg.shape[0]
    s_ref = np.linspace(0.0, s_total, n_ref + 1)
    s_nodes = np.linspace(0.0, s_total, num_nodes)

    def _resample(a):
        return np.interp(s_nodes, s_ref, np.append(a, a[0]))

    normvec_nodes = np.column_stack(
        (_resample(normvec[:, 0]), _resample(normvec[:, 1]))
    )
    normvec_nodes /= np.linalg.norm(normvec_nodes, axis=1, keepdims=True)

    return Track(
        name=csv_path.stem,
        s_total=s_total,
        s=s_nodes,
        kappa=_resample(kappa),
        x=_resample(track_reg[:, 0]),
        y=_resample(track_reg[:, 1]),
        w_right=_resample(track_reg[:, 2]),
        w_left=_resample(track_reg[:, 3]),
        normvec=normvec_nodes,
    )


def intervals_for_stepsize(name, stepsize):
    """Interval count for a fixed stepsize: ceil(track length / stepsize)."""
    csv_path = TRACKS_DIR / f"{name}.csv"
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return int(np.ceil(_closed_length(data[:, :2]) / stepsize))


def load_track(name, num_nodes=301):
    csv_path = TRACKS_DIR / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Track '{name}' not found. Available tracks: {list_tracks()}"
        )
    return _load_csv(csv_path, num_nodes)


def list_tracks():
    return sorted(p.stem for p in TRACKS_DIR.glob("*.csv"))
