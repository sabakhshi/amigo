"""
Friction map and per node linear friction fits.

Friction values are sampled from the map on a fine grid along the track
normals at the four tire positions, and a linear function mue(n) is
fitted per wheel and per node.

A friction map is a csv file with one grid cell per row, x;y;mue.
"""

import math
import numpy as np
from scipy.spatial import cKDTree


class FrictionMap:
    """Nearest neighbor lookup of mue values on the friction map grid."""

    def __init__(self, path):
        data = np.loadtxt(path, comments="#", delimiter=";")
        self._tree = cKDTree(data[:, :2])
        self._mue = data[:, 2]

    def mue_at(self, positions):
        _, idxs = self._tree.query(positions)
        return self._mue[idxs]


def fit_linear_friction(track, ni, map_path, veh_width, wb_front, wb_rear, dn=0.25):
    """Per node linear fits mue = w[0] * n + w[1] for each wheel.

    Returns four (ni + 1, 2) arrays for fl, fr, rl, rr; row k corresponds
    to track node k. The vehicle position is sampled at lateral offsets n
    with spacing dn across the drivable corridor, the four tire positions
    are offset by the wheelbase along the tangent and half the vehicle
    width along the normal, and mue is read from the map at each point.
    """
    fmap = FrictionMap(map_path)

    refline = np.column_stack((track.x[: ni + 1], track.y[: ni + 1]))
    w_right = track.w_right[: ni + 1]
    w_left = track.w_left[: ni + 1]
    normvec = track.normvec[: ni + 1]
    tangvec = np.column_stack((-normvec[:, 1], normvec[:, 0]))

    w_fl = np.zeros((ni + 1, 2))
    w_fr = np.zeros((ni + 1, 2))
    w_rl = np.zeros((ni + 1, 2))
    w_rr = np.zeros((ni + 1, 2))

    for i in range(ni + 1):
        num_right = math.floor((w_right[i] - 0.5 * veh_width - 0.5) / dn)
        num_left = math.floor((w_left[i] - 0.5 * veh_width - 0.5) / dn)
        n_pos = np.linspace(-dn * num_right, dn * num_left, num_right + num_left + 1)

        front = refline[i] + wb_front * tangvec[i]
        rear = refline[i] - wb_rear * tangvec[i]
        # normvec points right; positive n is left
        left_off = -(n_pos + 0.5 * veh_width)[:, None] * normvec[i]
        right_off = -(n_pos - 0.5 * veh_width)[:, None] * normvec[i]

        w_fl[i] = np.polyfit(n_pos, fmap.mue_at(front + left_off), 1)
        w_fr[i] = np.polyfit(n_pos, fmap.mue_at(front + right_off), 1)
        w_rl[i] = np.polyfit(n_pos, fmap.mue_at(rear + left_off), 1)
        w_rr[i] = np.polyfit(n_pos, fmap.mue_at(rear + right_off), 1)

    return w_fl, w_fr, w_rl, w_rr
