"""3D relativistic acceleration transformations and equations of motion.

Transforms proper acceleration measured in an accelerometer's co-moving rest frame
to coordinate 3-acceleration d^2r/dt^2 in the BCRS / lab frame, and vice versa.
"""

from __future__ import annotations

import math
import numpy as np

from relativistic_engine.constants import C_LIGHT


def proper_to_coordinate_acceleration(
    v_vec: np.ndarray | list[float],
    a_proper: np.ndarray | list[float],
) -> np.ndarray:
    """Transform 3D proper acceleration to coordinate acceleration dv/dt.

    Under Lorentz transformation from the instantaneous co-moving rest frame:
        a_coord,parallel = a_proper,parallel / gamma^3
        a_coord,perp     = a_proper,perp / gamma^2

    In vector notation:
        a_coord = (1 / gamma^2) * [ a_proper - (1 - 1/gamma) * (v . a_proper) * v / v^2 ]

    To avoid 0/0 division when v -> 0, the factor (1 - 1/gamma) / v^2 is stabilized as:
        (1 - 1/gamma) / v^2 = 1 / [ c^2 * (1 + sqrt(1 - beta^2)) ]

    Args:
        v_vec: Coordinate velocity vector [vx, vy, vz] in m/s.
        a_proper: Proper acceleration vector [ax, ay, az] in m/s^2 measured in rest frame.

    Returns:
        Coordinate acceleration vector dv/dt [ax, ay, az] in m/s^2.
    """
    v = np.asarray(v_vec, dtype=np.float64)
    a_p = np.asarray(a_proper, dtype=np.float64)

    v_sq = float(np.dot(v, v))
    c_sq = C_LIGHT * C_LIGHT
    beta_sq = v_sq / c_sq

    if beta_sq >= 1.0:
        # Guard against unphysical trial step overshoot in adaptive Runge-Kutta stages
        beta_sq = 1.0 - 1.0e-15

    gamma_inv = math.sqrt(1.0 - beta_sq)

    gamma_inv_sq = 1.0 - beta_sq

    # (1 - 1/gamma) / v^2 evaluated without division by v^2
    denom_stabilized = c_sq * (1.0 + gamma_inv)
    v_dot_a = float(np.dot(v, a_p))

    # a_coord = (1 / gamma^2) * a_proper - (1 / gamma^2) * (1 - 1/gamma) * (v . a) * v / v^2
    #         = gamma_inv_sq * a_proper - gamma_inv_sq * (v_dot_a / denom_stabilized) * v
    return gamma_inv_sq * (a_p - (v_dot_a / denom_stabilized) * v)


def coordinate_to_proper_acceleration(
    v_vec: np.ndarray | list[float],
    a_coord: np.ndarray | list[float],
) -> np.ndarray:
    """Transform 3D coordinate acceleration dv/dt to proper acceleration in rest frame.

    Inverse transformation:
        a_proper,parallel = gamma^3 * a_coord,parallel
        a_proper,perp     = gamma^2 * a_coord,perp

    In vector notation:
        a_proper = gamma^2 * [ a_coord + (gamma - 1) * (v . a_coord) * v / v^2 ]

    Stabilized factor:
        (gamma - 1) / v^2 = gamma / [ c^2 * (1 + sqrt(1 - beta^2)) ]

    Args:
        v_vec: Coordinate velocity vector [vx, vy, vz] in m/s.
        a_coord: Coordinate acceleration vector dv/dt [ax, ay, az] in m/s^2.

    Returns:
        Proper acceleration vector [ax, ay, az] in m/s^2 in instantaneous rest frame.
    """
    v = np.asarray(v_vec, dtype=np.float64)
    a_c = np.asarray(a_coord, dtype=np.float64)

    v_sq = float(np.dot(v, v))
    c_sq = C_LIGHT * C_LIGHT
    beta_sq = v_sq / c_sq

    if beta_sq >= 1.0:
        raise ValueError(f"Superluminal velocity in acceleration transform: beta^2 = {beta_sq:.8e} >= 1")

    gamma_inv = math.sqrt(1.0 - beta_sq)
    gamma = 1.0 / gamma_inv
    gamma_sq = gamma * gamma

    denom_stabilized = c_sq * (1.0 + gamma_inv)
    v_dot_a = float(np.dot(v, a_c))

    # a_proper = gamma^2 * [ a_coord + (gamma * v_dot_a / denom_stabilized) * v ]
    return gamma_sq * (a_c + (gamma * v_dot_a / denom_stabilized) * v)
