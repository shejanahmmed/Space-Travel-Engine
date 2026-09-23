"""IAU BCRS metric proper time and time deficit differential formulations.

Under IAU 2000 Resolution B1.3 and IAU 2006 Resolution 3, the metric signature
is (-, +, +, +) with line element:
    ds^2 = -c^2 dtau^2 = -(1 - 2w/c^2) c^2 dt^2 + (1 + 2w/c^2) dr^2 + O(c^-4)
where t is coordinate time (TDB/TCB) and w is the gravitational potential.
"""

from __future__ import annotations

import math
import numpy as np

from relativistic_engine.constants import C_LIGHT


def proper_time_rate(v_vec: np.ndarray | list[float], potential: float) -> float:
    """Evaluate instantaneous proper time differential rate dtau/dt.

    Args:
        v_vec: Coordinate velocity vector [vx, vy, vz] in m/s.
        potential: Gravitational potential w in m^2/s^2 (positive by convention: w = sum(GM/r)).

    Returns:
        dtau/dt (dimensionless).

    Raises:
        ValueError: If spacetime interval is spacelike (v >= c or 2w/c^2 >= 1).
    """
    v_arr = np.asarray(v_vec, dtype=np.float64)
    v_sq = float(np.dot(v_arr, v_arr))
    c_sq = C_LIGHT * C_LIGHT

    beta_sq = v_sq / c_sq
    phi_term = 2.0 * potential / c_sq

    radicand = (1.0 - phi_term) - (1.0 + phi_term) * beta_sq
    if radicand <= 0.0:
        raise ValueError(f"Spacelike trajectory encounter: metric radicand = {radicand:.8e} <= 0")

    return math.sqrt(radicand)


def coordinate_time_deficit_rate(v_vec: np.ndarray | list[float], potential: float) -> float:
    """Evaluate rate of coordinate time deficit d(t - tau)/dt = 1 - dtau/dt.

    In weak gravitational fields (w/c^2 ~ 10^-8 in the Solar System) and non-relativistic
    speeds (v/c ~ 10^-4), dtau/dt is within 10^-8 of 1.0. Direct evaluation of
    1.0 - sqrt(1 - ...) incurs catastrophic floating-point cancellation in float64,
    destroying 8 to 10 decimal digits of precision.

    This function uses the exact algebraic rationalization:
        1 - sqrt(1 - X) = X / (1 + sqrt(1 - X))
    where X = 2w/c^2 * (1 + v^2/c^2) + v^2/c^2.

    Args:
        v_vec: Coordinate velocity vector [vx, vy, vz] in m/s.
        potential: Gravitational potential w in m^2/s^2 (w >= 0).

    Returns:
        d(t - tau)/dt (dimensionless, positive for physical subluminal worldlines).
    """
    v_arr = np.asarray(v_vec, dtype=np.float64)
    v_sq = float(np.dot(v_arr, v_arr))
    c_sq = C_LIGHT * C_LIGHT

    beta_sq = v_sq / c_sq
    phi_term = 2.0 * potential / c_sq

    # Metric radicand is (1 - X), where X = phi_term * (1 + beta_sq) + beta_sq
    x_val = phi_term * (1.0 + beta_sq) + beta_sq
    if x_val >= 1.0:
        # Clamped guard for numerical trial steps in adaptive ODE integrators
        x_val = 1.0 - 1.0e-15

    radicand = max(0.0, 1.0 - x_val)
    return x_val / (1.0 + math.sqrt(radicand))

