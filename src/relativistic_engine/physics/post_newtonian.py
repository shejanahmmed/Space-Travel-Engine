"""Post-Newtonian dynamics: Lense-Thirring frame-dragging and 2.5PN gravitational radiation reaction.

Implements higher-order relativistic orbital dynamics:
1. 2.5PN quadrupole gravitational radiation reaction damping force a_2.5PN (Damour & Deruelle 1981).
2. Peters-Mathews (1963, 1964) analytical orbital decay rates <da/dt> and <de/dt>.
3. Lense-Thirring secular nodal precession rate dOmega/dt (Ciufolini & Wheeler 1995, LAGEOS).
4. De Sitter (geodetic) orbital curvature precession vector Omega_geodetic (Gravity Probe B).

Authoritative References:
- Damour, T., & Deruelle, N. (1981), "La méthode post-newtonienne et le problème des deux corps",
  C. R. Acad. Sci. Paris 293, 537.
- Peters, P. C., & Mathews, J. (1963), "Gravitational Radiation from Point Masses in a Keplerian Orbit",
  Phys. Rev. 131, 435.
- Peters, P. C. (1964), "Gravitational Radiation and the Motion of Two Point Masses", Phys. Rev. 136, B1224.
- Blanchet, L. (2014), "Gravitational Radiation from Post-Newtonian Sources and Inspirals of Compact Binaries",
  Living Rev. Relativity 17, 2, Eq. (227).
- Ciufolini, I., & Pavlis, E. C. (2004), "A confirmation of the general relativistic prediction of
  the Lense-Thirring effect", Nature 431, 958-960.
- Everitt, C. W. F., et al. (2011), "Gravity Probe B: Final Results of a Space Experiment to Test General
  Relativity", Phys. Rev. Lett. 106, 221101.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple
import numpy as np

from relativistic_engine.constants import (
    C_LIGHT,
    G_NEWTON,
)


def compute_2_5pn_radiation_reaction(
    r_vec: np.ndarray | Sequence[float],
    v_vec: np.ndarray | Sequence[float],
    m1: float,
    m2: float,
) -> np.ndarray:
    """Compute 2.5PN dissipative gravitational radiation reaction acceleration in m/s^2.

    Evaluates the Damour-Deruelle / Blanchet 2.5PN back-reaction force in harmonic coordinates:
        a_2.5PN = (8/5) * (G^2 M^2 eta / (c^5 r^3)) * [
            (dr_dt / r) * (3 v^2 + (17/3) GM/r) * r - (v^2 + 3 GM/r) * v
        ]
    where:
        M = m1 + m2 (total mass)
        eta = (m1 * m2) / M^2 (symmetric mass ratio)
        dr_dt = (r . v) / r (radial velocity)

    Args:
        r_vec: Relative position vector from m1 to m2 in meters [x, y, z].
        v_vec: Relative velocity vector [vx, vy, vz] in m/s.
        m1: Primary body mass in kg (e.g. central body).
        m2: Secondary body mass in kg (e.g. spacecraft or secondary).

    Returns:
        Cartesian acceleration vector [ax, ay, az] in m/s^2.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    v = np.asarray(v_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))

    if r_norm <= 0.0 or m1 <= 0.0 or m2 <= 0.0:
        return np.zeros(3, dtype=np.float64)

    total_m = m1 + m2
    eta = (m1 * m2) / (total_m * total_m)

    c5 = C_LIGHT ** 5
    gm = G_NEWTON * total_m
    v_sq = float(np.dot(v, v))
    r_dot_v = float(np.dot(r, v))
    dr_dt = r_dot_v / r_norm

    # Prefactor: (8/5) * G^2 * M^2 * eta / (c^5 * r^3) = (8/5) * (GM)^2 * eta / (c^5 * r^3)
    pref = (1.6 * (gm * gm) * eta) / (c5 * (r_norm ** 3))

    term_r_scalar = (dr_dt / r_norm) * (3.0 * v_sq + (17.0 / 3.0) * (gm / r_norm))
    term_v_scalar = v_sq + 3.0 * (gm / r_norm)

    return pref * (term_r_scalar * r - term_v_scalar * v)


def peters_orbital_decay_rates(
    semi_major_axis_m: float,
    eccentricity: float,
    m1: float,
    m2: float,
) -> Tuple[float, float]:
    """Compute orbit-averaged secular decay rates <da/dt> and <de/dt> via Peters (1964).

    Formulas:
        <da/dt> = - (64/5) * (G^3 m1 m2 M / (c^5 a^3 (1-e^2)^(7/2))) * (1 + (73/24) e^2 + (37/96) e^4)
        <de/dt> = - (304/15) * (G^3 m1 m2 M e / (c^5 a^4 (1-e^2)^(5/2))) * (1 + (121/304) e^2)

    Args:
        semi_major_axis_m: Orbital semi-major axis in meters (a > 0).
        eccentricity: Orbital eccentricity (0 <= e < 1).
        m1: Primary body mass in kg.
        m2: Secondary body mass in kg.

    Returns:
        Tuple of (<da/dt> in m/s, <de/dt> in 1/s). Strictly negative for bound orbits.
    """
    a = float(semi_major_axis_m)
    e = float(eccentricity)

    if a <= 0.0 or e < 0.0 or e >= 1.0 or m1 <= 0.0 or m2 <= 0.0:
        return 0.0, 0.0

    total_m = m1 + m2
    c5 = C_LIGHT ** 5
    g3 = G_NEWTON ** 3
    g3_mass_factor = g3 * m1 * m2 * total_m

    one_minus_e2 = 1.0 - e * e
    e2 = e * e
    e4 = e2 * e2

    # da/dt
    pref_a = - (64.0 / 5.0) * (g3_mass_factor / (c5 * (a ** 3) * (one_minus_e2 ** 3.5)))
    poly_a = 1.0 + (73.0 / 24.0) * e2 + (37.0 / 96.0) * e4
    da_dt = pref_a * poly_a

    # de/dt
    if e == 0.0:
        de_dt = 0.0
    else:
        pref_e = - (304.0 / 15.0) * (g3_mass_factor * e / (c5 * (a ** 4) * (one_minus_e2 ** 2.5)))
        poly_e = 1.0 + (121.0 / 304.0) * e2
        de_dt = pref_e * poly_e

    return da_dt, de_dt


def compute_lense_thirring_nodal_rate(
    semi_major_axis_m: float,
    eccentricity: float,
    spin_angular_momentum: float,
) -> float:
    """Compute secular Lense-Thirring nodal precession rate dOmega/dt in rad/s.

    Formula (Barker-O'Connell / Ciufolini & Wheeler):
        dOmega_LT/dt = (2 * G * S) / (c^2 * a^3 * (1 - e^2)^(3/2))

    Args:
        semi_major_axis_m: Semi-major axis in meters (a > 0).
        eccentricity: Eccentricity (0 <= e < 1).
        spin_angular_momentum: Central body spin angular momentum magnitude S in kg * m^2 / s.

    Returns:
        Secular nodal regression rate in radians per second.
    """
    a = float(semi_major_axis_m)
    e = float(eccentricity)
    s = float(spin_angular_momentum)

    if a <= 0.0 or e < 0.0 or e >= 1.0 or s == 0.0:
        return 0.0

    c2 = C_LIGHT * C_LIGHT
    denom = c2 * (a ** 3) * ((1.0 - e * e) ** 1.5)
    return (2.0 * G_NEWTON * s) / denom


def compute_geodetic_precession_vector(
    r_vec: np.ndarray | Sequence[float],
    v_vec: np.ndarray | Sequence[float],
    gm: float,
) -> np.ndarray:
    """Compute instantaneous de Sitter (geodetic) precession angular velocity vector in rad/s.

    Formula:
        Omega_geodetic = (3/2) * (GM / (c^2 * r^3)) * (r x v)

    Args:
        r_vec: Position vector from central body in meters [x, y, z].
        v_vec: Velocity vector in m/s [vx, vy, vz].
        gm: Central body gravitational parameter GM in m^3 / s^2.

    Returns:
        Precession angular velocity vector [Omegax, Omegay, Omegaz] in rad/s.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    v = np.asarray(v_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))

    if r_norm <= 0.0 or gm <= 0.0:
        return np.zeros(3, dtype=np.float64)

    c2 = C_LIGHT * C_LIGHT
    pref = 1.5 * gm / (c2 * (r_norm ** 3))
    r_cross_v = np.cross(r, v)
    return pref * r_cross_v
