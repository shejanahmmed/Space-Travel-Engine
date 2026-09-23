"""Spherical Harmonic Zonal Gravitational Fields (J2 through J8).

Evaluates high-degree axisymmetric gravitational potential and acceleration
multipoles for the Sun, Earth, Mars, Jupiter, and Saturn in body-centered coordinates.

Authoritative References:
- Folkner, W. M., et al. (2017), "Jupiter's Gravity Field from the Juno Mission",
  Geophysical Research Letters, 44(10), 4694-4700.
- Iess, L., et al. (2019), "Measurement and global modeling of the gravity field
  of Saturn", Science, 364(6445), eaat2965.
- Pavlis, N. K., et al. (2012), "The development and evaluation of the Earth
  Gravitational Model 2008 (EGM2008)", J. Geophys. Res., 117, B04406.
- Kaula, W. M. (1966), "Theory of Satellite Geodesy", Blaisdell Publishing Co., §3.2.
- Vallado, D. A. (2013), "Fundamentals of Astrodynamics and Applications", 4th ed., §8.6.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from relativistic_engine.constants import (
    GM_EARTH,
    GM_JUPITER,
    GM_MARS,
    GM_SUN,
    RADIUS_EARTH,
    RADIUS_JUPITER,
    RADIUS_SUN,
)

# Standard planetary zonal gravity models (unnormalized zonal coefficients J_n)
PLANETARY_ZONAL_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "Sun": {
        "gm": GM_SUN,
        "r_ref": RADIUS_SUN,
        "J2": 2.196e-7,  # IAU 2015
    },
    "Earth": {
        "gm": GM_EARTH,
        "r_ref": RADIUS_EARTH,
        "J2": 1.08262668e-3,  # EGM2008
        "J3": -2.53265648e-6,
        "J4": -1.61962159e-6,
        "J5": -2.27296082e-7,
        "J6": 5.40681239e-7,
    },
    "Mars": {
        "gm": GM_MARS,
        "r_ref": 3.39619e6,
        "J2": 1.96045e-3,  # MRO110J
        "J3": 3.145e-5,
        "J4": -1.5377e-5,
    },
    "Jupiter": {
        "gm": GM_JUPITER,
        "r_ref": RADIUS_JUPITER,
        "J2": 1.4696572e-2,   # Juno (Folkner et al. 2017)
        "J3": -0.042e-6,
        "J4": -5.86609e-4,
        "J5": -0.069e-6,
        "J6": 3.4199e-5,
        "J7": 0.124e-6,
        "J8": -2.426e-6,
    },
    "Saturn": {
        "gm": 3.79312077e16,
        "r_ref": 6.0268e7,
        "J2": 1.6290573e-2,  # Cassini Grand Finale (Iess et al. 2019)
        "J3": 0.059e-6,
        "J4": -9.3583e-4,
        "J5": -0.224e-6,
        "J6": 8.614e-5,
        "J8": -1.0e-5,
    },
}


def legendre_polynomials_and_derivatives(n_max: int, u: float) -> Tuple[List[float], List[float]]:
    """Compute Legendre polynomials P_n(u) and their derivatives P'_n(u) up to degree n_max.

    Uses Bonnet's recurrence relation:
        P_0(u) = 1
        P_1(u) = u
        n * P_n(u) = (2n - 1) * u * P_{n-1}(u) - (n - 1) * P_{n-2}(u)

    Derivatives:
        P'_n(u) = n / (1 - u^2) * (P_{n-1}(u) - u * P_n(u))  for |u| < 1
        At poles u = +/- 1, evaluates analytic limits.

    Args:
        n_max: Maximum degree (e.g. 8).
        u: Sine of latitude u = sin(phi) in [-1.0, 1.0].

    Returns:
        Tuple of (P, P_prime) where P[n] is P_n(u) and P_prime[n] is dP_n/du.
    """
    u_clamped = max(-1.0, min(1.0, float(u)))

    P = [0.0] * (n_max + 1)
    P_prime = [0.0] * (n_max + 1)

    P[0] = 1.0
    P_prime[0] = 0.0

    if n_max >= 1:
        P[1] = u_clamped
        P_prime[1] = 1.0

    for n in range(2, n_max + 1):
        P[n] = ((2 * n - 1) * u_clamped * P[n - 1] - (n - 1) * P[n - 2]) / n

    # Derivatives
    om_u2 = 1.0 - u_clamped * u_clamped
    if om_u2 > 1e-14:
        for n in range(2, n_max + 1):
            P_prime[n] = (n / om_u2) * (P[n - 1] - u_clamped * P[n])
    else:
        # Polar limit: u = +/- 1
        # P'_n(1) = n*(n+1)/2, P'_n(-1) = (-1)^(n-1) * n*(n+1)/2
        sign = 1.0 if u_clamped > 0 else -1.0
        for n in range(2, n_max + 1):
            base_val = n * (n + 1) / 2.0
            P_prime[n] = base_val if sign > 0 else base_val * ((-1.0) ** (n - 1))

    return P, P_prime


def compute_zonal_gravity_acceleration(
    r_body: np.ndarray,
    gm: float,
    r_ref: float,
    j_coeffs: Dict[int, float],
    *,
    max_degree: int = 8,
) -> np.ndarray:
    """Compute gravitational acceleration vector due to zonal harmonics J2 through J_max.

    Evaluates:
        a_zonal = sum_{n=2}^{n_max} (GM / r^2) * (R_ref / r)^n * [
            (n + 1) * P_n(u) * r_hat - P'_n(u) * (z_hat - u * r_hat)
        ]
    where u = sin(phi) = z / r, r_hat = r / r, z_hat = [0, 0, 1].

    Args:
        r_body: Cartesian position vector in body-fixed/centered frame [x, y, z] in meters.
        gm: Body gravitational parameter GM in m^3/s^2.
        r_ref: Body reference equatorial radius in meters.
        j_coeffs: Mapping of degree n -> unnormalized coefficient J_n (e.g. {2: 1.08263e-3, 4: -1.62e-6}).
        max_degree: Maximum degree to evaluate (up to 8).

    Returns:
        Cartesian acceleration vector [ax, ay, az] in m/s^2.
    """
    pos = np.asarray(r_body, dtype=np.float64)
    r = float(np.linalg.norm(pos))
    if r <= 0.0 or gm <= 0.0:
        return np.zeros(3, dtype=np.float64)

    u = pos[2] / r  # sin(latitude)
    r_hat = pos / r
    z_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_transverse = z_hat - u * r_hat

    n_limit = min(max_degree, max(j_coeffs.keys(), default=2))
    P, P_prime = legendre_polynomials_and_derivatives(n_limit, u)

    inv_r2 = 1.0 / (r * r)
    gm_over_r2 = gm * inv_r2
    ratio = r_ref / r

    a_zonal = np.zeros(3, dtype=np.float64)

    for n, j_n in j_coeffs.items():
        if n < 2 or n > n_limit or j_n == 0.0:
            continue

        scale = - j_n * (ratio ** n) * gm_over_r2
        term_radial = (n + 1) * P[n] * r_hat
        term_transverse = - P_prime[n] * z_transverse
        a_zonal += scale * (term_radial + term_transverse)

    return a_zonal


def compute_planetary_gravity_acceleration(
    body_name: str,
    r_body: np.ndarray,
    *,
    include_point_mass: bool = True,
    max_degree: int = 8,
) -> Dict[str, np.ndarray]:
    """Compute point-mass + zonal harmonic acceleration for a known celestial body.

    Args:
        body_name: Name of body ('Sun', 'Earth', 'Mars', 'Jupiter', 'Saturn').
        r_body: Position vector in body-centered frame in meters.
        include_point_mass: Whether to include central GM/r^2 acceleration.
        max_degree: Maximum zonal harmonic degree (2 through 8).

    Returns:
        Dictionary with:
            'point_mass': Acceleration from central point mass [m/s^2]
            'zonal': Acceleration from non-spherical zonal harmonics [m/s^2]
            'total': Combined acceleration [m/s^2]
    """
    pos = np.asarray(r_body, dtype=np.float64)
    r = float(np.linalg.norm(pos))

    if body_name not in PLANETARY_ZONAL_COEFFICIENTS:
        raise ValueError(f"Unknown body '{body_name}'. Available: {list(PLANETARY_ZONAL_COEFFICIENTS.keys())}")

    model = PLANETARY_ZONAL_COEFFICIENTS[body_name]
    gm = model["gm"]
    r_ref = model["r_ref"]

    j_dict = {}
    for key, val in model.items():
        if key.startswith("J") and key[1:].isdigit():
            j_dict[int(key[1:])] = val

    a_point = np.zeros(3, dtype=np.float64)
    if include_point_mass and r > 0.0:
        a_point = - (gm / (r**3)) * pos

    a_zonal = compute_zonal_gravity_acceleration(pos, gm, r_ref, j_dict, max_degree=max_degree)
    return {
        "point_mass": a_point,
        "zonal": a_zonal,
        "total": a_point + a_zonal,
    }


def secular_j2_nodal_precession_rate(
    semi_major_axis_m: float,
    eccentricity: float,
    inclination_rad: float,
    body_name: str = "Earth",
) -> float:
    """Compute the classical analytical secular nodal precession rate dOmega/dt in rad/s.

    Formula (Kaula 1966):
        dOmega/dt = - (3/2) * n * J2 * (R_ref / p)^2 * cos(i)
    where:
        p = a * (1 - e^2)
        n = sqrt(GM / a^3)

    Args:
        semi_major_axis_m: Semi-major axis in meters.
        eccentricity: Orbit eccentricity (0 <= e < 1).
        inclination_rad: Orbit inclination in radians.
        body_name: Central body name.

    Returns:
        Secular nodal regression rate in rad/s.
    """
    model = PLANETARY_ZONAL_COEFFICIENTS[body_name]
    gm = model["gm"]
    r_ref = model["r_ref"]
    j2 = model["J2"]

    p = semi_major_axis_m * (1.0 - eccentricity**2)
    n_mean = math.sqrt(gm / (semi_major_axis_m**3))

    rate = - 1.5 * n_mean * j2 * ((r_ref / p)**2) * math.cos(inclination_rad)
    return rate
