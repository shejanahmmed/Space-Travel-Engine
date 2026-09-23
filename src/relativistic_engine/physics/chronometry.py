"""High-precision relativistic chronometry and IAU BCRS metric tensor module.

Under IAU 2000 Resolution B1.3 and IAU 2006 Resolution 3, the metric signature
is (-, +, +, +) with line element:
    ds^2 = -c^2 dtau^2 = g_00 c^2 dt^2 + 2 g_0i c dt dx^i + g_ij dx^i dx^j
where t is BCRS coordinate time (TDB/TCB) and tau is the proper time measured
by an ideal atomic clock carried along the spacecraft worldline.

This module provides:
1. Full 1PN metric tensor components including central-body spin angular momentum
   (gravitomagnetism / Lense-Thirring off-diagonal term g_0i).
2. Zonal oblateness harmonics (J2) in gravitational potential and clock dilation.
3. Cancellation-free proper time and time deficit differential equations
   preserving 53-bit (or 113-bit) floating-point precision across all velocity regimes.

Authoritative References:
- IAU 2000 Resolution B1.3: Definition of BCRS metric tensor and gravitational potentials.
- Soffel, M., et al. (2003), AJ 126:2687-2706: Explanatory supplement to IAU 2000 resolutions.
- Ciufolini, I., & Wheeler, J. A. (1995), "Gravitation and Inertia", Princeton University Press.
- IERS Conventions (2010), IERS Technical Note No. 36.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence
import numpy as np

from relativistic_engine.constants import (
    C_LIGHT,
    G_NEWTON,
    GM_SUN,
    GM_EARTH,
    GM_MOON,
    GM_MARS,
    GM_JUPITER,
)

# Spin angular momentum magnitudes (kg * m^2 / s, IERS Conventions 2010 / Allen's Astrophysical Quantities)
# S_sun: derived from solar rotation period ~25.05 days and moment of inertia factor ~0.070
# S_earth: derived from Earth C-A moment of inertia and angular velocity 7.292115e-5 rad/s
# S_moon: derived from mean lunar rotation period 27.32166 days (C/MR^2 ~ 0.393)
# S_mars: derived from Mars sidereal rotation period 24.6229 hours (C/MR^2 ~ 0.365)
# S_jupiter: derived from Jupiter rotation period 9.925 hours
SPIN_ANGULAR_MOMENTUM: dict[str, float] = {
    "sun": 1.90e41,
    "earth": 7.07e33,
    "moon": 2.36e29,
    "mars": 1.92e32,
    "jupiter": 4.14e38,
}

# Unit spin pole vectors in ICRF / J2000 equatorial coordinates
# Earth rotational pole aligns with ICRF Z-axis by definition of J2000 equatorial frame.
# Sun: RA = 286.13 deg, Dec = 63.87 deg (IAU Working Group on Cartographic Coordinates, Archinal et al. 2018).
# Moon: RA = 269.995 deg, Dec = 66.539 deg (IAU WGCCRE mean pole).
# Mars: RA = 317.681 deg, Dec = 52.887 deg (IAU WGCCRE).
# Jupiter: RA = 268.05 deg, Dec = 64.49 deg (IAU WGCCRE).
_SUN_RA_RAD = math.radians(286.13)
_SUN_DEC_RAD = math.radians(63.87)
_MOON_RA_RAD = math.radians(269.9949)
_MOON_DEC_RAD = math.radians(66.5392)
_MARS_RA_RAD = math.radians(317.68143)
_MARS_DEC_RAD = math.radians(52.88650)
_JUP_RA_RAD = math.radians(268.05)
_JUP_DEC_RAD = math.radians(64.49)

BODY_SPIN_POLE: dict[str, np.ndarray] = {
    "sun": np.array(
        [
            math.cos(_SUN_DEC_RAD) * math.cos(_SUN_RA_RAD),
            math.cos(_SUN_DEC_RAD) * math.sin(_SUN_RA_RAD),
            math.sin(_SUN_DEC_RAD),
        ],
        dtype=np.float64,
    ),
    "earth": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "moon": np.array(
        [
            math.cos(_MOON_DEC_RAD) * math.cos(_MOON_RA_RAD),
            math.cos(_MOON_DEC_RAD) * math.sin(_MOON_RA_RAD),
            math.sin(_MOON_DEC_RAD),
        ],
        dtype=np.float64,
    ),
    "mars": np.array(
        [
            math.cos(_MARS_DEC_RAD) * math.cos(_MARS_RA_RAD),
            math.cos(_MARS_DEC_RAD) * math.sin(_MARS_RA_RAD),
            math.sin(_MARS_DEC_RAD),
        ],
        dtype=np.float64,
    ),
    "jupiter": np.array(
        [
            math.cos(_JUP_DEC_RAD) * math.cos(_JUP_RA_RAD),
            math.cos(_JUP_DEC_RAD) * math.sin(_JUP_RA_RAD),
            math.sin(_JUP_DEC_RAD),
        ],
        dtype=np.float64,
    ),
}

# Unnormalized zonal harmonics J2, J3, J4 (dimensionless, positive J2 by convention: V = (GM/r)[1 - sum J_n (R/r)^n P_n])
# Earth: EGM2008 / IERS Conventions (2010)
# Moon: GRAIL LP165P / GL0660B (Konopliv et al. 2013, Lemoine et al. 2014)
# Mars: GMM-3 / MRO120D (Genova et al. 2016, Konopliv et al. 2011)
# Sun: IAU 2015 B3 / Mecheri et al. (2004) / Rozelot et al. (2011)
# Jupiter: Juno 53-day orbit gravity solutions (Durante et al. 2020, Iess et al. 2018)
BODY_J2: dict[str, float] = {
    "sun": 2.20e-7,
    "earth": 1.0826359e-3,
    "moon": 2.027e-4,
    "mars": 1.96045e-3,
    "jupiter": 1.469657e-2,
}

BODY_J3: dict[str, float] = {
    "sun": 0.0,
    "earth": -2.53266e-6,
    "moon": 8.4e-6,
    "mars": 3.15e-5,
    "jupiter": -4.2e-8,
}

BODY_J4: dict[str, float] = {
    "sun": -4.0e-9,
    "earth": -1.61962e-6,
    "moon": -9.8e-6,
    "mars": -1.54e-5,
    "jupiter": -5.87e-4,
}

# Equatorial reference radii for zonal multipoles (meters, IAU 2015 B3 / IERS 2010)
BODY_EQUATORIAL_RADIUS: dict[str, float] = {
    "sun": 6.96342e8,
    "earth": 6.3781366e6,
    "moon": 1.7381e6,
    "mars": 3.39619e6,
    "jupiter": 7.1492e7,
}


def compute_gravitomagnetic_vector_potential(
    r_vec: np.ndarray | Sequence[float],
    spin_vector: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Compute post-Newtonian gravitomagnetic vector potential V_i in BCRS.

    In the IAU 2000 BCRS metric:
        g_0i = -(4 / c^3) * V_i
    where for an isolated body with spin angular momentum S:
        V_i = (G / 2 r^3) * (r x S)_i

    Args:
        r_vec: Position vector relative to central body center in meters [x, y, z].
        spin_vector: Spin angular momentum vector S in kg * m^2 / s [Sx, Sy, Sz].

    Returns:
        Vector potential V [Vx, Vy, Vz] in m^3 / s^3.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    s = np.asarray(spin_vector, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))

    if r_norm <= 0.0:
        return np.zeros(3, dtype=np.float64)

    r_cross_s = np.cross(r, s)
    return (G_NEWTON / (2.0 * r_norm * r_norm * r_norm)) * r_cross_s


def compute_metric_g0i(
    r_vec: np.ndarray | Sequence[float],
    spin_vector: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Compute off-diagonal metric tensor components g_0i (dimensionless).

    g_0i = -(2 * G / (c^3 * r^3)) * (r x S)_i

    Args:
        r_vec: Position vector relative to body center in meters.
        spin_vector: Spin angular momentum vector in kg * m^2 / s.

    Returns:
        Cartesian 3-vector [g_01, g_02, g_03] (dimensionless).
    """
    r = np.asarray(r_vec, dtype=np.float64)
    s = np.asarray(spin_vector, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))

    if r_norm <= 0.0:
        return np.zeros(3, dtype=np.float64)

    c_cubed = C_LIGHT * C_LIGHT * C_LIGHT
    r_cross_s = np.cross(r, s)
    return -((2.0 * G_NEWTON) / (c_cubed * r_norm * r_norm * r_norm)) * r_cross_s


def compute_zonal_potential(
    r_vec: np.ndarray | Sequence[float],
    body: str = "earth",
    max_degree: int = 4,
) -> float:
    """Compute zonal gravitational potential perturbation w_zonal in m^2 / s^2.

    Evaluates sum_{n=2}^{max_degree} w_n(r), where:
        w_n(r) = - (GM / r) * J_n * (R_eq / r)^n * P_n(u)
    with u = (r . s_pole) / r and P_n the standard Legendre polynomials.
    Positive by IAU BCRS metric convention: g_00 = -(1 - 2w/c^2).

    Authoritative Sources:
    - Earth: EGM2008 / IERS Conventions (2010)
    - Moon: GRAIL LP165P (Konopliv et al. 2013)
    - Mars: GMM-3 / MRO120D (Genova et al. 2016)
    - Sun: IAU 2015 B3 / Mecheri et al. (2004)
    - Jupiter: Juno gravity (Durante et al. 2020)

    Args:
        r_vec: Position vector relative to body center in meters [x, y, z].
        body: Central body identifier ('sun', 'earth', 'moon', 'mars', 'jupiter').
        max_degree: Maximum zonal harmonic degree to evaluate (2 <= max_degree <= 4).

    Returns:
        Scalar potential perturbation w_zonal in m^2 / s^2.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))

    b_key = body.lower()
    gm_map = {
        "sun": GM_SUN,
        "earth": GM_EARTH,
        "moon": GM_MOON,
        "mars": GM_MARS,
        "jupiter": GM_JUPITER,
    }
    gm = gm_map.get(b_key, GM_EARTH)
    r_eq = BODY_EQUATORIAL_RADIUS.get(b_key, 1.0)
    pole = BODY_SPIN_POLE.get(b_key, np.array([0.0, 0.0, 1.0]))

    if r_norm <= 0.0:
        return 0.0

    u = float(np.dot(r, pole)) / r_norm
    u_sq = u * u

    total_w = 0.0

    # Degree 2 (J2 Quadrupole)
    j2 = BODY_J2.get(b_key, 0.0)
    if max_degree >= 2 and j2 != 0.0:
        p2 = 0.5 * (3.0 * u_sq - 1.0)
        total_w -= (gm / r_norm) * j2 * ((r_eq / r_norm) ** 2) * p2

    # Degree 3 (J3 Pear-shaped / Odd Zonal)
    j3 = BODY_J3.get(b_key, 0.0)
    if max_degree >= 3 and j3 != 0.0:
        p3 = 0.5 * (5.0 * u * u_sq - 3.0 * u)
        total_w -= (gm / r_norm) * j3 * ((r_eq / r_norm) ** 3) * p3

    # Degree 4 (J4 Hexadecapole)
    j4 = BODY_J4.get(b_key, 0.0)
    if max_degree >= 4 and j4 != 0.0:
        p4 = 0.125 * (35.0 * (u_sq ** 2) - 30.0 * u_sq + 3.0)
        total_w -= (gm / r_norm) * j4 * ((r_eq / r_norm) ** 4) * p4

    return total_w


def compute_zonal_acceleration(
    r_vec: np.ndarray | Sequence[float],
    body: str = "earth",
    max_degree: int = 4,
) -> np.ndarray:
    """Compute 3D Cartesian gravitational acceleration from zonal harmonics in m/s^2.

    Evaluates exact vector gradient grad(w_zonal) = sum_{n=2}^{max_degree} a_n(r)
    in the BCRS/ICRF coordinate frame for arbitrary central body spin orientation.

    Formulas:
        a_2 = - (3 * GM * J2 * R_eq^2 / (2 * r^5)) * [ (1 - 5*u^2)*r + 2*u*r * s_pole ]
        a_3 = - (1 * GM * J3 * R_eq^3 / (2 * r^6)) * [ 5*u*(3 - 7*u^2)*r - 3*(1 - 5*u^2)*r * s_pole ]
        a_4 =   (5 * GM * J4 * R_eq^4 / (8 * r^7)) * [ 3*(21*u^4 - 14*u^2 + 1)*r + 4*u*(3 - 7*u^2)*r * s_pole ]

    Args:
        r_vec: Position vector relative to body center in meters [x, y, z].
        body: Central body identifier ('sun', 'earth', 'moon', 'mars', 'jupiter').
        max_degree: Maximum zonal harmonic degree (2 <= max_degree <= 4).

    Returns:
        Cartesian acceleration vector [ax, ay, az] in m/s^2.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))

    b_key = body.lower()
    gm_map = {
        "sun": GM_SUN,
        "earth": GM_EARTH,
        "moon": GM_MOON,
        "mars": GM_MARS,
        "jupiter": GM_JUPITER,
    }
    gm = gm_map.get(b_key, GM_EARTH)
    r_eq = BODY_EQUATORIAL_RADIUS.get(b_key, 1.0)
    pole = BODY_SPIN_POLE.get(b_key, np.array([0.0, 0.0, 1.0]))

    if r_norm <= 0.0:
        return np.zeros(3, dtype=np.float64)

    u = float(np.dot(r, pole)) / r_norm
    u_sq = u * u

    a_total = np.zeros(3, dtype=np.float64)

    # Degree 2 (J2 Oblateness)
    j2 = BODY_J2.get(b_key, 0.0)
    if max_degree >= 2 and j2 != 0.0:
        pref2 = - (1.5 * gm * j2 * r_eq * r_eq) / (r_norm ** 5)
        bracket2 = (1.0 - 5.0 * u_sq) * r + (2.0 * u * r_norm) * pole
        a_total += pref2 * bracket2

    # Degree 3 (J3 Pear-shaped)
    j3 = BODY_J3.get(b_key, 0.0)
    if max_degree >= 3 and j3 != 0.0:
        pref3 = - (0.5 * gm * j3 * (r_eq ** 3)) / (r_norm ** 6)
        bracket3 = (5.0 * u * (3.0 - 7.0 * u_sq)) * r - (3.0 * (1.0 - 5.0 * u_sq) * r_norm) * pole
        a_total += pref3 * bracket3

    # Degree 4 (J4 Hexadecapole)
    j4 = BODY_J4.get(b_key, 0.0)
    if max_degree >= 4 and j4 != 0.0:
        pref4 = (0.625 * gm * j4 * (r_eq ** 4)) / (r_norm ** 7)
        bracket4 = (3.0 * (21.0 * (u_sq ** 2) - 14.0 * u_sq + 1.0)) * r + (4.0 * u * (3.0 - 7.0 * u_sq) * r_norm) * pole
        a_total += pref4 * bracket4

    return a_total


def compute_j2_potential(
    r_vec: np.ndarray | Sequence[float],
    body: str = "earth",
) -> float:
    """Compute quadrupole gravitational potential perturbation w_J2 in m^2 / s^2."""
    return compute_zonal_potential(r_vec, body=body, max_degree=2)


def compute_j2_acceleration(
    r_vec: np.ndarray | Sequence[float],
    body: str = "earth",
) -> np.ndarray:
    """Compute gravitational acceleration gradient from J2 oblateness in m/s^2."""
    return compute_zonal_acceleration(r_vec, body=body, max_degree=2)


def compute_lense_thirring_acceleration(
    r_vec: np.ndarray | Sequence[float],
    v_vec: np.ndarray | Sequence[float],
    spin_vector: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Compute post-Newtonian Lense-Thirring gravitomagnetic acceleration in m/s^2.

    a_LT = (2 * G / (c^2 * r^3)) * [ (3 * (r . S) / r^2) * (r x v) + (v x S) ]

    Args:
        r_vec: Position vector relative to body center in meters [x, y, z].
        v_vec: Velocity vector relative to body center in m/s [vx, vy, vz].
        spin_vector: Central body spin angular momentum in kg * m^2 / s [Sx, Sy, Sz].

    Returns:
        Cartesian acceleration vector [ax, ay, az] in m/s^2.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    v = np.asarray(v_vec, dtype=np.float64)
    s = np.asarray(spin_vector, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))

    if r_norm <= 0.0:
        return np.zeros(3, dtype=np.float64)

    c_sq = C_LIGHT * C_LIGHT
    r_sq = r_norm * r_norm
    r_cross_v = np.cross(r, v)
    v_cross_s = np.cross(v, s)
    r_dot_s = float(np.dot(r, s))

    pref = (2.0 * G_NEWTON) / (c_sq * r_norm * r_sq)
    term1 = (3.0 * r_dot_s / r_sq) * r_cross_v
    term2 = v_cross_s

    return pref * (term1 + term2)




def proper_time_rate_full_1pn(
    v_vec: np.ndarray | Sequence[float],
    potential: float,
    *,
    r_vec: Optional[np.ndarray | Sequence[float]] = None,
    body: Optional[str] = None,
    spin_vector: Optional[np.ndarray | Sequence[float]] = None,
) -> float:
    """Evaluate exact proper time rate dtau/dt under full 1PN metric tensor.

    dtau/dt = sqrt( - (g_00 + 2 * g_0i * v^i / c + g_ij * v^i * v^j / c^2) )
            = sqrt( 1 - X )
    where:
        X = beta^2 + (2w/c^2) * (1 + beta^2) - 2 * (w/c^2)^2 + 2 * g_0i * (v^i / c)

    Args:
        v_vec: Coordinate velocity [vx, vy, vz] in m/s.
        potential: Total gravitational potential w in m^2 / s^2 (positive by convention).
        r_vec: Optional position vector relative to spinning body in meters.
        body: Optional body name to look up default spin vector ('sun', 'earth', 'jupiter').
        spin_vector: Optional explicit spin angular momentum vector in kg * m^2 / s.

    Returns:
        dtau/dt (dimensionless).
    """
    v_arr = np.asarray(v_vec, dtype=np.float64)
    v_norm = float(np.linalg.norm(v_arr))
    v_sq = float(np.dot(v_arr, v_arr))
    c_sq = C_LIGHT * C_LIGHT
    beta_sq = v_sq / c_sq

    phi_term = 2.0 * potential / c_sq
    phi_sq_term = 2.0 * (potential / c_sq) ** 2

    lt_term = 0.0
    if r_vec is not None:
        s_vec: Optional[np.ndarray] = None
        if spin_vector is not None:
            s_vec = np.asarray(spin_vector, dtype=np.float64)
        elif body is not None and body.lower() in SPIN_ANGULAR_MOMENTUM:
            s_mag = SPIN_ANGULAR_MOMENTUM[body.lower()]
            pole = BODY_SPIN_POLE[body.lower()]
            s_vec = s_mag * pole

        if s_vec is not None:
            g0i = compute_metric_g0i(r_vec, s_vec)
            lt_term = 2.0 * float(np.dot(g0i, v_arr)) / C_LIGHT

    # Dual-regime precision stabilization:
    # When beta >= 0.5, 1 - beta^2 suffers catastrophic cancellation if evaluated as 1 - beta^2.
    # We factor 1 - beta^2 = (c - v)(c + v) / c^2, preserving full 53-bit precision as v -> c.
    if v_norm >= 0.5 * C_LIGHT:
        one_minus_beta_sq = (C_LIGHT - v_norm) * (C_LIGHT + v_norm) / c_sq if v_norm < C_LIGHT else 0.0
        radicand = one_minus_beta_sq - phi_term * (1.0 + beta_sq) + phi_sq_term - lt_term
        if radicand <= 0.0:
            raise ValueError(f"Spacelike metric interval: radicand = {radicand:.8e} <= 0")
        return math.sqrt(radicand)
    else:
        x_val = beta_sq + phi_term * (1.0 + beta_sq) - phi_sq_term + lt_term
        if x_val >= 1.0:
            raise ValueError(f"Spacelike metric interval: X = {x_val:.8e} >= 1.0")
        radicand = max(0.0, 1.0 - x_val)
        return math.sqrt(radicand)


def coordinate_time_deficit_rate_full_1pn(
    v_vec: np.ndarray | Sequence[float],
    potential: float,
    *,
    r_vec: Optional[np.ndarray | Sequence[float]] = None,
    body: Optional[str] = None,
    spin_vector: Optional[np.ndarray | Sequence[float]] = None,
) -> float:
    """Evaluate rate of coordinate time deficit d(t - tau)/dt = 1 - dtau/dt.

    Eliminates catastrophic cancellation via:
        1 - sqrt(1 - X) = X / (1 + sqrt(1 - X))
    and factors 1 - beta^2 = (c - v)(c + v) / c^2 for beta >= 0.5,
    preserving full 53-bit precision across all velocity regimes (10^-8 <= beta < 1).

    Args:
        v_vec: Coordinate velocity [vx, vy, vz] in m/s.
        potential: Total gravitational potential w in m^2 / s^2.
        r_vec: Optional position vector relative to central body in meters.
        body: Optional body name ('sun', 'earth', 'jupiter').
        spin_vector: Optional explicit spin angular momentum vector.

    Returns:
        d(t - tau)/dt (dimensionless, strictly positive for subluminal trajectories).
    """
    v_arr = np.asarray(v_vec, dtype=np.float64)
    v_norm = float(np.linalg.norm(v_arr))
    v_sq = float(np.dot(v_arr, v_arr))
    c_sq = C_LIGHT * C_LIGHT
    beta_sq = v_sq / c_sq

    phi_term = 2.0 * potential / c_sq
    phi_sq_term = 2.0 * (potential / c_sq) ** 2

    lt_term = 0.0
    if r_vec is not None:
        s_vec: Optional[np.ndarray] = None
        if spin_vector is not None:
            s_vec = np.asarray(spin_vector, dtype=np.float64)
        elif body is not None and body.lower() in SPIN_ANGULAR_MOMENTUM:
            s_mag = SPIN_ANGULAR_MOMENTUM[body.lower()]
            pole = BODY_SPIN_POLE[body.lower()]
            s_vec = s_mag * pole

        if s_vec is not None:
            g0i = compute_metric_g0i(r_vec, s_vec)
            lt_term = 2.0 * float(np.dot(g0i, v_arr)) / C_LIGHT

    # Dual-regime cancellation elimination:
    if v_norm >= 0.5 * C_LIGHT:
        one_minus_beta_sq = (C_LIGHT - v_norm) * (C_LIGHT + v_norm) / c_sq if v_norm < C_LIGHT else 0.0
        radicand = max(0.0, one_minus_beta_sq - phi_term * (1.0 + beta_sq) + phi_sq_term - lt_term)
        sqrt_radicand = math.sqrt(radicand)
        x_val = 1.0 - radicand
        return x_val / (1.0 + sqrt_radicand)
    else:
        x_val = beta_sq + phi_term * (1.0 + beta_sq) - phi_sq_term + lt_term
        if x_val >= 1.0:
            x_val = 1.0 - 1.0e-15
        radicand = max(0.0, 1.0 - x_val)
        return x_val / (1.0 + math.sqrt(radicand))



def compute_frame_dragging_clock_shift_rate(
    r_vec: np.ndarray | Sequence[float],
    v_vec: np.ndarray | Sequence[float],
    spin_vector: np.ndarray | Sequence[float],
) -> float:
    """Compute differential proper-time shift rate delta(dtau/dt) from frame dragging.

    delta(dtau/dt)_LT = - g_0i * (v^i / c)
                     = (2 * G / (c^4 * r^3)) * (S . (r x v))
                     = (2 * G / (c^4 * r^3)) * (S . L)

    Positive for prograde orbits (clocks tick faster relative to non-rotating metric).
    Negative for retrograde orbits.

    Args:
        r_vec: Position vector in meters [x, y, z].
        v_vec: Velocity vector in m/s [vx, vy, vz].
        spin_vector: Spin angular momentum vector in kg * m^2 / s [Sx, Sy, Sz].

    Returns:
        delta(dtau/dt) (dimensionless fractional clock frequency shift).
    """
    r = np.asarray(r_vec, dtype=np.float64)
    v = np.asarray(v_vec, dtype=np.float64)
    s = np.asarray(spin_vector, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))

    if r_norm <= 0.0:
        return 0.0

    c_fourth = C_LIGHT ** 4
    r_cross_v = np.cross(r, v)
    s_dot_l = float(np.dot(s, r_cross_v))

    return (2.0 * G_NEWTON / (c_fourth * r_norm * r_norm * r_norm)) * s_dot_l
