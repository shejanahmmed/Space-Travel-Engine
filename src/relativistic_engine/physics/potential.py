"""Solar System gravitational potential and N-body relativistic acceleration.

Evaluates point-mass gravitational potential w(r, t) and coordinate acceleration
dv/dt from planetary and lunar ephemerides in the BCRS frame, including 1PN
(first post-Newtonian) general relativistic corrections from the Sun.

Authoritative References:
- IAU 2000 Resolution B1.3: Definition of BCRS metric and potentials.
- IAU 2015 Resolution B3: Nominal Solar System body radii and gravitational parameters.
- Soffel, M., et al. (2003), "The IAU 2000 Resolutions for Astrometry, Celestial Mechanics,
  and Metrology in the Relativistic Framework: Explanatory Supplement", AJ 126:2687-2706.
- Moyer, T. D. (2003), "Formulation for Observed and Computed Values of Deep Space Network
  Data (Part 1: Equations of Motion)", JPL Publication 00-7.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence
import numpy as np
from jplephem.spk import SPK

from relativistic_engine.constants import (
    C_LIGHT,
    GM_SUN,
    GM_MERCURY,
    GM_VENUS,
    GM_EARTH,
    GM_MOON,
    GM_MARS,
    GM_JUPITER,
    GM_SATURN,
)
from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state

# Authoritative standard gravitational parameters for point-mass bodies (m^3 / s^2)
STANDARD_BODY_GM: dict[str, float] = {
    "sun": GM_SUN,
    "mercury": GM_MERCURY,
    "venus": GM_VENUS,
    "earth": GM_EARTH,
    "moon": GM_MOON,
    "mars": GM_MARS,
    "jupiter": GM_JUPITER,
    "saturn": GM_SATURN,
}

# Nominal volumetric/mean radii for Solar System bodies (meters, IAU 2015 B3 / IERS 2010)
# Used for C^1 smooth potential regularization inside physical body boundaries (Gauss's law).
STANDARD_BODY_RADIUS: dict[str, float] = {
    "sun": 6.957e8,
    "mercury": 2.4397e6,
    "venus": 6.0518e6,
    "earth": 6.3710084e6,
    "moon": 1.7374e6,
    "mars": 3.3895e6,
    "jupiter": 6.9911e7,
    "saturn": 5.8232e7,
}


def solar_system_potential(
    r_bcrs: np.ndarray | Sequence[float],
    jd_tdb: float,
    spk: Optional[SPK] = None,
    *,
    jd_fraction: float = 0.0,
    bodies: Optional[Sequence[str]] = None,
    include_j2: bool = False,
    include_zonals: bool = False,
    max_zonal_degree: int = 4,
) -> float:
    """Compute gravitational potential w(r, t) in the BCRS frame.

    Outside body surface (|r - r_i| >= R_i):
        w_i = GM_i / |r - r_i|
    Inside body surface (|r - r_i| < R_i, Gauss's law for uniform sphere):
        w_i = (GM_i / (2 * R_i)) * [ 3 - (|r - r_i| / R_i)^2 ]

    Potential is positive by IAU BCRS convention: g_00 = -(1 - 2w/c^2).

    Args:
        r_bcrs: Cartesian position vector [x, y, z] in meters relative to SSB.
        jd_tdb: Epoch in Julian Date (TDB scale).
        spk: Optional pre-loaded SPK kernel.
        jd_fraction: Optional fractional day component for sub-second precision.
        bodies: Sequence of body names to include. If None, includes all standard bodies.
        include_j2: Whether to include quadrupole J2 oblateness perturbation (max_degree=2).
        include_zonals: Whether to evaluate multi-degree zonal harmonic potential (J2 through J4).
        max_zonal_degree: Maximum zonal harmonic degree when include_zonals is True (2 to 4).

    Returns:
        Scalar gravitational potential w in m^2 / s^2 (positive).
    """
    from relativistic_engine.physics.chronometry import compute_zonal_potential

    r = np.asarray(r_bcrs, dtype=np.float64)
    target_bodies = bodies if bodies is not None else STANDARD_BODY_GM.keys()

    eval_zonals = include_zonals or include_j2
    deg = 2 if (include_j2 and not include_zonals) else max_zonal_degree

    total_w = 0.0
    for name in target_bodies:
        gm = STANDARD_BODY_GM[name]
        radius = STANDARD_BODY_RADIUS.get(name, 1.0)
        body_state = get_body_barycentric_state(name, jd_tdb, spk=spk, jd_fraction=jd_fraction)
        dr = r - body_state.position
        dist = float(np.linalg.norm(dr))

        if dist >= radius:
            total_w += gm / dist
        else:
            # Interior Gauss potential: continuous and C^1 smooth at dist = radius
            total_w += (gm / (2.0 * radius)) * (3.0 - (dist / radius) ** 2)

        if eval_zonals and name in ("sun", "earth", "moon", "mars", "jupiter") and dist >= radius:
            # Zonal perturbation falls off as (R_eq/dist)^2; cutoff at 50 R_eq preserves < 10^-12 precision
            if dist < 50.0 * radius:
                total_w += compute_zonal_potential(dr, body=name, max_degree=deg)

    return total_w


def solar_system_gravitational_acceleration(
    r_bcrs: np.ndarray | Sequence[float],
    v_bcrs: np.ndarray | Sequence[float],
    jd_tdb: float,
    spk: Optional[SPK] = None,
    *,
    jd_fraction: float = 0.0,
    bodies: Optional[Sequence[str]] = None,
    include_1pn: bool = True,
    include_j2: bool = False,
    include_zonals: bool = False,
    max_zonal_degree: int = 4,
    include_frame_dragging: bool = False,
) -> np.ndarray:
    """Compute gravitational acceleration vector in BCRS (Newtonian + 1PN + Zonals + Lense-Thirring).

    Newtonian component (with interior Gauss regularization):
        dist >= R_i: a_i = - GM_i * (r - r_i) / dist^3
        dist <  R_i: a_i = - GM_i * (r - r_i) / R_i^3

    1PN Solar component (EIH / isotropic BCRS gauge, Moyer 2003):
        a_1pn = (GM_sun / (c^2 * r_rel^3)) * [
            (4 * GM_sun / r_rel - v_rel^2) * r_rel + 4 * (r_rel . v_rel) * v_rel
        ]
    where r_rel = r - r_sun, v_rel = v - v_sun.

    Args:
        r_bcrs: Position vector [x, y, z] in meters relative to SSB.
        v_bcrs: Velocity vector [vx, vy, vz] in m/s relative to SSB.
        jd_tdb: Epoch in Julian Date (TDB scale).
        spk: Optional pre-loaded SPK kernel.
        jd_fraction: Optional fractional day component.
        bodies: Sequence of bodies to include (defaults to all standard bodies).
        include_1pn: Whether to evaluate 1PN general relativistic solar correction.
        include_j2: Whether to evaluate J2 zonal oblateness acceleration (max_degree=2).
        include_zonals: Whether to evaluate multi-degree zonal harmonics (J2 through J4).
        max_zonal_degree: Maximum zonal harmonic degree (2 <= max_degree <= 4).
        include_frame_dragging: Whether to evaluate Lense-Thirring gravitomagnetic acceleration.

    Returns:
        Cartesian acceleration vector [ax, ay, az] in m/s^2.
    """
    from relativistic_engine.physics.chronometry import (
        compute_zonal_acceleration,
        compute_lense_thirring_acceleration,
        SPIN_ANGULAR_MOMENTUM,
        BODY_SPIN_POLE,
    )

    r = np.asarray(r_bcrs, dtype=np.float64)
    v = np.asarray(v_bcrs, dtype=np.float64)
    target_bodies = bodies if bodies is not None else STANDARD_BODY_GM.keys()

    eval_zonals = include_zonals or include_j2
    deg = 2 if (include_j2 and not include_zonals) else max_zonal_degree

    a_total = np.zeros(3, dtype=np.float64)
    r_sun_rel: Optional[np.ndarray] = None
    v_sun_rel: Optional[np.ndarray] = None
    r_sun_dist: float = 0.0

    for name in target_bodies:
        gm = STANDARD_BODY_GM[name]
        radius = STANDARD_BODY_RADIUS.get(name, 1.0)
        body_state = get_body_barycentric_state(name, jd_tdb, spk=spk, jd_fraction=jd_fraction)
        dr = r - body_state.position
        dist = float(np.linalg.norm(dr))

        if dist >= radius:
            inv_dist_cube = 1.0 / (dist * dist * dist)
            a_total -= (gm * inv_dist_cube) * dr
        else:
            # Interior Gauss gravitational field: linear in distance
            a_total -= (gm / (radius * radius * radius)) * dr

        if eval_zonals and name in ("sun", "earth", "moon", "mars", "jupiter") and dist >= radius:
            if dist < 50.0 * radius:
                a_total += compute_zonal_acceleration(dr, body=name, max_degree=deg)

        if include_frame_dragging and name in ("sun", "earth", "moon", "mars", "jupiter") and dist >= radius:
            if dist < 50.0 * radius and name in SPIN_ANGULAR_MOMENTUM and name in BODY_SPIN_POLE:
                s_mag = SPIN_ANGULAR_MOMENTUM[name]
                pole = BODY_SPIN_POLE[name]
                s_vec = s_mag * pole
                dv_rel = v - body_state.velocity
                a_total += compute_lense_thirring_acceleration(dr, dv_rel, s_vec)

        if name == "sun":
            r_sun_rel = dr
            v_sun_rel = v - body_state.velocity
            r_sun_dist = max(dist, radius)

    # Evaluate 1PN correction if requested and Sun is among evaluated bodies
    if include_1pn and r_sun_rel is not None and v_sun_rel is not None:
        c_sq = C_LIGHT * C_LIGHT
        gm_s = GM_SUN
        r_dist = r_sun_dist
        v_rel_sq = float(np.dot(v_sun_rel, v_sun_rel))
        r_dot_v = float(np.dot(r_sun_rel, v_sun_rel))

        pref = gm_s / (c_sq * r_dist * r_dist * r_dist)
        term_r = (4.0 * gm_s / r_dist - v_rel_sq) * r_sun_rel
        term_v = (4.0 * r_dot_v) * v_sun_rel

        a_total += pref * (term_r + term_v)

    return a_total

