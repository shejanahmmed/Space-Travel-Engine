"""Relativistic Spin-Orbit Couplings, Geodetic Precession, and Frame-Dragging.

Implements higher-order relativistic rotational and frame-dragging effects:
1. Geodetic (de Sitter) orbital curvature precession of gyroscopes (Schiff 1960).
2. Lense-Thirring frame-dragging precession from central body angular momentum (Lense & Thirring 1918).
3. Spin-orbit acceleration a_SO on orbital trajectories (Barker & O'Connell 1975; Kidder 1995).
4. Time-domain gyroscope spin vector integration: dS/dt = Omega x S.

Authoritative References:
- Barker, B. M., & O'Connell, R. F. (1975), "Gravitational two-body problem with
  arbitrary masses, spins, and quadrupole moments", Phys. Rev. D, 12(2), 329-355.
- Kidder, L. E. (1995), "Coalescing binary systems of compact objects to
  (post)^(5/2)-Newtonian order. V. Spin effects", Phys. Rev. D, 52(2), 821-847.
- Schiff, L. I. (1960), "Possible New Experimental Test of General Relativity
  Theory", Phys. Rev. Lett., 4(5), 215-217.
- Everitt, C. W. F., et al. (2011), "Gravity Probe B: Final Results of a Space
  Experiment to Test General Relativity", Phys. Rev. Lett., 106, 221101.
- Ciufolini, I., & Pavlis, E. C. (2004), "A confirmation of the general
  relativistic prediction of the Lense-Thirring effect", Nature, 431, 958-960.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np

from relativistic_engine.constants import (
    C_LIGHT,
    G_NEWTON,
    GM_EARTH,
    GM_JUPITER,
    GM_SUN,
    RADIUS_EARTH,
    RADIUS_JUPITER,
    RADIUS_SUN,
    SEC_PER_JULIAN_YEAR,
)

# Canonical planetary angular momentum vectors S = I * omega [kg m^2 / s] aligned with rotational pole
# Traceable to IERS Conventions (2010) and IAU Working Group on Cartographic Coordinates (2015)
PLANETARY_SPIN_VECTORS: Dict[str, Dict[str, Union[float, np.ndarray]]] = {
    "Earth": {
        "gm": GM_EARTH,
        "radius": RADIUS_EARTH,
        # I_zz ~ 8.034e37 kg m^2, omega = 7.292115e-5 rad/s -> S ~ 5.859e33 kg m^2 / s
        "spin_angular_momentum": np.array([0.0, 0.0, 5.859e33], dtype=np.float64),
    },
    "Sun": {
        "gm": GM_SUN,
        "radius": RADIUS_SUN,
        # Helioseismic angular momentum: S_sun ~ 1.9e41 kg m^2 / s
        "spin_angular_momentum": np.array([0.0, 0.0, 1.9e41], dtype=np.float64),
    },
    "Jupiter": {
        "gm": GM_JUPITER,
        "radius": RADIUS_JUPITER,
        # I_jup ~ 0.254 * M * R^2 ~ 2.45e42 kg m^2, omega = 1.758e-4 rad/s -> S ~ 4.14e38 kg m^2 / s
        "spin_angular_momentum": np.array([0.0, 0.0, 4.14e38], dtype=np.float64),
    },
}


def compute_geodetic_precession_vector(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    gm_central: float,
    *,
    c: float = C_LIGHT,
    gamma_ppn: float = 1.0,
) -> np.ndarray:
    """Compute instantaneous geodetic (de Sitter) precession angular velocity vector Omega_geodetic.

    Formula (Schiff 1960; Barker & O'Connell 1975):
        Omega_geodetic = ((1 + 2*gamma) / (2 * c^2)) * (v x grad_U)
                       = ((1 + 2*gamma) * GM / (2 * c^2 * r^3)) * (r x v)

    In General Relativity (gamma = 1):
        Omega_geodetic = (3 * GM / (2 * c^2 * r^3)) * (r x v)

    Args:
        r_vec: Spacecraft position vector [x, y, z] relative to central mass [m].
        v_vec: Spacecraft velocity vector [vx, vy, vz] relative to central mass [m/s].
        gm_central: Central gravitational parameter GM [m^3/s^2].
        c: Speed of light [m/s].
        gamma_ppn: PPN parameter gamma (1.0 in GR).

    Returns:
        Angular precession vector [Ox, Oy, Oz] in rad/s.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    v = np.asarray(v_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))

    if r_norm <= 0.0 or gm_central <= 0.0:
        return np.zeros(3, dtype=np.float64)

    c2 = c * c
    scale = ((1.0 + 2.0 * gamma_ppn) * gm_central) / (2.0 * c2 * (r_norm**3))
    r_cross_v = np.cross(r, v)
    return scale * r_cross_v


def compute_lense_thirring_precession_vector(
    r_vec: np.ndarray,
    spin_central: np.ndarray,
    *,
    c: float = C_LIGHT,
    gamma_ppn: float = 1.0,
    g: float = G_NEWTON,
) -> np.ndarray:
    """Compute instantaneous Lense-Thirring frame-dragging precession angular velocity vector.

    Formula (Lense & Thirring 1918; Schiff 1960):
        Omega_LT = ((1 + gamma) * G / (2 * c^2 * r^3)) * [ 3 * (S . n) * n - S ]
    where n = r / r.

    In General Relativity (gamma = 1):
        Omega_LT = (G / (c^2 * r^3)) * [ 3 * (S . n) * n - S ]

    Args:
        r_vec: Position vector [x, y, z] in meters.
        spin_central: Central body angular momentum vector S in kg m^2/s.
        c: Speed of light in m/s.
        gamma_ppn: PPN parameter gamma.
        g: Gravitational constant.

    Returns:
        Angular precession vector [Ox, Oy, Oz] in rad/s.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    s = np.asarray(spin_central, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))

    if r_norm <= 0.0 or float(np.linalg.norm(s)) <= 0.0:
        return np.zeros(3, dtype=np.float64)

    n_hat = r / r_norm
    s_dot_n = float(np.dot(s, n_hat))

    c2 = c * c
    prefactor = ((1.0 + gamma_ppn) * g) / (2.0 * c2 * (r_norm**3))
    bracket = 3.0 * s_dot_n * n_hat - s
    return prefactor * bracket


def compute_spin_orbit_acceleration(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    spin_central: np.ndarray,
    *,
    c: float = C_LIGHT,
    g: float = G_NEWTON,
) -> np.ndarray:
    """Compute 1.5PN spin-orbit relativistic acceleration a_SO exerted on an orbiting body.

    Evaluates the Barker & O'Connell (1975) / Kidder (1995) formulation:
        a_SO = (G / (c^2 * r^3)) * [
            (3 / r^2) * (r x v) * (r . S)
            + (v x S)
            - 3 * (r . v) * (r x S) / r^2
        ]

    Args:
        r_vec: Relative position vector [x, y, z] in meters.
        v_vec: Relative velocity vector [vx, vy, vz] in m/s.
        spin_central: Central body angular momentum vector S in kg m^2/s.
        c: Speed of light in m/s.
        g: Gravitational constant.

    Returns:
        Cartesian acceleration vector [ax, ay, az] in m/s^2.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    v = np.asarray(v_vec, dtype=np.float64)
    s = np.asarray(spin_central, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))

    if r_norm <= 0.0 or float(np.linalg.norm(s)) <= 0.0:
        return np.zeros(3, dtype=np.float64)

    r_sq = r_norm * r_norm
    r_cross_v = np.cross(r, v)
    r_dot_s = float(np.dot(r, s))
    v_cross_s = np.cross(v, s)
    r_dot_v = float(np.dot(r, v))
    r_cross_s = np.cross(r, s)

    term1 = (3.0 / r_sq) * r_cross_v * r_dot_s
    term2 = v_cross_s
    term3 = - (3.0 * r_dot_v / r_sq) * r_cross_s

    c2 = c * c
    prefactor = g / (c2 * (r_norm**3))
    return prefactor * (term1 + term2 + term3)


def propagate_gyroscope_spin(
    s_initial: np.ndarray,
    omega_prec: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Propagate gyroscope unit spin vector S under precession angular velocity: dS/dt = Omega x S.

    Uses Rodrigues' exact rotation formula over step dt to guarantee strict norm preservation:
        |S(t + dt)| = |S(t)| exactly.

    Args:
        s_initial: Gyroscope spin vector [Sx, Sy, Sz] at time t.
        omega_prec: Instantaneous precession vector [Ox, Oy, Oz] in rad/s.
        dt_s: Time step in seconds.

    Returns:
        Rotated gyroscope spin vector at time t + dt.
    """
    s_vec = np.asarray(s_initial, dtype=np.float64)
    omega = np.asarray(omega_prec, dtype=np.float64)
    omega_mag = float(np.linalg.norm(omega))

    if omega_mag <= 0.0 or dt_s == 0.0:
        return s_vec.copy()

    theta = omega_mag * dt_s
    k_hat = omega / omega_mag

    # Rodrigues' rotation formula: S_rot = S*cos(th) + (k x S)*sin(th) + k*(k . S)*(1 - cos(th))
    cos_th = math.cos(theta)
    sin_th = math.sin(theta)
    k_cross_s = np.cross(k_hat, s_vec)
    k_dot_s = float(np.dot(k_hat, s_vec))

    s_rot = s_vec * cos_th + k_cross_s * sin_th + k_hat * k_dot_s * (1.0 - cos_th)
    # Re-normalize to original magnitude to protect against float rounding
    norm_orig = float(np.linalg.norm(s_vec))
    norm_rot = float(np.linalg.norm(s_rot))
    if norm_rot > 0.0:
        s_rot = s_rot * (norm_orig / norm_rot)

    return s_rot


def secular_geodetic_precession_rate_arcsec_yr(
    semi_major_axis_m: float,
    eccentricity: float,
    gm_central: float = GM_EARTH,
) -> float:
    """Compute analytical orbit-averaged geodetic precession rate in arcseconds per year.

    Formula (Barker & O'Connell 1975; Everitt et al. 2011):
        <Omega_geodetic> = (3 * GM / (2 * c^2 * a * (1 - e^2))) * n
    where n = sqrt(GM / a^3).

    Args:
        semi_major_axis_m: Orbital semi-major axis in meters.
        eccentricity: Orbital eccentricity (0 <= e < 1).
        gm_central: Central gravitational parameter [m^3/s^2].

    Returns:
        Precession rate in arcseconds per Julian year.
    """
    a = semi_major_axis_m
    e = eccentricity
    p = a * (1.0 - e * e)
    n_mean = math.sqrt(gm_central / (a**3))

    c2 = C_LIGHT * C_LIGHT
    rate_rad_s = (1.5 * gm_central / (c2 * p)) * n_mean
    rate_rad_yr = rate_rad_s * SEC_PER_JULIAN_YEAR
    return math.degrees(rate_rad_yr) * 3600.0


def secular_lense_thirring_polar_rate_arcsec_yr(
    semi_major_axis_m: float,
    eccentricity: float,
    spin_central_mag: float = 5.859e33,
    gm_central: float = GM_EARTH,
) -> float:
    """Compute analytical orbit-averaged Lense-Thirring frame-dragging rate for a polar orbit.

    Formula (Barker & O'Connell 1975; Schiff 1960):
        <Omega_LT> = G * S / (2 * c^2 * a^3 * (1 - e^2)^(3/2))

    For Gravity Probe B (polar orbit along meridian):
        Precesses in the plane perpendicular to the orbital angular momentum.

    Args:
        semi_major_axis_m: Orbital semi-major axis in meters.
        eccentricity: Orbital eccentricity.
        spin_central_mag: Central body spin angular momentum magnitude [kg m^2 / s].
        gm_central: Central gravitational parameter.

    Returns:
        Frame-dragging rate in arcseconds per Julian year.
    """
    a = semi_major_axis_m
    e = eccentricity
    c2 = C_LIGHT * C_LIGHT
    denom = 2.0 * c2 * (a**3) * ((1.0 - e * e) ** 1.5)

    rate_rad_s = (G_NEWTON * spin_central_mag) / denom
    rate_rad_yr = rate_rad_s * SEC_PER_JULIAN_YEAR
    return math.degrees(rate_rad_yr) * 3600.0
