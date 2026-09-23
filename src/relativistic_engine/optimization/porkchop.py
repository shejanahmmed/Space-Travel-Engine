"""Relativistic Trajectory Optimization and Porkchop Launch Window Solver.

Solves the two-point orbital boundary-value problem (Lambert's problem) in BCRS
and evaluates 2D Porkchop grids (departure epoch x arrival epoch / TOF) for
interplanetary mission design:
- Ballistic / Impulsive Mode: Evaluates departure characteristic energy C3 = |v_inf|^2,
  arrival excess speed |v_inf_arr|, total delta-v, and traveler proper time.
- Continuous-Thrust Relativistic Brachistochrone Mode: Evaluates required proper
  acceleration alpha, coordinate duration t, traveler proper time tau, and proper
  time deficit Delta(t).
- Launch Window Optimization: Locates the Pareto-optimal launch window minimizing
  delta-v or proper time via multi-dimensional refinement.

Authoritative References:
- Battin, R. H. (1999), "An Introduction to the Mathematics and Methods of Astrodynamics",
  AIAA Education Series, Chapter 7 (Lambert's Problem).
- Bate, R. R., Mueller, D. D., & White, J. E. (1971), "Fundamentals of Astrodynamics",
  Dover Publications.
- Park, R. S., et al. (2021), "The JPL Planetary and Lunar Ephemerides DE440 and DE441",
  The Astronomical Journal, 161:105.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence, Literal
import numpy as np
from scipy.optimize import brentq, minimize

from relativistic_engine.constants import (
    C_LIGHT,
    GM_SUN,
    SEC_PER_DAY,
)
from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state
from relativistic_engine.ephemeris.jpl_loader import load_jpl_ephemeris
from relativistic_engine.physics.kinematics import brachistochrone_profile


@dataclass(frozen=True)
class LambertSolution:
    """Analytical solution to the orbital two-point boundary-value problem.

    Attributes
    ----------
    v1 : np.ndarray
        Heliocentric departure velocity vector in BCRS (m/s).
    v2 : np.ndarray
        Heliocentric arrival velocity vector in BCRS (m/s).
    semi_major_axis : float
        Semi-major axis a of the transfer conic (meters).
    eccentricity : float
        Orbital eccentricity e of the transfer orbit.
    transfer_angle_rad : float
        True anomaly change Delta theta between r1 and r2 (radians).
    proper_time_deficit_sec : float
        Relativistic proper time deficit Delta(t) = t - tau accumulated
        along the transfer orbit in the solar gravitational field (seconds).
    """

    v1: np.ndarray
    v2: np.ndarray
    semi_major_axis: float
    eccentricity: float
    transfer_angle_rad: float
    proper_time_deficit_sec: float


@dataclass(frozen=True)
class PorkchopResult:
    """2D launch window evaluation product over departure and arrival epochs.

    Attributes
    ----------
    origin_body : str
        Name of the departure body.
    target_body : str
        Name of the target body.
    dep_jds : np.ndarray
        1D array of departure epochs in Julian Date (TDB scale), shape (N_dep,).
    arr_jds : np.ndarray
        1D array of arrival epochs in Julian Date (TDB scale), shape (N_arr,).
    tof_days : np.ndarray
        2D mesh of time-of-flight in days, shape (N_dep, N_arr).
    c3_km2_s2 : np.ndarray
        2D mesh of departure characteristic energy C3 in km^2/s^2.
    v_inf_dep_km_s : np.ndarray
        2D mesh of departure hyperbolic excess speed in km/s.
    v_inf_arr_km_s : np.ndarray
        2D mesh of arrival hyperbolic excess speed in km/s.
    delta_v_total_km_s : np.ndarray
        2D mesh of total mission delta-v (v_inf_dep + v_inf_arr) in km/s.
    proper_time_days : np.ndarray
        2D mesh of traveler proper time tau in days.
    time_deficit_sec : np.ndarray
        2D mesh of relativistic proper time deficit Delta(t) = t - tau in seconds.
    best_window : dict
        Parameters of the minimum delta-v launch opportunity within the grid.
    """

    origin_body: str
    target_body: str
    dep_jds: np.ndarray
    arr_jds: np.ndarray
    tof_days: np.ndarray
    c3_km2_s2: np.ndarray
    v_inf_dep_km_s: np.ndarray
    v_inf_arr_km_s: np.ndarray
    delta_v_total_km_s: np.ndarray
    proper_time_days: np.ndarray
    time_deficit_sec: np.ndarray
    best_window: dict


def _stumpff_c(z: float) -> float:
    """Stumpff function C(z) = (1 - cos(sqrt(z))) / z."""
    if z > 1e-4:
        sq_z = math.sqrt(z)
        return (1.0 - math.cos(sq_z)) / z
    elif z < -1e-4:
        sq_mz = math.sqrt(-z)
        return (math.cosh(sq_mz) - 1.0) / (-z)
    else:
        # Taylor series expansion to 8th order: prevents 0/0 catastrophic cancellation
        return 0.5 - z / 24.0 + (z**2) / 720.0 - (z**3) / 40320.0


def _stumpff_s(z: float) -> float:
    """Stumpff function S(z) = (sqrt(z) - sin(sqrt(z))) / z^(3/2)."""
    if z > 1e-4:
        sq_z = math.sqrt(z)
        return (sq_z - math.sin(sq_z)) / (z * sq_z)
    elif z < -1e-4:
        sq_mz = math.sqrt(-z)
        return (math.sinh(sq_mz) - sq_mz) / ((-z) * sq_mz)
    else:
        # Taylor series expansion to 8th order
        return (1.0 / 6.0) - z / 120.0 + (z**2) / 5040.0 - (z**3) / 362880.0


def solve_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof_sec: float,
    mu: float = GM_SUN,
    *,
    prograde: bool = True,
    max_iter: int = 100,
) -> LambertSolution:
    """Solve Lambert's orbital boundary-value problem using Universal Variables.

    Finds the heliocentric conic transfer connecting r1 to r2 in flight time tof_sec.

    Parameters
    ----------
    r1 : np.ndarray
        Initial position vector in BCRS (meters), shape (3,).
    r2 : np.ndarray
        Final position vector in BCRS (meters), shape (3,).
    tof_sec : float
        Time of flight in coordinate seconds (TDB scale).
    mu : float, optional
        Gravitational parameter of the central body (default: GM_SUN in m^3/s^2).
    prograde : bool, optional
        Whether the transfer orbit is prograde (Delta theta < pi for standard transfers).
    max_iter : int, optional
        Maximum iterations for root finding.

    Returns
    -------
    LambertSolution
        Initial/final velocities, orbital elements, and relativistic proper time deficit.

    Raises
    ------
    ValueError
        If tof_sec is non-positive or if boundary vectors are colinear without specified plane.
    """
    if tof_sec <= 0.0:
        raise ValueError(f"Time of flight must be strictly positive, got {tof_sec} s.")

    r1_vec = np.asarray(r1, dtype=np.float64)
    r2_vec = np.asarray(r2, dtype=np.float64)

    r1_norm = float(np.linalg.norm(r1_vec))
    r2_norm = float(np.linalg.norm(r2_vec))

    cos_dtheta = float(np.dot(r1_vec, r2_vec) / (r1_norm * r2_norm))
    cos_dtheta = min(1.0, max(-1.0, cos_dtheta))

    cross_12 = np.cross(r1_vec, r2_vec)
    is_prograde_geom = cross_12[2] >= 0.0

    if prograde:
        if is_prograde_geom:
            dtheta = math.acos(cos_dtheta)
        else:
            dtheta = 2.0 * math.pi - math.acos(cos_dtheta)
    else:
        if is_prograde_geom:
            dtheta = 2.0 * math.pi - math.acos(cos_dtheta)
        else:
            dtheta = math.acos(cos_dtheta)

    sin_dtheta = math.sin(dtheta)

    # Geometry parameter A (Bate, Mueller, White Eq. 5.3-17 / Vallado Alg. 58)
    # A = sin(dtheta) * sqrt(r1 * r2 / (1 - cos(dtheta)))
    denom_a = 1.0 - cos_dtheta
    if denom_a < 1e-12:
        raise ValueError("Positions r1 and r2 are colinear; plane of transfer is undefined.")

    A = sin_dtheta * math.sqrt(r1_norm * r2_norm / denom_a)
    if A == 0.0:
        raise ValueError("Transfer angle is a multiple of pi; Lambert solution is singular.")

    def _y_func(z: float) -> float:
        c_val = _stumpff_c(z)
        s_val = _stumpff_s(z)
        return r1_norm + r2_norm + A * (z * s_val - 1.0) / math.sqrt(c_val)

    def _tof_func(z: float) -> float:
        c_val = _stumpff_c(z)
        s_val = _stumpff_s(z)
        y_val = _y_func(z)
        if y_val <= 0.0:
            return 0.0
        x_val = math.sqrt(y_val / c_val)
        return (x_val**3 * s_val + A * math.sqrt(y_val)) / math.sqrt(mu)

    # Parabolic transfer time (z = 0)
    t_parabolic = _tof_func(0.0)

    # Exact monotonic bracketing
    if abs(tof_sec - t_parabolic) < 1e-8:
        z_opt = 0.0
    elif tof_sec > t_parabolic:
        # Elliptic transfer: z in (0, 4*pi^2)
        z_low = 0.0
        z_high = 4.0 * math.pi**2 - 1e-5
        while _tof_func(z_high) < tof_sec and (4.0 * math.pi**2 - z_high) > 1e-12:
            z_high = 4.0 * math.pi**2 - (4.0 * math.pi**2 - z_high) * 0.5
        z_opt = brentq(lambda z: _tof_func(z) - tof_sec, z_low, z_high, xtol=1e-12, maxiter=max_iter)
    else:
        # Hyperbolic transfer: z in (z_lower, 0)
        z_high = 0.0
        z_low = -1.0
        while _y_func(z_low) > 0.0 and _tof_func(z_low) > tof_sec:
            z_low *= 2.0
            if _y_func(z_low) <= 0.0:
                z_lo_pos = z_low / 2.0
                z_lo_neg = z_low
                for _ in range(30):
                    z_mid = 0.5 * (z_lo_pos + z_lo_neg)
                    if _y_func(z_mid) > 0.0:
                        z_lo_pos = z_mid
                    else:
                        z_lo_neg = z_mid
                z_low = z_lo_pos
                break
        z_opt = brentq(lambda z: _tof_func(z) - tof_sec, z_low, z_high, xtol=1e-12, maxiter=max_iter)

    c_opt = _stumpff_c(z_opt)
    s_opt = _stumpff_s(z_opt)
    y_opt = _y_func(z_opt)
    if y_opt <= 0.0:
        y_opt = 1e-6

    # Lagrange coefficients (BMW Eq. 5.3-21)
    f = 1.0 - (y_opt / r1_norm)
    g = A * math.sqrt(y_opt / mu)
    g_dot = 1.0 - (y_opt / r2_norm)

    v1 = (r2_vec - f * r1_vec) / g
    v2 = (g_dot * r2_vec - r1_vec) / g

    # Semi-major axis from vis-viva equation: 1/a = 2/r1 - v1^2 / mu
    v1_sq = float(np.dot(v1, v1))
    inv_a = (2.0 / r1_norm) - (v1_sq / mu)
    a = (1.0 / inv_a) if abs(inv_a) > 1e-15 else float("inf")

    # Specific angular momentum and eccentricity
    h_vec = np.cross(r1_vec, v1)
    h_sq = float(np.dot(h_vec, h_vec))
    ecc = math.sqrt(max(0.0, 1.0 - h_sq / (mu * a))) if a > 0.0 else 1.0

    # Relativistic proper time deficit along the Keplerian worldline:
    # Time-averaged rate <dtau/dt> = 1 - 3*mu / (2*a*c^2) (Eddington / Robertson)
    # Deficit Delta(t) = t - tau = (3*mu / (2*a*c^2)) * tof_sec
    c2 = C_LIGHT**2
    if a > 0.0:
        proper_time_deficit = (1.5 * mu / (a * c2)) * tof_sec
    else:
        # Hyperbolic/parabolic estimate: integrate along chord
        v_avg_sq = 0.5 * (v1_sq + float(np.dot(v2, v2)))
        r_avg = 0.5 * (r1_norm + r2_norm)
        proper_time_deficit = ((0.5 * v_avg_sq + mu / r_avg) / c2) * tof_sec

    return LambertSolution(
        v1=v1,
        v2=v2,
        semi_major_axis=a,
        eccentricity=ecc,
        transfer_angle_rad=dtheta,
        proper_time_deficit_sec=proper_time_deficit,
    )


def compute_porkchop_grid(
    origin_body: str,
    target_body: str,
    dep_jds: Sequence[float],
    arr_jds: Sequence[float],
    *,
    mode: Literal["ballistic", "brachistochrone"] = "ballistic",
    alpha_thrust: float | None = None,
) -> PorkchopResult:
    """Compute a 2D Porkchop launch window mesh over departure and arrival epochs.

    Parameters
    ----------
    origin_body : str
        Departure planet name (e.g. 'earth').
    target_body : str
        Destination planet name (e.g. 'mars').
    dep_jds : Sequence[float]
        1D array of departure epochs in Julian Date (TDB scale).
    arr_jds : Sequence[float]
        1D array of arrival epochs in Julian Date (TDB scale).
    mode : {'ballistic', 'brachistochrone'}, optional
        Optimization mode (default: 'ballistic').
    alpha_thrust : float, optional
        Proper acceleration in m/s^2 (required if mode is 'brachistochrone').

    Returns
    -------
    PorkchopResult
        Complete 2D grid of energy, delta-v, time-of-flight, and proper time metrics.
    """
    dep_arr = np.asarray(dep_jds, dtype=np.float64)
    arr_arr = np.asarray(arr_jds, dtype=np.float64)

    n_dep = len(dep_arr)
    n_arr = len(arr_arr)

    # Pre-allocate 2D output matrices
    tof_days = np.zeros((n_dep, n_arr), dtype=np.float64)
    c3_km2_s2 = np.full((n_dep, n_arr), np.nan, dtype=np.float64)
    v_inf_dep_km_s = np.full((n_dep, n_arr), np.nan, dtype=np.float64)
    v_inf_arr_km_s = np.full((n_dep, n_arr), np.nan, dtype=np.float64)
    delta_v_total_km_s = np.full((n_dep, n_arr), np.nan, dtype=np.float64)
    proper_time_days = np.full((n_dep, n_arr), np.nan, dtype=np.float64)
    time_deficit_sec = np.full((n_dep, n_arr), np.nan, dtype=np.float64)

    kernel = load_jpl_ephemeris()

    # Precompute planetary ephemeris states to avoid redundant evaluations
    origin_states = [get_body_barycentric_state(origin_body, jd, spk=kernel) for jd in dep_arr]
    target_states = [get_body_barycentric_state(target_body, jd, spk=kernel) for jd in arr_arr]

    min_dv = float("inf")
    best_dep_jd = dep_arr[0]
    best_arr_jd = arr_arr[0]
    best_c3 = 0.0
    best_tof = 0.0
    best_tau_days = 0.0
    best_deficit_sec = 0.0

    for i, (jd_dep, s_orig) in enumerate(zip(dep_arr, origin_states)):
        for j, (jd_arr, s_targ) in enumerate(zip(arr_arr, target_states)):
            tof_d = jd_arr - jd_dep
            tof_days[i, j] = tof_d

            # Discard non-causal or excessively long transfers
            if tof_d <= 1.0 or tof_d > 1200.0:
                continue

            tof_s = tof_d * SEC_PER_DAY

            if mode == "ballistic":
                try:
                    sol = solve_lambert(s_orig.position, s_targ.position, tof_s, mu=GM_SUN)

                    # Asymptotic excess velocities relative to moving planets
                    v_inf_dep_vec = sol.v1 - s_orig.velocity
                    v_inf_arr_vec = sol.v2 - s_targ.velocity

                    v_inf_dep = float(np.linalg.norm(v_inf_dep_vec))
                    v_inf_arr = float(np.linalg.norm(v_inf_arr_vec))

                    c3 = (v_inf_dep / 1000.0) ** 2  # km^2 / s^2
                    v_dep_kms = v_inf_dep / 1000.0
                    v_arr_kms = v_inf_arr / 1000.0
                    dv_total = v_dep_kms + v_arr_kms

                    deficit = sol.proper_time_deficit_sec
                    tau_d = tof_d - (deficit / SEC_PER_DAY)

                    c3_km2_s2[i, j] = c3
                    v_inf_dep_km_s[i, j] = v_dep_kms
                    v_inf_arr_km_s[i, j] = v_arr_kms
                    delta_v_total_km_s[i, j] = dv_total
                    proper_time_days[i, j] = tau_d
                    time_deficit_sec[i, j] = deficit

                    if dv_total < min_dv:
                        min_dv = dv_total
                        best_dep_jd = jd_dep
                        best_arr_jd = jd_arr
                        best_c3 = c3
                        best_tof = tof_d
                        best_tau_days = tau_d
                        best_deficit_sec = deficit

                except (ValueError, RuntimeError):
                    continue

            elif mode == "brachistochrone":
                if alpha_thrust is None or alpha_thrust <= 0.0:
                    raise ValueError("alpha_thrust must be positive for brachistochrone mode.")

                # Distance between origin at departure and target at arrival
                dist = float(np.linalg.norm(s_targ.position - s_orig.position))
                profile = brachistochrone_profile(dist, alpha_thrust)

                req_coord_days = profile.total_coord_time / SEC_PER_DAY
                tau_d = profile.total_proper_time / SEC_PER_DAY
                deficit = profile.total_time_deficit

                # Difference between requested TOF and brachistochrone flight time
                dt_err = abs(tof_d - req_coord_days)

                c3_km2_s2[i, j] = (profile.peak_velocity / 1000.0) ** 2
                v_inf_dep_km_s[i, j] = 0.0
                v_inf_arr_km_s[i, j] = 0.0
                delta_v_total_km_s[i, j] = dt_err
                proper_time_days[i, j] = tau_d
                time_deficit_sec[i, j] = deficit

                if dt_err < min_dv:
                    min_dv = dt_err
                    best_dep_jd = jd_dep
                    best_arr_jd = jd_arr
                    best_c3 = (profile.peak_velocity / 1000.0) ** 2
                    best_tof = req_coord_days
                    best_tau_days = tau_d
                    best_deficit_sec = deficit

    best_window = {
        "departure_jd": best_dep_jd,
        "arrival_jd": best_arr_jd,
        "tof_days": best_tof,
        "c3_km2_s2": best_c3,
        "delta_v_total_km_s": min_dv if min_dv < float("inf") else None,
        "proper_time_days": best_tau_days,
        "time_deficit_sec": best_deficit_sec,
    }

    return PorkchopResult(
        origin_body=origin_body,
        target_body=target_body,
        dep_jds=dep_arr,
        arr_jds=arr_arr,
        tof_days=tof_days,
        c3_km2_s2=c3_km2_s2,
        v_inf_dep_km_s=v_inf_dep_km_s,
        v_inf_arr_km_s=v_inf_arr_km_s,
        delta_v_total_km_s=delta_v_total_km_s,
        proper_time_days=proper_time_days,
        time_deficit_sec=time_deficit_sec,
        best_window=best_window,
    )
