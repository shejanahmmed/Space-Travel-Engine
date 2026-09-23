"""Deep Space Network (DSN) relativistic tracking observables and ground station geodesy.

Implements high-precision general relativistic light-time solution, Solar/planetary
Shapiro gravitational time delay, 2-way coherent Doppler range-rate, and observation
sensitivity Jacobians across the three primary Deep Space Communications Complexes (DSCC):
- Goldstone (California, USA)
- Madrid (Robledo de Chavela, Spain)
- Canberra (Tidbinbilla, Australia)

Authoritative References:
- Moyer, T. D. (2000 / 2003), "Formulation for Observed and Computed Values of Deep Space
  Network Data (Part 1 & 2)", JPL Publication 00-7, National Aeronautics and Space Administration.
- JPL DSN Telecommunications Link Design Handbook (810-005, Module 301, Rev. L), "Coverage and Geometry".
- Shapiro, I. I. (1964), "Fourth Test of General Relativity", Phys. Rev. Lett. 13, 789.
- IERS Conventions (2010), Gérard Petit and Brian Luzum (eds.), IERS Technical Note No. 36.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
import math
import numpy as np
from jplephem.spk import SPK

from relativistic_engine.constants import (
    C_LIGHT,
    GM_SUN,
    GM_EARTH,
    SEC_PER_DAY,
)
from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state
from relativistic_engine.physics.potential import (
    STANDARD_BODY_RADIUS,
    solar_system_potential,
)


# Nominal Earth rotation angular velocity (rad / s, IERS Conventions 2010, Section 1.1)
EARTH_ROTATION_RATE: float = 7.292115146706979e-5

# J2000.0 epoch in Julian Date (TT / TDB scale)
J2000_JD: float = 2451545.0

# Greenwich Mean Sidereal Time at J2000.0 (radians, IAU 2000)
# theta_0 = 280.46061837 deg * (pi / 180)
GMST_J2000_RAD: float = 4.894961212823059


@dataclass(frozen=True)
class DSNStation:
    """Deep Space Network tracking ground station.

    Attributes:
        station_id: Canonical station identifier (e.g., 'DSS-14').
        name: Common name and antenna diameter (e.g., 'Goldstone 70m').
        complex_name: Deep Space Communications Complex name.
        itrf_pos: 3D Cartesian position vector [X, Y, Z] in ITRF frame (meters).
        radius: Geocentric radius |r_sta| (meters).
        latitude_rad: Geocentric latitude in radians.
        longitude_rad: East longitude in radians.
    """

    station_id: str
    name: str
    complex_name: str
    itrf_pos: np.ndarray
    radius: float
    latitude_rad: float
    longitude_rad: float

    def __repr__(self) -> str:
        return f"<DSNStation {self.station_id}: {self.name} ({self.complex_name})>"


def create_dsn_station(
    station_id: str,
    name: str,
    complex_name: str,
    itrf_x: float,
    itrf_y: float,
    itrf_z: float,
) -> DSNStation:
    """Construct a DSNStation from ITRF Cartesian coordinates."""
    pos = np.array([float(itrf_x), float(itrf_y), float(itrf_z)], dtype=np.float64)
    r = float(np.linalg.norm(pos))
    lat = math.asin(pos[2] / r)
    lon = math.atan2(pos[1], pos[0]) % (2.0 * math.pi)
    return DSNStation(
        station_id=station_id,
        name=name,
        complex_name=complex_name,
        itrf_pos=pos,
        radius=r,
        latitude_rad=lat,
        longitude_rad=lon,
    )


# Authoritative ITRF coordinates from JPL DSN Handbook 810-005, Module 301, Rev. L
# Table 1: Geocentric Coordinates of DSN Antennas
DSS_14 = create_dsn_station(
    station_id="DSS-14",
    name="Goldstone 70m",
    complex_name="Goldstone DSCC (California, USA)",
    itrf_x=-2353621.419,
    itrf_y=-4641341.542,
    itrf_z=3677052.321,
)

DSS_65 = create_dsn_station(
    station_id="DSS-65",
    name="Madrid 34m HEF",
    complex_name="Madrid DSCC (Robledo, Spain)",
    itrf_x=4849339.638,
    itrf_y=-360488.756,
    itrf_z=4114751.233,
)

DSS_43 = create_dsn_station(
    station_id="DSS-43",
    name="Canberra 70m",
    complex_name="Canberra DSCC (Tidbinbilla, Australia)",
    itrf_x=-4460894.917,
    itrf_y=2682361.507,
    itrf_z=-3674748.152,
)

DSN_STATIONS: dict[str, DSNStation] = {
    "DSS-14": DSS_14,
    "DSS-65": DSS_65,
    "DSS-43": DSS_43,
}


def compute_greenwich_sidereal_time(jd_tdb: float, t_sec: float = 0.0) -> float:
    """Compute Greenwich Mean Sidereal Time (GMST) angle in radians.

    Args:
        jd_tdb: Julian Date (TDB scale).
        t_sec: Additional elapsed seconds since jd_tdb.

    Returns:
        Sidereal rotation angle theta_gst in radians in [0, 2*pi).
    """
    total_jd = jd_tdb + (t_sec / SEC_PER_DAY)
    delta_days = total_jd - J2000_JD
    theta = GMST_J2000_RAD + (EARTH_ROTATION_RATE * delta_days * SEC_PER_DAY)
    return float(theta % (2.0 * math.pi))


def compute_station_gcrf_state(
    station: DSNStation,
    jd_tdb: float,
    t_sec: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute topocentric ground station state in Geocentric Celestial Reference Frame (GCRF).

    Accounts for Earth diurnal rotation about the instantaneous polar axis.

    Args:
        station: DSNStation instance.
        jd_tdb: Base Julian Date (TDB scale).
        t_sec: Elapsed seconds from jd_tdb.

    Returns:
        (r_gcrf, v_gcrf) in meters and meters per second.
    """
    theta = compute_greenwich_sidereal_time(jd_tdb, t_sec)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    x_itrf, y_itrf, z_itrf = station.itrf_pos
    x_gcrf = cos_t * x_itrf - sin_t * y_itrf
    y_gcrf = sin_t * x_itrf + cos_t * y_itrf
    z_gcrf = z_itrf
    r_gcrf = np.array([x_gcrf, y_gcrf, z_gcrf], dtype=np.float64)

    vx_gcrf = -EARTH_ROTATION_RATE * y_gcrf
    vy_gcrf = EARTH_ROTATION_RATE * x_gcrf
    vz_gcrf = 0.0
    v_gcrf = np.array([vx_gcrf, vy_gcrf, vz_gcrf], dtype=np.float64)

    return r_gcrf, v_gcrf


def compute_station_bcrs_state(
    station: DSNStation,
    jd_tdb: float,
    t_sec: float = 0.0,
    spk: Optional[SPK] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute DSN station position and velocity in Barycentric Celestial Reference System (BCRS).

    Combines Earth barycentric motion from JPL ephemeris with station topocentric rotation.

    Args:
        station: DSNStation instance.
        jd_tdb: Base Julian Date.
        t_sec: Elapsed seconds from jd_tdb.
        spk: Optional pre-loaded JPL SPK kernel.

    Returns:
        (r_bcrs, v_bcrs) in meters and meters per second.
    """
    r_gcrf, v_gcrf = compute_station_gcrf_state(station, jd_tdb, t_sec)

    jd_frac = t_sec / SEC_PER_DAY
    earth_state = get_body_barycentric_state("earth", jd_tdb, spk, jd_fraction=jd_frac)

    r_bcrs = earth_state.position + r_gcrf
    v_bcrs = earth_state.velocity + v_gcrf
    return r_bcrs, v_bcrs


def compute_shapiro_time_delay(
    r_tx: np.ndarray | Sequence[float],
    r_rx: np.ndarray | Sequence[float],
    gm_body: float = GM_SUN,
    r_body: Optional[np.ndarray | Sequence[float]] = None,
    gamma_ppn: float = 1.0,
) -> float:
    """Compute General Relativistic Shapiro gravitational time delay.

    For a photon traveling from transmitter r_tx to receiver r_rx in the gravitational
    field of a body at r_body (defaults to solar system barycenter [0,0,0]):
        Delta t_Shapiro = (1 + gamma_PPN) * (GM / c^3) * ln((r1 + r2 + rho) / (r1 + r2 - rho))

    Authoritative References:
    - Shapiro, I. I. (1964), Phys. Rev. Lett. 13, 789.
    - Moyer, T. D. (2000), JPL Pub 00-7, Eq. (8-50).

    Args:
        r_tx: Transmitter position vector (m).
        r_rx: Receiver position vector (m).
        gm_body: Gravitational parameter of gravitating body (m^3/s^2, default Sun).
        r_body: Position of gravitating body (m, default origin).
        gamma_ppn: Parameterized Post-Newtonian parameter (1.0 in General Relativity).

    Returns:
        Shapiro time delay in seconds.
    """
    p_tx = np.asarray(r_tx, dtype=np.float64)
    p_rx = np.asarray(r_rx, dtype=np.float64)

    if r_body is not None:
        p_body = np.asarray(r_body, dtype=np.float64)
        p_tx = p_tx - p_body
        p_rx = p_rx - p_body

    r1 = float(np.linalg.norm(p_tx))
    r2 = float(np.linalg.norm(p_rx))
    rho = float(np.linalg.norm(p_rx - p_tx))

    if r1 <= 0.0 or r2 <= 0.0 or rho <= 0.0:
        return 0.0

    # Numerical stabilization against catastrophic cancellation:
    # (r1 + r2 - rho) = ((r1 + r2)^2 - rho^2) / (r1 + r2 + rho)
    #                 = (2 * r1 * r2 * (1 + cos_theta)) / (r1 + r2 + rho)
    #                 = (4 * r1 * r2 * cos^2(theta / 2)) / (r1 + r2 + rho)
    dot_prod = float(np.dot(p_tx, p_rx))
    cos_theta = dot_prod / (r1 * r2)
    cos_theta = max(-1.0, min(1.0, cos_theta))

    num = r1 + r2 + rho
    # Guard against direct center intersection (theta = pi => cos_theta = -1)
    half_angle_term = 4.0 * r1 * r2 * (0.5 * (1.0 + cos_theta))
    den = half_angle_term / num

    # Regularization guard using nominal solar physical radius
    min_den = (4.0 * (STANDARD_BODY_RADIUS["sun"] ** 2)) / num
    den = max(den, min_den)

    arg = num / den
    if arg <= 1.0:
        return 0.0

    factor = (1.0 + gamma_ppn) * gm_body / (C_LIGHT ** 3)
    return float(factor * math.log(arg))


def solve_2way_light_time(
    station: DSNStation,
    sc_trajectory_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
    t3_receive: float,
    jd_base: float,
    spk: Optional[SPK] = None,
    include_shapiro: bool = True,
    max_iter: int = 10,
    tol_sec: float = 1e-13,
) -> Tuple[float, float, float, float, float, float]:
    """Iteratively solve the 2-way relativistic light-time equations.

    Given ground station reception epoch t3, solves for:
    - t2: Spacecraft signal reflection epoch
    - t1: Ground station signal transmission epoch

    Light-time equations:
        c * (t3 - t2) = |r_sta(t3) - r_sc(t2)| + c * Delta t_Shapiro(t2, t3)
        c * (t2 - t1) = |r_sc(t2) - r_sta(t1)| + c * Delta t_Shapiro(t1, t2)

    Args:
        station: DSN tracking station.
        sc_trajectory_fn: Callable mapping t_sec -> (r_bcrs, v_bcrs).
        t3_receive: Reception epoch at station (seconds from jd_base).
        jd_base: Base Julian Date (TDB scale).
        spk: Optional pre-loaded JPL SPK kernel.
        include_shapiro: Whether to evaluate Solar Shapiro delay.
        max_iter: Maximum fixed-point iterations.
        tol_sec: Convergence tolerance in seconds (default 0.1 picoseconds).

    Returns:
        (t1_transmit, t2_reflection, t3_receive, range_2way_m, shapiro_up_s, shapiro_down_s)
    """
    r_sta3, _ = compute_station_bcrs_state(station, jd_base, t3_receive, spk)

    # 1. Down-leg iteration: solve for t2 such that photon from sc arrives at sta3 at t3
    r_sc_guess, _ = sc_trajectory_fn(t3_receive)
    dist_guess = float(np.linalg.norm(r_sta3 - r_sc_guess))
    t2 = t3_receive - (dist_guess / C_LIGHT)

    shapiro_down = 0.0
    for _ in range(max_iter):
        r_sc2, _ = sc_trajectory_fn(t2)
        rho_down = float(np.linalg.norm(r_sta3 - r_sc2))
        if include_shapiro:
            shapiro_down = compute_shapiro_time_delay(r_sc2, r_sta3, GM_SUN)
        t2_new = t3_receive - (rho_down / C_LIGHT) - shapiro_down
        if abs(t2_new - t2) < tol_sec:
            t2 = t2_new
            break
        t2 = t2_new

    # 2. Up-leg iteration: solve for t1 such that photon from sta1 arrives at sc at t2
    r_sc2, _ = sc_trajectory_fn(t2)
    t1 = t2 - (dist_guess / C_LIGHT)

    shapiro_up = 0.0
    for _ in range(max_iter):
        r_sta1, _ = compute_station_bcrs_state(station, jd_base, t1, spk)
        rho_up = float(np.linalg.norm(r_sc2 - r_sta1))
        if include_shapiro:
            shapiro_up = compute_shapiro_time_delay(r_sta1, r_sc2, GM_SUN)
        t1_new = t2 - (rho_up / C_LIGHT) - shapiro_up
        if abs(t1_new - t1) < tol_sec:
            t1 = t1_new
            break
        t1 = t1_new

    rtt_sec = t3_receive - t1
    range_2way_m = 0.5 * C_LIGHT * rtt_sec

    return t1, t2, t3_receive, range_2way_m, shapiro_up, shapiro_down


def compute_2way_doppler_shift(
    station: DSNStation,
    sc_trajectory_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
    t3_receive: float,
    jd_base: float,
    spk: Optional[SPK] = None,
    k_trans: float = 1.0,
    include_gravitational_redshift: bool = True,
) -> Tuple[float, float]:
    """Compute relativistic 2-way coherent Doppler frequency shift and range-rate.

    Incorporates special relativistic kinematic Doppler on both legs plus
    the gravitational potential difference between ground station and spacecraft.

    Formula (Moyer 2000, Section 13):
        (1 + Delta f / f0) = [(1 - n_up . v_sc / c) / (1 - n_up . v_sta1 / c)]
                           * [(1 - n_down . v_sta3 / c) / (1 - n_down . v_sc / c)]
                           * [1 + (Phi_sta1 - Phi_sc2) / c^2 + (Phi_sc2 - Phi_sta3) / c^2]

    Args:
        station: DSN tracking station.
        sc_trajectory_fn: Callable mapping t_sec -> (r_bcrs, v_bcrs).
        t3_receive: Reception epoch at station (seconds).
        jd_base: Base Julian Date.
        spk: Optional pre-loaded JPL SPK kernel.
        k_trans: Spacecraft transponder turnaround ratio (default 1.0).
        include_gravitational_redshift: Include gravitational red/blueshift.

    Returns:
        (delta_f_over_f0, range_rate_mps)
        - delta_f_over_f0: Relative frequency shift (dimensionless).
        - range_rate_mps: Equivalent 2-way line-of-sight range-rate (m/s).
    """
    t1, t2, _, _, _, _ = solve_2way_light_time(
        station, sc_trajectory_fn, t3_receive, jd_base, spk
    )

    r_sta1, v_sta1 = compute_station_bcrs_state(station, jd_base, t1, spk)
    r_sc2, v_sc2 = sc_trajectory_fn(t2)
    r_sta3, v_sta3 = compute_station_bcrs_state(station, jd_base, t3_receive, spk)

    # Unit line-of-sight wavevectors
    d_up = r_sc2 - r_sta1
    n_up = d_up / np.linalg.norm(d_up)

    d_down = r_sta3 - r_sc2
    n_down = d_down / np.linalg.norm(d_down)

    # Kinematic Doppler factor
    beta_sta1 = np.dot(n_up, v_sta1) / C_LIGHT
    beta_sc_up = np.dot(n_up, v_sc2) / C_LIGHT
    up_factor = (1.0 - beta_sc_up) / (1.0 - beta_sta1)

    beta_sc_down = np.dot(n_down, v_sc2) / C_LIGHT
    beta_sta3 = np.dot(n_down, v_sta3) / C_LIGHT
    down_factor = (1.0 - beta_sta3) / (1.0 - beta_sc_down)

    freq_ratio = up_factor * down_factor

    # Relativistic gravitational redshift correction
    if include_gravitational_redshift:
        phi_sta1 = solar_system_potential(r_sta1, jd_base, spk, jd_fraction=t1 / SEC_PER_DAY)
        phi_sc2 = solar_system_potential(r_sc2, jd_base, spk, jd_fraction=t2 / SEC_PER_DAY)
        phi_sta3 = solar_system_potential(r_sta3, jd_base, spk, jd_fraction=t3_receive / SEC_PER_DAY)

        delta_phi = (phi_sta1 - phi_sc2) + (phi_sc2 - phi_sta3)
        grav_shift = 1.0 + (delta_phi / (C_LIGHT ** 2))
        freq_ratio *= grav_shift

    freq_ratio *= k_trans
    delta_f_over_f0 = float(freq_ratio - 1.0)

    # Convert relative frequency shift to equivalent 2-way range-rate observable:
    # In DSN tracking (Moyer 2000, Eq. 13-1), d(rho_2way)/dt3 = -0.5 * c * (Delta f / f0).
    range_rate_mps = float(-0.5 * C_LIGHT * delta_f_over_f0)

    return delta_f_over_f0, range_rate_mps


def compute_observation_jacobian(
    r_sta: np.ndarray | Sequence[float],
    v_sta: np.ndarray | Sequence[float],
    r_sc: np.ndarray | Sequence[float],
    v_sc: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Compute the 2x6 observation sensitivity matrix H = dy / dx.

    State vector: x = [x, y, z, vx, vy, vz]^T in R^6.
    Measurements: y = [rho (range), dot{rho} (range-rate)]^T in R^2.

    Partials:
        d(rho) / dr = (r_sc - r_sta)^T / rho = hat{rho}^T
        d(rho) / dv = 0_{1x3}
        d(dot{rho}) / dr = (v_rel - (v_rel . hat{rho}) * hat{rho})^T / rho
        d(dot{rho}) / dv = hat{rho}^T

    Args:
        r_sta: Station position vector [x, y, z] in meters.
        v_sta: Station velocity vector [vx, vy, vz] in m/s.
        r_sc: Spacecraft position vector [x, y, z] in meters.
        v_sc: Spacecraft velocity vector [vx, vy, vz] in m/s.

    Returns:
        2x6 numpy array representing the observation Jacobian H.
    """
    p_sta = np.asarray(r_sta, dtype=np.float64)
    u_sta = np.asarray(v_sta, dtype=np.float64)
    p_sc = np.asarray(r_sc, dtype=np.float64)
    u_sc = np.asarray(v_sc, dtype=np.float64)

    rel_pos = p_sc - p_sta
    rho = float(np.linalg.norm(rel_pos))
    if rho <= 0.0:
        return np.zeros((2, 6), dtype=np.float64)

    hat_rho = rel_pos / rho
    rel_vel = u_sc - u_sta

    # Partial of range: [hat_rho, 0, 0, 0]
    h_range = np.concatenate([hat_rho, np.zeros(3, dtype=np.float64)])

    # Partial of range-rate:
    # d(dot{rho}) / dr = (v_rel - (v_rel . hat_rho) * hat_rho) / rho
    # d(dot{rho}) / dv = hat_rho
    proj_vel = float(np.dot(rel_vel, hat_rho))
    d_range_rate_dr = (rel_vel - proj_vel * hat_rho) / rho
    h_doppler = np.concatenate([d_range_rate_dr, hat_rho])

    return np.vstack([h_range, h_doppler])
