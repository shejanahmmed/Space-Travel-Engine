"""Deep-Space Relativistic Atomic Clock Transport, Chronometric Geodesy, and Sagnac Delay.

Models the proper time dilation, gravitational redshift, second-order Doppler shift,
and rotating-frame Sagnac effects for high-precision spaceborne atomic clocks:
- NASA Deep Space Atomic Clock (DSAC, mercury-ion)
- ESA Atomic Clock Ensemble in Space (ACES / PHARAO, cold-atom cesium & rubidium)
- GPS / Galileo operational constellation standards
- Optical lattice space clocks (10^-17 to 10^-18 fractional accuracy)

Authoritative References:
- Ashby, N. (2003), "Relativity in the Global Positioning System", Living Reviews
  in Relativity, 6(1), §2-§4.
- Petit, G., & Luzum, B. (eds.) (2010), "IERS Conventions (2010)", IERS Technical
  Note 36, Chapter 10.
- Burt, E. A., et al. (2021), "Demonstration of a trapped-ion atomic clock in
  orbit", Nature, 595:43-47 (NASA DSAC flight results).
- Cacciapuoti, L., & Salomon, C. (2009), "Atomic clock ensemble in space",
  European Physical Journal Special Topics, 172:57-68.
- Allan, D. W. (1966), "Statistics of atomic frequency standards", Proc. IEEE, 54:221-230.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from relativistic_engine.constants import (
    C_LIGHT,
    G_NEWTON,
    GM_EARTH,
    GM_SUN,
    RADIUS_EARTH,
    SEC_PER_DAY,
)

# Reference geopotential value on the Earth geoid: W_0 / c^2 (IERS Conventions 2010, Eq. 10.2)
# W_0 = 62636856.0 m^2 / s^2 -> W_0 / c^2 = 6.969290134e-10
GEOID_POTENTIAL_W0: float = 62636856.0

# Earth mean rotational angular velocity in rad/s (IERS 2010)
OMEGA_EARTH_RAD_S: float = 7.292115146706979e-5

# Standard Allan deviation profiles at integration time tau = 1 day (86,400 s)
CLOCK_ALLAN_DEVIATIONS: Dict[str, float] = {
    "uso": 1.0e-13,          # Ultra-Stable Crystal Oscillator
    "rubidium": 2.0e-14,     # Standard spaceborne Rubidium standard
    "dsac": 3.0e-15,         # NASA Deep Space Atomic Clock (mercury-ion)
    "optical_lattice": 1e-17 # Next-gen space optical lattice clock
}


@dataclass(frozen=True)
class ClockTransportResult:
    """Chronometric evaluation output for an atomic clock along an orbital trajectory.

    Attributes:
        fractional_frequency_offset: Dimensionless fractional shift y = Delta_f / f0.
        gravitational_redshift: Gravitational contribution - Delta_Phi / c^2.
        kinematic_dilation: Second-order Doppler contribution - Delta(v^2) / (2 c^2).
        daily_drift_seconds: Net clock advance/retardation per Julian day [s/day].
        accumulated_phase_drift_seconds: Integrated proper-time deficit over trajectory [s].
        allan_deviation_1sigma_s: Estimated 1-sigma timing jitter from clock noise [s].
    """

    fractional_frequency_offset: float
    gravitational_redshift: float
    kinematic_dilation: float
    daily_drift_seconds: float
    accumulated_phase_drift_seconds: float
    allan_deviation_1sigma_s: float


def compute_fractional_frequency_offset(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    gm_central: float = GM_EARTH,
    *,
    reference_potential: float = GEOID_POTENTIAL_W0,
    reference_speed: float = 0.0,
    c: float = C_LIGHT,
) -> Tuple[float, float, float]:
    """Compute instantaneous fractional frequency shift y = Delta_f / f0 for an orbiting clock.

    Formula (Ashby 2003, Eq. 18):
        y = - (Phi(r) - Phi_ref) / c^2 - (v^2 - v_ref^2) / (2 * c^2)

    For Earth orbit relative to the surface geoid:
        Phi(r) = - GM / r
        Phi_ref = - W0
        y = (W0 - GM / r) / c^2 - v^2 / (2 * c^2)

    Positive y means the spacecraft clock runs FASTER than the Earth ground reference clock (blue-shift).

    Args:
        r_vec: Spacecraft position vector relative to central body [m].
        v_vec: Spacecraft velocity vector relative to central body [m/s].
        gm_central: Central gravitational parameter [m^3/s^2].
        reference_potential: Reference effective geopotential W0 in J/kg (m^2/s^2).
        reference_speed: Reference velocity of ground clock in m/s (default: 0 for geoid rotating frame).
        c: Speed of light in m/s.

    Returns:
        Tuple of (y_net, y_grav, y_kin) representing:
            y_net: Total fractional frequency offset
            y_grav: Gravitational redshift contribution
            y_kin: Kinematic time dilation contribution
    """
    pos = np.asarray(r_vec, dtype=np.float64)
    vel = np.asarray(v_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(pos))
    v_sq = float(np.dot(vel, vel))

    if r_norm <= 0.0:
        return 0.0, 0.0, 0.0

    c2 = c * c

    # Gravitational redshift: (Phi(r) - Phi_ref) / c^2
    # At high altitude, potential is less negative (higher) -> satellite clock runs FASTER (blue-shift)
    phi_space = - gm_central / r_norm
    phi_ref = - reference_potential
    y_grav = (phi_space - phi_ref) / c2

    # Kinematic second-order Doppler: - (v^2 - v_ref^2) / (2 c^2)
    # Moving clock runs SLOWER (red-shift)
    y_kin = - (v_sq - (reference_speed**2)) / (2.0 * c2)

    y_net = y_grav + y_kin
    return y_net, y_grav, y_kin


def compute_sagnac_delay(
    path_points: np.ndarray | Sequence[Sequence[float]],
    omega_rot_rad_s: float = OMEGA_EARTH_RAD_S,
    *,
    rot_axis: np.ndarray | Sequence[float] = (0.0, 0.0, 1.0),
    c: float = C_LIGHT,
) -> float:
    """Compute relativistic Sagnac time delay for an optical or radio signal link.

    Formula (Ashby 2003, Eq. 32; IERS 2010):
        Delta_t_Sagnac = (2 / c^2) * (omega . Area)
    where Area = 0.5 * sum_i (r_i x r_{i+1}) is the projected polygonal area
    swept by the ray path from origin.

    For closed loops or multi-station relay legs (e.g. Earth station -> Spacecraft -> Earth station):
        Delta_t = (2 * omega / c^2) * A_z

    Args:
        path_points: Array of shape (M, 3) representing sequence of path vertices in meters.
        omega_rot_rad_s: Angular rotation speed of the reference frame in rad/s.
        rot_axis: Unit rotation axis vector [wx, wy, wz].
        c: Speed of light in m/s.

    Returns:
        Sagnac time correction in seconds.
    """
    pts = np.asarray(path_points, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0

    axis = np.asarray(rot_axis, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm > 0.0:
        axis = axis / axis_norm

    # Compute vector area: Area = 0.5 * sum_i (r_i x r_{i+1})
    area_vec = np.zeros(3, dtype=np.float64)
    n_pts = pts.shape[0]

    for i in range(n_pts - 1):
        r1 = pts[i]
        r2 = pts[i + 1]
        area_vec += 0.5 * np.cross(r1, r2)

    omega_vec = omega_rot_rad_s * axis
    sagnac_sec = (2.0 / (c * c)) * float(np.dot(omega_vec, area_vec))
    return sagnac_sec


def compute_equatorial_closed_loop_sagnac(
    radius_m: float = RADIUS_EARTH,
    omega_rad_s: float = OMEGA_EARTH_RAD_S,
    *,
    c: float = C_LIGHT,
) -> float:
    """Compute the maximum theoretical Sagnac time delay for a circular equatorial transit.

    Formula:
        Delta_t = 2 * omega * pi * R^2 / c^2

    Args:
        radius_m: Equatorial radius in meters.
        omega_rad_s: Rotation speed in rad/s.
        c: Speed of light.

    Returns:
        Sagnac delay in seconds (nominally ~ 207.4 ns for Earth).
    """
    c2 = c * c
    area = math.pi * (radius_m**2)
    return (2.0 * omega_rad_s * area) / c2


def evaluate_trajectory_clock_drift(
    t_history_s: np.ndarray,
    r_history_m: np.ndarray,
    v_history_m_s: np.ndarray,
    clock_type: str = "dsac",
    gm_central: float = GM_EARTH,
) -> ClockTransportResult:
    """Evaluate integrated atomic clock drift, frequency shift, and noise bounds over a trajectory.

    Args:
        t_history_s: Array of coordinate time steps [s].
        r_history_m: Array of positions [x, y, z] of shape (N, 3) [m].
        v_history_m_s: Array of velocities [vx, vy, vz] of shape (N, 3) [m/s].
        clock_type: Clock standard ('uso', 'rubidium', 'dsac', 'optical_lattice').
        gm_central: Central gravitational parameter [m^3/s^2].

    Returns:
        ClockTransportResult with comprehensive frequency, phase, and Allan noise metrics.
    """
    t_arr = np.asarray(t_history_s, dtype=np.float64)
    r_arr = np.asarray(r_history_m, dtype=np.float64)
    v_arr = np.asarray(v_history_m_s, dtype=np.float64)

    n_pts = len(t_arr)
    if n_pts < 2:
        raise ValueError("Trajectory history must contain at least 2 points.")

    y_net_list = []
    y_grav_list = []
    y_kin_list = []

    for i in range(n_pts):
        yn, yg, yk = compute_fractional_frequency_offset(r_arr[i], v_arr[i], gm_central=gm_central)
        y_net_list.append(yn)
        y_grav_list.append(yg)
        y_kin_list.append(yk)

    y_net_arr = np.array(y_net_list, dtype=np.float64)
    y_grav_arr = np.array(y_grav_list, dtype=np.float64)
    y_kin_arr = np.array(y_kin_list, dtype=np.float64)

    # Integrated proper time advance: Delta_tau = integral(y_net * dt)
    integrated_phase_drift_s = float(np.trapezoid(y_net_arr, t_arr))

    mean_y_net = float(np.mean(y_net_arr))
    mean_y_grav = float(np.mean(y_grav_arr))
    mean_y_kin = float(np.mean(y_kin_arr))

    daily_drift_s = mean_y_net * SEC_PER_DAY

    # Allan deviation noise estimate over total duration T
    duration_s = float(t_arr[-1] - t_arr[0])
    sigma_y_1day = CLOCK_ALLAN_DEVIATIONS.get(clock_type.lower(), 3.0e-15)
    # White frequency noise scaling: sigma_y(T) ~ sigma_y(1 day) * sqrt(86400 / T)
    # Time jitter sigma_x(T) ~ T * sigma_y(T) = sigma_y(1 day) * sqrt(86400 * T)
    if duration_s > 0.0:
        time_jitter_s = sigma_y_1day * math.sqrt(SEC_PER_DAY * duration_s)
    else:
        time_jitter_s = 0.0

    return ClockTransportResult(
        fractional_frequency_offset=mean_y_net,
        gravitational_redshift=mean_y_grav,
        kinematic_dilation=mean_y_kin,
        daily_drift_seconds=daily_drift_s,
        accumulated_phase_drift_seconds=integrated_phase_drift_s,
        allan_deviation_1sigma_s=time_jitter_s,
    )
