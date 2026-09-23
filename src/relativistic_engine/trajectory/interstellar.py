"""Relativistic interstellar brachistochrone and worldline trajectory solver.

Computes 3D relativistic journeys from the Solar System to nearby interstellar star systems
(Proxima Centauri, Alpha Centauri A/B, Barnard's Star) under continuous proper acceleration.

Integrates the full worldline through the Solar System potential well into deep interstellar
space, computing:
- Earth coordinate transit time (t_Earth)
- Traveler proper time (tau_traveler)
- Relativistic time dilation deficit (t - tau)
- Relativistic Doppler factor and longitudinal beaming
- Moving-target star position and space velocity matching

Authoritative References:
- Forward, R. L. (1984), "Roundtrip Interstellar Travel Using Laser-Pushed Lightsails", J. Spacecraft, 21(2), 187-195.
- Misner, C. W., Thorne, K. S., & Wheeler, J. A. (1973), "Gravitation", W. H. Freeman, Chapter 6 (Accelerated Observers).
- Soffel, M., et al. (2003), "The IAU 2000 Resolutions for Astrometry, Celestial Mechanics, and Metrology in the Relativistic Framework: Explanatory Supplement", AJ 126, 2687-2706.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence
import numpy as np
from jplephem.spk import SPK

from relativistic_engine.constants import (
    C_LIGHT,
    G0,
    SEC_PER_DAY,
    LIGHT_YEAR,
)
from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state
from relativistic_engine.ephemeris.interstellar import (
    INTERSTELLAR_CATALOG,
    get_star_barycentric_state,
)
from relativistic_engine.numerical.trajectory import (
    TrajectoryResult3D,
    propagate_trajectory_3d,
)
from relativistic_engine.trajectory.profiles import TwoStageThrustProfile
from relativistic_engine.physics.kinematics import (
    lorentz_beta,
    lorentz_gamma,
)



# Seconds per Julian year
SEC_PER_YEAR = 365.25 * SEC_PER_DAY


@dataclass(frozen=True)
class InterstellarMissionResult:
    """Comprehensive physical solution for an interstellar relativistic flight.

    Attributes
    ----------
    target_star : str
        Target star system identifier.
    departure_epoch_jd_tdb : float
        Departure epoch in Julian Date (TDB scale).
    arrival_epoch_jd_tdb : float
        Arrival epoch at destination in Julian Date (TDB scale).
    coordinate_flight_time_years : float
        Elapsed coordinate time in Julian years as measured in the BCRS / Earth frame.
    proper_flight_time_years : float
        Elapsed proper time in Julian years experienced by travelers aboard the spacecraft.
    time_deficit_years : float
        Cumulative relativistic time deficit (t_Earth - tau_traveler) in Julian years.
    distance_light_years : float
        Initial Euclidean distance to target star at departure epoch in light years.
    max_velocity_c : float
        Peak coordinate velocity achieved at midpoint turnaround, normalized to c (beta = v/c).
    max_lorentz_factor : float
        Peak relativistic Lorentz factor gamma = 1 / sqrt(1 - beta^2) at turnaround.
    max_doppler_redshift_earth : float
        Maximum relativistic Doppler factor (1 + z) for signals transmitted to Earth (redshift).
    max_doppler_blueshift_target : float
        Maximum relativistic Doppler factor for radiation encountered from target star (blueshift).
    miss_distance_meters : float
        Residual position miss distance at arrival relative to the moving star in meters.
    relative_arrival_velocity_m_s : float
        Residual velocity difference at arrival relative to the star's space velocity in m/s.
    trajectory : TrajectoryResult3D
        Complete 3D worldline trajectory integration solution.
    """

    target_star: str
    departure_epoch_jd_tdb: float
    arrival_epoch_jd_tdb: float
    coordinate_flight_time_years: float
    proper_flight_time_years: float
    time_deficit_years: float
    distance_light_years: float
    max_velocity_c: float
    max_lorentz_factor: float
    max_doppler_redshift_earth: float
    max_doppler_blueshift_target: float
    miss_distance_meters: float
    relative_arrival_velocity_m_s: float
    trajectory: TrajectoryResult3D


def solve_interstellar_brachistochrone(
    target_star: str = "proxima_centauri",
    accel_proper: float = G0,
    departure_epoch_jd_tdb: float = 2451545.0,  # J2000.0
    *,
    departure_body: str = "earth",
    spk: Optional[SPK] = None,
    gravitational_bodies: Optional[Sequence[str]] = ("sun",),
    include_1pn: bool = True,
    rtol: float = 1e-8,
    atol: float = 1e-9,
) -> InterstellarMissionResult:
    """Compute a high-precision 3D relativistic brachistochrone to an interstellar star.

    The spacecraft departs the Solar System from the barycentric state of `departure_body`
    at `departure_epoch_jd_tdb`, accelerating continuously at `accel_proper` toward the
    moving target star for half the transit time, then reverses thrust direction to decelerate
    and match the star's BCRS space velocity at arrival.

    Parameters
    ----------
    target_star : str
        Name of destination star from `INTERSTELLAR_CATALOG` (e.g. 'proxima_centauri').
    accel_proper : float
        Constant proper acceleration magnitude in m/s^2 (default: 1.0 g0 = 9.80665 m/s^2).
    departure_epoch_jd_tdb : float
        Departure epoch in Julian Date (TDB scale).
    departure_body : str
        Departure body name (default: 'earth').
    spk : Optional[SPK]
        NASA/JPL SPK ephemeris kernel.
    gravitational_bodies : Optional[Sequence[str]]
        Bodies included in Solar System potential during departure.
    include_1pn : bool
        Whether to evaluate 1PN general relativistic solar acceleration.
    rtol : float
        ODE relative tolerance.
    atol : float
        ODE absolute tolerance.

    Returns
    -------
    InterstellarMissionResult
        Physical mission parameters, time dilation, Doppler factors, and worldline solution.
    """
    # 1. Evaluate departure state in BCRS
    dep_state = get_body_barycentric_state(departure_body, departure_epoch_jd_tdb, spk=spk)
    r_dep = dep_state.position
    v_dep = dep_state.velocity

    # 2. Evaluate target star initial state at departure epoch
    star_dep_state = get_star_barycentric_state(target_star, departure_epoch_jd_tdb)
    r_star_init = star_dep_state.position
    v_star = star_dep_state.velocity

    dr_init = r_star_init - r_dep
    dist_init = float(np.linalg.norm(dr_init))

    # 3. Analytical relativistic brachistochrone estimate for flight duration
    # d_half = dist / 2
    # t_half = (c / alpha) * sqrt((1 + alpha * d_half / c^2)^2 - 1)
    d_half = 0.5 * dist_init
    u = (accel_proper * d_half) / (C_LIGHT * C_LIGHT)
    gamma_half = 1.0 + u
    t_half_est = (C_LIGHT / accel_proper) * math.sqrt(gamma_half * gamma_half - 1.0)
    t_total_est = 2.0 * t_half_est

    # 4. Predict star position at estimated arrival epoch to account for proper motion
    arrival_epoch_est = departure_epoch_jd_tdb + (t_total_est / SEC_PER_DAY)
    star_arr_state = get_star_barycentric_state(target_star, arrival_epoch_est)
    r_star_arr = star_arr_state.position

    # Refine vector toward predicted arrival position
    dr_target = r_star_arr - r_dep
    dist_refined = float(np.linalg.norm(dr_target))
    d_half_refined = 0.5 * dist_refined
    u_ref = (accel_proper * d_half_refined) / (C_LIGHT * C_LIGHT)
    gamma_half_ref = 1.0 + u_ref
    t_half = (C_LIGHT / accel_proper) * math.sqrt(gamma_half_ref * gamma_half_ref - 1.0)
    t_total = 2.0 * t_half

    # Unit vector along line of flight
    u_flight = dr_target / dist_refined

    # 5. Construct two-stage continuous thrust profile: +alpha then -alpha
    a_boost = accel_proper * u_flight
    a_brake = -accel_proper * u_flight

    profile = TwoStageThrustProfile(
        t1=t_half,
        t2=t_half,
        a1=a_boost,
        a2=a_brake,
    )

    # 6. Integrate full 3D relativistic worldline
    traj = propagate_trajectory_3d(
        r0=r_dep,
        v0=v_dep,
        t_span=(0.0, t_total),
        thrust_func=profile,
        epoch_jd_tdb=departure_epoch_jd_tdb,
        spk=spk,
        gravitational_bodies=gravitational_bodies,
        include_1pn=include_1pn,
        rtol=rtol,
        atol=atol,
    )

    # 7. Physical observables extraction
    final_pos = traj.r[-1]
    final_vel = traj.v[-1]
    final_deficit = traj.time_deficit[-1]
    final_tau = traj.tau[-1]

    # Target star exact state at final arrival epoch
    actual_arrival_epoch = departure_epoch_jd_tdb + (t_total / SEC_PER_DAY)
    star_actual_arr = get_star_barycentric_state(target_star, actual_arrival_epoch)

    miss_dist = float(np.linalg.norm(final_pos - star_actual_arr.position))
    rel_vel = float(np.linalg.norm(final_vel - star_actual_arr.velocity))

    # Peak velocity and Lorentz factor at turnaround
    speeds = np.linalg.norm(traj.v, axis=1)
    v_peak = float(np.max(speeds))
    beta_peak = lorentz_beta(v_peak)
    gamma_peak = lorentz_gamma(v_peak)


    # Relativistic Doppler factors
    # Toward Earth (redshifted): D_earth = sqrt((1 - beta) / (1 + beta))
    # Toward Target (blueshifted): D_target = sqrt((1 + beta) / (1 - beta))
    doppler_redshift_earth = math.sqrt((1.0 - beta_peak) / (1.0 + beta_peak))
    doppler_blueshift_target = math.sqrt((1.0 + beta_peak) / (1.0 - beta_peak))

    return InterstellarMissionResult(
        target_star=star_actual_arr.name,
        departure_epoch_jd_tdb=departure_epoch_jd_tdb,
        arrival_epoch_jd_tdb=actual_arrival_epoch,
        coordinate_flight_time_years=t_total / SEC_PER_YEAR,
        proper_flight_time_years=final_tau / SEC_PER_YEAR,
        time_deficit_years=final_deficit / SEC_PER_YEAR,
        distance_light_years=dist_init / LIGHT_YEAR,
        max_velocity_c=beta_peak,
        max_lorentz_factor=gamma_peak,
        max_doppler_redshift_earth=doppler_redshift_earth,
        max_doppler_blueshift_target=doppler_blueshift_target,
        miss_distance_meters=miss_dist,
        relative_arrival_velocity_m_s=rel_vel,
        trajectory=traj,
    )
