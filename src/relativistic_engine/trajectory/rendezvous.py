"""Interplanetary Relativistic Rendezvous & Dynamic Trajectory Solver.

Solves the moving-target two-point boundary value problem (BVP) between celestial
bodies moving along independent JPL DE440 ephemeris orbits in the BCRS frame.

Authoritative Standards:
- IAU 2000 Resolution B1.3 / IAU 2006 Resolution 3 (BCRS metric and time scales).
- Lawden, D. F. (1963), "Optimal Trajectories for Space Navigation", Butterworths.
- Park, R. S., et al. (2021), "The JPL Planetary and Lunar Ephemerides DE440 and DE441", AJ 161:105.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Sequence
import numpy as np
from scipy.optimize import least_squares
from jplephem.spk import SPK

from relativistic_engine.constants import C_LIGHT, G0, SEC_PER_DAY
from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state
from relativistic_engine.physics.kinematics import coordinate_time_from_distance
from relativistic_engine.physics.potential import STANDARD_BODY_RADIUS
from relativistic_engine.numerical.trajectory import (
    TrajectoryResult3D,
    propagate_trajectory_3d,
)
from relativistic_engine.trajectory.profiles import (
    TwoStageThrustProfile,
    create_brachistochrone_steering,
)


class RendezvousMode(Enum):
    """Target arrival boundary condition mode."""

    SOFT_RENDEZVOUS = "soft_rendezvous"  # Match both position and velocity at arrival
    INTERCEPT = "intercept"  # Match position only at arrival (flyby)


@dataclass(frozen=True)
class RendezvousSolution:
    """Complete numerical solution of an interplanetary relativistic rendezvous.

    Attributes
    ----------
    success : bool
        Whether the shooting solver successfully converged to boundary conditions.
    message : str
        Solver termination status message.
    trajectory : TrajectoryResult3D
        Full 3D numerical worldline solution.
    departure_epoch_jd : float
        Departure epoch in Julian Date (TDB scale).
    arrival_epoch_jd : float
        Arrival epoch in Julian Date (TDB scale).
    flight_time_seconds : float
        Coordinate flight time Delta t in seconds.
    flight_time_days : float
        Coordinate flight time Delta t in Julian days.
    crew_proper_time_seconds : float
        Proper flight time Delta tau in seconds experienced by the crew.
    crew_proper_time_days : float
        Proper flight time Delta tau in Julian days.
    coordinate_time_deficit_seconds : float
        Difference between Earth coordinate time and crew proper time (Delta t - Delta tau).
    departure_position : np.ndarray
        Spacecraft position vector [x, y, z] at departure in meters.
    departure_velocity : np.ndarray
        Spacecraft velocity vector [vx, vy, vz] at departure in m/s.
    arrival_position : np.ndarray
        Spacecraft position vector [x, y, z] at arrival in meters.
    arrival_velocity : np.ndarray
        Spacecraft velocity vector [vx, vy, vz] at arrival in m/s.
    target_arrival_position : np.ndarray
        Target body position vector [x, y, z] at arrival epoch in meters.
    target_arrival_velocity : np.ndarray
        Target body velocity vector [vx, vy, vz] at arrival epoch in m/s.
    position_error_m : float
        Residual position miss distance |r_ship - r_target| at arrival in meters.
    velocity_error_mps : float
        Residual velocity error |v_ship - v_target| at arrival in m/s.
    max_speed_mps : float
        Maximum coordinate speed |v| achieved during trajectory in m/s.
    max_beta : float
        Maximum relativistic velocity fraction v / c.
    thrust_profile : TwoStageThrustProfile
        Thrust profile that generates this trajectory.
    """

    success: bool
    message: str
    trajectory: TrajectoryResult3D
    departure_epoch_jd: float
    arrival_epoch_jd: float
    flight_time_seconds: float
    flight_time_days: float
    crew_proper_time_seconds: float
    crew_proper_time_days: float
    coordinate_time_deficit_seconds: float
    departure_position: np.ndarray
    departure_velocity: np.ndarray
    arrival_position: np.ndarray
    arrival_velocity: np.ndarray
    target_arrival_position: np.ndarray
    target_arrival_velocity: np.ndarray
    position_error_m: float
    velocity_error_mps: float
    max_speed_mps: float
    max_beta: float
    thrust_profile: TwoStageThrustProfile


def solve_interplanetary_rendezvous(
    departure_body: str,
    target_body: str,
    departure_epoch_jd: float,
    *,
    accel_magnitude: float = G0,
    mode: RendezvousMode = RendezvousMode.SOFT_RENDEZVOUS,
    spk: Optional[SPK] = None,
    gravitational_bodies: Optional[Sequence[str]] = ("sun",),
    include_1pn: bool = True,
    parking_altitude_m: Optional[float] = None,
    rtol: float = 1e-7,
    atol: float = 1e-8,
    max_fev: int = 80,
) -> RendezvousSolution:
    """Solve the relativistic 3D rendezvous trajectory to a moving planetary target.

    Formulates a continuous-thrust boundary value problem and solves the
    non-linear shooting equations using the Levenberg-Marquardt algorithm.

    Parameters
    ----------
    departure_body : str
        Name of departure celestial body (e.g. 'earth').
    target_body : str
        Name of target celestial body (e.g. 'mars').
    departure_epoch_jd : float
        Departure epoch in Julian Date (TDB scale).
    accel_magnitude : float, optional
        Constant proper acceleration magnitude alpha in m/s^2 (default: 1 g_0 = 9.80665).
    mode : RendezvousMode, optional
        Target arrival boundary condition mode (default: SOFT_RENDEZVOUS).
    spk : Optional[SPK], optional
        Pre-loaded NASA/JPL SPK kernel handle.
    gravitational_bodies : Optional[Sequence[str]], optional
        Bodies to include in the gravitational potential field (default: ('sun',)).
    include_1pn : bool, optional
        Whether to evaluate 1PN general relativistic solar acceleration (default: True).
    parking_altitude_m : Optional[float], optional
        Departure orbital parking altitude above body radius in meters.
        If None, defaults to 400 km for Earth, 250 km for Mars, or 0.05 * radius.
    rtol : float, optional
        Relative integration tolerance for ODE propagation (default: 1e-7).
    atol : float, optional
        Absolute integration tolerance for ODE propagation (default: 1e-8).
    max_fev : int, optional
        Maximum function evaluations for root-finding solver (default: 80).

    Returns
    -------
    RendezvousSolution
        Complete solution containing worldline, timing breakdown, and residuals.
    """
    dep_name = departure_body.lower().strip()
    tgt_name = target_body.lower().strip()

    # 1. Evaluate departure body state at departure epoch
    dep_state = get_body_barycentric_state(dep_name, departure_epoch_jd, spk=spk)
    tgt_state_0 = get_body_barycentric_state(tgt_name, departure_epoch_jd, spk=spk)

    # Offset initial position to parking orbit radius
    r_body = STANDARD_BODY_RADIUS.get(dep_name, 6.371e6)
    if parking_altitude_m is not None:
        alt = parking_altitude_m
    elif dep_name == "earth":
        alt = 400.0e3  # 400 km LEO
    elif dep_name == "mars":
        alt = 250.0e3  # 250 km LMO
    else:
        alt = 0.05 * r_body

    r_orbit = r_body + alt
    # Offset radially outward from the Sun
    sun_dir = dep_state.position / np.linalg.norm(dep_state.position)
    r0 = dep_state.position + r_orbit * sun_dir
    v0 = dep_state.velocity.copy()

    # 2. Scaling and initial parameter estimates
    initial_dist = float(np.linalg.norm(tgt_state_0.position - r0))
    r_scale = 1.0e8  # 100,000 km
    v_scale = 1.0e4  # 10 km/s

    if mode == RendezvousMode.SOFT_RENDEZVOUS:
        # Two-stage brachistochrone: 6 parameters (w1 in R^3, w2 in R^3)
        t_half_1d = coordinate_time_from_distance(initial_dist / 2.0, accel_magnitude)
        t_flight_guess = 2.0 * t_half_1d

        # Predict target position at arrival
        tgt_pred = get_body_barycentric_state(
            tgt_name,
            departure_epoch_jd,
            spk=spk,
            jd_fraction=t_flight_guess / SEC_PER_DAY,
        )
        d_transfer = tgt_pred.position - (r0 + v0 * t_flight_guess)
        transfer_dir = d_transfer / np.linalg.norm(d_transfer)

        w1_init = accel_magnitude * t_half_1d * transfer_dir
        w2_init = -accel_magnitude * t_half_1d * transfer_dir
        p0 = np.concatenate([w1_init, w2_init])

        def objective(p: np.ndarray) -> np.ndarray:
            w1 = p[0:3]
            w2 = p[3:6]
            norm1 = float(np.linalg.norm(w1))
            norm2 = float(np.linalg.norm(w2))

            if norm1 < 1e-3 or norm2 < 1e-3:
                return np.ones(6) * 1e6

            t1 = norm1 / accel_magnitude
            t2 = norm2 / accel_magnitude
            a1 = accel_magnitude * (w1 / norm1)
            a2 = accel_magnitude * (w2 / norm2)
            total_time = t1 + t2

            profile = TwoStageThrustProfile(t1=t1, t2=t2, a1=a1, a2=a2)

            res = propagate_trajectory_3d(
                r0=r0,
                v0=v0,
                t_span=(0.0, total_time),
                thrust_func=profile,
                epoch_jd_tdb=departure_epoch_jd,
                spk=spk,
                gravitational_bodies=gravitational_bodies,
                include_1pn=include_1pn,
                rtol=rtol,
                atol=atol,
            )

            tgt_arr = get_body_barycentric_state(
                tgt_name,
                departure_epoch_jd,
                spk=spk,
                jd_fraction=total_time / SEC_PER_DAY,
            )

            r_err = (res.r[-1] - tgt_arr.position) / r_scale
            v_err = (res.v[-1] - tgt_arr.velocity) / v_scale
            return np.concatenate([r_err, v_err])

        opt_sol = least_squares(
            objective,
            p0,
            method="lm",
            diff_step=1e-4,
            max_nfev=max_fev,
        )

        w1_sol = opt_sol.x[0:3]
        w2_sol = opt_sol.x[3:6]
        norm1_sol = float(np.linalg.norm(w1_sol))
        norm2_sol = float(np.linalg.norm(w2_sol))
        t1_sol = norm1_sol / accel_magnitude
        t2_sol = norm2_sol / accel_magnitude
        a1_sol = accel_magnitude * (w1_sol / norm1_sol)
        a2_sol = accel_magnitude * (w2_sol / norm2_sol)

        final_profile = TwoStageThrustProfile(
            t1=t1_sol,
            t2=t2_sol,
            a1=a1_sol,
            a2=a2_sol,
        )
        final_flight_time = t1_sol + t2_sol

    else:
        # Intercept (flyby): single continuous burn, 3 parameters (w1 in R^3)
        t_burn_guess = coordinate_time_from_distance(initial_dist, accel_magnitude)
        tgt_pred = get_body_barycentric_state(
            tgt_name,
            departure_epoch_jd,
            spk=spk,
            jd_fraction=t_burn_guess / SEC_PER_DAY,
        )
        d_transfer = tgt_pred.position - (r0 + v0 * t_burn_guess)
        transfer_dir = d_transfer / np.linalg.norm(d_transfer)
        p0 = accel_magnitude * t_burn_guess * transfer_dir

        def objective_intercept(p: np.ndarray) -> np.ndarray:
            norm = float(np.linalg.norm(p))
            if norm < 1e-3:
                return np.ones(3) * 1e6
            t1 = norm / accel_magnitude
            a1 = accel_magnitude * (p / norm)
            profile = TwoStageThrustProfile(t1=t1, t2=0.0, a1=a1, a2=np.zeros(3))

            res = propagate_trajectory_3d(
                r0=r0,
                v0=v0,
                t_span=(0.0, t1),
                thrust_func=profile,
                epoch_jd_tdb=departure_epoch_jd,
                spk=spk,
                gravitational_bodies=gravitational_bodies,
                include_1pn=include_1pn,
                rtol=rtol,
                atol=atol,
            )

            tgt_arr = get_body_barycentric_state(
                tgt_name,
                departure_epoch_jd,
                spk=spk,
                jd_fraction=t1 / SEC_PER_DAY,
            )

            return (res.r[-1] - tgt_arr.position) / r_scale

        opt_sol = least_squares(
            objective_intercept,
            p0,
            method="lm",
            diff_step=1e-4,
            max_nfev=max_fev,
        )

        w1_sol = opt_sol.x
        norm1_sol = float(np.linalg.norm(w1_sol))
        t1_sol = norm1_sol / accel_magnitude
        a1_sol = accel_magnitude * (w1_sol / norm1_sol)

        final_profile = TwoStageThrustProfile(
            t1=t1_sol,
            t2=0.0,
            a1=a1_sol,
            a2=np.zeros(3),
        )
        final_flight_time = t1_sol

    # 4. Propagate final high-precision worldline
    final_traj = propagate_trajectory_3d(
        r0=r0,
        v0=v0,
        t_span=(0.0, final_flight_time),
        thrust_func=final_profile,
        epoch_jd_tdb=departure_epoch_jd,
        spk=spk,
        gravitational_bodies=gravitational_bodies,
        include_1pn=include_1pn,
        rtol=rtol,
        atol=atol,
    )

    # 5. Evaluate target state at final arrival epoch
    arrival_epoch_jd = departure_epoch_jd + final_flight_time / SEC_PER_DAY
    tgt_final = get_body_barycentric_state(
        tgt_name,
        departure_epoch_jd,
        spk=spk,
        jd_fraction=final_flight_time / SEC_PER_DAY,
    )

    r_final = final_traj.r[-1]
    v_final = final_traj.v[-1]
    pos_err = float(np.linalg.norm(r_final - tgt_final.position))
    vel_err = float(np.linalg.norm(v_final - tgt_final.velocity))

    speeds = np.linalg.norm(final_traj.v, axis=1)
    max_speed = float(np.max(speeds))
    max_beta = max_speed / C_LIGHT

    delta_tau = final_traj.tau[-1]
    time_deficit = final_traj.time_deficit[-1]

    return RendezvousSolution(
        success=bool(opt_sol.success),
        message=str(opt_sol.message),
        trajectory=final_traj,
        departure_epoch_jd=departure_epoch_jd,
        arrival_epoch_jd=arrival_epoch_jd,
        flight_time_seconds=final_flight_time,
        flight_time_days=final_flight_time / SEC_PER_DAY,
        crew_proper_time_seconds=delta_tau,
        crew_proper_time_days=delta_tau / SEC_PER_DAY,
        coordinate_time_deficit_seconds=time_deficit,
        departure_position=r0,
        departure_velocity=v0,
        arrival_position=r_final,
        arrival_velocity=v_final,
        target_arrival_position=tgt_final.position,
        target_arrival_velocity=tgt_final.velocity,
        position_error_m=pos_err,
        velocity_error_mps=vel_err,
        max_speed_mps=max_speed,
        max_beta=max_beta,
        thrust_profile=final_profile,
    )
