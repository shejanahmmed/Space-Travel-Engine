"""Verification and Benchmark Suite for Interplanetary Relativistic Rendezvous.

Validation Levels:
- Level 1: Flat-space moving target rendezvous matching analytical kinematics.
- Level 2: Real NASA/JPL DE440 Earth-to-Mars 2030 rendezvous mission at 1g.
- Level 3: Earth-to-Mars at 0.1g scaling analysis.
- Level 4: Relativistic time dilation budget (crew proper time vs Earth coordinate time).
- Level 5: Intercept (flyby) mode vs soft rendezvous comparison.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    C_LIGHT,
    G0,
    AU,
    SEC_PER_DAY,
)
from relativistic_engine.time.time_scales import datetime_to_jd
from relativistic_engine.trajectory.rendezvous import (
    RendezvousMode,
    RendezvousSolution,
    solve_interplanetary_rendezvous,
)
from relativistic_engine.trajectory.profiles import TwoStageThrustProfile
from relativistic_engine.numerical.trajectory import propagate_trajectory_3d


def test_level1_flat_space_moving_target_rendezvous():
    """Level 1: 3D rendezvous solver against moving target in flat Minkowski space."""
    r_target_0 = np.array([1.0e9, 0.0, 0.0])
    v_target_0 = np.array([0.0, 20000.0, 0.0])

    r0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.0])
    alpha = G0

    t_half_guess = math.sqrt(1.0e9 / alpha)
    w1_init = alpha * t_half_guess * np.array([1.0, 0.0, 0.0])
    w2_init = -alpha * t_half_guess * np.array([1.0, 0.0, 0.0])
    p0 = np.concatenate([w1_init, w2_init])

    r_scale = 1.0e6
    v_scale = 1.0e3

    from scipy.optimize import root

    def obj(p):
        w1 = p[0:3]
        w2 = p[3:6]
        norm1 = float(np.linalg.norm(w1))
        norm2 = float(np.linalg.norm(w2))
        t1 = norm1 / alpha
        t2 = norm2 / alpha
        profile = TwoStageThrustProfile(t1=t1, t2=t2, a1=alpha * (w1 / norm1), a2=alpha * (w2 / norm2))
        T = t1 + t2
        res = propagate_trajectory_3d(r0, v0, (0.0, T), thrust_func=profile, rtol=1e-9, atol=1e-10)
        r_tgt = r_target_0 + v_target_0 * T
        v_tgt = v_target_0
        return np.concatenate([(res.r[-1] - r_tgt) / r_scale, (res.v[-1] - v_tgt) / v_scale])

    sol = root(obj, p0, method="hybr")
    assert sol.success, f"Solver failed: {sol.message}"

    w1_sol = sol.x[0:3]
    w2_sol = sol.x[3:6]
    t1 = np.linalg.norm(w1_sol) / alpha
    t2 = np.linalg.norm(w2_sol) / alpha
    T_final = t1 + t2

    final_profile = TwoStageThrustProfile(
        t1=t1,
        t2=t2,
        a1=alpha * (w1_sol / np.linalg.norm(w1_sol)),
        a2=alpha * (w2_sol / np.linalg.norm(w2_sol)),
    )
    res = propagate_trajectory_3d(r0, v0, (0.0, T_final), thrust_func=final_profile, rtol=1e-10, atol=1e-11)

    r_tgt_final = r_target_0 + v_target_0 * T_final
    v_tgt_final = v_target_0

    pos_err = np.linalg.norm(res.r[-1] - r_tgt_final)
    vel_err = np.linalg.norm(res.v[-1] - v_tgt_final)

    assert pos_err < 100.0, f"Position error {pos_err:.2f} m exceeds 100 m"
    assert vel_err < 0.1, f"Velocity error {vel_err:.4f} m/s exceeds 0.1 m/s"


def test_level2_earth_mars_2030_rendezvous_1g():
    """Level 2: Earth to Mars rendezvous departing May 1, 2030 at 1g continuous thrust."""
    jd_dep = datetime_to_jd(2030, 5, 1, 0, 0, 0.0)

    sol = solve_interplanetary_rendezvous(
        departure_body="earth",
        target_body="mars",
        departure_epoch_jd=jd_dep,
        accel_magnitude=G0,
        mode=RendezvousMode.SOFT_RENDEZVOUS,
        gravitational_bodies=["sun"],
        rtol=1e-7,
        atol=1e-8,
        max_fev=60,
    )

    assert sol.success, f"Rendezvous solver failed to converge: {sol.message}"

    # Flight time at 1g for ~370 M km must be roughly 4 to 5 days
    assert 3.5 < sol.flight_time_days < 6.0, f"Unexpected flight time: {sol.flight_time_days:.2f} days"

    # Residual errors: miss distance < 100 km across 370 million km; vel error < 1 m/s
    assert sol.position_error_m < 100000.0, f"Position error too large: {sol.position_error_m:.1f} m"
    assert sol.velocity_error_mps < 1.0, f"Velocity error too large: {sol.velocity_error_mps:.3f} m/s"

    # Maximum velocity: v_max ~ 1000 - 2500 km/s (beta ~ 0.005 - 0.008)
    assert 1.0e6 < sol.max_speed_mps < 3.0e6, f"Peak speed out of bounds: {sol.max_speed_mps:.2e} m/s"
    assert 0.003 < sol.max_beta < 0.010, f"Max beta out of bounds: {sol.max_beta:.4f}"

    # Relativistic time dilation:
    # Crew proper time must be strictly less than Earth coordinate time
    assert sol.crew_proper_time_seconds < sol.flight_time_seconds
    assert sol.coordinate_time_deficit_seconds > 0.0

    # At beta ~ 0.006, beta^2 / 2 ~ 1.8e-5. Over 4.5 days (~3.9e5 s), deficit ~ 2 - 10 seconds
    assert 2.0 < sol.coordinate_time_deficit_seconds < 15.0, (
        f"Time deficit unexpected: {sol.coordinate_time_deficit_seconds:.3f} s"
    )


def test_level3_earth_mars_2030_rendezvous_scaling_0_1g():
    """Level 3: Verify T ~ 1/sqrt(alpha) scaling for Earth to Mars rendezvous at 0.1g."""
    jd_dep = datetime_to_jd(2030, 5, 1, 0, 0, 0.0)

    sol_0_1g = solve_interplanetary_rendezvous(
        departure_body="earth",
        target_body="mars",
        departure_epoch_jd=jd_dep,
        accel_magnitude=0.1 * G0,
        mode=RendezvousMode.SOFT_RENDEZVOUS,
        gravitational_bodies=["sun"],
        rtol=1e-7,
        atol=1e-8,
        max_fev=60,
    )

    assert sol_0_1g.success, f"0.1g rendezvous solver failed: {sol_0_1g.message}"

    # At 0.1g, flight time scales as roughly sqrt(10) * 4.5 days ~ 14 days
    assert 10.0 < sol_0_1g.flight_time_days < 20.0, (
        f"Flight time at 0.1g unexpected: {sol_0_1g.flight_time_days:.2f} days"
    )
    assert sol_0_1g.position_error_m < 500000.0, f"Position error too large: {sol_0_1g.position_error_m:.1f} m"
    assert sol_0_1g.velocity_error_mps < 2.0, f"Velocity error too large: {sol_0_1g.velocity_error_mps:.3f} m/s"


def test_level5_intercept_mode_comparison():
    """Level 5: Verify intercept mode matches arrival position with single continuous burn."""
    jd_dep = datetime_to_jd(2030, 5, 1, 0, 0, 0.0)

    sol_intercept = solve_interplanetary_rendezvous(
        departure_body="earth",
        target_body="mars",
        departure_epoch_jd=jd_dep,
        accel_magnitude=G0,
        mode=RendezvousMode.INTERCEPT,
        gravitational_bodies=["sun"],
        rtol=1e-7,
        atol=1e-8,
        max_fev=50,
    )

    assert sol_intercept.success, f"Intercept solver failed: {sol_intercept.message}"
    assert sol_intercept.position_error_m < 100000.0
    # Flyby flight time is shorter than rendezvous because no braking stage
    assert 2.5 < sol_intercept.flight_time_days < 4.0
