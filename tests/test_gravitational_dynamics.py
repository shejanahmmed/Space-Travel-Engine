"""Verification and Benchmark Suite for 3D Relativistic Dynamics and Gravitational Dilation.

Validation Levels:
- Level 1: Flat-space limit vs 1D exact analytical hyperbolic motion & acceleration transforms.
- Level 2: Keplerian two-body orbit energy and angular momentum conservation.
- Level 3: Static gravitational time dilation vs exact general relativistic formula.
- Level 4: Combined SR + GR clock rate in circular orbit vs theoretical (3/2) GM / (c^2 r).
- Level 4B: Real JPL DE440 ephemeris potential evaluation and 1PN solar acceleration magnitude.
- Level 5: Numerical stabilization against catastrophic cancellation at non-relativistic speeds.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    C_LIGHT,
    G0,
    AU,
    SEC_PER_JULIAN_YEAR,
    GM_SUN,
    GM_EARTH,
)
from relativistic_engine.physics.kinematics import (
    distance_from_coordinate_time,
    velocity_from_coordinate_time,
    coordinate_to_proper_time,
)
from relativistic_engine.physics.metrics import (
    proper_time_rate,
    coordinate_time_deficit_rate,
)
from relativistic_engine.physics.dynamics import (
    proper_to_coordinate_acceleration,
    coordinate_to_proper_acceleration,
)
from relativistic_engine.physics.potential import (
    solar_system_potential,
    solar_system_gravitational_acceleration,
)
from relativistic_engine.numerical.trajectory import propagate_trajectory_3d
from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state


# ==============================================================================
# LEVEL 1: ACCELERATION TRANSFORMS & FLAT-SPACE ANALYTICAL BENCHMARK
# ==============================================================================


def test_acceleration_transforms_parallel_and_transverse():
    """Verify 3D acceleration transformation matches SR longitudinal and transverse laws."""
    v_mag = 0.8 * C_LIGHT  # gamma = 1 / sqrt(1 - 0.64) = 1 / 0.6 = 5/3
    gamma = 1.0 / math.sqrt(1.0 - 0.8**2)

    # 1. Purely parallel acceleration
    v_vec = np.array([v_mag, 0.0, 0.0])
    a_prop_par = np.array([10.0, 0.0, 0.0])
    a_coord_par = proper_to_coordinate_acceleration(v_vec, a_prop_par)

    expected_a_par = 10.0 / (gamma**3)
    assert np.isclose(a_coord_par[0], expected_a_par, rtol=1e-14)
    assert np.isclose(a_coord_par[1], 0.0, atol=1e-15)
    assert np.isclose(a_coord_par[2], 0.0, atol=1e-15)

    # 2. Purely transverse acceleration
    a_prop_perp = np.array([0.0, 10.0, 0.0])
    a_coord_perp = proper_to_coordinate_acceleration(v_vec, a_prop_perp)

    expected_a_perp = 10.0 / (gamma**2)
    assert np.isclose(a_coord_perp[0], 0.0, atol=1e-15)
    assert np.isclose(a_coord_perp[1], expected_a_perp, rtol=1e-14)
    assert np.isclose(a_coord_perp[2], 0.0, atol=1e-15)


def test_acceleration_transform_roundtrip_invertibility():
    """Verify proper <-> coordinate acceleration round-trip invertibility."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        # Random velocity with beta in [0.01, 0.99]
        v_dir = rng.normal(size=3)
        v_dir /= np.linalg.norm(v_dir)
        beta = rng.uniform(0.01, 0.99)
        v_vec = v_dir * (beta * C_LIGHT)

        # Random proper acceleration
        a_prop = rng.uniform(-100.0, 100.0, size=3)

        a_coord = proper_to_coordinate_acceleration(v_vec, a_prop)
        a_prop_recovered = coordinate_to_proper_acceleration(v_vec, a_coord)

        rel_err = np.linalg.norm(a_prop_recovered - a_prop) / np.linalg.norm(a_prop)
        assert rel_err < 1e-14, f"Invertibility failed: relative error {rel_err:.2e}"


def test_level1_zero_gravity_matches_1d_analytical():
    """Level 1: 3D propagator in zero-g matches exact 1D analytical hyperbolic motion."""
    alpha = G0
    t_flight = SEC_PER_JULIAN_YEAR  # 1 Julian year

    r0 = [0.0, 0.0, 0.0]
    v0 = [0.0, 0.0, 0.0]

    def thrust(t, r, v):
        return np.array([alpha, 0.0, 0.0])

    res = propagate_trajectory_3d(
        r0=r0,
        v0=v0,
        t_span=(0.0, t_flight),
        thrust_func=thrust,
        rtol=1e-12,
        atol=1e-13,
    )

    t_end = res.t[-1]
    x_numerical = res.r[-1, 0]
    v_numerical = res.v[-1, 0]
    tau_numerical = res.tau[-1]

    # Exact analytical benchmarks from kinematics.py
    x_exact = distance_from_coordinate_time(t_end, alpha)
    v_exact = velocity_from_coordinate_time(t_end, alpha)
    tau_exact = coordinate_to_proper_time(t_end, alpha)

    rel_err_x = abs(x_numerical - x_exact) / x_exact
    rel_err_v = abs(v_numerical - v_exact) / v_exact
    rel_err_tau = abs(tau_numerical - tau_exact) / tau_exact

    assert rel_err_x < 1e-11, f"Position error too large: {rel_err_x:.2e}"
    assert rel_err_v < 1e-11, f"Velocity error too large: {rel_err_v:.2e}"
    assert rel_err_tau < 1e-11, f"Proper time error too large: {rel_err_tau:.2e}"


# ==============================================================================
# LEVEL 2: KEPLERIAN TWO-BODY CONSERVATION BENCHMARKS
# ==============================================================================


def test_level2_keplerian_energy_and_momentum_conservation():
    """Level 2: Free-fall in 1/r potential conserves orbital energy and angular momentum."""
    # Circular orbit at 1 AU around 1 Solar mass (Newtonian point-mass)
    r_orbit = AU
    v_circ = math.sqrt(GM_SUN / r_orbit)
    orbital_period = 2.0 * math.pi * math.sqrt(r_orbit**3 / GM_SUN)

    r0 = [r_orbit, 0.0, 0.0]
    v0 = [0.0, v_circ, 0.0]

    # Isolated Newtonian gravity function
    def newton_gravity(r, v, t):
        r_norm = float(np.linalg.norm(r))
        w = GM_SUN / r_norm
        a = - (GM_SUN / (r_norm**3)) * r
        return a, w

    res = propagate_trajectory_3d(
        r0=r0,
        v0=v0,
        t_span=(0.0, orbital_period),
        custom_gravity_func=newton_gravity,
        rtol=1e-12,
        atol=1e-13,
    )

    # Specific orbital energy: E = 0.5 * v^2 - GM / r
    r_norms = np.linalg.norm(res.r, axis=1)
    v_norms = np.linalg.norm(res.v, axis=1)
    energies = 0.5 * v_norms**2 - GM_SUN / r_norms
    e0 = energies[0]

    energy_errors = np.abs(energies - e0) / abs(e0)
    max_energy_err = float(np.max(energy_errors))
    assert max_energy_err < 1e-10, f"Orbital energy drift too large: {max_energy_err:.2e}"

    # Specific angular momentum: h = r x v
    h_vectors = np.cross(res.r, res.v)
    h0 = h_vectors[0]
    h_errors = np.linalg.norm(h_vectors - h0, axis=1) / np.linalg.norm(h0)
    max_h_err = float(np.max(h_errors))
    assert max_h_err < 1e-10, f"Angular momentum drift too large: {max_h_err:.2e}"

    # Orbit closure after 1 full period
    r_final_err = np.linalg.norm(res.r[-1] - res.r[0]) / r_orbit
    assert r_final_err < 1e-9, f"Orbit closure error too large: {r_final_err:.2e}"


# ==============================================================================
# LEVEL 3: STATIC GRAVITATIONAL TIME DILATION BENCHMARK
# ==============================================================================


def test_level3_static_gravitational_time_dilation():
    """Level 3: Stationary clock in potential w matches exact Schwarzschild dilation."""
    r_test = AU
    duration = 1.0e7  # 10 million seconds (~115 days)

    r0 = [r_test, 0.0, 0.0]
    v0 = [0.0, 0.0, 0.0]

    # Custom static gravity field with no thrust and fixed clock (zero coordinate acceleration)
    def static_clock(r, v, t):
        w = GM_SUN / r_test
        # Fixed clock held stationary (e.g. by rocket hover)
        return np.zeros(3), w

    res = propagate_trajectory_3d(
        r0=r0,
        v0=v0,
        t_span=(0.0, duration),
        custom_gravity_func=static_clock,
        rtol=1e-12,
        atol=1e-13,
    )

    integrated_deficit = res.time_deficit[-1]

    # Exact GR rate: d(t - tau)/dt = 1 - sqrt(1 - 2*GM / (c^2 * r))
    phi = 2.0 * GM_SUN / (C_LIGHT**2 * r_test)
    exact_rate = phi / (1.0 + math.sqrt(1.0 - phi))
    expected_deficit = exact_rate * duration

    rel_err = abs(integrated_deficit - expected_deficit) / expected_deficit
    assert rel_err < 1e-12, f"Static gravitational dilation error: {rel_err:.2e}"


def test_level3_einstein_perihelion_precession():
    """Level 3: 1PN solar acceleration matches Einstein's perihelion advance formula."""
    from scipy.integrate import solve_ivp

    # Mercury orbital parameters
    a_merc = 0.38709893 * AU
    e_merc = 0.20563069
    c_sq = C_LIGHT**2

    # Einstein's theoretical shift: d_omega = 6*pi*GM / (c^2 * a * (1 - e^2))
    d_omega_theory = (6.0 * math.pi * GM_SUN) / (c_sq * a_merc * (1.0 - e_merc**2))

    r_p = a_merc * (1.0 - e_merc)
    v_p = math.sqrt(GM_SUN / a_merc * (1.0 + e_merc) / (1.0 - e_merc))
    t_period = 2.0 * math.pi * math.sqrt(a_merc**3 / GM_SUN)

    def eom_1pn(t, y):
        r = y[0:3]
        v = y[3:6]
        r_mag = np.linalg.norm(r)
        v_mag_sq = np.dot(v, v)
        r_dot_v = np.dot(r, v)
        a_newt = -(GM_SUN / (r_mag**3)) * r
        pref = GM_SUN / (c_sq * r_mag**3)
        term_r = (4.0 * GM_SUN / r_mag - v_mag_sq) * r
        term_v = (4.0 * r_dot_v) * v
        return np.concatenate([v, a_newt + pref * (term_r + term_v)])

    def perihelion_event(t, y):
        if t < 0.5 * t_period:
            return -1.0
        return float(np.dot(y[0:3], y[3:6]))

    perihelion_event.direction = 1

    y0 = np.array([r_p, 0.0, 0.0, 0.0, v_p, 0.0])
    sol = solve_ivp(
        eom_1pn,
        (0.0, 1.5 * t_period),
        y0,
        method="DOP853",
        rtol=1e-13,
        atol=1e-14,
        events=perihelion_event,
    )

    assert len(sol.t_events[0]) > 0, "Perihelion passage event not detected"
    y_peri = sol.y_events[0][0]
    theta_peri = math.atan2(y_peri[1], y_peri[0])

    rel_err = abs(theta_peri - d_omega_theory) / d_omega_theory
    # Matches theoretical Einstein shift to within 5 parts per million
    assert rel_err < 5e-6, f"1PN perihelion advance error: {rel_err:.2e}"


# ==============================================================================
# LEVEL 4: CIRCULAR ORBIT COMBINED SR + GR CLOCK RATE
# ==============================================================================


def test_level4_circular_orbit_clock_rate_sr_plus_gr():
    """Level 4: Circular orbit clock combines potential (-GM/r) and kinematic (-v^2/2) dilation."""
    r_orbit = AU
    v_circ = math.sqrt(GM_SUN / r_orbit)
    orbital_period = 2.0 * math.pi * math.sqrt(r_orbit**3 / GM_SUN)

    r0 = [r_orbit, 0.0, 0.0]
    v0 = [0.0, v_circ, 0.0]

    def newton_gravity(r, v, t):
        r_norm = float(np.linalg.norm(r))
        w = GM_SUN / r_norm
        a = - (GM_SUN / (r_norm**3)) * r
        return a, w

    res = propagate_trajectory_3d(
        r0=r0,
        v0=v0,
        t_span=(0.0, orbital_period),
        custom_gravity_func=newton_gravity,
        rtol=1e-12,
        atol=1e-13,
    )

    integrated_deficit = res.time_deficit[-1]

    # Theoretical rate:
    # d(t - tau)/dt = (w/c^2) + (v^2 / (2 c^2)) + O(c^-4)
    # Since v^2 = GM/r = w, total rate is (3/2) * GM / (c^2 * r) + higher orders
    # Let's compute the exact BCRS rate at r_orbit, v_circ:
    exact_rate = coordinate_time_deficit_rate([0.0, v_circ, 0.0], GM_SUN / r_orbit)
    expected_deficit = exact_rate * orbital_period

    rel_err = abs(integrated_deficit - expected_deficit) / expected_deficit
    assert rel_err < 1e-10, f"Circular orbit clock rate error: {rel_err:.2e}"

    # Verify fractional rate is ~ 1.48e-8 (standard solar system twin paradox value)
    fractional_rate = integrated_deficit / orbital_period
    assert np.isclose(fractional_rate, 1.5 * GM_SUN / (C_LIGHT**2 * r_orbit), rtol=1e-7)


def test_level4b_real_ephemeris_potential_and_1pn_acceleration():
    """Level 4B: Validate solar system potential and 1PN acceleration on Earth at J2000.0."""
    jd_j2000 = 2451545.0
    earth_state = get_body_barycentric_state("earth", jd_j2000)

    # Potential at Earth's center from other bodies (Sun, Jupiter, etc.)
    w_pot = solar_system_potential(earth_state.position, jd_j2000, bodies=["sun", "jupiter"])

    # Dominant term is the Sun: GM_sun / r_sun
    sun_state = get_body_barycentric_state("sun", jd_j2000)
    dist_sun = float(np.linalg.norm(earth_state.position - sun_state.position))
    w_sun_expected = GM_SUN / dist_sun

    # Must match to within 0.1% (Jupiter adds ~ 10^-4 contribution)
    assert np.isclose(w_pot, w_sun_expected, rtol=1e-3)
    # Gravitational potential depth w/c^2 at 1 AU must be ~ 9.87e-9
    assert np.isclose(w_pot / (C_LIGHT**2), 9.87e-9, rtol=5e-2)

    # 1PN solar acceleration comparison
    a_with_1pn = solar_system_gravitational_acceleration(
        earth_state.position,
        earth_state.velocity,
        jd_j2000,
        bodies=["sun"],
        include_1pn=True,
    )
    a_without_1pn = solar_system_gravitational_acceleration(
        earth_state.position,
        earth_state.velocity,
        jd_j2000,
        bodies=["sun"],
        include_1pn=False,
    )

    delta_a_1pn = np.linalg.norm(a_with_1pn - a_without_1pn)
    a_newton_mag = np.linalg.norm(a_without_1pn)

    # 1PN correction in Earth's orbit is on the order of (v/c)^2 ~ 10^-8 of Newtonian gravity
    ratio_1pn = delta_a_1pn / a_newton_mag
    assert 1e-9 < ratio_1pn < 1e-7, f"1PN acceleration ratio out of expected range: {ratio_1pn:.2e}"


# ==============================================================================
# LEVEL 5: NUMERICAL STABILITY AT NON-RELATIVISTIC SPEEDS
# ==============================================================================


def test_level5_stabilized_time_deficit_underflow_avoidance():
    """Level 5: Stabilized rate avoids float64 underflow/cancellation at human walking speed."""
    v_walk = 1.0  # 1 m/s (beta ~ 3.3e-9, beta^2 / 2 ~ 5.56e-18)
    w_zero = 0.0

    # Direct naive formula loses all precision in float64
    naive_beta_sq = (v_walk / C_LIGHT) ** 2
    naive_rate = 1.0 - math.sqrt(1.0 - naive_beta_sq)
    assert naive_rate == 0.0, "Naive float64 evaluation surprisingly did not underflow"

    # Stabilized formula preserves full precision
    stabilized_rate = coordinate_time_deficit_rate([v_walk, 0.0, 0.0], w_zero)
    expected_rate = 0.5 * (v_walk / C_LIGHT) ** 2

    rel_err = abs(stabilized_rate - expected_rate) / expected_rate
    assert rel_err < 1e-14, f"Stabilized rate failed: relative error {rel_err:.2e}"
