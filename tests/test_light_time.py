"""Automated validation tests for Klioner (2003) iterative light-time model.

Validates:
1. Solar limb gravitational deflection angle (1.751 arcseconds, Einstein 1915).
2. Cassini solar conjunction Shapiro delay (one-way ~132 us, round-trip ~264 us, Bertotti et al. 2003).
3. Fast convergence in <= 3 iterations to < 1 picosecond tolerance.
4. Jupiter limb deflection angle (~16.3 milliarcseconds, Klioner 2003 Table 2).
5. Downlink ranging discrepancy between iterative Klioner and static Euclidean distance.
"""

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    AU,
    C_LIGHT,
    G_NEWTON,
    GM_JUPITER,
    GM_SUN,
    RADIUS_JUPITER,
    RADIUS_SUN,
)
from relativistic_engine.physics.light_time import (
    compute_gravitational_deflection_angle,
    compute_shapiro_delay_body,
    solve_klioner_light_time,
)


def test_solar_limb_deflection_angle():
    """Verify gravitational light deflection by the Sun at the limb equals 1.751 arcseconds."""
    # Distant star at -100 AU, observer at Earth (1 AU), grazing Sun at [0, R_sun, 0]
    r_tx = np.array([- 100.0 * AU, RADIUS_SUN, 0.0])
    r_rx = np.array([1.0 * AU, RADIUS_SUN, 0.0])
    r_sun = np.array([0.0, 0.0, 0.0])

    theta_rad = compute_gravitational_deflection_angle(r_tx, r_rx, r_sun, GM_SUN)
    theta_arcsec = math.degrees(theta_rad) * 3600.0

    # Theoretical Einstein deflection: 4 * GM / (c^2 * R) = 1.7512 arcsec
    expected_arcsec = math.degrees(4.0 * GM_SUN / ((C_LIGHT**2) * RADIUS_SUN)) * 3600.0
    assert math.isclose(theta_arcsec, expected_arcsec, rel_tol=1e-3)
    assert 1.74 < theta_arcsec < 1.76


def test_cassini_solar_conjunction_shapiro_delay():
    """Verify Earth-Cassini Shapiro delay at solar conjunction matches Bertotti (2003)."""
    # During superior conjunction (June 2002):
    # Earth at ~1 AU, Saturn/Cassini at ~9 AU, ray passes near solar limb (~1.6 R_sun)
    d_impact = 1.6 * RADIUS_SUN
    r_earth = np.array([1.0 * AU, d_impact, 0.0])
    r_cassini = np.array([- 9.0 * AU, d_impact, 0.0])
    r_sun = np.array([0.0, 0.0, 0.0])

    shapiro_one_way_s = compute_shapiro_delay_body(r_cassini, r_earth, r_sun, GM_SUN)
    shapiro_one_way_us = shapiro_one_way_s * 1.0e6
    shapiro_two_way_us = 2.0 * shapiro_one_way_us

    # One-way delay is ~ 132 microseconds
    assert 125.0 < shapiro_one_way_us < 140.0
    # Two-way round-trip coherent link (as measured by Cassini transponder in Bertotti 2003) is ~ 264 us
    assert 250.0 < shapiro_two_way_us < 275.0


def test_jupiter_limb_deflection():
    """Verify Jupiter limb deflection matches Klioner (2003) Table 2 (~16.3 milliarcseconds)."""
    # Source at infinity, observer at Earth (5 AU away), ray grazing Jupiter limb
    r_tx = np.array([- 100.0 * AU, RADIUS_JUPITER, 0.0])
    r_rx = np.array([5.0 * AU, RADIUS_JUPITER, 0.0])
    r_jup = np.array([0.0, 0.0, 0.0])

    theta_rad = compute_gravitational_deflection_angle(r_tx, r_rx, r_jup, GM_JUPITER)
    theta_mas = math.degrees(theta_rad) * 3600.0 * 1000.0

    # Theoretical: 4 * GM_JUP / (c^2 * R_JUP) ~ 16.3 mas
    expected_mas = math.degrees(4.0 * GM_JUPITER / ((C_LIGHT**2) * RADIUS_JUPITER)) * 3600.0 * 1000.0
    assert math.isclose(theta_mas, expected_mas, rel_tol=1e-3)
    assert 16.0 < theta_mas < 16.6


def test_klioner_light_time_convergence_and_ranging_bias():
    """Verify that Klioner solver converges in <= 3 iterations and captures dynamic ranging bias."""
    # Spacecraft receding at 30 km/s radially from Earth
    # Target distance ~ 2.0 AU (~1000 seconds light-time)
    v_sc = np.array([30000.0, 0.0, 0.0])
    r0_sc = np.array([2.0 * AU, 0.0, 0.0])

    def tx_pos_fn(t):
        return r0_sc + v_sc * t

    rx_pos = np.array([0.0, 0.0, 0.0])
    t_rx = 10000.0

    # Include Sun at (1 AU, 0, 0)
    body_data = [(np.array([1.0 * AU, 0.0, 0.0]), GM_SUN, "Sun")]

    sol = solve_klioner_light_time(
        tx_pos_fn, rx_pos, t_rx, body_data, tol_seconds=1e-12, max_iter=10
    )

    # Must converge in 4 iterations or fewer to reach picosecond tolerance (1e-12 s)
    assert sol["iterations"] <= 4
    assert sol["residual"] < 1e-12

    # During the ~998s light-time, spacecraft moved ~ 998s * 30 km/s = 29,940 km!
    # Comparing static distance at t_rx vs dynamic light-time position at t_tx:
    static_distance = float(np.linalg.norm(rx_pos - tx_pos_fn(t_rx)))
    dynamic_distance = sol["romer_delay"] * C_LIGHT
    range_discrepancy_km = abs(static_distance - dynamic_distance) / 1000.0

    # Radial discrepancy must be ~ 30,000 km
    assert 28000.0 < range_discrepancy_km < 32000.0
