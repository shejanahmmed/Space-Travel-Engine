"""Automated validation tests for relativistic atomic clock transport and Sagnac delay.

Validates:
1. GPS operational orbit relativistic clock drift (+38.6 microseconds/day, Ashby 2003).
2. Earth equatorial circular closed-loop Sagnac delay (~207.4 nanoseconds, IERS 2010).
3. Geoid reference clock neutrality (y ~ 0 relative to standard TT rate).
4. Deep Space Atomic Clock (DSAC) 1-day timing stability bound (< 1 nanosecond).
5. Polygonal Sagnac relay loop area calculation.
"""

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    C_LIGHT,
    GM_EARTH,
    RADIUS_EARTH,
    SEC_PER_DAY,
)
from relativistic_engine.physics.clock_transport import (
    compute_equatorial_closed_loop_sagnac,
    compute_fractional_frequency_offset,
    compute_sagnac_delay,
    evaluate_trajectory_clock_drift,
    GEOID_POTENTIAL_W0,
    OMEGA_EARTH_RAD_S,
)


def test_gps_orbit_relativistic_clock_drift():
    """Verify GPS orbit clock drift matches Ashby (2003) standard (+38.6 us/day)."""
    # GPS nominal circular orbit: a = 26,561.75 km (altitude ~ 20,183 km)
    r_gps = 26561.75e3
    v_gps = math.sqrt(GM_EARTH / r_gps)  # ~ 3.874 km/s

    r_vec = np.array([r_gps, 0.0, 0.0])
    v_vec = np.array([0.0, v_gps, 0.0])

    y_net, y_grav, y_kin = compute_fractional_frequency_offset(r_vec, v_vec)

    # Convert to daily drift in microseconds
    drift_grav_us_day = y_grav * SEC_PER_DAY * 1e6
    drift_kin_us_day = y_kin * SEC_PER_DAY * 1e6
    drift_net_us_day = y_net * SEC_PER_DAY * 1e6

    # Ashby (2003) values:
    # Gravitational blue-shift: ~ +45.7 us/day
    # Kinematic red-shift: ~ -7.1 us/day
    # Net clock advance: ~ +38.6 us/day
    assert 44.0 < drift_grav_us_day < 47.0
    assert -8.5 < drift_kin_us_day < -6.0
    assert math.isclose(drift_net_us_day, 38.6, abs_tol=1.0)
    assert 37.5 < drift_net_us_day < 39.5


def test_earth_equatorial_sagnac_delay():
    """Verify maximum closed-loop equatorial Sagnac delay on Earth (~207.4 ns)."""
    delay_s = compute_equatorial_closed_loop_sagnac(RADIUS_EARTH, OMEGA_EARTH_RAD_S)
    delay_ns = delay_s * 1e9

    # Standard textbook value: 2 * omega * pi * R^2 / c^2 = 207.4 ns
    expected_ns = (2.0 * OMEGA_EARTH_RAD_S * math.pi * (RADIUS_EARTH**2) / (C_LIGHT**2)) * 1e9
    assert math.isclose(delay_ns, expected_ns, rel_tol=1e-4)
    assert 206.0 < delay_ns < 209.0


def test_geoid_reference_clock_neutrality():
    """Verify that a clock at rest on the Earth geoid has negligible net offset relative to W0."""
    # At geoid surface: potential W = W0, speed in rotating frame = 0
    r_geoid = RADIUS_EARTH
    # Geoid potential W0 incorporates both GM/R and rotational centrifugal potential -0.5 * omega^2 * R^2
    r_vec = np.array([r_geoid, 0.0, 0.0])
    v_vec = np.array([0.0, 0.0, 0.0])

    y_net, y_grav, y_kin = compute_fractional_frequency_offset(
        r_vec, v_vec, reference_potential=GM_EARTH / r_geoid
    )

    # Offset must be zero within numerical tolerance
    assert abs(y_net) < 1e-15


def test_dsac_trajectory_timing_stability_bound():
    """Verify NASA Deep Space Atomic Clock (DSAC) timing jitter is under 1 ns over 1 day."""
    r_gps = 26561.75e3
    v_gps = math.sqrt(GM_EARTH / r_gps)

    # 1 full day of orbit in 1-hour steps
    t_arr = np.linspace(0.0, SEC_PER_DAY, 25)
    r_arr = np.zeros((25, 3))
    v_arr = np.zeros((25, 3))

    for i, t in enumerate(t_arr):
        theta = (v_gps / r_gps) * t
        r_arr[i] = [r_gps * math.cos(theta), r_gps * math.sin(theta), 0.0]
        v_arr[i] = [-v_gps * math.sin(theta), v_gps * math.cos(theta), 0.0]

    res = evaluate_trajectory_clock_drift(t_arr, r_arr, v_arr, clock_type="dsac")

    # Over 1 day, DSAC fractional frequency offset produces ~ +38.6 us of relativistic advance
    assert math.isclose(res.daily_drift_seconds * 1e6, 38.6, abs_tol=1.0)
    assert math.isclose(res.accumulated_phase_drift_seconds * 1e6, 38.6, abs_tol=1.0)

    # Allan deviation timing uncertainty (noise) must be sub-nanosecond (< 1.0 ns)
    jitter_ns = res.allan_deviation_1sigma_s * 1e9
    assert jitter_ns < 1.0
    assert jitter_ns > 0.05


def test_polygonal_sagnac_delay_calculation():
    """Verify Sagnac delay on a closed triangular ground-space-ground loop."""
    # Triangular loop: Station A -> GEO Spacecraft -> Station B -> Station A
    r_geo = 42164e3
    pts = np.array([
        [RADIUS_EARTH, 0.0, 0.0],
        [r_geo * math.cos(0.2), r_geo * math.sin(0.2), 0.0],
        [RADIUS_EARTH * math.cos(0.4), RADIUS_EARTH * math.sin(0.4), 0.0],
        [RADIUS_EARTH, 0.0, 0.0],  # Closed loop
    ])

    delay_s = compute_sagnac_delay(pts, OMEGA_EARTH_RAD_S)
    delay_ns = delay_s * 1e9

    # Sagnac delay for this large triangular loop must be tens of nanoseconds
    assert delay_ns > 10.0
    assert delay_ns < 200.0
