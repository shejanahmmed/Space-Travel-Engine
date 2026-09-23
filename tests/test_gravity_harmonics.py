"""Automated validation tests for spherical harmonic zonal gravity fields (J2-J8).

Validates:
1. Exact Legendre polynomial evaluations and polar boundary limits.
2. Earth LEO J2 perturbation magnitude matching EGM2008 standard.
3. Analytical secular nodal regression rate (Kaula 1966).
4. Jupiter Juno gravity field hierarchy (J2 >> J4 >> J6 >> J8).
5. Gravitational oblateness asymmetry (g_pole > g_equator).
"""

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    GM_EARTH,
    RADIUS_EARTH,
)
from relativistic_engine.physics.gravity_harmonics import (
    compute_planetary_gravity_acceleration,
    compute_zonal_gravity_acceleration,
    legendre_polynomials_and_derivatives,
    secular_j2_nodal_precession_rate,
    PLANETARY_ZONAL_COEFFICIENTS,
)


def test_legendre_polynomial_exact_values():
    """Verify Legendre polynomials and derivatives against analytical expressions."""
    u_vals = [0.0, 0.5, -0.70710678, 1.0, -1.0]

    for u in u_vals:
        P, P_prime = legendre_polynomials_and_derivatives(8, u)

        # P0(u) = 1, P1(u) = u
        assert math.isclose(P[0], 1.0, abs_tol=1e-15)
        assert math.isclose(P[1], u, abs_tol=1e-15)

        # P2(u) = 0.5 * (3*u^2 - 1), P2'(u) = 3*u
        expected_p2 = 0.5 * (3.0 * u * u - 1.0)
        expected_p2_prime = 3.0 * u
        assert math.isclose(P[2], expected_p2, abs_tol=1e-14)
        assert math.isclose(P_prime[2], expected_p2_prime, abs_tol=1e-13)

        # P4(u) = 1/8 * (35*u^4 - 30*u^2 + 3)
        expected_p4 = 0.125 * (35.0 * (u**4) - 30.0 * (u**2) + 3.0)
        assert math.isclose(P[4], expected_p4, abs_tol=1e-14)


def test_earth_leo_j2_acceleration():
    """Verify Earth J2 perturbation at 400 km LEO altitude."""
    r_leo = RADIUS_EARTH + 400e3  # 6778.137 km
    pos_equator = np.array([r_leo, 0.0, 0.0])

    res = compute_planetary_gravity_acceleration("Earth", pos_equator, max_degree=2)
    a_point = res["point_mass"]
    a_zonal = res["zonal"]

    g_point = float(np.linalg.norm(a_point))  # GM / r^2 ~ 8.69 m/s^2
    g_zonal = float(np.linalg.norm(a_zonal))  # J2 perturbation ~ 1.5 * J2 * (R/r)^2 * g_point ~ 0.012 m/s^2

    assert 8.5 < g_point < 9.0
    assert 0.005 < g_zonal < 0.02
    # J2 to point-mass ratio is ~ 1.5 * 1.08e-3 ~ 1.5e-3
    ratio = g_zonal / g_point
    assert 1e-3 < ratio < 2e-3


def test_kaula_nodal_precession_rate():
    """Verify Kaula's analytical secular nodal precession rate for ISS orbit."""
    # ISS orbit: a ~ 6778 km, e ~ 0.0005, i = 51.6 degrees
    a_iss = 6778.137e3
    e_iss = 0.0005
    inc_rad = math.radians(51.6)

    rate_rad_s = secular_j2_nodal_precession_rate(a_iss, e_iss, inc_rad, body_name="Earth")
    rate_deg_day = math.degrees(rate_rad_s) * 86400.0

    # ISS nodal precession is well-known to be ~ -5.0 degrees/day
    assert -5.5 < rate_deg_day < -4.5


def test_jupiter_juno_gravity_multipole_hierarchy():
    """Verify Juno J2 through J8 multipole hierarchy around Jupiter."""
    r_jovian = 75000e3  # Just above cloud tops (r_ref ~ 71492 km)
    pos = np.array([r_jovian / math.sqrt(2), 0.0, r_jovian / math.sqrt(2)])

    res_j2 = compute_planetary_gravity_acceleration("Jupiter", pos, max_degree=2)["zonal"]
    res_j4 = compute_planetary_gravity_acceleration("Jupiter", pos, max_degree=4)["zonal"]
    res_j6 = compute_planetary_gravity_acceleration("Jupiter", pos, max_degree=6)["zonal"]
    res_j8 = compute_planetary_gravity_acceleration("Jupiter", pos, max_degree=8)["zonal"]

    diff_j4_j2 = float(np.linalg.norm(res_j4 - res_j2))
    diff_j6_j4 = float(np.linalg.norm(res_j6 - res_j4))
    diff_j8_j6 = float(np.linalg.norm(res_j8 - res_j6))

    # Hierarchy: J2 >> J4 >> J6 >> J8
    assert float(np.linalg.norm(res_j2)) > diff_j4_j2 > diff_j6_j4 > diff_j8_j6
    assert diff_j8_j6 > 0.0  # J8 contribution is resolved and non-zero


def test_oblate_gravity_asymmetry():
    """Verify that effective gravity is stronger at the poles than at the equator due to J2."""
    r = RADIUS_EARTH
    pos_pole = np.array([0.0, 0.0, r])
    pos_eq = np.array([r, 0.0, 0.0])

    g_pole = float(np.linalg.norm(compute_planetary_gravity_acceleration("Earth", pos_pole)["total"]))
    g_eq = float(np.linalg.norm(compute_planetary_gravity_acceleration("Earth", pos_eq)["total"]))

    # Due to equatorial bulge (J2 > 0), gravity at the poles is stronger than at the equator
    assert g_pole > g_eq
    diff = g_pole - g_eq
    assert 0.02 < diff < 0.08  # ~ 0.03 m/s^2 J2 gravitational asymmetry
