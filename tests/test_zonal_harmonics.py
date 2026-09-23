"""Verification Suite for Planetary Gravitational Zonal Harmonics (J2 through J4).

Validates:
- Level 1: Exact gradient consistency between analytical Cartesian acceleration a_n
           and central finite-difference gradient of scalar potential w_n across
           Earth, Moon, Mars, Sun, and Jupiter.
- Level 2: Strict conservation of axial angular momentum L_spin = (r x v) . s_pole
           under axisymmetric zonal harmonic fields over multi-orbit integration.
- Level 3: Brouwer (1959) / King-Hele secular nodal regression rate dOmega/dt
           matching numerical orbit propagation for retrograde and prograde inclinations.
- Level 4: Relativistic chronometric clock frequency shift from zonal quadrupole potential.
- Level 5: Full trajectory propagation with NASA/JPL DE440 ephemerides and zonal harmonics.

Authoritative Standards:
- EGM2008 / IERS Conventions (2010): Earth gravity field and J2-J4 parameters.
- GRAIL LP165P (Konopliv et al. 2013): Lunar gravitational coefficients.
- GMM-3 / MRO120D (Genova et al. 2016): Martian gravitational coefficients.
- IAU 2015 Resolution B3 / IAU WGCCRE (Archinal et al. 2018): Rotational poles and radii.
- Brouwer, D. (1959), "Solution of the problem of artificial satellite theory without drag", AJ 64:378-397.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from relativistic_engine.constants import (
    C_LIGHT,
    G0,
    GM_EARTH,
    GM_MOON,
    GM_MARS,
    GM_SUN,
    GM_JUPITER,
    SEC_PER_DAY,
)
from relativistic_engine.physics.chronometry import (
    compute_zonal_potential,
    compute_zonal_acceleration,
    compute_j2_potential,
    compute_j2_acceleration,
    coordinate_time_deficit_rate_full_1pn,
    BODY_SPIN_POLE,
    BODY_EQUATORIAL_RADIUS,
    BODY_J2,
    BODY_J3,
    BODY_J4,
)
from relativistic_engine.physics.potential import (
    solar_system_potential,
    solar_system_gravitational_acceleration,
)
from relativistic_engine.numerical.trajectory import propagate_trajectory_3d


# ==============================================================================
# LEVEL 1: FINITE-DIFFERENCE GRADIENT CONSISTENCY
# ==============================================================================


@pytest.mark.parametrize("body", ["earth", "moon", "mars", "sun", "jupiter"])
@pytest.mark.parametrize("max_deg", [2, 3, 4])
def test_level1_zonal_gradient_matches_finite_difference(body: str, max_deg: int):
    """Level 1: Verify analytical a_zonal matches numerical grad(w_zonal) to < 10^-8."""
    r_eq = BODY_EQUATORIAL_RADIUS[body]
    pole = BODY_SPIN_POLE[body]

    # Test position at 1.5 R_eq with arbitrary off-axis latitude (45 deg)
    # Construct orthogonal basis with pole
    rand_v = np.array([0.312, -0.741, 0.594], dtype=np.float64)
    perp1 = np.cross(pole, rand_v)
    perp1 /= np.linalg.norm(perp1)

    r_test = (1.5 * r_eq) * (math.cos(math.radians(45.0)) * perp1 + math.sin(math.radians(45.0)) * pole)

    # Analytical acceleration
    a_anal = compute_zonal_acceleration(r_test, body=body, max_degree=max_deg)

    # Numerical gradient of w_zonal via 4th-order central finite differences
    # Scale h with radius (h/r ~ 10^-6) to balance roundoff (eps * r / h) and truncation (h^4 / r^4)
    r_norm = float(np.linalg.norm(r_test))
    h = max(5.0, 1.0e-6 * r_norm)
    a_num = np.zeros(3, dtype=np.float64)

    for i in range(3):
        e_i = np.zeros(3, dtype=np.float64)
        e_i[i] = 1.0

        w_p2 = compute_zonal_potential(r_test + 2.0 * h * e_i, body=body, max_degree=max_deg)
        w_p1 = compute_zonal_potential(r_test + 1.0 * h * e_i, body=body, max_degree=max_deg)
        w_m1 = compute_zonal_potential(r_test - 1.0 * h * e_i, body=body, max_degree=max_deg)
        w_m2 = compute_zonal_potential(r_test - 2.0 * h * e_i, body=body, max_degree=max_deg)

        # 4th-order central difference: (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
        a_num[i] = (-w_p2 + 8.0 * w_p1 - 8.0 * w_m1 + w_m2) / (12.0 * h)

    a_mag = float(np.linalg.norm(a_anal))
    if a_mag > 0.0:
        rel_diff = float(np.linalg.norm(a_anal - a_num)) / a_mag
        assert rel_diff < 1.0e-8, f"Gradient mismatch for {body} (deg {max_deg}): rel_diff = {rel_diff:.2e}"


def test_level1_backward_compatibility_j2_wrappers():
    """Verify compute_j2_potential and compute_j2_acceleration match degree 2 exactly."""
    r_test = np.array([7.0e6, -2.0e6, 3.5e6], dtype=np.float64)
    w_j2 = compute_j2_potential(r_test, body="earth")
    w_z2 = compute_zonal_potential(r_test, body="earth", max_degree=2)
    assert np.isclose(w_j2, w_z2, rtol=1e-15, atol=1e-15)

    a_j2 = compute_j2_acceleration(r_test, body="earth")
    a_z2 = compute_zonal_acceleration(r_test, body="earth", max_degree=2)
    assert np.allclose(a_j2, a_z2, rtol=1e-15, atol=1e-15)


# ==============================================================================
# LEVEL 2: CONSERVATION OF AXIAL ANGULAR MOMENTUM (L_spin)
# ==============================================================================


@pytest.mark.parametrize("body,gm,alt_km", [
    ("earth", GM_EARTH, 600.0),
    ("mars", GM_MARS, 400.0),
    ("moon", GM_MOON, 150.0),
])
def test_level2_axial_angular_momentum_conservation(body: str, gm: float, alt_km: float):
    """Level 2: Due to axial symmetry, L . s_pole must be conserved to < 10^-11."""
    r_eq = BODY_EQUATORIAL_RADIUS[body]
    pole = BODY_SPIN_POLE[body]
    r_orbit = r_eq + alt_km * 1000.0
    v_circ = math.sqrt(gm / r_orbit)

    # Construct arbitrary inclined orbit (i = 50 deg)
    # Unit vector perpendicular to pole
    ref = np.array([0.5, 0.2, -0.7], dtype=np.float64)
    p1 = np.cross(pole, ref)
    p1 /= np.linalg.norm(p1)
    p2 = np.cross(pole, p1)

    inc = math.radians(50.0)
    r0 = r_orbit * p1
    # Velocity tilted by inclination relative to equatorial plane (p2 is in-plane, pole is normal)
    v0 = v_circ * (math.cos(inc) * p2 + math.sin(inc) * pole)

    period = 2.0 * math.pi * math.sqrt((r_orbit**3) / gm)
    t_span = (0.0, 3.0 * period)  # 3 full orbits

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        r = y[0:3]
        v = y[3:6]
        r_norm = float(np.linalg.norm(r))
        a_mono = - (gm / (r_norm**3)) * r
        a_zonal = compute_zonal_acceleration(r, body=body, max_degree=4)
        return np.concatenate([v, a_mono + a_zonal])

    y0 = np.concatenate([r0, v0])
    sol = solve_ivp(rhs, t_span, y0, method="DOP853", rtol=1e-12, atol=1e-14)

    # Initial axial angular momentum: (r0 x v0) . pole
    l0_vec = np.cross(r0, v0)
    l0_spin = float(np.dot(l0_vec, pole))

    # Check at final state and intermediate steps
    for k in range(sol.y.shape[1]):
        rk = sol.y[0:3, k]
        vk = sol.y[3:6, k]
        lk_vec = np.cross(rk, vk)
        lk_spin = float(np.dot(lk_vec, pole))
        rel_err = abs(lk_spin - l0_spin) / abs(l0_spin)
        assert rel_err < 1.0e-11, f"L_spin drift {rel_err:.2e} on {body} exceeds tolerance at t={sol.t[k]:.1f}s"


# ==============================================================================
# LEVEL 3: BROUWER / KING-HELE SECULAR NODAL REGRESSION (dOmega/dt)
# ==============================================================================


def test_level3_brouwer_secular_nodal_regression():
    """Level 3: Compare numerical orbit precession against Brouwer (1959) first-order secular rate.

    Brouwer first-order secular rate of the right ascension of the ascending node:
        dOmega/dt = - (3/2) * J2 * (R_eq / p)^2 * n * cos(i)
    where p = a(1 - e^2), n = sqrt(GM / a^3), and i is orbit inclination.
    """
    body = "earth"
    gm = GM_EARTH
    r_eq = BODY_EQUATORIAL_RADIUS[body]
    j2 = BODY_J2[body]

    # Circular Low Earth Orbit (altitude 600 km, inc = 60 deg)
    alt = 600.0e3
    a_orbit = r_eq + alt
    n_mean = math.sqrt(gm / (a_orbit**3))
    inc = math.radians(60.0)
    v_circ = math.sqrt(gm / a_orbit)

    # Analytical Brouwer secular rate (rad / s)
    domega_dt_analytic = - 1.5 * j2 * ((r_eq / a_orbit)**2) * n_mean * math.cos(inc)

    # Initial position: ascending node crossing along X-axis
    r0 = np.array([a_orbit, 0.0, 0.0], dtype=np.float64)
    # Velocity tilted by inclination: vy = v_circ * cos(inc), vz = v_circ * sin(inc)
    v0 = np.array([0.0, v_circ * math.cos(inc), v_circ * math.sin(inc)], dtype=np.float64)

    period = 2.0 * math.pi / n_mean  # ~5800 seconds (~96.6 min)
    # Propagate for 10 orbits to average out short-period oscillations
    num_orbits = 10
    t_final = num_orbits * period
    t_span = (0.0, t_final)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        r = y[0:3]
        v = y[3:6]
        r_norm = float(np.linalg.norm(r))
        a_mono = - (gm / (r_norm**3)) * r
        a_j2 = compute_zonal_acceleration(r, body=body, max_degree=2)
        return np.concatenate([v, a_mono + a_j2])

    y0 = np.concatenate([r0, v0])
    sol = solve_ivp(rhs, t_span, y0, method="DOP853", rtol=1e-12, atol=1e-14, max_step=60.0)

    # Calculate RAAN (Omega) at each ascending node crossing (z crosses 0 from negative to positive)
    # Since node is defined by line of nodes vector N = Z x L = [-Ly, Lx, 0]:
    # Omega = atan2(N_y, N_x) = atan2(Lx, -Ly)
    r_final = sol.y[0:3, -1]
    v_final = sol.y[3:6, -1]
    l_final = np.cross(r_final, v_final)
    omega_final = math.atan2(l_final[0], -l_final[1])

    l_init = np.cross(r0, v0)
    omega_init = math.atan2(l_init[0], -l_init[1])

    delta_omega_numeric = omega_final - omega_init
    # Wrap to [-pi, pi]
    delta_omega_numeric = (delta_omega_numeric + math.pi) % (2.0 * math.pi) - math.pi

    expected_delta_omega = domega_dt_analytic * t_final

    # Secular rate must match within 0.5% (difference is due to 2nd-order J2^2 and short-period terms)
    rel_err = abs(delta_omega_numeric - expected_delta_omega) / abs(expected_delta_omega)
    assert rel_err < 5.0e-3, f"Brouwer nodal regression mismatch: numeric {delta_omega_numeric:.6f} vs analytic {expected_delta_omega:.6f} (rel err {rel_err:.2e})"


# ==============================================================================
# LEVEL 4: RELATIVISTIC CHRONOMETRY IN OBLATE GRAVITATIONAL POTENTIAL
# ==============================================================================


def test_level4_zonal_proper_time_dilation_rate():
    """Level 4: Verify proper time clock frequency shifts from quadrupole potential.

    Earth's oblateness creates a potential difference between pole and equator:
        delta_w_J2(R_eq) = w_J2(equator) - w_J2(pole) = (3/2) * (GM/R_eq) * J2
    Leading to a fractional clock frequency shift of:
        delta_nu / nu = - delta_w / c^2 ~ - 1.0e-13.
    """
    r_eq = BODY_EQUATORIAL_RADIUS["earth"]
    pole = BODY_SPIN_POLE["earth"]

    # Position on equator at distance r = R_eq
    r_eq_vec = np.array([r_eq, 0.0, 0.0], dtype=np.float64)
    # Position on pole at distance r = R_eq
    r_pole_vec = r_eq * pole

    w_eq = compute_zonal_potential(r_eq_vec, body="earth", max_degree=2)
    w_pole = compute_zonal_potential(r_pole_vec, body="earth", max_degree=2)

    delta_w = w_eq - w_pole
    expected_delta_w = 1.5 * (GM_EARTH / r_eq) * BODY_J2["earth"]
    assert np.isclose(delta_w, expected_delta_w, rtol=1e-12)

    # Coordinate time deficit rate shift for a clock at rest:
    # d(t - tau)/dt = 1 - sqrt(1 - 2w/c^2) ~ w/c^2
    rate_eq = coordinate_time_deficit_rate_full_1pn([0.0, 0.0, 0.0], potential=GM_EARTH / r_eq + w_eq)
    rate_pole = coordinate_time_deficit_rate_full_1pn([0.0, 0.0, 0.0], potential=GM_EARTH / r_eq + w_pole)

    clock_rate_diff = rate_eq - rate_pole
    expected_clock_diff = delta_w / (C_LIGHT**2)

    assert np.isclose(clock_rate_diff, expected_clock_diff, rtol=1e-6)
    # Clock frequency shift delta_w / c^2 is ~ 1.13e-12 (1.13 picoseconds per second)
    assert 5.0e-13 < abs(clock_rate_diff) < 2.0e-12


# ==============================================================================
# LEVEL 5: END-TO-END TRAJECTORY PROPAGATION WITH ZONAL HARMONICS
# ==============================================================================


def test_level5_propagate_trajectory_with_zonals():
    """Level 5: Verify propagate_trajectory_3d integrates with include_zonals=True."""
    from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state

    jd_epoch = 2451545.0  # J2000.0
    earth_state = get_body_barycentric_state("earth", jd_epoch)

    # LEO position and velocity relative to Earth center, placed in BCRS
    r_earth_leo = np.array([6.978e6, 0.0, 0.0], dtype=np.float64)
    v_earth_leo = np.array([0.0, 7560.0, 0.0], dtype=np.float64)

    r_init_bcrs = earth_state.position + r_earth_leo
    v_init_bcrs = earth_state.velocity + v_earth_leo

    # 1. Propagate with zonals enabled
    res_zonals = propagate_trajectory_3d(
        r0=r_init_bcrs,
        v0=v_init_bcrs,
        t_span=(0.0, 3600.0),
        epoch_jd_tdb=jd_epoch,
        gravitational_bodies=["earth"],
        include_1pn=False,
        include_zonals=True,
        max_zonal_degree=4,
        rtol=1e-10,
        atol=1e-12,
    )

    # 2. Propagate without zonals (pure point-mass)
    res_point = propagate_trajectory_3d(
        r0=r_init_bcrs,
        v0=v_init_bcrs,
        t_span=(0.0, 3600.0),
        epoch_jd_tdb=jd_epoch,
        gravitational_bodies=["earth"],
        include_1pn=False,
        include_zonals=False,
        rtol=1e-10,
        atol=1e-12,
    )

    # Verify both solved successfully
    assert res_zonals.status == 0
    assert res_point.status == 0

    # J2 creates a physical trajectory departure of several kilometers over 1 hour
    delta_r = np.linalg.norm(res_zonals.r[-1] - res_point.r[-1])
    assert delta_r > 50.0, f"Zonals produced negligible difference: delta_r = {delta_r:.2f} m"

    # Proper time deficit must be physically non-zero and smooth
    assert res_zonals.time_deficit[-1] > 0.0
    assert np.all(np.diff(res_zonals.tau) > 0.0)
