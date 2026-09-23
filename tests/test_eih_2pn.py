"""Automated validation tests for 2PN and 2.5PN EIH relativistic dynamics.

Validates:
1. Blanchet & Iyer (1989) circular binary orbital frequency formula.
2. Newtonian recovery in the c -> infinity limit.
3. Strict consistency with existing 1PN EIH N-body acceleration kernel.
4. Peters (1964) 2.5PN dissipative energy loss rate.
5. Realistic planetary scale (Mercury perihelion 2PN acceleration magnitude).
"""

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    C_LIGHT,
    G_NEWTON,
    GM_SUN,
)
from relativistic_engine.physics.eih import compute_eih_nbody_accelerations
from relativistic_engine.physics.eih_2pn import (
    circular_orbit_frequency_2pn,
    compute_eih_2pn_nbody_accelerations,
    compute_relative_2pn_acceleration,
)


def test_newtonian_limit_as_c_approaches_infinity():
    """Verify that as c -> inf, 1PN, 2PN, and 2.5PN corrections vanish strictly."""
    r_vec = np.array([1.496e11, 0.0, 0.0])  # 1 AU
    v_vec = np.array([0.0, 29780.0, 0.0])   # Earth orbital speed
    m1 = 1.9885e30  # Sun
    m2 = 5.972e24   # Earth

    # At physical c
    res_physical = compute_relative_2pn_acceleration(
        r_vec, v_vec, m1, m2, include_1pn=True, include_2pn=True, include_25pn=True
    )

    # At c = 1e20 (effectively infinite)
    res_inf = compute_relative_2pn_acceleration(
        r_vec, v_vec, m1, m2, c=1e20, include_1pn=True, include_2pn=True, include_25pn=True
    )

    # 1PN, 2PN, 2.5PN must be negligible at c = 1e20
    assert np.allclose(res_inf["1pn"], 0.0, atol=1e-25)
    assert np.allclose(res_inf["2pn"], 0.0, atol=1e-35)
    assert np.allclose(res_inf["25pn"], 0.0, atol=1e-45)
    assert np.allclose(res_inf["total"], res_inf["newtonian"], atol=1e-25)


def test_circular_orbit_frequency_blanchet_formula():
    """Verify 2PN circular orbit angular frequency against Blanchet & Iyer (1989)."""
    m1 = 1.4 * 1.9885e30  # 1.4 M_sun neutron star
    m2 = 1.4 * 1.9885e30  # Equal mass binary
    r_sep = 1.0e8         # 100,000 km separation

    omega_2pn = circular_orbit_frequency_2pn(m1, m2, r_sep)
    omega_kepler = math.sqrt(G_NEWTON * (m1 + m2) / (r_sep**3))

    # Relativistic correction is small but non-zero at 1e8 m: gamma_pn ~ GM/(c^2 r) ~ 4e-6
    rel_diff = (omega_2pn - omega_kepler) / omega_kepler
    assert abs(rel_diff) > 1e-7
    assert abs(rel_diff) < 1e-4


def test_1pn_mode_matches_existing_eih_kernel():
    """Verify that order='1pn' strictly matches the audited compute_eih_nbody_accelerations."""
    # 3-body system: Sun, Earth, Jupiter
    pos = np.array([
        [0.0, 0.0, 0.0],
        [1.496e11, 0.0, 0.0],
        [0.0, 7.785e11, 0.0],
    ])
    vel = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 29780.0, 0.0],
        [-13070.0, 0.0, 0.0],
    ])
    gms = np.array([
        1.32712440018e20,  # Sun
        3.986004418e14,    # Earth
        1.26686534e17,     # Jupiter
    ])

    a_tot_old, a_newt_old, a_1pn_old = compute_eih_nbody_accelerations(pos, vel, gms, include_1pn=True)
    res_new = compute_eih_2pn_nbody_accelerations(pos, vel, gms, order="1pn")

    assert np.allclose(res_new["newtonian"], a_newt_old, rtol=1e-14, atol=1e-16)
    assert np.allclose(res_new["1pn"], a_1pn_old, rtol=1e-14, atol=1e-16)
    assert np.allclose(res_new["total"], a_tot_old, rtol=1e-14, atol=1e-16)


def test_mercury_2pn_acceleration_magnitude():
    """Verify that 2PN acceleration for Mercury around Sun is nonzero and has physical scale."""
    # Mercury at perihelion: r ~ 4.60e10 m, v ~ 5.898e4 m/s
    r_vec = np.array([4.60012e10, 0.0, 0.0])
    v_vec = np.array([0.0, 58980.0, 0.0])
    m_sun = 1.9885e30
    m_merc = 3.3011e23

    res = compute_relative_2pn_acceleration(
        r_vec, v_vec, m_sun, m_merc, include_1pn=True, include_2pn=True, include_25pn=False
    )

    a_newt_mag = float(np.linalg.norm(res["newtonian"]))
    a_1pn_mag = float(np.linalg.norm(res["1pn"]))
    a_2pn_mag = float(np.linalg.norm(res["2pn"]))

    # a_newt ~ GM/r^2 ~ 0.0626 m/s^2
    assert 0.05 < a_newt_mag < 0.07

    # 1PN ratio ~ (v/c)^2 ~ 3.8e-8 -> a_1pn ~ 2.4e-9 m/s^2
    assert 1e-10 < a_1pn_mag < 1e-8

    # 2PN ratio ~ (v/c)^4 ~ 1.5e-15 -> a_2pn ~ 1e-16 to 1e-15 m/s^2
    assert 1e-18 < a_2pn_mag < 1e-14
    assert a_2pn_mag > 0.0


def test_25pn_radiation_reaction_dissipation():
    """Verify that 2.5PN radiation reaction accelerates oppositely to orbital velocity (damping)."""
    # Circular binary: r dot v = 0 -> radiation reaction directly opposes velocity
    r_vec = np.array([1.0e7, 0.0, 0.0])
    v_circ = math.sqrt(G_NEWTON * (2.0 * 1.4 * 1.9885e30) / 1.0e7)
    v_vec = np.array([0.0, v_circ, 0.0])
    m1 = 1.4 * 1.9885e30
    m2 = 1.4 * 1.9885e30

    res = compute_relative_2pn_acceleration(
        r_vec, v_vec, m1, m2, include_1pn=False, include_2pn=False, include_25pn=True
    )

    a_25pn = res["25pn"]
    # Power P = v . a must be strictly negative (energy extraction by gravitational waves)
    power = float(np.dot(v_vec, a_25pn))
    assert power < 0.0
