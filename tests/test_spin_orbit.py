"""Automated validation tests for relativistic spin-orbit couplings and frame-dragging.

Validates:
1. NASA Gravity Probe B geodetic precession (6.604 arcsec/yr, Everitt et al. 2011).
2. NASA Gravity Probe B Lense-Thirring frame-dragging (0.039-0.041 arcsec/yr, Everitt et al. 2011).
3. Strict norm conservation of gyroscope spin vectors under Rodrigues rotation.
4. Newtonian limit: spin-orbit acceleration vanishes when spin=0 or c -> infinity.
5. Planetary angular momentum catalog properties (Earth, Sun, Jupiter).
"""

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    C_LIGHT,
    GM_EARTH,
    RADIUS_EARTH,
)
from relativistic_engine.physics.spin_orbit import (
    compute_geodetic_precession_vector,
    compute_lense_thirring_precession_vector,
    compute_spin_orbit_acceleration,
    propagate_gyroscope_spin,
    secular_geodetic_precession_rate_arcsec_yr,
    secular_lense_thirring_polar_rate_arcsec_yr,
    PLANETARY_SPIN_VECTORS,
)


def test_gravity_probe_b_geodetic_precession():
    """Verify Gravity Probe B geodetic precession matches Everitt et al. (2011)."""
    # GP-B flight orbit: a = 7027.4 km, e = 0.0014 (Everitt et al. 2011 PRL 106, 221101)
    a_gpb = 7027.4e3
    e_gpb = 0.0014

    rate_arcsec_yr = secular_geodetic_precession_rate_arcsec_yr(a_gpb, e_gpb, gm_central=GM_EARTH)

    # General Relativity theoretical prediction: ~6.606 arcsec/year
    # GP-B measured: 6.6018 +/- 0.0183 arcsec/year
    assert math.isclose(rate_arcsec_yr, 6.604, rel_tol=2e-3)
    assert 6.58 < rate_arcsec_yr < 6.62


def test_gravity_probe_b_lense_thirring_precession():
    """Verify Gravity Probe B frame-dragging rate matches Everitt et al. (2011)."""
    # GP-B flight orbit: a = 7027.4 km, e = 0.0014
    a_gpb = 7027.4e3
    e_gpb = 0.0014

    rate_arcsec_yr = secular_lense_thirring_polar_rate_arcsec_yr(a_gpb, e_gpb)

    # General Relativity prediction is ~ 0.039 - 0.041 arcsec/year
    # GP-B experimental measurement: 0.0372 +/- 0.0072 arcsec/year
    assert 0.035 < rate_arcsec_yr < 0.045


def test_gyroscope_spin_norm_strict_conservation():
    """Verify that numerical integration of gyroscope spin preserves unit norm exactly."""
    s0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # Fast precession vector for numerical stress test
    omega = np.array([1e-4, 2e-4, -1.5e-4], dtype=np.float64)
    dt = 10.0

    s_curr = s0.copy()
    for _ in range(1000):
        s_curr = propagate_gyroscope_spin(s_curr, omega, dt)

    norm_final = float(np.linalg.norm(s_curr))
    # Norm must remain exactly 1.0 within machine epsilon
    assert math.isclose(norm_final, 1.0, abs_tol=1e-15)


def test_spin_orbit_acceleration_limits():
    """Verify that spin-orbit acceleration vanishes for zero spin and infinite c."""
    r_vec = np.array([7000e3, 0.0, 0.0])
    v_vec = np.array([0.0, 7500.0, 0.0])
    s_earth = PLANETARY_SPIN_VECTORS["Earth"]["spin_angular_momentum"]

    # Physical Earth spin
    a_so = compute_spin_orbit_acceleration(r_vec, v_vec, s_earth)
    a_mag = float(np.linalg.norm(a_so))
    assert a_mag > 0.0
    # For LEO, a_SO is typically ~ 5e-11 to 1e-8 m/s^2
    assert 1e-11 < a_mag < 1e-7

    # Zero spin -> zero acceleration
    a_zero = compute_spin_orbit_acceleration(r_vec, v_vec, np.zeros(3))
    assert np.allclose(a_zero, 0.0, atol=1e-25)

    # c -> infinity -> zero acceleration
    a_inf = compute_spin_orbit_acceleration(r_vec, v_vec, s_earth, c=1e20)
    assert np.allclose(a_inf, 0.0, atol=1e-25)


def test_planetary_spin_catalog_consistency():
    """Verify physical consistency of planetary spin vectors."""
    for body in ["Earth", "Sun", "Jupiter"]:
        assert body in PLANETARY_SPIN_VECTORS
        entry = PLANETARY_SPIN_VECTORS[body]
        s_vec = entry["spin_angular_momentum"]
        s_mag = float(np.linalg.norm(s_vec))
        assert s_mag > 0.0

    # Jupiter spin angular momentum dwarfs Earth by ~5 orders of magnitude
    s_earth = float(np.linalg.norm(PLANETARY_SPIN_VECTORS["Earth"]["spin_angular_momentum"]))
    s_jup = float(np.linalg.norm(PLANETARY_SPIN_VECTORS["Jupiter"]["spin_angular_momentum"]))
    assert s_jup / s_earth > 5e4
