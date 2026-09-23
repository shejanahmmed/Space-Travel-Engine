"""5-Level Automated Validation Suite for Einstein-Infeld-Hoffmann (EIH) N-Body Dynamics.

Validates the self-consistent multi-body 1PN relativistic equations of motion
according to the operational scientific doctrine:

Level 1: Exact Analytical & Two-Body Reduction
Level 2: Strict Relativistic Conservation Laws (Energy, Momentum, Angular Momentum)
Level 3: Relativistic Precession & Asymptotic Benchmarks (Mercury 42.98 arcsec/century)
Level 4: Symmetry & Coordinate Frame Invariance (Rotations, Translations, Time-Reversal)
Level 5: Real Astronomical Ephemeris & Cross-Tool Standards (JPL Horizons, DE440)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    AU,
    C_LIGHT,
    G_NEWTON,
    GM_EARTH,
    GM_JUPITER,
    GM_SUN,
    SEC_PER_DAY,
    SEC_PER_JULIAN_YEAR,
)
from relativistic_engine.physics.eih import (
    compute_eih_nbody_accelerations,
    compute_eih_conserved_quantities,
    compute_mercury_perihelion_advance_rate,
    propagate_eih_nbody_system,
)
from benchmarks.reference_cross_validation import (
    verify_horizons_1pn_acceleration_agreement,
    run_sun_jupiter_saturn_eih_benchmark,
    run_mercury_perihelion_benchmark,
)


class TestEIHLevel1AnalyticalReduction:
    """Level 1: Analytical reductions and asymptotic limits."""

    def test_eih_test_particle_limit(self):
        """When m_2 << m_1, EIH acceleration matches 1-body 1PN equation to machine precision."""
        r_vec = np.array([1.0 * AU, 0.0, 0.0], dtype=np.float64)
        v_vec = np.array([0.0, 29780.0, 0.0], dtype=np.float64)

        pos = np.array([[0.0, 0.0, 0.0], r_vec], dtype=np.float64)
        vel = np.array([[0.0, 0.0, 0.0], v_vec], dtype=np.float64)
        gms = np.array([GM_SUN, 0.0], dtype=np.float64)

        _, a_newt, a_1pn = compute_eih_nbody_accelerations(pos, vel, gms, include_1pn=True)

        # Standard 1-body 1PN equation (Moyer 2003 / Will 2014)
        r = np.linalg.norm(r_vec)
        v2 = np.dot(v_vec, v_vec)
        r_dot_v = np.dot(r_vec, v_vec)
        c2 = C_LIGHT**2

        expected_1pn = (GM_SUN / (c2 * (r**3))) * (
            (4.0 * GM_SUN / r - v2) * r_vec + 4.0 * r_dot_v * v_vec
        )

        assert np.allclose(a_1pn[1], expected_1pn, rtol=1e-15, atol=1e-20)
        # Central body experiences zero acceleration when test particle mass is zero
        assert np.allclose(a_1pn[0], 0.0, atol=1e-25)

    def test_eih_static_limit(self):
        """When all velocities are zero, velocity-dependent 1PN terms vanish identically."""
        pos = np.array([[0.0, 0.0, 0.0], [1.0 * AU, 0.0, 0.0]], dtype=np.float64)
        vel = np.zeros((2, 3), dtype=np.float64)
        gms = np.array([GM_SUN, GM_EARTH], dtype=np.float64)

        _, _, a_1pn = compute_eih_nbody_accelerations(pos, vel, gms, include_1pn=True)

        # In static limit, 1PN acceleration on body 1 points strictly along radial separation
        assert abs(a_1pn[1, 1]) == 0.0
        assert abs(a_1pn[1, 2]) == 0.0
        assert a_1pn[1, 0] != 0.0

    def test_eih_newtonian_switch(self):
        """When include_1pn=False, 1PN acceleration is identically zero."""
        pos = np.array([[0.0, 0.0, 0.0], [1.0 * AU, 0.0, 0.0]], dtype=np.float64)
        vel = np.array([[0.0, 0.0, 0.0], [0.0, 29780.0, 0.0]], dtype=np.float64)
        gms = np.array([GM_SUN, GM_EARTH], dtype=np.float64)

        a_tot, a_newt, a_1pn = compute_eih_nbody_accelerations(pos, vel, gms, include_1pn=False)

        assert np.array_equal(a_tot, a_newt)
        assert np.all(a_1pn == 0.0)


class TestEIHLevel2ConservationLaws:
    """Level 2: Relativistic energy, linear momentum, and angular momentum conservation."""

    def test_eih_two_body_energy_conservation(self):
        """Total 1PN relativistic energy conserved to < 1e-11 over a complete orbit."""
        a = 1.0 * AU
        e = 0.2
        r_peri = a * (1.0 - e)
        v_peri = math.sqrt(GM_SUN * (1.0 + e) / (a * (1.0 - e)))

        pos_0 = np.array([[0.0, 0.0, 0.0], [r_peri, 0.0, 0.0]], dtype=np.float64)
        vel_0 = np.array([[0.0, 0.0, 0.0], [0.0, v_peri, 0.0]], dtype=np.float64)
        gms = np.array([GM_SUN, GM_EARTH], dtype=np.float64)

        # Period T = 2 * pi * sqrt(a^3 / GM)
        t_orb = 2.0 * math.pi * math.sqrt((a**3) / (GM_SUN + GM_EARTH))

        inv_0 = compute_eih_conserved_quantities(pos_0, vel_0, gms)
        e_0 = inv_0["energy_total"]

        prop = propagate_eih_nbody_system(
            pos_0,
            vel_0,
            gms,
            (0.0, t_orb),
            include_1pn=True,
            rtol=1e-12,
            atol=1e-14,
            max_step=t_orb / 50.0,
        )

        pos_end = prop["positions"][-1]
        vel_end = prop["velocities"][-1]
        inv_end = compute_eih_conserved_quantities(pos_end, vel_end, gms)
        e_end = inv_end["energy_total"]

        rel_error = abs(e_end - e_0) / abs(e_0)
        assert rel_error < 1e-11

    def test_eih_two_body_angular_momentum_conservation(self):
        """Total 1PN relativistic angular momentum J conserved to < 1e-11."""
        a = 1.0 * AU
        e = 0.3
        r_peri = a * (1.0 - e)
        v_peri = math.sqrt(GM_SUN * (1.0 + e) / (a * (1.0 - e)))

        pos_0 = np.array([[0.0, 0.0, 0.0], [r_peri, 0.0, 0.0]], dtype=np.float64)
        vel_0 = np.array([[0.0, 0.0, 0.0], [0.0, v_peri, 0.0]], dtype=np.float64)
        gms = np.array([GM_SUN, GM_EARTH], dtype=np.float64)

        t_orb = 2.0 * math.pi * math.sqrt((a**3) / (GM_SUN + GM_EARTH))
        inv_0 = compute_eih_conserved_quantities(pos_0, vel_0, gms)
        j_0 = inv_0["angular_momentum_magnitude"]

        prop = propagate_eih_nbody_system(
            pos_0,
            vel_0,
            gms,
            (0.0, t_orb),
            include_1pn=True,
            rtol=1e-12,
            atol=1e-14,
            max_step=t_orb / 50.0,
        )

        inv_end = compute_eih_conserved_quantities(prop["positions"][-1], prop["velocities"][-1], gms)
        j_end = inv_end["angular_momentum_magnitude"]

        rel_error = abs(j_end - j_0) / j_0
        assert rel_error < 1e-11

    def test_eih_linear_momentum_conservation(self):
        """Total 1PN linear momentum P conserved across arbitrary isolated interactions."""
        pos = np.array([[0.0, 0.0, 0.0], [1.0 * AU, 0.0, 0.0]], dtype=np.float64)
        # Center-of-mass frame velocity configuration
        m1 = GM_SUN / G_NEWTON
        m2 = GM_JUPITER / G_NEWTON
        v_rel = 13000.0
        v1 = - (m2 / (m1 + m2)) * v_rel
        v2 = + (m1 / (m1 + m2)) * v_rel

        vel = np.array([[0.0, v1, 0.0], [0.0, v2, 0.0]], dtype=np.float64)
        gms = np.array([GM_SUN, GM_JUPITER], dtype=np.float64)

        inv_0 = compute_eih_conserved_quantities(pos, vel, gms)
        p_0 = inv_0["momentum_vector"]

        prop = propagate_eih_nbody_system(
            pos,
            vel,
            gms,
            (0.0, 30.0 * SEC_PER_DAY),
            include_1pn=True,
            rtol=1e-12,
            atol=1e-14,
            max_step=SEC_PER_DAY,
        )

        inv_end = compute_eih_conserved_quantities(prop["positions"][-1], prop["velocities"][-1], gms)
        p_end = inv_end["momentum_vector"]

        dp = np.linalg.norm(p_end - p_0)
        p_scale = m1 * abs(v1) + m2 * abs(v2)
        assert dp / p_scale < 1e-7


class TestEIHLevel3RelativisticPrecession:
    """Level 3: Einstein secular perihelion precession and asymptotic scaling."""

    def test_mercury_analytical_perihelion_formula(self):
        """Analytical perihelion precession formula matches Einstein (1915) 42.98 arcsec/century."""
        a = 5.790905e10  # 0.387098 AU
        e = 0.20563069
        res = compute_mercury_perihelion_advance_rate(a, e, GM_SUN)

        # Expected value from literature: 42.98 arcseconds per century
        assert pytest.approx(42.98, rel=1e-3) == res["rate_arcsec_per_century"]
        assert res["advance_per_orbit_rad"] > 0.0
        assert res["orbital_period_s"] > 0.0

    def test_mercury_numerical_perihelion_advance(self):
        """Multi-orbit numerical integration reproduces Einstein's secular advance to < 10 ppm."""
        res = run_mercury_perihelion_benchmark(n_orbits=5)
        assert res["status"] == "PASS"
        assert res["relative_error"] < 1.0e-5  # Within 10 ppm

    def test_eih_speed_of_light_scaling(self):
        """1PN acceleration scales quadratically with inverse speed of light (1/c^2)."""
        pos = np.array([[0.0, 0.0, 0.0], [1.0 * AU, 0.0, 0.0]], dtype=np.float64)
        vel = np.array([[0.0, 0.0, 0.0], [0.0, 29780.0, 0.0]], dtype=np.float64)
        gms = np.array([GM_SUN, GM_EARTH], dtype=np.float64)

        _, _, a_1pn_c1 = compute_eih_nbody_accelerations(pos, vel, gms, c=C_LIGHT)
        _, _, a_1pn_c2 = compute_eih_nbody_accelerations(pos, vel, gms, c=2.0 * C_LIGHT)

        # Doubling c should reduce 1PN acceleration by exactly 4
        assert np.allclose(a_1pn_c1 / 4.0, a_1pn_c2, rtol=1e-15)


class TestEIHLevel4SymmetryAndInvariance:
    """Level 4: Physical symmetries (rotations, translations, time-reversal)."""

    def test_eih_spatial_rotational_invariance(self):
        """EIH accelerations transform covariantly under arbitrary 3D spatial rotations."""
        pos = np.array([
            [1.0e10, 2.0e10, 0.5e10],
            [-3.0e10, 1.5e10, -0.8e10],
            [0.2e10, -4.0e10, 2.1e10],
        ], dtype=np.float64)
        vel = np.array([
            [10000.0, -5000.0, 2000.0],
            [3000.0, 12000.0, -4000.0],
            [-8000.0, 2000.0, 7000.0],
        ], dtype=np.float64)
        gms = np.array([GM_SUN, GM_JUPITER, GM_EARTH], dtype=np.float64)

        a_tot, _, _ = compute_eih_nbody_accelerations(pos, vel, gms, include_1pn=True)

        # Arbitrary 3D Euler rotation matrix (around X and Z)
        theta = 0.731
        phi = 1.245
        rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(theta), -math.sin(theta)],
            [0.0, math.sin(theta), math.cos(theta)],
        ])
        rz = np.array([
            [math.cos(phi), -math.sin(phi), 0.0],
            [math.sin(phi), math.cos(phi), 0.0],
            [0.0, 0.0, 1.0],
        ])
        rot = rz @ rx

        # Apply rotation to positions and velocities
        pos_rot = pos @ rot.T
        vel_rot = vel @ rot.T

        a_tot_rot, _, _ = compute_eih_nbody_accelerations(pos_rot, vel_rot, gms, include_1pn=True)

        # Expected rotated acceleration: a_rot == a @ rot.T
        expected_a_rot = a_tot @ rot.T
        assert np.allclose(a_tot_rot, expected_a_rot, rtol=1e-13, atol=1e-18)

    def test_eih_spatial_translation_invariance(self):
        """Translating all bodies uniformly does not alter mutual accelerations."""
        pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0 * AU, 0.0, 0.0],
            [0.0, 5.2 * AU, 0.0],
        ], dtype=np.float64)
        vel = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 29780.0, 0.0],
            [-13000.0, 0.0, 0.0],
        ], dtype=np.float64)
        gms = np.array([GM_SUN, GM_EARTH, GM_JUPITER], dtype=np.float64)

        a_tot_1, _, _ = compute_eih_nbody_accelerations(pos, vel, gms, include_1pn=True)

        shift = np.array([1.5e11, -2.8e11, 4.3e11], dtype=np.float64)
        a_tot_2, _, _ = compute_eih_nbody_accelerations(pos + shift, vel, gms, include_1pn=True)

        assert np.allclose(a_tot_1, a_tot_2, rtol=1e-15, atol=1e-22)

    def test_eih_time_reversal_symmetry(self):
        """Time reversal t -> -t, v -> -v preserves 1PN acceleration (conservative Hamiltonian)."""
        pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0 * AU, 0.5 * AU, -0.2 * AU],
        ], dtype=np.float64)
        vel = np.array([
            [0.0, 0.0, 0.0],
            [-15000.0, 25000.0, 3000.0],
        ], dtype=np.float64)
        gms = np.array([GM_SUN, GM_EARTH], dtype=np.float64)

        _, _, a_1pn_forward = compute_eih_nbody_accelerations(pos, vel, gms, include_1pn=True)
        _, _, a_1pn_reversed = compute_eih_nbody_accelerations(pos, -vel, gms, include_1pn=True)

        # In 1PN conservative dynamics, terms are quadratic in velocity (v^2, v_A . v_B, etc.)
        # Hence acceleration is exactly even under velocity inversion: a(-v) == a(v)
        assert np.allclose(a_1pn_forward, a_1pn_reversed, rtol=1e-15, atol=1e-22)


class TestEIHLevel5RealEphemerisAndCrossValidation:
    """Level 5: Real astronomical benchmarks against JPL Horizons and DE440."""

    def test_horizons_1pn_acceleration_agreement(self):
        """EIH 1PN acceleration matches JPL Horizons test particle formula to < 1e-15."""
        res = verify_horizons_1pn_acceleration_agreement()
        assert res["status"] == "PASS"
        assert res["relative_error"] < 1.0e-15

    def test_sun_jupiter_saturn_eih_conservation_and_shift(self):
        """Coupled Sun-Jupiter-Saturn EIH propagation conserves energy and quantifies 1PN shift."""
        res = run_sun_jupiter_saturn_eih_benchmark(duration_days=30.0)
        assert res["status"] == "PASS"
        assert res["energy_relative_error"] < 1.0e-11
        assert res["angular_momentum_relative_error"] < 1.0e-11
        assert res["jupiter_1pn_trajectory_shift_m"] > 0.0  # Confirms real non-zero relativistic perturbation

    def test_eih_invalid_inputs(self):
        """Engine guards against unphysical and malformed inputs."""
        pos = np.zeros((3, 3))
        vel = np.zeros((2, 3))  # Shape mismatch
        gms = np.zeros(3)

        with pytest.raises(ValueError, match="Incompatible input shapes"):
            compute_eih_nbody_accelerations(pos, vel, gms)

        with pytest.raises(ValueError, match="Eccentricity must be in"):
            compute_mercury_perihelion_advance_rate(1.0 * AU, 1.2)

        with pytest.raises(ValueError, match="Semi-major axis must be positive"):
            compute_mercury_perihelion_advance_rate(-1.0 * AU, 0.1)
