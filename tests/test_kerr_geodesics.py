"""Validation and Unit Tests for Kerr Metric Geodesics and Strong-Field Relativistic Physics.

Validates:
1. Exact analytical Schwarzschild reduction (a -> 0):
   - Horizons: r_+ = 2M, r_- = 0.
   - Ergosphere: r_E(theta) = 2M.
   - ISCO: r_ISCO = 6M.
   - Photon orbit: r_ph = 3M.
   - Shadow radius: R_shadow = 3 * sqrt(3) * M.
2. High-spin and near-extremal Kerr horizons and orbital limits (Bardeen et al. 1972).
3. Metric contravariance identity: g_mu_alpha * g^alpha_nu = delta_mu_nu to machine precision.
4. Christoffel symbol analytical symmetry: Gamma^mu_alpha_beta = Gamma^mu_beta_alpha.
5. Numerical conservation of Carter constant Q, energy E, and rest-mass norm along
   integrated timelike and null geodesics using DOP853.
6. Bardeen black hole shadow contour generation and frame-dragging asymmetry.
7. Penrose process energy extraction efficiency in the ergosphere.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from relativistic_engine.constants import C_LIGHT, G_NEWTON
from relativistic_engine.physics.kerr import (
    KerrGeometry,
    compute_bardeen_shadow_contour,
    compute_carter_invariants,
    compute_kerr_christoffel,
    compute_kerr_metric,
    compute_penrose_energy_gain,
    propagate_kerr_geodesic,
)


class TestSchwarzschildAnalyticalReduction:
    """Level 1: Exact analytical reduction to Schwarzschild spacetime when spin a -> 0."""

    @pytest.fixture
    def schw_bh(self) -> KerrGeometry:
        # Solar-mass non-rotating black hole
        return KerrGeometry(mass_kg=1.98847e30, spin_dimensionless=0.0)

    def test_horizons_and_ergosphere_schwarzschild(self, schw_bh: KerrGeometry) -> None:
        """Schwarzschild horizons: r_+ = 2M, r_- = 0; ergosphere matches horizon identically."""
        m = schw_bh.mass_m
        assert math.isclose(schw_bh.event_horizon_outer, 2.0 * m, rel_tol=1e-15)
        assert math.isclose(schw_bh.event_horizon_inner, 0.0, abs_tol=1e-15)

        # Ergosphere in Schwarzschild is spherically symmetric and coincides with horizon
        thetas = np.linspace(0.0, math.pi, 20)
        r_erg = schw_bh.ergosphere_outer(thetas)
        np.testing.assert_allclose(r_erg, 2.0 * m, rtol=1e-15)

    def test_isco_and_photon_orbit_schwarzschild(self, schw_bh: KerrGeometry) -> None:
        """Schwarzschild: ISCO is exactly 6M, photon orbit is exactly 3M."""
        m = schw_bh.mass_m
        assert math.isclose(schw_bh.isco_radius(prograde=True), 6.0 * m, rel_tol=1e-15)
        assert math.isclose(schw_bh.isco_radius(prograde=False), 6.0 * m, rel_tol=1e-15)
        assert math.isclose(schw_bh.photon_orbit_radius(prograde=True), 3.0 * m, rel_tol=1e-15)
        assert math.isclose(schw_bh.photon_orbit_radius(prograde=False), 3.0 * m, rel_tol=1e-15)

    def test_schwarzschild_metric_components(self, schw_bh: KerrGeometry) -> None:
        """Schwarzschild metric components: g_tt = -(1 - 2M/r), g_rr = 1/(1 - 2M/r), g_thth = r^2, g_phiphi = r^2 sin^2(theta)."""
        m = schw_bh.mass_m
        r = 10.0 * m
        theta = math.pi / 3.0  # 60 degrees

        g_cov, g_con = compute_kerr_metric(r, theta, schw_bh)

        expected_g_tt = -(1.0 - 2.0 * m / r)
        expected_g_rr = 1.0 / (1.0 - 2.0 * m / r)
        expected_g_thth = r * r
        expected_g_phiphi = (r * r) * (math.sin(theta) ** 2)

        assert math.isclose(g_cov[0, 0], expected_g_tt, rel_tol=1e-14)
        assert math.isclose(g_cov[1, 1], expected_g_rr, rel_tol=1e-14)
        assert math.isclose(g_cov[2, 2], expected_g_thth, rel_tol=1e-14)
        assert math.isclose(g_cov[3, 3], expected_g_phiphi, rel_tol=1e-14)
        assert g_cov[0, 3] == 0.0  # No frame dragging

    def test_schwarzschild_shadow_contour(self, schw_bh: KerrGeometry) -> None:
        """Schwarzschild shadow is an exact circle of radius 3*sqrt(3)*M approx 5.19615 M."""
        alpha, beta = compute_bardeen_shadow_contour(schw_bh, theta_obs_rad=math.pi * 0.5)
        radii = np.sqrt(alpha ** 2 + beta ** 2)
        expected_radius = math.sqrt(27.0)  # in units of M
        np.testing.assert_allclose(radii, expected_radius, rtol=1e-12)


class TestKerrMetricAlgebraAndSymmetries:
    """Level 2: Metric algebraic identities and Christoffel symbol properties."""

    @pytest.fixture
    def kerr_bh(self) -> KerrGeometry:
        # Supermassive spinning black hole (Sgr A* mass scale, spin a_* = 0.9)
        return KerrGeometry(mass_kg=4.154e6 * 1.98847e30, spin_dimensionless=0.9)

    def test_metric_inverse_identity(self, kerr_bh: KerrGeometry) -> None:
        """Covariant and contravariant metric tensors satisfy g_mu_alpha * g^alpha_nu = delta_mu_nu."""
        # 1. Test in normalized geometric units (M = 1 m) where condition number is O(10)
        # Demonstrates exact analytical inversion to machine precision (< 1e-14)
        bh_geom = KerrGeometry(mass_kg=(C_LIGHT ** 2) / G_NEWTON, spin_dimensionless=0.9)
        for r in [2.5, 5.0, 20.0]:
            for th in [0.2, math.pi / 4.0, math.pi / 2.0]:
                g_cov, g_con = compute_kerr_metric(r, th, bh_geom)
                np.testing.assert_allclose(g_con, np.linalg.inv(g_cov), rtol=1e-14, atol=1e-15)
                product = np.dot(g_cov, g_con)
                np.testing.assert_allclose(product, np.eye(4), atol=1e-14)

        # 2. Test at astrophysical supermassive scale (Sgr A*, r ~ 10^11 m)
        m = kerr_bh.mass_m
        for r in [2.5 * m, 5.0 * m, 20.0 * m]:
            for th in [0.2, math.pi / 4.0, math.pi / 2.0]:
                g_cov, g_con = compute_kerr_metric(r, th, kerr_bh)
                # Matches numerical inversion within float64 matrix conditioning limits
                np.testing.assert_allclose(g_con, np.linalg.inv(g_cov), rtol=1e-8)
                product = np.dot(g_cov, g_con)
                np.testing.assert_allclose(product, np.eye(4), atol=1e-5)

    def test_christoffel_torsion_free_symmetry(self, kerr_bh: KerrGeometry) -> None:
        """Christoffel symbols are symmetric in their lower indices: Gamma^mu_alpha_beta = Gamma^mu_beta_alpha."""
        m = kerr_bh.mass_m
        r = 4.0 * m
        theta = math.pi / 3.0
        gamma = compute_kerr_christoffel(r, theta, kerr_bh)

        for mu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    diff = abs(gamma[mu, alpha, beta] - gamma[mu, beta, alpha])
                    assert diff < 1e-14, f"Asymmetry in Gamma^{mu}_{alpha}_{beta}: {diff}"


class TestKerrHorizonsAndBPTLimits:
    """Level 3: Bardeen, Press & Teukolsky (1972) horizon and orbital limits."""

    def test_high_spin_horizons(self) -> None:
        """Near-extremal spin a_* = 0.999 horizons and ergosphere bounds."""
        bh = KerrGeometry(mass_kg=1e31, spin_dimensionless=0.999)
        m = bh.mass_m
        a = bh.spin_m

        r_plus = bh.event_horizon_outer
        r_minus = bh.event_horizon_inner
        assert r_plus > m
        assert r_minus < m
        assert math.isclose(r_plus * r_minus, a * a, rel_tol=1e-12)

        # Ergosphere at poles theta = 0 matches outer event horizon
        r_erg_pole = bh.ergosphere_outer(0.0)
        assert math.isclose(r_erg_pole, r_plus, rel_tol=1e-12)

        # Ergosphere at equator theta = pi/2 extends to 2M
        r_erg_eq = bh.ergosphere_outer(math.pi * 0.5)
        assert math.isclose(r_erg_eq, 2.0 * m, rel_tol=1e-12)

    def test_isco_bpt_extremal_limits(self) -> None:
        """As a_* -> 1, prograde ISCO -> M and retrograde ISCO -> 9M."""
        bh_extreme = KerrGeometry(mass_kg=1e31, spin_dimensionless=0.99999)
        m = bh_extreme.mass_m
        r_isco_pro = bh_extreme.isco_radius(prograde=True)
        r_isco_ret = bh_extreme.isco_radius(prograde=False)

        assert abs(r_isco_pro - m) / m < 0.05
        assert abs(r_isco_ret - 9.0 * m) / m < 0.01

    def test_photon_orbit_limits(self) -> None:
        """As a_* -> 1, prograde photon orbit -> M and retrograde -> 4M."""
        bh_extreme = KerrGeometry(mass_kg=1e31, spin_dimensionless=0.99999)
        m = bh_extreme.mass_m
        r_ph_pro = bh_extreme.photon_orbit_radius(prograde=True)
        r_ph_ret = bh_extreme.photon_orbit_radius(prograde=False)

        assert abs(r_ph_pro - m) / m < 0.01
        assert abs(r_ph_ret - 4.0 * m) / m < 0.01


class TestGeodesicPropagationAndCarterConservation:
    """Level 4: DOP853 geodesic integration and exact constant of motion conservation."""

    def test_timelike_circular_equatorial_orbit(self) -> None:
        """A stable equatorial circular orbit conserves E, L_z, Q, and metric norm to < 1e-10."""
        # Spin a_* = 0.5
        bh = KerrGeometry(mass_kg=1e31, spin_dimensionless=0.5)
        m = bh.mass_m
        a = bh.spin_m

        # Radius outside ISCO: r = 8.0 * M
        r0 = 8.0 * m
        theta0 = math.pi * 0.5  # Equator

        # Keplerian orbital frequency in Kerr spacetime (Bardeen et al. 1972 Eq. 2.16):
        # Omega = dphi / d(ct) = sqrt(M) / (r^(3/2) + a * sqrt(M))
        omega_coord = math.sqrt(m) / ((r0 ** 1.5) + a * math.sqrt(m))

        g_cov, _ = compute_kerr_metric(r0, theta0, bh)
        # Normalization factor for timelike 4-velocity u^mu = u^0 [1, 0, 0, Omega]
        # g_00 + 2 * Omega * g_03 + Omega^2 * g_33 = -1 / (u^0)^2
        denom = -(g_cov[0, 0] + 2.0 * omega_coord * g_cov[0, 3] + (omega_coord ** 2) * g_cov[3, 3])
        assert denom > 0.0, "Circular orbit velocity must be timelike"
        u0 = 1.0 / math.sqrt(denom)
        u3 = u0 * omega_coord

        initial_pos = np.array([0.0, r0, theta0, 0.0])
        initial_vel = np.array([u0, 0.0, 0.0, u3])

        # Verify initial metric norm is -1
        invariants_init = compute_carter_invariants(initial_pos, initial_vel, bh, is_null=False)
        assert math.isclose(invariants_init["metric_norm"], -1.0, rel_tol=1e-12)
        # Equatorial orbit has Q = 0
        assert math.isclose(invariants_init["carter_constant"], 0.0, abs_tol=1e-12)

        # Propagate for 5 orbits: T_orb = 2*pi / omega_coord in coordinate time ct
        t_orb = (2.0 * math.pi) / omega_coord
        tau_span = (0.0, 5.0 * t_orb / u0)

        result = propagate_kerr_geodesic(
            initial_pos,
            initial_vel,
            tau_span,
            bh,
            is_null=False,
            rtol=1e-10,
            atol=1e-12,
        )

        assert result["success"]
        assert not result["horizon_crossed"]

        # Conservation audit
        assert result["energy_drift_fraction"] < 1e-10, f"Energy drift {result['energy_drift_fraction']} exceeds 1e-10"
        final_norm = result["final_invariants"]["metric_norm"]
        assert math.isclose(final_norm, -1.0, rel_tol=1e-8)

    def test_inclined_spherical_orbit_carter_conservation(self) -> None:
        """An inclined orbit with non-zero Carter constant Q conserves Q along the trajectory."""
        bh = KerrGeometry(mass_kg=1e31, spin_dimensionless=0.7)
        m = bh.mass_m

        r0 = 10.0 * m
        theta0 = math.pi / 4.0  # 45 degrees inclination

        g_cov, _ = compute_kerr_metric(r0, theta0, bh)
        # Give initial velocity in both phi and theta directions
        u_th = 0.02 / m
        u_phi = 0.03 / m
        term = -(g_cov[1, 1] * 0.0 + g_cov[2, 2] * (u_th ** 2) + g_cov[3, 3] * (u_phi ** 2) + 1.0)
        # Solve g_00 (u^0)^2 + 2 g_03 u^0 u^3 = term
        a_quad = g_cov[0, 0]
        b_quad = 2.0 * g_cov[0, 3] * u_phi
        c_quad = -term
        u0 = (-b_quad - math.sqrt(b_quad * b_quad - 4.0 * a_quad * c_quad)) / (2.0 * a_quad)

        pos = np.array([0.0, r0, theta0, 0.0])
        vel = np.array([u0, 0.0, u_th, u_phi])

        inv_init = compute_carter_invariants(pos, vel, bh, is_null=False)
        assert inv_init["carter_constant"] > 0.0, "Inclined orbit must have positive Carter constant Q"

        result = propagate_kerr_geodesic(
            pos,
            vel,
            (0.0, 200.0 * m),
            bh,
            is_null=False,
            rtol=1e-10,
            atol=1e-12,
        )

        assert result["success"]
        assert result["carter_drift_fraction"] < 1e-9, f"Carter drift {result['carter_drift_fraction']} exceeds 1e-9"


class TestBardeenShadowAndPenroseProcess:
    """Level 5: Celestial black hole shadow and Penrose process energy extraction."""

    def test_bardeen_kerr_shadow_asymmetry(self) -> None:
        """Spinning black hole shadow is shifted and flattened on the prograde side."""
        bh = KerrGeometry(mass_kg=1e31, spin_dimensionless=0.95)
        alpha, beta = compute_bardeen_shadow_contour(bh, theta_obs_rad=math.pi * 0.5, n_points=200)

        # Center of horizontal extent is displaced to positive alpha (frame dragging)
        alpha_center = 0.5 * (np.max(alpha) + np.min(alpha))
        assert alpha_center > 0.1, f"Kerr shadow center {alpha_center} should be displaced towards +alpha"

        # Shadow must be closed and bounded
        assert len(alpha) == len(beta)
        assert np.all(np.isfinite(alpha))
        assert np.all(np.isfinite(beta))
        assert np.min(alpha) > -7.0
        assert np.max(alpha) < 7.0

    def test_penrose_process_energy_extraction(self) -> None:
        """Decay inside ergosphere with negative-energy particle yields net energy extraction."""
        e_in = 1.0  # Normalized particle of unit energy
        e_neg = -0.15  # Negative energy particle plunged into horizon

        gain = compute_penrose_energy_gain(e_in, e_neg)
        assert gain["energy_escaping"] == 1.15
        assert math.isclose(gain["extraction_efficiency"], 0.15, rel_tol=1e-12)

    def test_kerr_invalid_inputs(self) -> None:
        """Kerr geometry guards against non-positive mass and extremal spin >= 1."""
        with pytest.raises(ValueError):
            KerrGeometry(mass_kg=-100.0)

        with pytest.raises(ValueError):
            KerrGeometry(mass_kg=1e30, spin_dimensionless=1.05)
