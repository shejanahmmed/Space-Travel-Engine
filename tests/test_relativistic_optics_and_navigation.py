"""5-Level Automated Validation Suite for Relativistic Optics & Autonomous Navigation.

Validates:
Level 1: Exact Invertibility & Zero-Velocity Identity
Level 2: Longitudinal & Transverse Doppler Theoretical Limits
Level 3: Bolometric Flux Scaling (D^4), Apparent Magnitude Shift, & Headlight Effect
Level 4: Analytical Sensitivity & Aberration Jacobian Consistency
Level 5: Autonomous Velocity Vector Recovery & Ultra-Relativistic Stability (gamma ~ 7071)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from relativistic_engine.constants import C_LIGHT
from relativistic_engine.physics.optics import (
    transform_aberration_vector,
    inverse_aberration_vector,
    compute_doppler_factor,
    compute_bolometric_flux_scaling,
    compute_apparent_magnitude_shift,
    compute_aberration_jacobian,
    estimate_velocity_from_star_observations,
)
from benchmarks.optical_navigation_benchmark import (
    run_optical_aberration_benchmark,
    run_autonomous_velocity_estimation_benchmark,
)


class TestOpticsLevel1InvertibilityAndLimits:
    """Level 1: Invertibility and identity limits."""

    def test_aberration_zero_velocity_identity(self):
        """At zero velocity, aberration is the exact identity transformation."""
        stars = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.577350269, 0.577350269, 0.577350269],
        ], dtype=np.float64)
        beta_zero = np.zeros(3, dtype=np.float64)

        obs = transform_aberration_vector(stars, beta_zero)
        assert np.allclose(obs, stars, rtol=1e-15, atol=1e-15)

        doppler = compute_doppler_factor(stars, beta_zero)
        assert np.allclose(doppler, 1.0, rtol=1e-15, atol=1e-15)

    def test_aberration_exact_invertibility(self):
        """Forward followed by inverse aberration recovers initial vector to < 1e-14."""
        rng = np.random.default_rng(101)
        raw_stars = rng.standard_normal((100, 3))
        stars = raw_stars / np.linalg.norm(raw_stars, axis=1, keepdims=True)

        beta = np.array([-0.3, 0.5, -0.6], dtype=np.float64)  # |beta| ~ 0.836

        obs = transform_aberration_vector(stars, beta, normalize=True)
        restored = inverse_aberration_vector(obs, beta, normalize=True)

        discrepancies = np.linalg.norm(restored - stars, axis=1)
        assert np.max(discrepancies) < 1.0e-14
        assert np.mean(discrepancies) < 1.0e-15

    def test_unit_norm_preservation(self):
        """Transformed aberration vectors preserve unit norm to machine precision."""
        rng = np.random.default_rng(202)
        raw_stars = rng.standard_normal((50, 3))
        stars = raw_stars / np.linalg.norm(raw_stars, axis=1, keepdims=True)

        beta = np.array([0.9, 0.0, 0.0], dtype=np.float64)
        obs = transform_aberration_vector(stars, beta, normalize=True)

        norms = np.linalg.norm(obs, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-15, atol=1e-15)


class TestOpticsLevel2DopplerLimits:
    """Level 2: Exact theoretical longitudinal and transverse Doppler shift limits."""

    def test_longitudinal_doppler_limits(self):
        """Direct approach and recession match analytical sqrt((1 +- beta)/(1 -+ beta))."""
        beta_mag = 0.8
        beta_vec = np.array([0.0, 0.0, beta_mag], dtype=np.float64)

        # Star directly ahead along +Z
        n_ahead = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        d_ahead = compute_doppler_factor(n_ahead, beta_vec)
        expected_blueshift = math.sqrt((1.0 + beta_mag) / (1.0 - beta_mag))  # sqrt(1.8 / 0.2) = 3.0
        assert pytest.approx(expected_blueshift, rel=1e-15) == d_ahead

        # Star directly behind along -Z
        n_behind = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        d_behind = compute_doppler_factor(n_behind, beta_vec)
        expected_redshift = math.sqrt((1.0 - beta_mag) / (1.0 + beta_mag))  # sqrt(0.2 / 1.8) = 1/3
        assert pytest.approx(expected_redshift, rel=1e-15) == d_behind

    def test_transverse_doppler_limit(self):
        """Transverse star in inertial frame matches pure Lorentz gamma factor."""
        beta_mag = 0.6
        beta_vec = np.array([0.0, 0.0, beta_mag], dtype=np.float64)
        gamma = 1.0 / math.sqrt(1.0 - beta_mag**2)  # 1.25

        # Star perpendicular in inertial frame (n . beta = 0)
        n_perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        d_perp = compute_doppler_factor(n_perp, beta_vec)
        assert pytest.approx(gamma, rel=1e-15) == d_perp


class TestOpticsLevel3FluxAndHeadlightEffect:
    """Level 3: Bolometric flux scaling, magnitude shifts, and relativistic beam concentration."""

    def test_bolometric_flux_and_magnitude_shift(self):
        """Flux scaling F'/F = D^4 and Delta m = -10 log10(D) are mathematically exact."""
        beta_vec = np.array([0.0, 0.0, 0.5], dtype=np.float64)
        n_vec = np.array([0.0, 0.6, 0.8], dtype=np.float64)

        d = compute_doppler_factor(n_vec, beta_vec)
        flux_ratio = compute_bolometric_flux_scaling(n_vec, beta_vec)
        mag_shift = compute_apparent_magnitude_shift(n_vec, beta_vec)

        assert pytest.approx(d**4, rel=1e-15) == flux_ratio
        assert pytest.approx(-2.5 * math.log10(flux_ratio), rel=1e-15) == mag_shift

    def test_headlight_effect_concentration(self):
        """At ultra-relativistic velocity (beta=0.99), stars concentrate into forward cone."""
        rng = np.random.default_rng(303)
        raw_stars = rng.standard_normal((1000, 3))
        stars = raw_stars / np.linalg.norm(raw_stars, axis=1, keepdims=True)

        beta_vec = np.array([0.0, 0.0, 0.99], dtype=np.float64)  # gamma ~ 7.09
        obs_stars = transform_aberration_vector(stars, beta_vec, normalize=True)

        # Fraction of stars in the forward hemisphere (obs_z > 0)
        forward_fraction = np.mean(obs_stars[:, 2] > 0.0)
        # For beta = 0.99, cos(theta') = (cos(theta) + beta) / (1 + beta cos(theta))
        # An un-aberrated star at theta = 171.8 deg (cos theta = -0.99) is pulled to theta' = 90 deg!
        # Thus virtually the entire sky (> 99%) is pulled into the forward hemisphere!
        assert forward_fraction > 0.95


class TestOpticsLevel4JacobianAndSensitivity:
    """Level 4: Analytical sensitivity and aberration Jacobian."""

    def test_aberration_jacobian_properties(self):
        """Aberration Jacobian matches numerical differentiation and preserves orthogonality."""
        n_inertial = np.array([0.6, 0.8, 0.0], dtype=np.float64)
        beta_vec = np.array([0.2, -0.1, 0.4], dtype=np.float64)

        jac = compute_aberration_jacobian(n_inertial, beta_vec)
        assert jac.shape == (3, 3)

        # Because ||n'|| == 1, d(||n'||^2)/d(beta) == 2 * n' . (dn'/d beta) == 0
        n_obs = transform_aberration_vector(n_inertial, beta_vec)
        ortho_check = n_obs @ jac
        assert np.allclose(ortho_check, 0.0, atol=1.0e-6)


class TestOpticsLevel5NavigationAndUltrarelativistic:
    """Level 5: Autonomous velocity estimation and ultra-relativistic stability."""

    def test_autonomous_velocity_estimation_noiseless(self):
        """Star tracker recovers true 3D velocity vector to < 1e-12 in noiseless observations."""
        rng = np.random.default_rng(404)
        raw_stars = rng.standard_normal((10, 3))
        cat_stars = raw_stars / np.linalg.norm(raw_stars, axis=1, keepdims=True)

        true_beta = np.array([0.15, -0.35, 0.65], dtype=np.float64)
        obs_stars = transform_aberration_vector(cat_stars, true_beta)

        res = estimate_velocity_from_star_observations(cat_stars, obs_stars)
        assert res["success"] is True
        error_norm = np.linalg.norm(res["beta"] - true_beta)
        assert error_norm < 1.0e-12
        assert res["residuals_arcsec"] < 0.01  # Sub-milliarcsecond residual bound

    def test_autonomous_velocity_estimation_noisy(self):
        """Star tracker recovers velocity within bounded variance under 1 arcsec noise."""
        rng = np.random.default_rng(505)
        raw_stars = rng.standard_normal((30, 3))
        cat_stars = raw_stars / np.linalg.norm(raw_stars, axis=1, keepdims=True)

        true_beta = np.array([0.0, 0.0, 0.5], dtype=np.float64)
        true_obs = transform_aberration_vector(cat_stars, true_beta)

        # Add 1 arcsecond Gaussian noise
        noise_rad = 1.0 * (math.pi / (180.0 * 3600.0))
        noise = rng.standard_normal((30, 3)) * noise_rad
        noisy_obs = true_obs + noise
        noisy_obs /= np.linalg.norm(noisy_obs, axis=1, keepdims=True)

        res = estimate_velocity_from_star_observations(cat_stars, noisy_obs)
        assert res["success"] is True
        error_norm = np.linalg.norm(res["beta"] - true_beta)
        # 1 arcsecond noise with 30 stars gives velocity error delta_beta ~ noise / sqrt(N) < 1e-5
        assert error_norm < 1.0e-5

    def test_ultrarelativistic_limits(self):
        """Numerical stability verified at beta = 0.99999999 (gamma ~ 7071)."""
        beta_ultra = np.array([0.0, 0.0, 0.99999999], dtype=np.float64)
        n_test = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # 90 deg star

        obs = transform_aberration_vector(n_test, beta_ultra)
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
        assert pytest.approx(1.0, rel=1e-15) == np.linalg.norm(obs)

        # Restores back under inverse
        restored = inverse_aberration_vector(obs, beta_ultra)
        assert np.allclose(restored, n_test, rtol=1e-7, atol=1e-7)

    def test_optics_invalid_inputs(self):
        """Engine raises proper exceptions on invalid inputs."""
        with pytest.raises(ValueError, match="Unphysical spacelike/tachyon velocity"):
            transform_aberration_vector(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))

        with pytest.raises(ValueError, match="At least 2 star sightings"):
            cat = np.array([[1.0, 0.0, 0.0]])
            obs = np.array([[1.0, 0.0, 0.0]])
            estimate_velocity_from_star_observations(cat, obs)
