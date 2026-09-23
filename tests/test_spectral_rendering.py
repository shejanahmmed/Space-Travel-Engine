"""Validation and Unit Tests for Relativistic Spectral Rendering and Ray-Tracing.

Validates:
1. Planck spectral radiance and numerical stability.
2. Wien displacement peak and Stefan-Boltzmann integral convergence.
3. Peebles-Wilkinson blackbody temperature invariance under Lorentz boost:
   I'_lambda(lambda', D * T_0) * D^4 = I_lambda(lambda'/D, T_0) * D^5.
4. CIE 1931 color matching functions and Illuminant E white-point neutrality.
5. McKinley & Doherty (1979) full-sky photometric flux boost invariant:
   gamma * (1 + 1/3 * beta^2).
6. Relativistic starbow chromatic temperature gradient across line of sight.
7. Pinhole camera ray-tracer star projection, aberration displacement, and PSF integration.
8. Full-sky equirectangular panorama rendering.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    C_LIGHT,
    H_PLANCK,
    K_BOLTZMANN,
    SIGMA_SB,
    WIEN_B,
)
from relativistic_engine.physics.optics import compute_doppler_factor
from relativistic_engine.physics.spectral_rendering import (
    CIE_WAVELENGTHS_M,
    CelestialStar,
    analytical_sky_flux_boost_factor,
    blackbody_to_srgb,
    boosted_blackbody_temperature,
    cie_xyz_to_srgb,
    numerical_sky_flux_boost_factor,
    planck_spectral_radiance,
    render_relativistic_panorama,
    render_relativistic_starfield_pinhole,
    spectrum_to_cie_xyz,
    wien_peak_wavelength,
)


class TestPlanckRadianceAndThermodynamics:
    """Level 1: Fundamental analytical thermodynamic and radiation laws."""

    def test_planck_radiance_positive_and_stable(self) -> None:
        """Planck radiance must be strictly non-negative and finite."""
        lambdas = np.geomspace(1e-9, 1e-3, 500)
        temp = 5778.0  # Solar effective temperature
        b_lam = planck_spectral_radiance(lambdas, temp)
        
        assert np.all(b_lam >= 0.0)
        assert np.all(np.isfinite(b_lam))
        assert np.max(b_lam) > 0.0

    def test_planck_extreme_frequency_overflow_guard(self) -> None:
        """High-frequency / cryogenic limits should smoothly underflow to 0."""
        # Extremely short wavelength: 1e-12 m (gamma-ray) at 1 Kelvin
        b_lam = planck_spectral_radiance(1e-12, 1.0)
        assert b_lam == 0.0

        # Zero or negative inputs
        assert planck_spectral_radiance(0.0, 5000.0) == 0.0
        assert planck_spectral_radiance(500e-9, 0.0) == 0.0
        assert planck_spectral_radiance(-500e-9, 5000.0) == 0.0

    def test_wien_displacement_law(self) -> None:
        """Peak of B_lambda must match Wien's displacement law lambda_max = b / T."""
        temperatures = [2000.0, 5778.0, 10000.0, 30000.0]
        for t in temperatures:
            lam_peak_analytical = wien_peak_wavelength(t)
            # Evaluate fine grid around analytical peak
            fine_lambdas = np.linspace(lam_peak_analytical * 0.95, lam_peak_analytical * 1.05, 5000)
            rad_profile = planck_spectral_radiance(fine_lambdas, t)
            peak_idx = int(np.argmax(rad_profile))
            lam_peak_numerical = fine_lambdas[peak_idx]

            rel_err = abs(lam_peak_numerical - lam_peak_analytical) / lam_peak_analytical
            assert rel_err < 1e-4, f"Wien peak error {rel_err} exceeds 1e-4 for T={t}"

    def test_stefan_boltzmann_law_integration(self) -> None:
        """Integral over all wavelengths must equal sigma * T^4 / pi."""
        temp = 5778.0
        # Integrate from 50 nm to 50 microns with dense geometric sampling
        lambdas = np.geomspace(5e-8, 5e-5, 20000)
        rad = planck_spectral_radiance(lambdas, temp)
        integral_numerical = np.trapezoid(rad, lambdas)

        # Theoretical hemispherical radiance: B = sigma * T^4 / pi
        b_theoretical = (SIGMA_SB * (temp ** 4)) / math.pi

        # Trapezoidal quadrature across finite window captures > 99.9% of total flux
        rel_diff = abs(integral_numerical - b_theoretical) / b_theoretical
        assert rel_diff < 0.002, f"Stefan-Boltzmann integral error {rel_diff} exceeds 0.2%"


class TestPeeblesWilkinsonInvariance:
    """Level 2: Peebles-Wilkinson (1968) relativistic blackbody transformation."""

    def test_blackbody_temperature_boost(self) -> None:
        """T' = D * T_0 matches analytical formula."""
        t0 = 5778.0
        d_vals = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
        t_boosted = boosted_blackbody_temperature(t0, d_vals)
        expected = t0 * d_vals
        np.testing.assert_allclose(t_boosted, expected, rtol=1e-15)

    def test_planck_doppler_invariance_identity(self) -> None:
        """Lorentz-boosted Planck spectrum satisfies I'_lambda = D^5 * B_lambda(D * lambda', T_0) == B_lambda(lambda', D * T_0).

        By relativistic phase space invariance I_nu / nu^3 = inv:
            I'_lambda(lambda') = D^5 * B_lambda(D * lambda', T_0)
        Peebles & Wilkinson (1968) proved this is identically equal to the unperturbed
        Planck distribution at the boosted temperature T' = D * T_0:
            B_lambda(lambda', D * T_0)
        """
        t0 = 4500.0
        d = 2.5
        lambdas_prime = np.geomspace(200e-9, 2000e-9, 100)

        # Direct Lorentz boost transformation from rest-frame spectrum
        i_prime_direct = (d ** 5) * planck_spectral_radiance(d * lambdas_prime, t0)

        # Blackbody evaluated at Peebles-Wilkinson boosted temperature T' = D * T0
        t_prime = t0 * d
        i_prime_peebles = planck_spectral_radiance(lambdas_prime, t_prime)

        # Must be identical to machine precision across all wavelengths
        np.testing.assert_allclose(i_prime_direct, i_prime_peebles, rtol=1e-14)


class TestCIEColorimetryAndStarbow:
    """Level 3: Colorimetric perception, Illuminant E, and starbow progression."""

    def test_illuminant_e_white_point(self) -> None:
        """An equal-energy flat spectrum should produce CIE Illuminant E (x=1/3, y=1/3)."""
        flat_spectrum = np.ones_like(CIE_WAVELENGTHS_M)
        xyz, xy = spectrum_to_cie_xyz(CIE_WAVELENGTHS_M, flat_spectrum)

        # Standard CIE 1931 tabulated 5 nm quadrature reproduces (1/3, 1/3) within 5e-4
        assert abs(xy[0] - 1.0 / 3.0) < 5e-4
        assert abs(xy[1] - 1.0 / 3.0) < 5e-4

    def test_stellar_temperature_color_progression(self) -> None:
        """Hot stars must be blue-dominated, cool stars must be red-dominated."""
        # Check normalized CIE tristimulus ratios (Z / X)
        rad_hot = planck_spectral_radiance(CIE_WAVELENGTHS_M, 25000.0)
        xyz_hot, _ = spectrum_to_cie_xyz(CIE_WAVELENGTHS_M, rad_hot)
        # Blue (Z) dominates over Red (X) for hot O-star
        assert xyz_hot[2] > xyz_hot[0], f"Hot star XYZ {xyz_hot} should have Z > X"

        rad_cool = planck_spectral_radiance(CIE_WAVELENGTHS_M, 3000.0)
        xyz_cool, _ = spectrum_to_cie_xyz(CIE_WAVELENGTHS_M, rad_cool)
        # Red (X) dominates over Blue (Z) for cool M-star
        assert xyz_cool[0] > xyz_cool[2], f"Cool star XYZ {xyz_cool} should have X > Z"

        # Check tone-mapped sRGB colors at balanced exposure
        rgb_hot = blackbody_to_srgb(25000.0, exposure=1e-14)
        rgb_cool = blackbody_to_srgb(3000.0, exposure=1e-11)
        assert rgb_hot[2] > rgb_hot[0], f"Hot star sRGB {rgb_hot} should have B > R"
        assert rgb_cool[0] > rgb_cool[2], f"Cool star sRGB {rgb_cool} should have R > B"

    def test_starbow_monotonic_chromatic_shift(self) -> None:
        """At beta = 0.8, Doppler factor decreases monotonically from ahead to astern."""
        beta = np.array([0.0, 0.0, 0.8])
        t0 = 5778.0  # Sun-like G2V star

        # Sample stars at theta = 0 (ahead), 45 deg, 90 deg (transverse), 135 deg, 180 deg (behind)
        angles_deg = [0.0, 45.0, 90.0, 135.0, 180.0]
        dopplers = []
        temps = []

        for deg in angles_deg:
            rad = math.radians(deg)
            # n points toward source
            n = np.array([math.sin(rad), 0.0, math.cos(rad)])
            d = compute_doppler_factor(n, beta)
            t_boosted = float(boosted_blackbody_temperature(t0, d))
            dopplers.append(d)
            temps.append(t_boosted)

        # Monotonically decreasing Doppler and temperature from head to tail
        for i in range(len(angles_deg) - 1):
            assert dopplers[i] > dopplers[i + 1]
            assert temps[i] > temps[i + 1]

        # Forward star (theta = 0): D = sqrt((1+beta)/(1-beta)) = 3.0
        assert math.isclose(dopplers[0], 3.0, rel_tol=1e-12)
        assert math.isclose(temps[0], 3.0 * t0, rel_tol=1e-12)

        # Transverse star (theta = 90): D = gamma = 1 / sqrt(1 - 0.64) = 1 / 0.6 = 5/3 approx 1.6667
        assert math.isclose(dopplers[2], 5.0 / 3.0, rel_tol=1e-12)

        # Astern star (theta = 180): D = sqrt((1-beta)/(1+beta)) = 1/3
        assert math.isclose(dopplers[4], 1.0 / 3.0, rel_tol=1e-12)
        assert math.isclose(temps[4], t0 / 3.0, rel_tol=1e-12)


class TestFullSkyFluxInvariant:
    """Level 4: McKinley & Doherty (1979) full-sky flux invariant."""

    @pytest.mark.parametrize("beta", [0.0, 0.1, 0.3, 0.5, 0.8, 0.95])
    def test_sky_flux_numerical_vs_analytical(self, beta: float) -> None:
        """Numerical integration of D^4 over sphere matches gamma * (1 + 1/3 * beta^2)."""
        exact = analytical_sky_flux_boost_factor(beta)
        num = numerical_sky_flux_boost_factor(beta, n_theta=15000)

        rel_err = abs(num - exact) / exact
        assert rel_err < 1e-4, f"Sky flux error {rel_err} exceeds 1e-4 at beta={beta}"

    def test_sky_flux_rest_frame_identity(self) -> None:
        """At beta = 0, flux ratio must be exactly 1.0."""
        assert analytical_sky_flux_boost_factor(0.0) == 1.0
        assert math.isclose(numerical_sky_flux_boost_factor(0.0), 1.0, rel_tol=1e-12)


class TestRayTracingAndCameraProjection:
    """Level 5: Pinhole camera viewport and panoramic equirectangular ray-tracing."""

    def test_pinhole_empty_starfield(self) -> None:
        """Empty star catalog produces pure zero image."""
        beta = np.array([0.0, 0.0, 0.5])
        img = render_relativistic_starfield_pinhole([], beta, resolution=(64, 64))
        assert img.shape == (64, 64, 3)
        assert np.all(img == 0.0)

    def test_pinhole_center_star_projection(self) -> None:
        """A star located directly along optical axis (+Z) projects to image center."""
        # Single star at +Z
        star = CelestialStar(
            direction=np.array([0.0, 0.0, 1.0]),
            temperature_k=5778.0,
            visual_magnitude=0.0,
            name="CentroidStar",
        )
        beta_rest = np.zeros(3)
        img = render_relativistic_starfield_pinhole([star], beta_rest, resolution=(65, 65))
        
        # Center pixel (32, 32) must be the maximum intensity
        center_val = np.sum(img[32, 32, :])
        corner_val = np.sum(img[0, 0, :])
        assert center_val > 0.0
        assert center_val > corner_val * 100.0

    def test_pinhole_relativistic_headlight_pull(self) -> None:
        """Aberration pulls off-axis stars inward toward the center of the camera FOV."""
        # Star at 35 degrees off-axis in inertial frame (outside a 40-deg FOV camera)
        angle_rad = math.radians(35.0)
        n_inertial = np.array([math.sin(angle_rad), 0.0, math.cos(angle_rad)])
        star = CelestialStar(
            direction=n_inertial,
            temperature_k=6000.0,
            visual_magnitude=1.0,
        )

        # At rest, star is outside 40-degree horizontal FOV (half-angle 20 deg < 35 deg)
        img_rest = render_relativistic_starfield_pinhole(
            [star],
            np.zeros(3),
            fov_deg=40.0,
            resolution=(64, 64),
        )
        assert np.max(img_rest) == 0.0, "Star should be outside FOV at rest"

        # Moving forward at beta = 0.8:
        # tan(theta'/2) = sqrt((1-beta)/(1+beta)) * tan(theta/2)
        # theta' = 12.3 degrees < 20 degrees! Star enters the camera FOV.
        beta_fwd = np.array([0.0, 0.0, 0.8])
        img_boost = render_relativistic_starfield_pinhole(
            [star],
            beta_fwd,
            fov_deg=40.0,
            resolution=(64, 64),
        )
        assert np.max(img_boost) > 0.0, "Star should enter FOV under relativistic aberration"

    def test_panoramic_equirectangular_rendering(self) -> None:
        """Panoramic full-sky render executes cleanly and produces normalized [0, 1] sRGB."""
        stars = [
            CelestialStar(direction=np.array([1.0, 0.0, 0.0]), temperature_k=9000.0, visual_magnitude=0.0),
            CelestialStar(direction=np.array([0.0, 1.0, 0.0]), temperature_k=4000.0, visual_magnitude=1.5),
            CelestialStar(direction=np.array([0.0, 0.0, 1.0]), temperature_k=6000.0, visual_magnitude=-0.5),
        ]
        beta = np.array([0.0, 0.0, 0.7])
        pano = render_relativistic_panorama(stars, beta, resolution=(32, 64))

        assert pano.shape == (32, 64, 3)
        assert np.all(pano >= 0.0)
        assert np.all(pano <= 1.0)
        assert np.max(pano) > 0.0
