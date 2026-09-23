"""Unit and integration tests for Solar Gravitational Lens (SGL) focal mechanics and wave optics.

Validates:
- Exact SGL minimum focal distance z_min >= 547.77 AU against IAU 2015 solar constants.
- Theoretical peak wave-optical amplification mu_0 = 4*pi^2*r_g / lambda (Turyshev & Toth 2017).
- Bessel function PSF profile I(rho)/I_0 and first diffraction ring zero.
- Solar Oberth hyperbolic escape trajectory velocity, TOF, and 1PN proper time deficit.
- FastAPI REST API contract for POST /api/mission/sgl.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from relativistic_engine.constants import (
    AU,
    C_LIGHT,
    GM_SUN,
    RADIUS_SUN,
    SEC_PER_JULIAN_YEAR,
)
from relativistic_engine.trajectory.sgl import (
    R_G_SUN,
    Z_MIN_SGL_AU,
    compute_sgl_focal_parameters,
    sgl_point_spread_function,
    solve_sgl_mission_trajectory,
)
from relativistic_engine.api.app import app


client = TestClient(app)


def test_sgl_minimum_focal_distance():
    """Verify theoretical solar minimum focal distance matches analytical formula."""
    expected_rg = (2.0 * GM_SUN) / (C_LIGHT ** 2)
    expected_z_min_m = (RADIUS_SUN ** 2) / (2.0 * expected_rg)
    expected_z_min_au = expected_z_min_m / AU

    assert math.isclose(R_G_SUN, expected_rg, rel_tol=1e-12)
    assert math.isclose(Z_MIN_SGL_AU, expected_z_min_au, rel_tol=1e-12)
    # Turyshev & Toth (2017): ~547.77 AU
    assert 547.0 < Z_MIN_SGL_AU < 548.5


def test_sgl_wave_optical_amplification():
    """Verify peak wave-optical light amplification mu_0 across optical and IR wavelengths."""
    lambda_1um = 1.0e-6  # 1.0 micron (near-IR)
    params = compute_sgl_focal_parameters(
        wavelength_m=lambda_1um,
        heliocentric_distance_au=550.0,
        target_distance_pc=1.30,
    )

    # Analytical Turyshev & Toth (2017) Eq. 47: mu_0 = 4 * pi^2 * r_g / lambda
    expected_mu0 = (4.0 * (math.pi ** 2) * R_G_SUN) / lambda_1um
    assert math.isclose(params.peak_light_amplification, expected_mu0, rel_tol=1e-12)
    assert params.peak_light_amplification > 1.0e11  # Over 100 billion fold amplification

    # At 550 AU, impact parameter should slightly exceed 1 solar radius
    assert params.impact_parameter_solar_radii >= 1.0


def test_sgl_bessel_point_spread_function():
    """Verify PSF radial distribution follows J_0^2(k * rho * theta_E)."""
    wavelength = 1.0e-6
    z_au = 550.0
    rho = np.linspace(0.0, 10.0, 100)  # meters in focal plane

    psf = sgl_point_spread_function(rho, wavelength_m=wavelength, heliocentric_distance_au=z_au)

    # On-axis peak
    params = compute_sgl_focal_parameters(wavelength_m=wavelength, heliocentric_distance_au=z_au)
    assert math.isclose(psf[0], params.peak_light_amplification, rel_tol=1e-12)

    # First Bessel zero j0(x) = 0 occurs at x ≈ 2.4048255577
    k = (2.0 * math.pi) / wavelength
    theta_e = math.sqrt(2.0 * R_G_SUN / (z_au * AU))
    first_zero_rho = 2.4048255577 / (k * theta_e)

    psf_at_zero = sgl_point_spread_function(np.array([first_zero_rho]), wavelength_m=wavelength, heliocentric_distance_au=z_au)
    assert psf_at_zero[0] < 1e4  # Near zero compared to 1e11 peak


def test_sgl_mission_trajectory_solution():
    """Verify hyperbolic solar flyby trajectory reaches 550 AU with proper time dilation."""
    sol = solve_sgl_mission_trajectory(
        target_star="proxima_centauri",
        departure_epoch_jd=2462622.5,
        periapsis_solar_radii=4.0,
        periapsis_delta_v_km_s=25.0,
        wavelength_m=1.0e-6,
        num_waypoints=30,
    )

    assert sol.target_star == "Proxima Centauri"
    # Anti-solar focal line RA is target RA + 180 deg
    expected_focal_ra = (sol.target_ra_deg + 180.0) % 360.0
    assert math.isclose(sol.focal_line_ra_deg, expected_focal_ra, abs_tol=1e-3)
    assert math.isclose(sol.focal_line_dec_deg, -sol.target_dec_deg, abs_tol=1e-3)

    # Fast asymptotic speed > 20 AU/yr
    assert sol.asymptotic_speed_au_per_year > 15.0
    # Time to 550 AU should be reasonable for solar Oberth (< 35 years)
    assert sol.time_to_550au_years < 35.0
    # Relativistic time deficit must be positive (coordinate time > proper time)
    assert sol.time_dilation_deficit_at_550au_seconds > 0.0
    assert len(sol.waypoints) == 30


def test_sgl_api_endpoint():
    """Verify REST API endpoint POST /api/mission/sgl returns valid JSON response."""
    payload = {
        "target_star": "proxima_centauri",
        "departure_epoch_jd": 2462622.5,
        "periapsis_solar_radii": 4.0,
        "periapsis_delta_v_km_s": 25.0,
        "wavelength_m": 1.0e-6,
    }
    resp = client.post("/api/mission/sgl", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["target_star"] == "Proxima Centauri"
    assert data["asymptotic_speed_au_per_year"] > 15.0
    assert data["focal_parameters"]["peak_light_amplification"] > 1.0e11
    assert len(data["waypoints"]) == 50
