"""Verification suite for Interstellar Astrometry, Relativistic Brachistochrones, and Benchmarks.

Validation Levels:
- Level 1: Gaia DR3 astrometric catalog and space-motion propagation.
- Level 2: Interstellar relativistic brachistochrone to Proxima Centauri (1.0g).
- Level 3: NASA JPL Horizons DE440 planetary state cross-validation (< 1 mm).
- Level 4: Automated validation certificate and cryptographic SHA-256 fingerprinting.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from relativistic_engine.constants import (
    C_LIGHT,
    G0,
    LIGHT_YEAR,
    SEC_PER_DAY,
)
from relativistic_engine.ephemeris.interstellar import (
    INTERSTELLAR_CATALOG,
    get_star_barycentric_state,
)
from relativistic_engine.trajectory.interstellar import (
    solve_interstellar_brachistochrone,
)
from relativistic_engine.validation.horizons_validator import (
    validate_ephemeris_against_horizons,
)
from relativistic_engine.validation.report_generator import (
    generate_validation_certificate,
    run_full_benchmark_suite,
)


# ==============================================================================
# LEVEL 1: GAIA DR3 ASTROMETRY & STELLAR KINEMATICS
# ==============================================================================


def test_level1_gaia_dr3_star_astrometry():
    """Level 1: Verify Gaia DR3 coordinates, parallax distance, and space motion."""
    # 1. Proxima Centauri distance check
    prox = INTERSTELLAR_CATALOG["proxima_centauri"]
    assert np.isclose(prox.distance_light_years, 4.2465, atol=0.01)

    # 2. State evaluation at catalog epoch vs 10 years later
    state_t0 = get_star_barycentric_state("proxima_centauri", prox.epoch_jd)
    assert np.isclose(state_t0.distance_light_years, 4.2465, atol=0.01)

    # 10 Julian years later
    jd_10yr = prox.epoch_jd + 10.0 * 365.25
    state_10yr = get_star_barycentric_state("proxima_centauri", jd_10yr)

    # Radial velocity is negative (-22.2 km/s approaching), so distance must strictly decrease
    assert state_10yr.distance_meters < state_t0.distance_meters
    expected_delta_r = prox.radial_velocity_m_s * (10.0 * 365.25 * SEC_PER_DAY)
    observed_delta_dist = state_10yr.distance_meters - state_t0.distance_meters

    # Over 10 years at 4.25 ly, radial motion dominates distance change
    assert np.isclose(observed_delta_dist, expected_delta_r, rtol=1e-3)

    # 3. Barnard's Star high proper motion
    barnard = INTERSTELLAR_CATALOG["barnards_star"]
    assert np.isclose(barnard.distance_light_years, 5.963, atol=0.02)
    state_b0 = get_star_barycentric_state("barnards_star", barnard.epoch_jd)
    state_b10 = get_star_barycentric_state("barnards_star", barnard.epoch_jd + 10.0 * 365.25)
    # Barnard's star approaches at ~110.5 km/s
    assert state_b10.distance_meters < state_b0.distance_meters


# ==============================================================================
# LEVEL 2: INTERSTELLAR RELATIVISTIC BRACHISTOCHRONE
# ==============================================================================


def test_level2_interstellar_brachistochrone_proxima_centauri():
    """Level 2: Earth -> Proxima Centauri 1g brachistochrone matches analytical SR & moving star."""
    res = solve_interstellar_brachistochrone(
        target_star="proxima_centauri",
        accel_proper=G0,
        departure_epoch_jd_tdb=2451545.0,  # J2000.0
        rtol=1e-8,
        atol=1e-9,
    )

    # 1. Flight times for Alpha Centauri at 1g:
    # Earth coordinate time ~ 5.8 - 6.0 years
    # Traveler proper time ~ 3.5 - 3.7 years
    assert 5.7 < res.coordinate_flight_time_years < 6.1
    assert 3.4 < res.proper_flight_time_years < 3.8

    # 2. Relativistic time dilation deficit: Delta t - Delta tau ~ 2.2 - 2.5 years
    assert 2.1 < res.time_deficit_years < 2.6
    assert np.isclose(
        res.time_deficit_years,
        res.coordinate_flight_time_years - res.proper_flight_time_years,
        rtol=1e-6,
    )

    # 3. Peak velocity at turnaround: v_peak ~ 0.95c, gamma ~ 3.2
    assert 0.94 < res.max_velocity_c < 0.96
    assert 3.0 < res.max_lorentz_factor < 3.5

    # 4. Relativistic Doppler factors
    # Redshifted to Earth: D < 0.20
    # Blueshifted toward star: D > 5.0
    assert 0.10 < res.max_doppler_redshift_earth < 0.20
    assert 5.0 < res.max_doppler_blueshift_target < 7.0

    # 5. Miss distance: must hit within 35 AU (< 0.012% relative distance error) over 4.25 light years
    assert res.miss_distance_meters < 5.0e12



# ==============================================================================
# LEVEL 3: NASA JPL HORIZONS DE440 CROSS-VALIDATION
# ==============================================================================


def test_level3_jpl_horizons_cross_validation():
    """Level 3: Cross-validate engine BCRS states against NASA JPL Horizons records."""
    report = validate_ephemeris_against_horizons(
        position_tolerance_m=1.0e-3,  # 1 mm
        velocity_tolerance_m_s=1.0e-6, # 1 um/s
    )

    assert report.is_valid, (
        f"Horizons validation failed: max dr = {report.max_position_residual_meters:.2e} m, "
        f"max dv = {report.max_velocity_residual_m_s:.2e} m/s"
    )
    assert report.total_checks >= 8
    assert report.max_position_residual_meters < 1.0e-3
    assert report.max_velocity_residual_m_s < 1.0e-6


# ==============================================================================
# LEVEL 4: VALIDATION CERTIFICATE & SHA-256 GENERATION
# ==============================================================================


def test_level4_validation_certificate_generation(tmp_path: Path):
    """Level 4: Verify certificate generation, SHA-256 signatures, and report persistence."""
    cert = generate_validation_certificate(output_dir=tmp_path)

    # 1. Certificate object status
    assert cert.is_certified is True
    assert cert.total_failed == 0
    assert cert.total_passed >= 5

    # 2. Verify file outputs
    json_file = tmp_path / "validation_certificate.json"
    md_file = tmp_path / "VALIDATION_CERTIFICATE.md"

    assert json_file.exists(), "JSON certificate not generated"
    assert md_file.exists(), "Markdown certificate not generated"

    # 3. Inspect JSON schema
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["is_certified"] is True
    assert "source_checksums" in data
    assert len(data["source_checksums"]) >= 10

    # 4. Verify SHA-256 hex format (64 hex characters)
    for filepath, sha in data["source_checksums"].items():
        if sha != "FILE_NOT_FOUND":
            assert len(sha) == 64
            assert all(c in "0123456789abcdefABCDEF" for c in sha)

    # 5. Inspect Markdown content
    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "# Relativistic Space Travel Computational Engine" in md_text
    assert "PASSED & CERTIFIED" in md_text
    assert "Level 1: SR Hyperbolic Time Invertibility" in md_text
    assert "Level 4: Earth -> Proxima Centauri Relativistic Flight" in md_text
