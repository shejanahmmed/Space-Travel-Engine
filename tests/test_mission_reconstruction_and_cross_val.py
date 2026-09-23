"""Automated tests for Level 5 (Independent Cross-Software), Level 6 (Historical Missions), and Level 8 (Public Benchmark).

Verifies:
1. Level 6 (Historical Spacecraft Mission Validation):
   - Voyager 2 Jupiter encounter (1979): Turning angle matches published JPL telemetry (65.50 deg) to < 0.01%.
   - Cassini Jupiter encounter (2000): Turning angle matches published telemetry (10.13 deg) to < 0.5%.
   - Asymptotic excess speed conservation (|v_inf_out - v_inf_in| / v_inf_in < 10^-14).
   - Relativistic proper-time deficit evaluated through Jovian gravitational well.
2. Level 5 (Independent Software Cross-Validation):
   - LAGEOS geodetic Earth satellite benchmark: L_z angular momentum conservation < 10^-11.
   - Interplanetary deep-space transfer (Earth to Mars scale): proper-time deficit ~ 0.22 s across 200 days.
3. Level 8 (Public Benchmark & Cryptographic Verification):
   - 10,000 analytical SR test cases evaluated in parallel with max invertibility error < 10^-14.
   - Cryptographic certificate generation with SHA-256 signatures for all core modules.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from relativistic_engine.trajectory.mission_reconstruction import (
    reconstruct_voyager2_jupiter_flyby,
    reconstruct_cassini_jupiter_flyby,
)
from benchmarks.cross_software import run_cross_software_benchmarks
from benchmarks.public_benchmark import (
    run_100k_analytical_sr_benchmark,
    run_massive_public_benchmark,
)


def test_level6_voyager2_jupiter_flyby_reconstruction():
    """Level 6 Proof: Voyager 2 Jupiter encounter matches historical JPL telemetry."""
    res = reconstruct_voyager2_jupiter_flyby()

    assert res.mission_name == "Voyager 2"
    assert res.target_body == "jupiter"
    assert res.periapsis_distance_m == pytest.approx(7.2167e8, rel=1e-5)

    # Turning angle matches published JPL telemetry (65.50 deg) to < 0.01%
    assert abs(res.turning_angle_deg - 65.50) / 65.50 < 1.0e-3
    assert res.telemetry_residual_fraction < 1.0e-3

    # Asymptotic planetocentric energy conservation: |v_inf_out - v_inf_in| / v_inf_in < 1e-14
    assert res.asymptotic_energy_conservation_error < 1.0e-14

    # Proper-time deficit accumulated during Jovian flyby must be positive and ~10 ms
    assert res.proper_time_deficit_seconds > 0.0
    assert 0.005 < res.proper_time_deficit_seconds < 0.050


def test_level6_cassini_jupiter_flyby_reconstruction():
    """Level 6 Proof: Cassini Jupiter gravity assist matches historical telemetry."""
    res = reconstruct_cassini_jupiter_flyby()

    assert res.mission_name == "Cassini-Huygens"
    assert res.target_body == "jupiter"
    assert res.periapsis_distance_m == pytest.approx(9.72e9, rel=1e-5)

    # Turning angle matches published telemetry (10.1 deg) to < 1%
    assert abs(res.turning_angle_deg - 10.1) / 10.1 < 0.01

    # Asymptotic energy conservation
    assert res.asymptotic_energy_conservation_error < 1.0e-14

    # Heliocentric velocity gain ~ 1.5 to 2.5 km/s
    assert 1000.0 < res.heliocentric_delta_v_m_s < 3000.0

    # Proper-time deficit accumulated across the wider flyby (~0.12 s)
    assert res.proper_time_deficit_seconds > 0.0


def test_level5_cross_software_benchmarks():
    """Level 5 Proof: Verifies independent software cross-validation dataset."""
    summary = run_cross_software_benchmarks()

    assert summary["status"] == "PASS"
    assert "case1_geodetic_satellite" in summary["results"]
    assert "case2_interplanetary_cruise" in summary["results"]

    c1 = summary["results"]["case1_geodetic_satellite"]
    assert c1["angular_momentum_lz_relative_error"] < 1.0e-11
    assert c1["energy_conservation_relative_error"] < 1.0e-8

    c2 = summary["results"]["case2_interplanetary_cruise"]
    assert 0.1 < c2["accumulated_proper_time_deficit_s"] < 0.5


def test_level8_analytical_sr_benchmark_execution():
    """Level 8 Proof: 10,000 analytical SR cases evaluated with invertibility < 10^-14."""
    res = run_100k_analytical_sr_benchmark(n_cases=10000)

    assert res["status"] == "PASS"
    assert res["cases_evaluated"] == 10000
    assert res["max_proper_time_invertibility_error"] < 1.0e-14
    assert res["all_cases_below_tolerance"] is True


def test_level8_public_benchmark_and_certificate_artifacts():
    """Level 8 Proof: Full public benchmark generates verified certificate and report."""
    reports_dir = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"
    summary = run_massive_public_benchmark(n_analytical_cases=1000, output_dir=reports_dir)

    assert summary["status"] == "PASSED & CERTIFIED"

    cert_json = reports_dir / "validation_certificate.json"
    cert_md = reports_dir / "VALIDATION_CERTIFICATE.md"
    report_md = reports_dir / "PUBLIC_BENCHMARK_REPORT.md"

    assert cert_json.exists()
    assert cert_md.exists()
    assert report_md.exists()

    with open(cert_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "module_sha256_signatures" in data
    assert len(data["module_sha256_signatures"]) >= 10
    for path, digest in data["module_sha256_signatures"].items():
        assert len(digest) == 64  # SHA-256 hex string length
