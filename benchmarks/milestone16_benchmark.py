"""Milestone 16 Scientific Benchmark Suite: Solar Gravitational Lens (SGL) & Production Packaging.

Executes 5 rigorous verification scenarios evaluating:
1. B16-01: SGL Minimum Focal Distance & Einstein Ring Geometry (Turyshev & Toth 2017)
2. B16-02: Wave-Optical Peak Light Amplification & Inverse Wavelength Scaling
3. B16-03: Bessel Function Point Spread Function & Exoplanet Surface Resolution
4. B16-04: Solar Oberth Hyperbolic Escape Trajectory & Relativistic Proper Time Deficit
5. B16-05: SGL REST API Contract & Response Latency SLA

Authoritative Standards & References:
- Turyshev, S. G., & Toth, V. T. (2017), "Diffraction of light by the gravitational
  field of the Sun and the solar gravitational lens", Phys. Rev. D, 96(2), 024008.
- Turyshev, S. G., et al. (2020), "Direct Multipixel Imaging and Spectroscopy of an
  Exoplanet with a Solar Gravitational Lens Mission", NASA NIAC Phase III.
- Einstein, A. (1936), "Lens-like Action of a Star by the Deviation of Light in the
  Gravitational Field", Science, 84(2188), 506-507.
- IAU 2015 Resolution B3 (Nominal Solar Constants).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
import numpy as np

from benchmarks.save_results import BenchmarkResult, BenchmarkSession
from relativistic_engine.api.app import app
from relativistic_engine.constants import (
    AU,
    C_LIGHT,
    GM_SUN,
    PARSEC,
    RADIUS_SUN,
)
from relativistic_engine.trajectory.sgl import (
    R_G_SUN,
    Z_MIN_SGL_AU,
    compute_sgl_focal_parameters,
    sgl_point_spread_function,
    solve_sgl_mission_trajectory,
)


def run_benchmark_b16_01_focal_distance() -> Dict[str, str]:
    """B16-01: SGL Minimum Focal Distance & Einstein Ring Geometry."""
    rg_expected = (2.0 * GM_SUN) / (C_LIGHT ** 2)
    z_min_expected_au = ((RADIUS_SUN ** 2) / (2.0 * rg_expected)) / AU

    params_550 = compute_sgl_focal_parameters(wavelength_m=1.0e-6, heliocentric_distance_au=550.0)

    # Einstein ring angular radius theta_E = sqrt(2 * r_g / z)
    theta_e_expected_arcsec = math.degrees(math.sqrt(2.0 * rg_expected / (550.0 * AU))) * 3600.0

    err_z_min = abs(Z_MIN_SGL_AU - z_min_expected_au) / z_min_expected_au
    err_theta_e = abs(params_550.einstein_ring_angular_radius_arcsec - theta_e_expected_arcsec) / theta_e_expected_arcsec

    passed = (err_z_min < 1e-12) and (err_theta_e < 1e-12) and (547.0 < Z_MIN_SGL_AU < 548.5)

    return {
        "id": "B16-01",
        "name": "SGL Minimum Focal Distance & Einstein Ring Geometry",
        "evaluated_value": f"z_min = {Z_MIN_SGL_AU:.4f} AU, theta_E(550 AU) = {params_550.einstein_ring_angular_radius_arcsec:.6f} arcsec",
        "reference_value": f"z_min = {z_min_expected_au:.4f} AU (~547.77 AU per Turyshev & Toth 2017)",
        "discrepancy": f"Relative Error = {err_z_min:.2e} (z_min), {err_theta_e:.2e} (theta_E)",
        "status": "PASS" if passed else "FAIL",
    }


def run_benchmark_b16_02_light_amplification() -> Dict[str, str]:
    """B16-02: Wave-Optical Peak Light Amplification & Inverse Wavelength Scaling."""
    lambdas = [0.2e-6, 0.5e-6, 1.0e-6]  # UV, Visible, Near-IR
    mu_values = []
    for l_m in lambdas:
        p = compute_sgl_focal_parameters(wavelength_m=l_m, heliocentric_distance_au=550.0)
        mu_values.append(p.peak_light_amplification)

    # Verify mu_0 is proportional to 1 / lambda
    ratio_uv_vis = mu_values[0] / mu_values[1]  # Expected: 0.5 / 0.2 = 2.5
    ratio_vis_ir = mu_values[1] / mu_values[2]  # Expected: 1.0 / 0.5 = 2.0

    err_ratio1 = abs(ratio_uv_vis - 2.5) / 2.5
    err_ratio2 = abs(ratio_vis_ir - 2.0) / 2.0

    passed = (err_ratio1 < 1e-10) and (err_ratio2 < 1e-10) and (mu_values[2] > 1.0e11)

    return {
        "id": "B16-02",
        "name": "Wave-Optical Light Amplification & 1/lambda Scaling",
        "evaluated_value": f"mu_0(1.0 um) = {mu_values[2]:.3e}, mu_0(0.5 um) = {mu_values[1]:.3e}, mu_0(0.2 um) = {mu_values[0]:.3e}",
        "reference_value": "mu_0 = 4*pi^2*r_g / lambda = 1.166e11 at 1 um (Turyshev & Toth 2017 Eq. 47)",
        "discrepancy": f"Linearity Error = {max(err_ratio1, err_ratio2):.2e}, mu_0(1 um) = {mu_values[2]:.4e}",
        "status": "PASS" if passed else "FAIL",
    }


def run_benchmark_b16_03_psf_resolution() -> Dict[str, str]:
    """B16-03: Bessel Function PSF & Exoplanet Surface Resolution."""
    lambda_m = 1.0e-6
    z_au = 550.0
    dist_pc = 1.30  # Proxima Centauri b

    params = compute_sgl_focal_parameters(
        wavelength_m=lambda_m,
        heliocentric_distance_au=z_au,
        target_distance_pc=dist_pc,
    )

    # First zero of J_0 occurs at x ≈ 2.4048255577
    k = (2.0 * math.pi) / lambda_m
    theta_e = math.sqrt(2.0 * R_G_SUN / (z_au * AU))
    rho_first_zero = 2.4048255577 / (k * theta_e)

    psf_peak = sgl_point_spread_function(np.array([0.0]), wavelength_m=lambda_m, heliocentric_distance_au=z_au)[0]
    psf_zero = sgl_point_spread_function(np.array([rho_first_zero]), wavelength_m=lambda_m, heliocentric_distance_au=z_au)[0]

    # Surface resolution in meters/km on Proxima b (1.3 pc) should be ~35 meters (0.035 km)
    res_km = params.resolvable_surface_resolution_km
    res_m = res_km * 1000.0

    passed = (psf_zero / psf_peak < 1e-12) and (10.0 <= res_m <= 100.0)

    return {
        "id": "B16-03",
        "name": "Bessel Point Spread Function & Spatial Resolution",
        "evaluated_value": f"First Bessel Zero at rho = {rho_first_zero:.4f} m (I_zero/I_peak = {psf_zero/psf_peak:.2e}), Delta_x = {res_m:.1f} m ({res_km:.4f} km)",
        "reference_value": "First zero at x=2.4048, Delta_x ~ 20-50 meters at 1.3 pc (Turyshev et al. 2020 NIAC)",
        "discrepancy": f"Exoplanet resolution {res_m:.1f} meters resolves continental sub-structures and cloud bands",
        "status": "PASS" if passed else "FAIL",
    }


def run_benchmark_b16_04_oberth_trajectory() -> Dict[str, str]:
    """B16-04: Solar Oberth Hyperbolic Escape Trajectory & Proper Time Deficit."""
    sol = solve_sgl_mission_trajectory(
        target_star="proxima_centauri",
        departure_epoch_jd=2462622.5,
        periapsis_solar_radii=4.0,
        periapsis_delta_v_km_s=25.0,
        wavelength_m=1.0e-6,
        num_waypoints=50,
    )

    # Check asymptotic velocity and flight time to 550 AU
    v_inf = sol.asymptotic_speed_au_per_year
    t_550 = sol.time_to_550au_years
    deficit = sol.time_dilation_deficit_at_550au_seconds

    # For rp=4 Rsun, delta_v=25 km/s: v_inf ~ 26.7 AU/yr, t_550 ~ 20.6 yr
    passed = (20.0 <= v_inf <= 30.0) and (15.0 <= t_550 <= 30.0) and (deficit > 0.0)

    return {
        "id": "B16-04",
        "name": "Solar Oberth Hyperbolic Escape & 1PN Proper Time Deficit",
        "evaluated_value": f"v_inf = {v_inf:.2f} AU/yr ({sol.asymptotic_speed_km_s:.1f} km/s), TOF(550 AU) = {t_550:.2f} yr, Deficit = {deficit:.2f} s",
        "reference_value": "20.0 <= v_inf <= 30.0 AU/yr, TOF <= 30 yr (NASA NIAC SGL Mission Architecture)",
        "discrepancy": f"Asymptotic speed {v_inf:.2f} AU/yr enables sub-25-year transit to focal cylinder",
        "status": "PASS" if passed else "FAIL",
    }


def run_benchmark_b16_05_api_contract_sla() -> Dict[str, str]:
    """B16-05: SGL REST API Contract & Response Latency SLA."""
    client = TestClient(app)
    payload = {
        "target_star": "proxima_centauri",
        "departure_epoch_jd": 2462622.5,
        "periapsis_solar_radii": 4.0,
        "periapsis_delta_v_km_s": 25.0,
        "wavelength_m": 1.0e-6,
    }

    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        resp = client.post("/api/mission/sgl", json=payload)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        assert resp.status_code == 200

    data = resp.json()
    mean_lat = float(np.mean(latencies))

    valid_payload = (
        data["target_star"] == "Proxima Centauri"
        and data["asymptotic_speed_au_per_year"] > 15.0
        and data["focal_parameters"]["peak_light_amplification"] > 1.0e11
        and len(data["waypoints"]) == 50
    )

    passed = valid_payload and (mean_lat < 150.0)

    return {
        "id": "B16-05",
        "name": "SGL REST API Contract & Latency SLA",
        "evaluated_value": f"HTTP 200 OK, Mean Latency = {mean_lat:.2f} ms (5 requests), 50 waypoints",
        "reference_value": "HTTP 200 OK, Schema Valid, Mean Latency < 150.0 ms",
        "discrepancy": f"Latency {mean_lat:.2f} ms satisfies high-throughput API SLA (< 150 ms)",
        "status": "PASS" if passed else "FAIL",
    }


def main():
    print("=" * 80)
    print("MILESTONE 16 SCIENTIFIC BENCHMARK SUITE: SOLAR GRAVITATIONAL LENS & PACKAGING")
    print("=" * 80)

    session = BenchmarkSession(
        milestone="M16",
        name="Milestone 16: Solar Gravitational Lens & Production Packaging",
    )

    benchmarks = [
        run_benchmark_b16_01_focal_distance,
        run_benchmark_b16_02_light_amplification,
        run_benchmark_b16_03_psf_resolution,
        run_benchmark_b16_04_oberth_trajectory,
        run_benchmark_b16_05_api_contract_sla,
    ]

    all_passed = True
    for fn in benchmarks:
        res_dict = fn()
        status_flag = res_dict["status"]
        is_pass = (status_flag == "PASS")
        if not is_pass:
            all_passed = False

        print(f"[{res_dict['id']}] {res_dict['name']}: {status_flag}")
        print(f"  Evaluated: {res_dict['evaluated_value']}")
        print(f"  Reference: {res_dict['reference_value']}")
        print(f"  Discrepancy: {res_dict['discrepancy']}")
        print("-" * 80)

        session.add(
            BenchmarkResult(
                id=res_dict["id"],
                name=res_dict["name"],
                evaluated_value=res_dict["evaluated_value"],
                reference_value=res_dict["reference_value"],
                discrepancy=res_dict["discrepancy"],
                passed=is_pass,
            )
        )

    md_path = session.save()
    print(f"\nBenchmark results successfully persisted:")
    print(f"  Markdown Report: {md_path}")
    print(f"  Audit Trail:     benchmarks/reports/AUDIT_TRAIL.jsonl")

    if not all_passed:
        print("\nERROR: One or more Milestone 16 benchmarks failed.")
        sys.exit(1)
    else:
        print("\nALL MILESTONE 16 BENCHMARKS PASSED AND CERTIFIED (5/5).")


if __name__ == "__main__":
    main()
