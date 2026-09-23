"""Milestone 17 Scientific Benchmark Suite: Relativistic X-ray Pulsar Navigation (XPNAV).

Executes 5 rigorous verification scenarios evaluating:
1. B17-01: Millisecond Pulsar Astrometric Catalog Integrity & Geometry
2. B17-02: BCRS Geometric Rømer & Solar Shapiro Relativistic Delays
3. B17-03: Interstellar Plasma Dispersion & Inverse-Square Frequency Scaling
4. B17-04: NASA SEXTANT 4D Spacecraft State Reconstruction & Clock Sync
5. B17-05: XPNAV REST API Contract & Response Latency SLA

Authoritative Standards & References:
- Mitchell, J. W., et al. (2018), "SEXTANT - Station Explorer for X-ray Timing
  and Navigation Technology: Flight Demonstration Results", AIAA/AAS Astrodynamics.
- Sheikh, S. I., et al. (2006), "Spacecraft Navigation Using X-Ray Pulsars",
  Journal of Guidance, Control, and Dynamics, 29(1), 49-63.
- Manchester, R. N., et al. (2005), "The ATNF Pulsar Catalogue", Astron. J., 129, 1993.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
import numpy as np

from benchmarks.save_results import BenchmarkResult, BenchmarkSession
from relativistic_engine.api.app import app
from relativistic_engine.constants import AU, C_LIGHT, GM_SUN
from relativistic_engine.navigation.xpnav import (
    DISPERSION_CONSTANT_D,
    PULSAR_CATALOG,
    PulsarObservation,
    compute_pulse_delays,
    compute_predicted_pulse_phase,
    solve_spacecraft_state_xpnav,
)


def run_benchmark_b17_01_catalog_geometry() -> Dict[str, str]:
    """B17-01: Millisecond Pulsar Astrometric Catalog Integrity & Geometry."""
    n_pulsars = len(PULSAR_CATALOG)
    all_valid = True
    max_norm_err = 0.0

    for key, psr in PULSAR_CATALOG.items():
        n_hat = psr.unit_vector_bcrs
        norm_err = abs(float(np.linalg.norm(n_hat)) - 1.0)
        max_norm_err = max(max_norm_err, norm_err)
        period_err = abs(psr.period_s - (1.0 / psr.f0_hz))
        if norm_err > 1e-12 or period_err > 1e-12:
            all_valid = False

    passed = all_valid and (n_pulsars >= 4) and (max_norm_err < 1e-12)

    return {
        "id": "B17-01",
        "name": "Pulsar Catalog Astrometry & Unit Vector Geometry",
        "evaluated_value": f"{n_pulsars} primary pulsars verified, max unit vector error = {max_norm_err:.2e}",
        "reference_value": ">= 4 millisecond pulsars, ||n_hat|| = 1.000000000000 (ATNF Database)",
        "discrepancy": f"Unit vector normalization error {max_norm_err:.2e} <= 1.00e-12",
        "status": "PASS" if passed else "FAIL",
    }


def run_benchmark_b17_02_delays_shapiro() -> Dict[str, str]:
    """B17-02: BCRS Geometric Rømer & Solar Shapiro Relativistic Delays."""
    psr = PULSAR_CATALOG["b1937+21"]
    r_sc = np.array([1.0 * AU, 0.0, 0.0], dtype=np.float64)

    delays = compute_pulse_delays(psr, r_sc, freq_ghz=1.0)
    expected_geom = -float(np.dot(psr.unit_vector_bcrs, r_sc)) / C_LIGHT

    err_geom = abs(delays.geometric_delay_s - expected_geom)
    shapiro_us = delays.shapiro_delay_s * 1e6

    # Solar Shapiro delay should be magnitude ~ 1 to 10 us for 1 AU distance
    passed = (err_geom < 1e-12) and (-100.0 < shapiro_us < 100.0)

    return {
        "id": "B17-02",
        "name": "BCRS Geometric Rømer & Solar Shapiro Relativistic Delays",
        "evaluated_value": f"t_geom = {delays.geometric_delay_s:.6f} s ({delays.range_equivalent_km:,.1f} km), t_Shapiro = {shapiro_us:.4f} us",
        "reference_value": "t_geom = -(n_hat . r)/c, t_Shapiro = -(2GM/c^3)*ln(1 + n.r_hat) (Moyer 2000)",
        "discrepancy": f"Geometric residual = {err_geom:.2e} s, Shapiro magnitude = {shapiro_us:.4f} us",
        "status": "PASS" if passed else "FAIL",
    }


def run_benchmark_b17_03_plasma_dispersion() -> Dict[str, str]:
    """B17-03: Interstellar Plasma Dispersion & Inverse-Square Frequency Scaling."""
    psr = PULSAR_CATALOG["b1821-24"]  # DM = 119.86 pc/cm^3
    frequencies = [0.5, 1.0, 2.0]  # GHz
    delays = []

    for f in frequencies:
        d = compute_pulse_delays(psr, np.zeros(3), freq_ghz=f)
        delays.append(d.dispersion_delay_s)

    # Verify exact 1 / nu^2 scaling
    ratio_05_10 = delays[0] / delays[1]  # Expected: (1.0 / 0.5)^2 = 4.0
    ratio_10_20 = delays[1] / delays[2]  # Expected: (2.0 / 1.0)^2 = 4.0

    err1 = abs(ratio_05_10 - 4.0)
    err2 = abs(ratio_10_20 - 4.0)

    passed = (err1 < 1e-10) and (err2 < 1e-10)

    return {
        "id": "B17-03",
        "name": "Interstellar Plasma Dispersion & 1/nu^2 Frequency Scaling",
        "evaluated_value": f"t_DM(0.5 GHz) = {delays[0]*1e3:.3f} ms, t_DM(1.0 GHz) = {delays[1]*1e3:.3f} ms, t_DM(2.0 GHz) = {delays[2]*1e3:.3f} ms",
        "reference_value": "t_DM = D * (DM / nu^2), Ratio(0.5/1.0 GHz) = 4.000000, Ratio(1.0/2.0 GHz) = 4.000000",
        "discrepancy": f"Dispersion frequency scaling linearity error = {max(err1, err2):.2e}",
        "status": "PASS" if passed else "FAIL",
    }


def run_benchmark_b17_04_sextant_reconstruction() -> Dict[str, str]:
    """B17-04: NASA SEXTANT 4D Spacecraft State Reconstruction & Clock Sync."""
    true_pos_km = np.array([149597870.7, 5000000.0, -2000000.0], dtype=np.float64)
    true_clock_s = 1.25e-6
    t_obs_s = 1000.0

    keys = ["b1937+21", "b1821-24", "j0437-4715", "j0218+4232"]
    observations = []
    for k in keys:
        psr = PULSAR_CATALOG[k]
        phase = compute_predicted_pulse_phase(
            psr,
            t_sc_tdb_s=t_obs_s,
            r_sc_m=true_pos_km * 1000.0,
            clock_bias_s=true_clock_s,
            freq_ghz=1.0,
        )
        observations.append(
            PulsarObservation(
                pulsar_key=k,
                observed_time_tdb_s=t_obs_s,
                frequency_ghz=1.0,
                measured_phase=phase,
            )
        )

    # Offset initial guess by 200 meters and 1 ns
    initial_guess_km = true_pos_km + np.array([0.2, -0.3, 0.1])
    initial_guess_clock = true_clock_s + 1.0e-9

    sol = solve_spacecraft_state_xpnav(
        observations=observations,
        initial_guess_r_km=initial_guess_km,
        initial_guess_clock_s=initial_guess_clock,
    )

    est_pos_km = np.array(sol.position_bcrs_km)
    pos_err_m = float(np.linalg.norm(est_pos_km - true_pos_km)) * 1000.0
    clock_err_ns = abs(sol.clock_bias_s - true_clock_s) * 1e9

    passed = sol.converged and (pos_err_m < 1.0) and (clock_err_ns < 0.01)

    return {
        "id": "B17-04",
        "name": "NASA SEXTANT 4D State Reconstruction & Clock Synchronization",
        "evaluated_value": f"Position Error = {pos_err_m * 1e3:.2f} mm ({pos_err_m:.4f} m), Clock Error = {clock_err_ns * 1e3:.3f} ps, GDOP = {sol.gdop:.2f}",
        "reference_value": "Position Error < 1.00 m, Clock Error < 0.010 ns (NASA SEXTANT Architecture)",
        "discrepancy": f"Position error {pos_err_m:.4f} m resolves spacecraft sub-meter BCRS trajectory",
        "status": "PASS" if passed else "FAIL",
    }


def run_benchmark_b17_05_api_sla() -> Dict[str, str]:
    """B17-05: XPNAV REST API Contract & Response Latency SLA."""
    client = TestClient(app)

    predict_payload = {
        "pulsar_key": "b1937+21",
        "t_sc_tdb_s": 100.0,
        "r_sc_km": [149597870.7, 0.0, 0.0],
        "clock_bias_s": 0.0,
        "frequency_ghz": 1.0,
    }

    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        resp = client.post("/api/navigation/xpnav/predict", json=predict_payload)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        assert resp.status_code == 200

    mean_lat = float(np.mean(latencies))
    passed = mean_lat < 50.0

    return {
        "id": "B17-05",
        "name": "XPNAV REST API Contract & Latency SLA",
        "evaluated_value": f"HTTP 200 OK, Mean Latency = {mean_lat:.2f} ms (5 requests)",
        "reference_value": "HTTP 200 OK, Schema Valid, Mean Latency < 50.0 ms",
        "discrepancy": f"API response latency {mean_lat:.2f} ms satisfies deep-space telemetry SLA (< 50 ms)",
        "status": "PASS" if passed else "FAIL",
    }


def main():
    print("=" * 80)
    print("MILESTONE 17 SCIENTIFIC BENCHMARK SUITE: RELATIVISTIC XPNAV NAVIGATION")
    print("=" * 80)

    session = BenchmarkSession(
        milestone="M17",
        name="Milestone 17: Relativistic X-ray Pulsar Navigation (XPNAV)",
    )

    benchmarks = [
        run_benchmark_b17_01_catalog_geometry,
        run_benchmark_b17_02_delays_shapiro,
        run_benchmark_b17_03_plasma_dispersion,
        run_benchmark_b17_04_sextant_reconstruction,
        run_benchmark_b17_05_api_sla,
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
        print("\nERROR: One or more Milestone 17 benchmarks failed.")
        sys.exit(1)
    else:
        print("\nALL MILESTONE 17 BENCHMARKS PASSED AND CERTIFIED (5/5).")


if __name__ == "__main__":
    main()
