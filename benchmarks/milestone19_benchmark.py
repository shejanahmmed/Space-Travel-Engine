"""Milestone 19 Scientific Benchmark Suite: Relativistic Closed-Loop Autonomous Guidance (ZEM/ZEV).

Executes 5 rigorous verification scenarios evaluating:
1. B19-01: ZEM/ZEV Linear Guidance Analytical Optimality (< 1 cm terminal miss)
2. B19-02: 1PN Relativistic Solar Gravity Compensation & Schiff Precession
3. B19-03: Relativistic Mass Depletion & Rocket Energy Conservation
4. B19-04: Closed-Loop Guidance Dispersion Recovery (5000 m -> < 50 m terminal miss)
5. B19-05: Guidance Simulation REST API Throughput & Latency SLA (< 150 ms)

Authoritative Standards & References:
- D'Souza, C. N. (1997), "An Optimal Guidance Law for Planetary Landing",
  AIAA Guidance, Navigation, and Control Conference.
- Guo, Y., Hawkins, M., & Wie, B. (2013), "Waypoint-Optimized Zero-Effort-Miss/
  Zero-Effort-Velocity Guidance for Mars Landing", J. Guidance, Control, and Dynamics.
- Schiff, L. I. (1960), "Possible New Experimental Tests of General Relativity Theory",
  Phys. Rev. Lett. 4, 215.
- Damour, T., & Deruelle, N. (1985), "General relativistic celestial mechanics
  of binary systems", Ann. Inst. Henri Poincaré Phys. Théor. 43, 107-132.
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
from relativistic_engine.constants import AU, C_LIGHT, G0, GM_SUN
from relativistic_engine.guidance.zem_zev import (
    GuidanceCommand,
    ZEMZEVGuidanceLaw,
    compute_1pn_gravity_acceleration,
    compute_schiff_gyro_precession_rate,
    simulate_closed_loop_mission,
    vector_to_unit_quaternion,
)


def run_benchmark_b19_01_linear_optimality() -> Dict[str, str]:
    """B19-01: ZEM/ZEV Linear Guidance Analytical Optimality (< 1 cm terminal miss)."""
    r_target = np.array([1.0e6, 2.0e6, -5.0e5], dtype=np.float64)
    v_target = np.array([100.0, -50.0, 20.0], dtype=np.float64)
    t_f = 1000.0

    r_init = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    v_init = np.array([50.0, 100.0, 0.0], dtype=np.float64)

    controller = ZEMZEVGuidanceLaw(
        target_position_m=r_target,
        target_velocity_ms=v_target,
        target_arrival_epoch_s=t_f,
        thrust_max_n=1e8,
        isp_sec=4500.0,
        min_time_to_go_s=0.01,
        include_gravity=False,
    )

    r = r_init.copy()
    v = v_init.copy()
    dt = 0.1
    steps = int(t_f / dt)
    for i in range(steps):
        t = i * dt
        cmd = controller.compute_command(t, r, v, 1000.0)
        r += v * dt + 0.5 * cmd.clamped_accel_ms2 * (dt**2)
        v += cmd.clamped_accel_ms2 * dt

    miss_m = float(np.linalg.norm(r - r_target))
    vel_err_ms = float(np.linalg.norm(v - v_target))
    passed = (miss_m < 0.01) and (vel_err_ms < 0.01)

    return {
        "id": "B19-01",
        "name": "ZEM/ZEV Linear Guidance Analytical Convergence & Optimality",
        "evaluated_value": f"Terminal miss = {miss_m:.6e} m, Vel error = {vel_err_ms:.6e} m/s",
        "reference_value": "Terminal miss < 0.01 m, Vel error < 0.01 m/s (D'Souza 1997 / Guo et al. 2013)",
        "discrepancy": f"Miss {miss_m:.4e} m <= 0.01 m, Vel error {vel_err_ms:.4e} m/s <= 0.01 m/s",
        "status": "PASS" if passed else "FAIL",
    }


def run_benchmark_b19_02_relativistic_gravity_and_schiff() -> Dict[str, str]:
    """B19-02: 1PN Relativistic Solar Gravity Compensation & Schiff Precession."""
    r_test = np.array([AU, 0.0, 0.0], dtype=np.float64)
    v_test = np.array([0.0, 29780.0, 0.0], dtype=np.float64)

    a_1pn = compute_1pn_gravity_acceleration(r_test, v_test)
    a_newton = -GM_SUN / (AU**2)
    rel_diff = abs(float(a_1pn[0]) - a_newton) / abs(a_newton)

    omega_schiff = compute_schiff_gyro_precession_rate(r_test, v_test)
    arcsec_yr = float(omega_schiff[2]) * (180.0 / math.pi) * 3600.0 * (86400.0 * 365.25)

    passed = (1e-9 < rel_diff < 1e-7) and (0.01 < arcsec_yr < 0.03)

    return {
        "id": "B19-02",
        "name": "1PN Relativistic Gravity & Schiff Geodetic Precession",
        "evaluated_value": f"1PN Delta-a/a = {rel_diff:.3e}, Schiff Precession = {arcsec_yr:.4f} arcsec/yr",
        "reference_value": "1PN correction ~ O(v^2/c^2) ~ 1e-8, Schiff geodetic ~ 0.019 arcsec/yr (Schiff 1960)",
        "discrepancy": f"1PN rel_diff {rel_diff:.3e} within [1e-9, 1e-7], Precession {arcsec_yr:.4f} in [0.01, 0.03]",
        "status": "PASS" if passed else "FAIL",
    }


def run_benchmark_b19_03_relativistic_mass_depletion() -> Dict[str, str]:
    """B19-03: Relativistic Mass Depletion & Rocket Energy Conservation."""
    controller = ZEMZEVGuidanceLaw(
        target_position_m=[AU, 1e7, 0.0],
        target_velocity_ms=[0.0, 29780.0, 0.0],
        target_arrival_epoch_s=86400.0,
        thrust_max_n=10.0,
        isp_sec=3000.0,
        min_time_to_go_s=10.0,
    )

    r = np.array([AU, 0.0, 0.0], dtype=np.float64)
    v = np.array([0.0, 29780.0, 0.0], dtype=np.float64)
    cmd = controller.compute_command(0.0, r, v, current_mass_kg=1000.0)

    expected_mdot = (cmd.thrust_n / (3000.0 * G0)) * math.sqrt(1.0 - (29780.0 / C_LIGHT)**2)
    diff = abs(cmd.mass_flow_rate_kg_s - expected_mdot)
    passed = diff < 1e-12

    return {
        "id": "B19-03",
        "name": "Relativistic Mass Depletion & Propulsion Formulation",
        "evaluated_value": f"m_dot = {cmd.mass_flow_rate_kg_s:.10f} kg/s, diff = {diff:.2e} kg/s",
        "reference_value": "m_dot = T / (Isp * g0) * sqrt(1 - v^2/c^2) (Relativistic Rocket Equation)",
        "discrepancy": f"Mass flow rate error {diff:.2e} kg/s <= 1.00e-12 kg/s",
        "status": "PASS" if passed else "FAIL",
    }


def run_benchmark_b19_04_dispersion_recovery() -> Dict[str, str]:
    """B19-04: Closed-Loop Guidance Dispersion Recovery (5000 m -> < 50 m)."""
    res = simulate_closed_loop_mission(
        duration_days=15.0,
        step_hours=6.0,
        initial_pos_dispersion_m=5000.0,
        initial_vel_dispersion_ms=0.5,
        wet_mass_kg=1500.0,
        thrust_max_n=10.0,
        isp_sec=4500.0,
    )

    final_miss = res["final_miss_distance_m"]
    final_vel_err = res["final_velocity_error_ms"]
    propellant_used = res["total_propellant_used_kg"]
    passed = (final_miss < 50.0) and (final_vel_err < 0.1) and (propellant_used > 0.0)

    return {
        "id": "B19-04",
        "name": "Closed-Loop Trajectory Correction & 5km Dispersion Intercept",
        "evaluated_value": f"Final Miss = {final_miss:.2f} m, Vel Error = {final_vel_err:.4f} m/s, Propellant = {propellant_used:.2f} kg",
        "reference_value": "Final Miss < 50.0 m, Vel Error < 0.10 m/s from 5000 m / 0.5 m/s initial dispersion",
        "discrepancy": f"Final miss {final_miss:.2f} m <= 50.0 m, Vel error {final_vel_err:.4f} m/s <= 0.10 m/s",
        "status": "PASS" if passed else "FAIL",
    }


def run_benchmark_b19_05_api_sla() -> Dict[str, str]:
    """B19-05: Guidance Simulation REST API Throughput & Latency SLA (< 150 ms)."""
    client = TestClient(app)
    payload = {
        "duration_days": 15.0,
        "step_hours": 6.0,
        "wet_mass_kg": 1200.0,
        "thrust_max_n": 8.0,
        "isp_sec": 4000.0,
        "initial_pos_dispersion_m": 3000.0,
        "initial_vel_dispersion_ms": 0.3,
    }

    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        resp = client.post("/api/guidance/zem_zev/simulate", json=payload)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        latencies.append((t1 - t0) * 1000.0)

    mean_lat = float(np.mean(latencies))
    passed = mean_lat < 150.0

    return {
        "id": "B19-05",
        "name": "Guidance Simulation REST API Throughput & Latency SLA",
        "evaluated_value": f"HTTP 200 OK, Mean Latency = {mean_lat:.2f} ms over 5 runs",
        "reference_value": "HTTP 200 OK, Latency SLA < 150.0 ms (Real-time G&C Simulation SLA)",
        "discrepancy": f"Mean latency {mean_lat:.2f} ms <= 150.0 ms",
        "status": "PASS" if passed else "FAIL",
    }


def run_all_milestone19_benchmarks() -> BenchmarkSession:
    """Execute complete Milestone 19 benchmark suite and generate certified report."""
    print("=" * 80)
    print("EXECUTING MILESTONE 19 BENCHMARK SUITE: CLOSED-LOOP AUTONOMOUS GUIDANCE (ZEM/ZEV)")
    print("=" * 80)

    benchmarks_to_run = [
        run_benchmark_b19_01_linear_optimality,
        run_benchmark_b19_02_relativistic_gravity_and_schiff,
        run_benchmark_b19_03_relativistic_mass_depletion,
        run_benchmark_b19_04_dispersion_recovery,
        run_benchmark_b19_05_api_sla,
    ]

    session = BenchmarkSession(
        milestone="M19",
        name="Milestone 19: Relativistic Closed-Loop Autonomous Guidance (ZEM/ZEV)",
    )

    for bench_fn in benchmarks_to_run:
        res_dict = bench_fn()
        passed_bool = (res_dict["status"] == "PASS")
        bench_res = BenchmarkResult(
            id=res_dict["id"],
            name=res_dict["name"],
            evaluated_value=res_dict["evaluated_value"],
            reference_value=res_dict["reference_value"],
            discrepancy=res_dict["discrepancy"],
            passed=passed_bool,
        )
        session.add(bench_res)
        tag = f"[{'PASS' if passed_bool else 'FAIL'}]"
        print(f"  {bench_res.id} : {bench_res.name:<58} {tag}")
        print(f"         Evaluated: {bench_res.evaluated_value}")
        print(f"         Reference: {bench_res.reference_value}")
        print(f"         Result   : {bench_res.discrepancy}\n")

    return session


if __name__ == "__main__":
    session = run_all_milestone19_benchmarks()
    session.save()
    print("=" * 80)
    print(f"MILESTONE 19 BENCHMARK SUMMARY: {session.score} PASSED")
    print("Artifacts generated:")
    print("  - benchmarks/reports/M19_BENCHMARK_REPORT.md")
    print("  - benchmarks/reports/M19_BENCHMARK_DATA.json")
    print("  - benchmarks/reports/AUDIT_TRAIL.jsonl")
    print("=" * 80)
    if not session.all_passed:
        sys.exit(1)
    sys.exit(0)
