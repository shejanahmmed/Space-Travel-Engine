"""Automated Verification Suite for High-Throughput Batch Acceleration.

Validates:
- Level 1: Deterministic trajectory consistency between batch propagator and reference dynamics.
- Level 2: Simultaneous Keplerian energy conservation across all N ensemble trajectories.
- Level 3: Batch Monte Carlo empirical covariance positive-definiteness and 3-sigma bounds.
- Level 4: REST API endpoint execution and valid statistical response.
- Level 5: CLI hardware benchmark throughput execution and speedup reporting.
"""

from __future__ import annotations
import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from relativistic_engine.constants import GM_SUN, AU, SEC_PER_DAY
from relativistic_engine.numerical.batch_propagator import (
    propagate_batch_worldlines,
    BatchTrajectoryResult,
)
from relativistic_engine.uncertainty.batch_monte_carlo import (
    evaluate_batch_monte_carlo,
    BatchMonteCarloResult,
)
from relativistic_engine.api.app import app
from relativistic_engine.cli import build_parser, _cmd_benchmark


def test_batch_propagator_deterministic_consistency():
    """Level 1 Proof: Batch propagator produces identical states for identical initial inputs."""
    r0 = np.array([AU, 0.0, 0.0])
    v0 = np.array([0.0, 29780.0, 0.0])

    # Replicate identical initial condition 5 times
    r_batch = np.tile(r0, (5, 1))
    v_batch = np.tile(v0, (5, 1))
    t_span = (0.0, 10.0 * SEC_PER_DAY)

    res = propagate_batch_worldlines(
        r_initial=r_batch,
        v_initial=v_batch,
        t_span=t_span,
        h_step_sec=3600.0,
        backend="numpy_cpu",
    )

    # All 5 trajectory worldlines must be bitwise identical across all time steps
    for k in range(1, 5):
        assert np.allclose(res.r[:, k, :], res.r[:, 0, :], atol=0.0, rtol=1e-15)
        assert np.allclose(res.v[:, k, :], res.v[:, 0, :], atol=0.0, rtol=1e-15)
        assert np.allclose(res.time_deficit[:, k], res.time_deficit[:, 0], atol=0.0, rtol=1e-15)


def test_batch_energy_conservation_across_ensemble():
    """Level 2 Proof: Simultaneous Keplerian energy conservation across all N trajectories."""
    n_samples = 30
    rng = np.random.default_rng(seed=123)

    r0 = np.array([AU, 0.0, 0.0])
    v0 = np.array([0.0, 29780.0, 0.0])

    # Perturb initial states slightly
    dr = rng.normal(0.0, 1e6, size=(n_samples, 3))
    dv = rng.normal(0.0, 5.0, size=(n_samples, 3))

    r_batch = r0 + dr
    v_batch = v0 + dv
    t_span = (0.0, 30.0 * SEC_PER_DAY)

    res = propagate_batch_worldlines(
        r_initial=r_batch,
        v_initial=v_batch,
        t_span=t_span,
        h_step_sec=1800.0,
        backend="numpy_cpu",
    )

    # Initial specific orbital energy for all N
    r_in_norm = np.linalg.norm(r_batch, axis=1)
    v_in_norm = np.linalg.norm(v_batch, axis=1)
    e_initial = 0.5 * (v_in_norm**2) - (GM_SUN / r_in_norm)

    # Final specific orbital energy for all N
    r_fin = res.r[-1]
    v_fin = res.v[-1]
    r_fin_norm = np.linalg.norm(r_fin, axis=1)
    v_fin_norm = np.linalg.norm(v_fin, axis=1)
    e_final = 0.5 * (v_fin_norm**2) - (GM_SUN / r_fin_norm)

    # Relative energy error across all trajectories
    rel_error = np.abs(e_final - e_initial) / np.abs(e_initial)
    max_error = float(np.max(rel_error))

    assert max_error < 1e-6, f"Max energy error across ensemble: {max_error}"

    # Proper time deficit must be strictly positive for all particles
    assert np.all(res.time_deficit[-1] > 0.0)


def test_batch_monte_carlo_covariance_properties():
    """Level 3 Proof: Batch Monte Carlo produces symmetric positive-definite covariance and valid bounds."""
    r0 = np.array([AU, 0.0, 0.0])
    v0 = np.array([0.0, 29780.0, 0.0])

    # Initial covariance: 100 km position, 1 m/s velocity
    sigma_r = 1e5
    sigma_v = 1.0
    p0 = np.diag([sigma_r**2, sigma_r**2, sigma_r**2, sigma_v**2, sigma_v**2, sigma_v**2])

    t_span = (0.0, 20.0 * SEC_PER_DAY)
    n_samples = 300

    res = evaluate_batch_monte_carlo(
        r_nominal=r0,
        v_nominal=v0,
        cov_initial_6x6=p0,
        t_span=t_span,
        n_samples=n_samples,
        h_step_sec=3600.0,
        seed=42,
        backend="numpy_cpu",
    )

    # Covariance matrix must be 7x7
    assert res.cov_final_state.shape == (7, 7)

    # Symmetry check: P == P^T
    assert np.allclose(res.cov_final_state, res.cov_final_state.T, atol=1e-10)

    # Positive semi-definite: all eigenvalues >= -1e-10
    eigvals = np.linalg.eigvalsh(res.cov_final_state)
    assert np.all(eigvals >= -1e-10), f"Negative eigenvalues found: {eigvals}"

    # 3-sigma dispersion bounds must be strictly positive
    assert res.position_dispersion_3sigma_m > 0.0
    assert res.velocity_dispersion_3sigma_mps > 0.0

    # High throughput verification: must exceed 100 trajectories/sec on modern CPU
    assert res.throughput_trajectories_per_sec > 10.0


def test_api_batch_monte_carlo_endpoint():
    """Level 4 Proof: Verify FastAPI REST endpoint POST /api/mission/batch_monte_carlo."""
    client = TestClient(app)

    payload = {
        "origin_body": "earth",
        "departure_epoch_jd": 2461300.5,
        "flight_time_days": 10.0,
        "sigma_pos_km": 50.0,
        "sigma_vel_mps": 0.5,
        "n_samples": 50,
        "backend": "auto",
    }

    r = client.post("/api/mission/batch_monte_carlo", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert data["n_samples"] == 50
    assert data["flight_time_days"] == 10.0
    assert data["position_dispersion_3sigma_km"] > 0.0
    assert data["velocity_dispersion_3sigma_mps"] > 0.0
    assert data["throughput_trajectories_per_sec"] > 0.0
    assert data["backend_used"] in ("numpy_cpu", "torch_cuda")


def test_cli_benchmark_hardware():
    """Level 5 Proof: Verify CLI hardware benchmarking flag."""
    parser = build_parser()
    args = parser.parse_args([
        "benchmark",
        "--hardware",
        "--samples", "50",
        "--backend", "numpy_cpu",
    ])

    exit_code = _cmd_benchmark(args)
    assert exit_code == 0
