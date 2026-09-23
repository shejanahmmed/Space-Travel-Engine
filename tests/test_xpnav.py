"""Unit and integration tests for Relativistic X-ray Pulsar Navigation (XPNAV).

Validates:
- Pulsar catalog astrometric parameters and unit vector normalization.
- Geometric Rømer delay, solar Shapiro delay, and plasma dispersion delay formulas.
- Pulse phase prediction and phase wrapping properties.
- 4D spacecraft state reconstruction (position [x, y, z] and clock bias) from synthetic pulsar observations.
- FastAPI REST API contracts for XPNAV predict and solve endpoints.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from relativistic_engine.constants import AU, C_LIGHT, GM_SUN
from relativistic_engine.navigation.xpnav import (
    DISPERSION_CONSTANT_D,
    PULSAR_CATALOG,
    PulsarObservation,
    compute_pulse_delays,
    compute_predicted_pulse_phase,
    solve_spacecraft_state_xpnav,
)
from relativistic_engine.api.app import app


client = TestClient(app)


def test_pulsar_catalog_properties():
    """Verify millisecond pulsar catalog entries, spin periods, and unit vector norms."""
    assert len(PULSAR_CATALOG) >= 4
    for key, psr in PULSAR_CATALOG.items():
        assert psr.f0_hz > 0.0
        assert math.isclose(psr.period_s, 1.0 / psr.f0_hz, rel_tol=1e-12)
        n_vec = psr.unit_vector_bcrs
        assert math.isclose(float(np.linalg.norm(n_vec)), 1.0, rel_tol=1e-12)
        assert psr.dispersion_measure_pc_cm3 >= 0.0


def test_xpnav_delays_and_shapiro():
    """Verify geometric, Shapiro, and dispersion delay equations."""
    psr = PULSAR_CATALOG["b1937+21"]
    r_sc = np.array([1.0 * AU, 0.0, 0.0], dtype=np.float64)

    delays = compute_pulse_delays(psr, r_sc, freq_ghz=1.0)

    # Geometric delay: -(n_hat . r_sc) / c
    expected_geom = -float(np.dot(psr.unit_vector_bcrs, r_sc)) / C_LIGHT
    assert math.isclose(delays.geometric_delay_s, expected_geom, rel_tol=1e-12)

    # Shapiro delay: Solar Shapiro should be microsecond-scale and negative/zero depending on geometry
    assert abs(delays.shapiro_delay_s) < 1.0e-3

    # Dispersion delay at 1.0 GHz (1000 MHz): D * DM / (1000^2)
    expected_dm = DISPERSION_CONSTANT_D * (psr.dispersion_measure_pc_cm3 / (1000.0 ** 2))
    assert math.isclose(delays.dispersion_delay_s, expected_dm, rel_tol=1e-12)


def test_xpnav_4d_state_reconstruction():
    """Verify 4D state reconstruction (3D position + clock bias) from synthetic observations."""
    true_pos_km = np.array([149597870.7, 5000000.0, -2000000.0], dtype=np.float64)
    true_clock_s = 1.25e-6  # 1.25 microsecond clock offset
    t_obs_s = 1000.0

    # Generate synthetic observations from 4 distinct pulsars
    pulsar_keys = ["b1937+21", "b1821-24", "j0437-4715", "j0218+4232"]
    observations = []

    for pkey in pulsar_keys:
        psr = PULSAR_CATALOG[pkey]
        true_phase = compute_predicted_pulse_phase(
            psr,
            t_sc_tdb_s=t_obs_s,
            r_sc_m=true_pos_km * 1000.0,
            clock_bias_s=true_clock_s,
            freq_ghz=1.0,
        )
        observations.append(
            PulsarObservation(
                pulsar_key=pkey,
                observed_time_tdb_s=t_obs_s,
                frequency_ghz=1.0,
                measured_phase=true_phase,
            )
        )

    # Perturbed initial guess (offset within unambiguous half-pulse domain: 20 km and 50 ns)
    initial_guess_km = true_pos_km + np.array([15.0, -10.0, 12.0])
    initial_guess_clock = true_clock_s + 5.0e-8

    sol = solve_spacecraft_state_xpnav(
        observations=observations,
        initial_guess_r_km=initial_guess_km,
        initial_guess_clock_s=initial_guess_clock,
    )

    assert sol.converged is True
    est_pos = np.array(sol.position_bcrs_km)
    pos_error_km = float(np.linalg.norm(est_pos - true_pos_km))
    clock_error_s = abs(sol.clock_bias_s - true_clock_s)

    # Sub-meter/sub-kilometer precision reconstruction from exact synthetic phases
    assert pos_error_km < 1.0
    assert clock_error_s < 1.0e-10
    assert sol.gdop > 0.0


def test_xpnav_api_endpoints():
    """Verify REST API contracts for XPNAV predict and solve routes."""
    # 1. Predict endpoint
    predict_payload = {
        "pulsar_key": "b1937+21",
        "t_sc_tdb_s": 100.0,
        "r_sc_km": [149597870.7, 0.0, 0.0],
        "clock_bias_s": 0.0,
        "frequency_ghz": 1.0,
    }
    resp_pred = client.post("/api/navigation/xpnav/predict", json=predict_payload)
    assert resp_pred.status_code == 200
    pred_data = resp_pred.json()
    assert "PSR B1937+21" in pred_data["pulsar_name"]
    assert "geometric_delay_s" in pred_data
    assert 0.0 <= pred_data["predicted_phase"] <= 1.0

    # 2. Solve endpoint
    solve_payload = {
        "observations": [
            {"pulsar_key": "b1937+21", "observed_time_tdb_s": 0.0, "frequency_ghz": 1.0, "measured_phase": 0.12},
            {"pulsar_key": "b1821-24", "observed_time_tdb_s": 0.0, "frequency_ghz": 1.0, "measured_phase": 0.45},
            {"pulsar_key": "j0437-4715", "observed_time_tdb_s": 0.0, "frequency_ghz": 1.0, "measured_phase": 0.78},
            {"pulsar_key": "j0218+4232", "observed_time_tdb_s": 0.0, "frequency_ghz": 1.0, "measured_phase": 0.33},
        ],
        "initial_guess_r_km": [149597870.7, 0.0, 0.0],
        "initial_guess_clock_s": 0.0,
    }
    resp_solve = client.post("/api/navigation/xpnav/solve", json=solve_payload)
    assert resp_solve.status_code == 200
    solve_data = resp_solve.json()
    assert len(solve_data["position_bcrs_km"]) == 3
    assert solve_data["num_pulsars_used"] == 4
    assert solve_data["converged"] is True
