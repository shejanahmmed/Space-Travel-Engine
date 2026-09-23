"""Validation test suite for interactive mission workstation API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from relativistic_engine.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_porkchop_heatmap_matrix_shape(client: TestClient):
    """Verify 2D porkchop returns NxN matrix of C3 and well-formed optimal window."""
    n_steps = 8
    payload = {
        "origin_body": "earth",
        "target_body": "mars",
        "dep_start_jd": 2461300.5,
        "dep_end_jd": 2461350.5,
        "arr_start_jd": 2461500.5,
        "arr_end_jd": 2461600.5,
        "grid_steps": n_steps,
        "mode": "ballistic",
    }
    response = client.post("/api/mission/porkchop", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["origin_body"] == "earth"
    assert data["target_body"] == "mars"
    assert len(data["dep_jds"]) == n_steps
    assert len(data["arr_jds"]) == n_steps
    assert len(data["c3_km2_s2"]) == n_steps
    assert len(data["c3_km2_s2"][0]) == n_steps
    assert len(data["delta_v_total_km_s"]) == n_steps
    assert len(data["tof_days"]) == n_steps

    best = data["best_window"]
    assert "departure_jd" in best
    assert "arrival_jd" in best
    assert "c3_km2_s2" in best
    assert best["c3_km2_s2"] > 0.0
    assert best["tof_days"] > 0.0


def test_tour_legs_consistency(client: TestClient):
    """Verify multi-leg planetary tour returns matching leg sequence and positive total delta-V."""
    legs_cfg = [
        {
            "origin_body": "earth",
            "target_body": "venus",
            "departure_jd": 2461300.5,
            "arrival_jd": 2461450.5,
            "is_flyby": True,
            "periapsis_altitude_km": 500.0,
        },
        {
            "origin_body": "venus",
            "target_body": "mars",
            "departure_jd": 2461450.5,
            "arrival_jd": 2461750.5,
            "is_flyby": False,
        },
    ]
    payload = {
        "mission_name": "Earth-Venus-Mars Tour",
        "legs": legs_cfg,
    }
    response = client.post("/api/mission/tour", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["mission_name"] == "Earth-Venus-Mars Tour"
    assert len(data["legs"]) == 2
    assert data["total_delta_v_km_s"] > 0.0
    assert data["total_coordinate_time_days"] > 0.0
    assert data["total_proper_time_days"] > 0.0

    leg1 = data["legs"][0]
    assert leg1["origin_body"] == "earth"
    assert leg1["target_body"] == "venus"
    assert leg1["flyby_turning_angle_deg"] is not None
    assert leg1["r_dep_bcrs_km"] is not None
    assert len(leg1["r_dep_bcrs_km"]) == 3

    leg2 = data["legs"][1]
    assert leg2["origin_body"] == "venus"
    assert leg2["target_body"] == "mars"
    assert leg2["r_arr_bcrs_km"] is not None


def test_low_thrust_convergence_fields(client: TestClient):
    """Verify continuous low-thrust collocation returns propellant telemetry and trajectory points."""
    payload = {
        "departure_body": "earth",
        "target_body": "mars",
        "departure_epoch_jd": 2462622.5,
        "tof_days": 180.0,
        "initial_mass_kg": 1500.0,
        "dry_mass_kg": 500.0,
        "thrust_max_n": 2.5,
        "isp_sec": 4500.0,
        "num_segments": 8,
    }
    response = client.post("/api/mission/low_thrust", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["departure_body"] == "earth"
    assert data["target_body"] == "mars"
    assert "converged" in data
    assert data["propellant_used_kg"] >= 0.0
    assert data["final_mass_kg"] <= data["initial_mass_kg"]
    assert data["mass_ratio"] >= 1.0
    assert data["max_defect"] >= 0.0
    assert len(data["trajectory_points"]) > 0


def test_monte_carlo_dispersion_bounds(client: TestClient):
    """Verify batch Monte Carlo dispersion produces positive 3-sigma bounds and throughput."""
    payload = {
        "origin_body": "earth",
        "departure_epoch_jd": 2461300.5,
        "flight_time_days": 15.0,
        "sigma_pos_km": 50.0,
        "sigma_vel_mps": 1.0,
        "n_samples": 100,
        "backend": "numpy_cpu",
    }
    response = client.post("/api/mission/batch_monte_carlo", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["n_samples"] == 100
    assert data["position_dispersion_3sigma_km"] > 0.0
    assert data["velocity_dispersion_3sigma_mps"] > 0.0
    assert data["throughput_trajectories_per_sec"] > 0.0
    assert data["elapsed_wall_time_sec"] > 0.0
    assert data["backend_used"] == "numpy_cpu"
