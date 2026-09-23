"""Test suite for the Relativistic Engine REST API and Web Endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from relativistic_engine.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client: TestClient):
    """Verify API health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["engine"] == "deterministic"


def test_api_catalog(client: TestClient):
    """Verify celestial catalog returns planets and interstellar stars."""
    response = client.get("/api/catalog")
    assert response.status_code == 200
    data = response.json()
    assert "planetary_bodies" in data
    assert "interstellar_stars" in data
    assert "mars" in data["planetary_bodies"]
    assert any(s["id"] == "proxima_centauri" for s in data["interstellar_stars"])


def test_api_interplanetary_mission(client: TestClient):
    """Verify interplanetary rendezvous endpoint."""
    payload = {
        "departure_body": "earth",
        "target_body": "mars",
        "departure_epoch_jd_tdb": 2462622.5,  # 2030-May-01
        "accel_proper_g": 1.0,
        "mode": "soft",
    }
    response = client.post("/api/mission/interplanetary", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["departure_body"] == "earth"
    assert data["target_body"] == "mars"
    # Earth -> Mars at 1g is ~4.5 days
    assert 4.0 < data["coordinate_flight_time_days"] < 5.0
    assert 4.0 < data["proper_flight_time_days"] < 5.0
    assert data["time_deficit_seconds"] > 2.0
    assert len(data["trajectory_points"]) > 20
    assert "±" in data["formatted_coordinate_time"]


def test_api_interstellar_mission(client: TestClient):
    """Verify interstellar brachistochrone endpoint."""
    payload = {
        "target_star": "proxima_centauri",
        "departure_epoch_jd_tdb": 2451545.0,
        "accel_proper_g": 1.0,
    }
    response = client.post("/api/mission/interstellar", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["target_star"] == "Proxima Centauri"
    # Proxima Centauri at 1g is ~5.87 years coordinate, ~3.54 years proper
    assert 5.5 < data["coordinate_flight_time_years"] < 6.2
    assert 3.3 < data["proper_flight_time_years"] < 3.8
    assert data["max_speed_c"] > 0.90
    assert data["max_lorentz_gamma"] > 3.0
    assert len(data["trajectory_points"]) > 20
    assert "±" in data["formatted_coordinate_time"]


def test_api_certificate(client: TestClient):
    """Verify cryptographic certificate endpoint."""
    response = client.get("/api/certificate")
    assert response.status_code == 200
    data = response.json()
    assert data["is_certified"] is True
    assert "source_checksums" in data
    assert len(data["proof_records"]) >= 5


def test_web_index_html_serving(client: TestClient):
    """Verify static index.html is served at root."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Relativistic Space Travel" in response.text
    assert "main-canvas" in response.text


def test_api_kerr_lensing(client: TestClient):
    """Verify Kerr black hole geometry, invariants, and Bardeen shadow endpoint."""
    payload = {
        "mass_solar": 10.0,
        "spin_dimensionless": 0.9,
        "inclination_deg": 85.0,
        "render_preview": False,
    }
    response = client.post("/api/lensing/kerr", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["spin_dimensionless"] == 0.9
    assert data["gravitational_radius_m"] > 14000.0
    assert data["r_horizon_outer_m"] > data["r_horizon_inner_m"]
    assert data["r_isco_prograde_m"] < data["r_isco_retrograde_m"]
    assert 14.0 < data["penrose_theoretical_max_efficiency_pct"] < 16.0
    assert len(data["shadow_contour_alpha"]) > 50
    assert len(data["shadow_contour_beta"]) > 50
    assert data["shadow_pixels_count"] is None


def test_api_navigation_simulate_arc(client: TestClient):
    """Verify DSN relativistic tracking arc and filtering simulation endpoint."""
    payload = {
        "station": "goldstone",
        "duration_hours": 2.0,
        "sampling_interval_s": 1800.0,
        "sigma_range_m": 1.0,
        "target_profile": "mars_approach",
    }
    response = client.post("/api/navigation/simulate_arc", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["station_id"] == "DSS-14"
    assert data["target_profile"] == "mars_approach"
    assert data["n_points"] >= 4
    assert len(data["telemetry_points"]) == data["n_points"]
    p0 = data["telemetry_points"][0]
    assert p0["t_elapsed_hours"] == 0.0
    assert p0["range_km"] > 1e7
    assert p0["shapiro_delay_us"] > 1.0
    assert p0["sigma_r_m"] > 0.0
    assert data["mean_nis"] >= 0.0
