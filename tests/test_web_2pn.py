"""Validation test suite for Web Workstation 2PN & Quad-Precision Interface."""

from fastapi.testclient import TestClient
from relativistic_engine.api.app import app, WEB_DIR


def test_web_static_assets_exist():
    """Verify web assets exist and are mounted properly."""
    assert (WEB_DIR / "index.html").is_file()
    assert (WEB_DIR / "style.css").is_file()
    assert (WEB_DIR / "app.js").is_file()


def test_index_html_contains_2pn_elements():
    """Verify index.html contains the 2PN mode tab, config form, and visualizer view tabs."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert 'data-mode="pn2"' in html
    assert 'id="tab-pn2"' in html
    assert 'id="form-pn2"' in html
    assert 'id="select-pn2-preset"' in html
    assert 'id="select-pn2-precision"' in html
    assert 'id="vtab-pn2-orbit"' in html
    assert 'id="vtab-energy-drift"' in html


def test_app_js_contains_2pn_logic():
    """Verify app.js contains 2PN dispatch, preset handlers, and dual visualizers."""
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "computeTrajectory2PN" in js
    assert "applyPn2Preset" in js
    assert "render2PNOrbit" in js
    assert "renderEnergyDrift" in js
    assert "vtabPn2Orbit" in js
    assert "vtabEnergyDrift" in js


def test_api_2pn_endpoint_end_to_end():
    """Verify POST /api/trajectory/2pn returns valid symplectic payload for web consumption."""
    client = TestClient(app)
    payload = {
        "central_body": "Sun",
        "pn_order": "2pn",
        "precision": "float64",
        "gravity_harmonics_degree": 0,
        "r0_km": [46001200.0, 0.0, 0.0],
        "v0_km_s": [0.0, 58.98, 0.0],
        "duration_days": 1.0,
        "step_size_s": 3600.0,
    }
    response = client.post("/api/trajectory/2pn", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["pn_order"] == "2pn"
    assert data["precision"] == "float64"
    assert data["central_body"] == "Sun"
    assert data["duration_days"] == 1.0
    assert len(data["points"]) > 0
    assert abs(data["energy_drift_relative"]) < 1e-8
