"""Automated validation tests for end-to-end 2PN integration, API, and CLI.

Validates:
1. End-to-end 2PN trajectory propagation in float64 DOP853 mode.
2. End-to-end 2PN symplectic trajectory propagation in quad precision mode.
3. REST API endpoint POST /api/trajectory/2pn schema conformance and physics outputs.
4. CLI command relativistic-engine orbit-2pn execution.
"""

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from relativistic_engine.api.app import app
from relativistic_engine.cli import build_parser, _cmd_orbit_2pn
from relativistic_engine.trajectory.pn2_propagator import (
    propagate_2pn_trajectory,
)


def test_propagate_2pn_orbit_float64():
    """Verify 2PN orbit propagation over 5 days in float64 mode."""
    r0 = [1.495978707e11, 0.0, 0.0]
    v0 = [0.0, 29780.0, 0.0]
    duration_s = 5.0 * 86400.0
    step_s = 3600.0

    res = propagate_2pn_trajectory(
        r0=r0,
        v0=v0,
        duration_s=duration_s,
        step_size_s=step_s,
        central_body="Sun",
        pn_order="2pn",
        precision="float64",
        max_zonal_degree=2,
    )

    assert len(res.t) > 10
    assert res.pn_order == "2pn"
    assert res.precision == "float64"
    assert res.energy_drift_relative < 1e-8

    # Time deficit Delta(t) = t - tau must be positive and growing
    assert res.time_deficit[-1] > 0.0
    assert res.tau[-1] < res.t[-1]


def test_propagate_2pn_orbit_quad_precision():
    """Verify 2PN symplectic orbit propagation in quad precision mode."""
    r0 = [1.495978707e11, 0.0, 0.0]
    v0 = [0.0, 29780.0, 0.0]
    duration_s = 2.0 * 86400.0
    step_s = 43200.0  # 12-hour steps

    res = propagate_2pn_trajectory(
        r0=r0,
        v0=v0,
        duration_s=duration_s,
        step_size_s=step_s,
        central_body="Sun",
        pn_order="2pn",
        precision="quad",
        max_zonal_degree=0,
        dps_quad=34,
    )

    assert len(res.t) == 5
    assert res.precision == "quad"
    assert res.energy_drift_relative < 1e-12
    assert res.time_deficit[-1] > 0.0


def test_api_2pn_trajectory_endpoint():
    """Verify POST /api/trajectory/2pn returns valid scientific response."""
    client = TestClient(app)
    payload = {
        "pn_order": "2pn",
        "precision": "float64",
        "central_body": "Sun",
        "r0_km": [149597870.7, 0.0, 0.0],
        "v0_km_s": [0.0, 29.78, 0.0],
        "duration_days": 3.0,
        "step_size_s": 7200.0,
        "gravity_harmonics_degree": 2,
    }

    response = client.post("/api/trajectory/2pn", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["pn_order"] == "2pn"
    assert data["precision"] == "float64"
    assert data["central_body"] == "Sun"
    assert data["duration_days"] == 3.0
    assert len(data["points"]) > 0
    assert data["accumulated_time_deficit_s"] > 0.0
    assert data["energy_drift_relative"] < 1e-8


def test_cli_orbit_2pn_execution():
    """Verify CLI orbit-2pn command runs successfully."""
    parser = build_parser()
    args = parser.parse_args([
        "orbit-2pn",
        "--central", "Earth",
        "--pn-order", "2pn",
        "--precision", "float64",
        "--days", "0.5",
        "--step-s", "1800.0",
        "--r0-km", "7000.0,0.0,0.0",
        "--v0-km-s", "0.0,7.546,0.0",
        "--harmonics", "2",
    ])

    ret_code = _cmd_orbit_2pn(args)
    assert ret_code == 0
