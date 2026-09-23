"""Unit test suite for Autonomous Flight Management System (FMS) mission executive."""

from __future__ import annotations

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from relativistic_engine.api.app import app
from relativistic_engine.constants import AU
from relativistic_engine.fms.executive import MissionExecutive
from relativistic_engine.fms.timeline import (
    EventType,
    FlightPhase,
    MissionEvent,
    SequenceOfEvents,
)


def test_sequence_of_events_and_propellant_accounting() -> None:
    """Verify Sequence of Events chronological sorting and exact propellant ledger conservation."""
    initial_mass = 1200.0
    soe = SequenceOfEvents(initial_wet_mass_kg=initial_mass)

    # Add events out of order to verify sorting
    soe.add_event(
        epoch_seconds=86400.0,
        phase=FlightPhase.TRAJECTORY_CORRECTION,
        event_type=EventType.MANEUVER,
        description="TCM-1 Burn",
        delta_v_ms=15.0,
        mass_depleted_kg=5.2,
    )
    soe.add_event(
        epoch_seconds=0.0,
        phase=FlightPhase.INJECTION,
        event_type=EventType.MILESTONE,
        description="Mission Start",
        delta_v_ms=0.0,
        mass_depleted_kg=0.0,
    )
    soe.add_event(
        epoch_seconds=172800.0,
        phase=FlightPhase.TERMINAL_APPROACH,
        event_type=EventType.MANEUVER,
        description="Terminal Brake Burn",
        delta_v_ms=50.0,
        mass_depleted_kg=17.8,
    )

    events = soe.events_as_dicts()
    assert len(events) == 3
    # Verify strict ascending chronological ordering
    assert events[0]["epoch_seconds"] == 0.0
    assert events[1]["epoch_seconds"] == 86400.0
    assert events[2]["epoch_seconds"] == 172800.0

    # Total Delta-v should be exactly sum
    assert math.isclose(soe.total_delta_v_ms, 65.0, abs_tol=1e-9)
    # Total propellant expended
    assert math.isclose(soe.total_propellant_used_kg, 23.0, abs_tol=1e-9)
    # Final mass
    assert math.isclose(soe.final_mass_kg, initial_mass - 23.0, abs_tol=1e-9)


def test_mission_executive_state_continuity() -> None:
    """Verify that multi-phase mission telemetry maintains strict physical continuity."""
    executive = MissionExecutive(
        mission_name="Continuity Test Flight",
        initial_wet_mass_kg=1000.0,
        thrust_max_n=10.0,
        isp_sec=4000.0,
    )

    res = executive.execute_mission(
        duration_days=10.0,
        step_hours=6.0,
        initial_pos_dispersion_m=1000.0,
        initial_vel_dispersion_ms=0.1,
    )

    telem = res["telemetry"]
    assert len(telem) > 0

    # Verify no unphysical state jumps across steps
    for i in range(len(telem) - 1):
        pt_curr = telem[i]
        pt_next = telem[i + 1]

        r_curr_m = np.array(pt_curr["r_au"]) * AU
        r_next_m = np.array(pt_next["r_au"]) * AU
        dr = float(np.linalg.norm(r_next_m - r_curr_m))

        v_curr_mps = np.array(pt_curr["v_kms"]) * 1000.0
        dt = (pt_next["day"] - pt_curr["day"]) * 86400.0

        # Maximum physical displacement cannot exceed v * dt + 0.5 * a_max * dt^2 + margin
        max_allowed_dr = float(np.linalg.norm(v_curr_mps)) * dt + 0.5 * 0.05 * (dt**2) + 1e5
        assert dr < max_allowed_dr, f"Discontinuity detected between day {pt_curr['day']} and {pt_next['day']}: dr = {dr} m"


def test_fms_dsn_blackout_and_tcm_triggers() -> None:
    """Verify that DSN blackout transitions and TCM burns generate discrete timeline events."""
    executive = MissionExecutive(
        mission_name="Blackout & TCM Trigger Test",
        initial_wet_mass_kg=1500.0,
    )

    res = executive.execute_mission(
        duration_days=20.0,
        step_hours=6.0,
        dsn_blackout_start_day=5.0,
        dsn_blackout_end_day=12.0,
    )

    events = res["events"]
    descriptions = [e["description"] for e in events]

    assert any("blackout commenced" in d.lower() for d in descriptions)
    assert any("tracking re-established" in d.lower() for d in descriptions)
    assert any("tcm-1" in d.lower() for d in descriptions)


def test_fms_api_endpoint() -> None:
    """Verify FastAPI endpoint POST /api/fms/execute_mission returns 200 OK and valid schema."""
    client = TestClient(app)
    payload = {
        "mission_name": "API Automated Test Flight",
        "duration_days": 15.0,
        "step_hours": 6.0,
        "initial_wet_mass_kg": 1200.0,
        "thrust_max_n": 8.0,
        "isp_sec": 4200.0,
        "initial_pos_dispersion_m": 2000.0,
        "initial_vel_dispersion_ms": 0.2,
        "dsn_blackout_start_day": 3.0,
        "dsn_blackout_end_day": 7.0,
    }

    resp = client.post("/api/fms/execute_mission", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["mission_name"] == "API Automated Test Flight"
    assert "events" in data and len(data["events"]) >= 4
    assert "telemetry" in data and len(data["telemetry"]) > 0
    assert data["total_propellant_used_kg"] > 0.0
    assert data["final_mass_kg"] < 1200.0
