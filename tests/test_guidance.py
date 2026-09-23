"""Unit test suite for relativistic closed-loop autonomous guidance (ZEM/ZEV)."""

from __future__ import annotations

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

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


def test_zem_zev_unperturbed_linear_optimality() -> None:
    """Verify analytical optimality and terminal convergence in unperturbed space."""
    r_target = np.array([1.0e6, 2.0e6, -5.0e5], dtype=np.float64)
    v_target = np.array([100.0, -50.0, 20.0], dtype=np.float64)
    t_f = 1000.0  # seconds

    # Initial state with significant miss
    r_init = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    v_init = np.array([50.0, 100.0, 0.0], dtype=np.float64)

    controller = ZEMZEVGuidanceLaw(
        target_position_m=r_target,
        target_velocity_ms=v_target,
        target_arrival_epoch_s=t_f,
        thrust_max_n=1e8,  # unconstrained for optimality proof
        isp_sec=4500.0,
        min_time_to_go_s=0.01,
        include_gravity=False,
    )

    r_curr = r_init.copy()
    v_curr = v_init.copy()
    dt = 0.1
    num_steps = int(t_f / dt)

    for step in range(num_steps):
        t = step * dt
        cmd = controller.compute_command(t, r_curr, v_curr, current_mass_kg=1000.0)
        # In flat unperturbed space, acceleration is pure commanded control
        r_curr += v_curr * dt + 0.5 * cmd.clamped_accel_ms2 * (dt**2)
        v_curr += cmd.clamped_accel_ms2 * dt

    miss_m = float(np.linalg.norm(r_curr - r_target))
    vel_err_ms = float(np.linalg.norm(v_curr - v_target))

    assert miss_m < 0.01, f"Terminal miss {miss_m:.6f} m exceeds 1 cm in linear unperturbed regime"
    assert vel_err_ms < 0.01, f"Terminal velocity error {vel_err_ms:.6f} m/s exceeds 1 cm/s"


def test_1pn_gravity_acceleration_and_schiff_precession() -> None:
    """Verify 1PN gravitational acceleration and Schiff de Sitter precession."""
    r_test = np.array([AU, 0.0, 0.0], dtype=np.float64)
    v_test = np.array([0.0, 29780.0, 0.0], dtype=np.float64)

    a_1pn = compute_1pn_gravity_acceleration(r_test, v_test)
    a_newton = -GM_SUN / (AU**2)

    # 1PN correction should be on the order of v^2/c^2 ~ 10^-8
    rel_diff = abs(float(a_1pn[0]) - a_newton) / abs(a_newton)
    assert 1e-9 < rel_diff < 1e-7, f"1PN relativistic relative acceleration correction {rel_diff:.3e} out of physical bounds"

    omega_schiff = compute_schiff_gyro_precession_rate(r_test, v_test)
    assert omega_schiff[0] == 0.0 and omega_schiff[1] == 0.0
    # Precession should be in +Z direction (r x v)
    assert omega_schiff[2] > 0.0
    arcsec_per_yr = float(omega_schiff[2]) * (180.0 / math.pi) * 3600.0 * (86400.0 * 365.25)
    # Solar geodetic precession at 1 AU is ~ 0.019 arcsec/yr
    assert 0.01 < arcsec_per_yr < 0.03, f"Schiff precession rate {arcsec_per_yr:.4f} arcsec/yr outside theoretical Earth orbit expectations"


def test_quaternion_attitude_pointing_unit_norm() -> None:
    """Verify unit quaternion conversion preserves strict unity norm."""
    test_vectors = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-3.5, 8.2, -1.1]),
        np.array([0.0, 0.0, 0.0]),
    ]

    for vec in test_vectors:
        q = vector_to_unit_quaternion(vec)
        q_norm = float(np.linalg.norm(q))
        assert abs(q_norm - 1.0) < 1e-14, f"Quaternion norm error {abs(q_norm - 1.0):.2e} violates unit invariant"


def test_relativistic_mass_depletion_and_thrust_clamping() -> None:
    """Verify relativistic mass depletion and acceleration limits under thrust saturation."""
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
    mass = 1000.0

    cmd = controller.compute_command(0.0, r, v, current_mass_kg=mass)

    # Commanded thrust should not exceed thrust_max
    assert cmd.thrust_n <= 10.0 + 1e-12
    # Mass flow rate should follow m_dot = T / (Isp * g0) * sqrt(1 - v^2/c^2)
    expected_mdot = (cmd.thrust_n / (3000.0 * G0)) * math.sqrt(1.0 - (29780.0 / C_LIGHT)**2)
    assert abs(cmd.mass_flow_rate_kg_s - expected_mdot) < 1e-12


def test_simulate_closed_loop_mission_convergence() -> None:
    """Verify closed-loop trajectory steering achieves sub-50m intercept under 5km dispersion."""
    res = simulate_closed_loop_mission(
        duration_days=15.0,
        step_hours=6.0,
        initial_pos_dispersion_m=5000.0,
        initial_vel_dispersion_ms=0.5,
        wet_mass_kg=1500.0,
        thrust_max_n=10.0,
        isp_sec=4500.0,
    )

    assert res["final_miss_distance_m"] < 50.0, f"Final miss {res['final_miss_distance_m']:.2f} m exceeds 50 m requirement"
    assert res["total_propellant_used_kg"] > 0.0
    assert res["final_mass_kg"] < 1500.0
    assert len(res["telemetry"]) > 0


def test_guidance_api_endpoint() -> None:
    """Verify FastAPI guidance simulation endpoint."""
    client = TestClient(app)
    payload = {
        "duration_days": 10.0,
        "step_hours": 6.0,
        "wet_mass_kg": 1200.0,
        "thrust_max_n": 8.0,
        "isp_sec": 4000.0,
        "initial_pos_dispersion_m": 2000.0,
        "initial_vel_dispersion_ms": 0.2,
    }

    resp = client.post("/api/guidance/zem_zev/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "final_miss_distance_m" in data
    assert "telemetry" in data
    assert len(data["telemetry"]) > 0
