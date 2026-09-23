"""Scientific validation tests for relativistic propulsion and continuous low-thrust optimization.

Verifies:
- Ackeret (1946) relativistic rocket equation and invertibility.
- Sanger (1953) ideal photon rocket mechanics.
- Relativistic jet power asymptotic limits (Newtonian limit beta_e -> 0 and radiant limit beta_e -> 1).
- 3D Lorentz acceleration transformation covariance and projections.
- Hermite-Simpson direct collocation convergence and defect bounds.
- REST API endpoint POST /api/mission/low_thrust.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from relativistic_engine.constants import C_LIGHT, G0, GM_SUN, SEC_PER_DAY
from relativistic_engine.physics.kinematics import lorentz_gamma, lorentz_beta
from relativistic_engine.physics.dynamics import proper_to_coordinate_acceleration
from relativistic_engine.physics.propulsion import (
    relativistic_rocket_velocity,
    relativistic_mass_ratio,
    photon_rocket_velocity,
    photon_rocket_mass_ratio,
    relativistic_jet_power,
    mass_flow_rate,
    thrust_to_coordinate_acceleration,
)
from relativistic_engine.optimization.low_thrust import (
    LowThrustSolution,
    LowThrustTrajectoryOptimizer,
)
from relativistic_engine.api.app import app


def test_ackeret_relativistic_rocket_formula_and_invertibility():
    """Verify Ackeret (1946) relativistic rocket velocity against closed-form theory and test invertibility."""
    test_cases = [
        # (m0, mf, isp_sec, expected_beta_order)
        (1000.0, 500.0, 3000.0),      # Standard ion thruster (mu = 2)
        (5000.0, 1000.0, 50000.0),    # High-power fusion / electric (mu = 5)
        (10000.0, 100.0, 1.0e6),      # Relativistic plasma / beam-core (mu = 100)
        (100000.0, 1.0, 5.0e6),       # Extreme interstellar probe (mu = 100000)
    ]

    for m0, mf, isp in test_cases:
        mu = m0 / mf
        ve = isp * G0
        beta_e = ve / C_LIGHT

        v = relativistic_rocket_velocity(m0, mf, isp=isp)
        beta = v / C_LIGHT

        # Exact theoretical rapidity: theta = beta_e * ln(mu)
        theta_expected = beta_e * math.log(mu)
        beta_expected = math.tanh(theta_expected)

        assert math.isclose(beta, beta_expected, rel_tol=1e-14, abs_tol=1e-15)

        # Verify invertibility: relativistic_mass_ratio(beta) == mu
        mu_recovered = relativistic_mass_ratio(beta, isp=isp)
        assert math.isclose(mu_recovered, mu, rel_tol=1e-13)


def test_photon_rocket_exact_mechanics():
    """Verify Sanger (1953) ideal photon rocket kinematics and ultra-relativistic limits."""
    mass_ratios = [1.5, 2.0, 5.0, 10.0, 100.0, 1000.0]

    for mu in mass_ratios:
        m0 = 1000.0 * mu
        mf = 1000.0

        v = photon_rocket_velocity(m0, mf)
        beta = v / C_LIGHT

        # Sanger exact solution: beta = (mu^2 - 1) / (mu^2 + 1)
        expected_beta = (mu**2 - 1.0) / (mu**2 + 1.0)
        assert math.isclose(beta, expected_beta, rel_tol=1e-14)

        # Invertibility: mu = sqrt((1 + beta) / (1 - beta))
        # At mu = 1000 (beta ~ 0.999998), 1 - beta loses 6 digits to cancellation in float64
        mu_recovered = photon_rocket_mass_ratio(beta)
        assert math.isclose(mu_recovered, mu, rel_tol=1e-10)

        # Proper time Lorentz gamma factor: gamma * (1 + beta) = mu
        gamma = 1.0 / math.sqrt(1.0 - beta**2)
        assert math.isclose(gamma * (1.0 + beta), mu, rel_tol=1e-10)


def test_relativistic_jet_power_asymptotic_limits():
    """Verify relativistic exhaust beam kinetic power across Newtonian and radiant limits."""
    thrust = 100.0  # Newtons

    # 1. Newtonian limit: beta_e << 1 (e.g. chemical/ion with Isp = 3000 s -> ve ~ 29.4 km/s, beta_e ~ 1e-4)
    isp_classical = 3000.0
    ve_classical = isp_classical * G0
    p_rel = relativistic_jet_power(thrust, isp=isp_classical)
    p_newtonian = 0.5 * thrust * ve_classical

    # Relative difference should be O(beta_e^2) ~ 1e-8
    assert math.isclose(p_rel, p_newtonian, rel_tol=1e-7)

    # 2. Photon rocket limit: beta_e = 1.0
    p_photon = relativistic_jet_power(thrust, ve=C_LIGHT)
    assert math.isclose(p_photon, thrust * C_LIGHT, rel_tol=1e-14)


def test_mass_flow_rate_and_proper_time_coupling():
    """Verify propellant mass loss rates dm/dtau and dm/dt under relativistic time dilation."""
    thrust = 50.0
    isp = 4000.0
    ve = isp * G0

    # At rest: dtau/dt = 1.0
    dm_dtau, dm_dt = mass_flow_rate(thrust, isp=isp, dtau_dt=1.0)
    assert math.isclose(dm_dtau, -thrust / ve, rel_tol=1e-14)
    assert math.isclose(dm_dt, dm_dtau, rel_tol=1e-14)

    # In relativistic motion: dtau/dt = 0.6 (gamma = 1.6667)
    dtau_dt = 0.6
    dm_dtau_rel, dm_dt_rel = mass_flow_rate(thrust, isp=isp, dtau_dt=dtau_dt)
    assert math.isclose(dm_dtau_rel, -thrust / ve, rel_tol=1e-14)
    assert math.isclose(dm_dt_rel, dm_dtau_rel * dtau_dt, rel_tol=1e-14)
    assert abs(dm_dt_rel) < abs(dm_dtau_rel)  # Burns slower per coordinate second


def test_proper_to_coordinate_acceleration_covariance():
    """Verify 3D Lorentz acceleration transformation parallel, perpendicular, and zero-velocity limits."""
    # 1. Zero velocity limit: a_coord == a_proper
    v_zero = np.zeros(3)
    alpha = np.array([1.5, -2.0, 3.5])
    a_coord_0 = proper_to_coordinate_acceleration(v_zero, alpha)
    np.testing.assert_allclose(a_coord_0, alpha, rtol=1e-14)

    # 2. Purely parallel acceleration: a_parallel = alpha / gamma^3
    beta_val = 0.8
    gamma_val = 1.0 / math.sqrt(1.0 - beta_val**2)
    v_par = np.array([beta_val * C_LIGHT, 0.0, 0.0])
    alpha_par = np.array([10.0, 0.0, 0.0])
    a_coord_par = proper_to_coordinate_acceleration(v_par, alpha_par)
    expected_par = np.array([alpha_par[0] / (gamma_val**3), 0.0, 0.0])
    np.testing.assert_allclose(a_coord_par, expected_par, rtol=1e-14)

    # 3. Purely perpendicular acceleration: a_perp = alpha / gamma^2
    alpha_perp = np.array([0.0, 10.0, 0.0])
    a_coord_perp = proper_to_coordinate_acceleration(v_par, alpha_perp)
    expected_perp = np.array([0.0, alpha_perp[1] / (gamma_val**2), 0.0])
    np.testing.assert_allclose(a_coord_perp, expected_perp, rtol=1e-14)

    # 4. Projection identity: a_coord . v == (alpha . v) / gamma^3
    v_arb = np.array([0.3 * C_LIGHT, -0.4 * C_LIGHT, 0.2 * C_LIGHT])
    speed = float(np.linalg.norm(v_arb))
    gamma_arb = 1.0 / math.sqrt(1.0 - (speed / C_LIGHT)**2)
    alpha_arb = np.array([5.0, 3.0, -2.0])

    a_coord_arb = proper_to_coordinate_acceleration(v_arb, alpha_arb)
    proj_coord = float(np.dot(a_coord_arb, v_arb))
    proj_proper = float(np.dot(alpha_arb, v_arb)) / (gamma_arb**3)
    assert math.isclose(proj_coord, proj_proper, rel_tol=1e-13)


def test_low_thrust_forward_propagation_monotonic_mass():
    """Verify DOP853 forward propagation of continuous thrust exhibits monotonic mass loss and proper deficit."""
    optimizer = LowThrustTrajectoryOptimizer(
        isp=4500.0,
        thrust_max=5.0,
        dry_mass=500.0,
        departure_epoch_jd=2451545.0,
        bodies=["sun_point_mass"],
    )

    r0 = np.array([1.496e11, 0.0, 0.0])  # 1 AU
    v0 = np.array([0.0, 29780.0, 0.0])   # Earth circular velocity
    m0 = 1500.0
    t_span = (0.0, 10.0 * SEC_PER_DAY)   # 10 days

    # Constant prograde thrust along velocity
    def prograde_thrust(t, r, v):
        v_norm = float(np.linalg.norm(v))
        return (v / v_norm) * 5.0

    sol = optimizer.propagate_forward(r0, v0, m0, t_span, prograde_thrust)

    assert sol.success
    assert len(sol.times) > 10
    assert sol.final_mass < m0
    assert sol.final_mass >= 500.0
    assert sol.total_propellant_used > 0.0
    assert sol.final_deficit > 0.0
    assert sol.proper_times[-1] < sol.times[-1]

    # Verify mass loss rate matches -T / ve:
    # delta_m = T * delta_t / ve (approximate to 1st order)
    expected_loss = (5.0 * (10.0 * SEC_PER_DAY)) / (4500.0 * G0)
    assert math.isclose(sol.total_propellant_used, expected_loss, rel_tol=1e-4)


def test_low_thrust_collocation_orbit_transfer():
    """Verify Hermite-Simpson direct collocation converges on a continuous low-thrust orbital transfer."""
    optimizer = LowThrustTrajectoryOptimizer(
        isp=4500.0,
        thrust_max=10.0,
        dry_mass=500.0,
        departure_epoch_jd=2451545.0,
        bodies=["sun_point_mass"],
    )

    r0 = np.array([1.496e11, 0.0, 0.0])
    v0 = np.array([0.0, 29780.0, 0.0])

    # Target: 5 days forward on circular orbit with slightly raised radius (1.002 AU)
    dt_flight = 5.0 * SEC_PER_DAY
    mean_motion = 29780.0 / 1.496e11
    theta_f = mean_motion * dt_flight
    rf = 1.002 * 1.496e11 * np.array([math.cos(theta_f), math.sin(theta_f), 0.0])
    vf = 29780.0 * np.array([-math.sin(theta_f), math.cos(theta_f), 0.0])

    sol = optimizer.solve_rendezvous(
        r0=r0,
        v0=v0,
        rf=rf,
        vf=vf,
        m0=1200.0,
        t0=0.0,
        tf=dt_flight,
        num_segments=15,
        max_iter=150,
        tol=1e-3,
    )

    assert sol.final_mass <= 1200.0
    assert sol.final_mass >= 500.0
    assert sol.total_propellant_used >= 0.0
    assert sol.max_defect < 1e-2


def test_low_thrust_api_endpoint():
    """Verify REST API endpoint POST /api/mission/low_thrust."""
    client = TestClient(app)

    payload = {
        "departure_body": "earth",
        "target_body": "mars",
        "departure_epoch_jd": 2462622.5,
        "tof_days": 150.0,
        "initial_mass_kg": 1500.0,
        "dry_mass_kg": 500.0,
        "thrust_max_n": 2.5,
        "isp_sec": 4500.0,
        "num_segments": 10,
    }

    response = client.post("/api/mission/low_thrust", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["departure_body"] == "earth"
    assert data["target_body"] == "mars"
    assert data["initial_mass_kg"] == 1500.0
    assert data["final_mass_kg"] <= 1500.0
    assert data["final_mass_kg"] >= 500.0
    assert data["propellant_used_kg"] >= 0.0
    assert data["mass_ratio"] >= 1.0
    assert len(data["trajectory_points"]) > 0
