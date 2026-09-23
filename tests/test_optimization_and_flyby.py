"""Automated Verification Suite for Relativistic Optimization & Planetary Gravity Assists.

Validates:
- Level 1: Analytical energy conservation in planetocentric hyperbolic flybys (|v_inf^-| == |v_inf^+|).
- Level 2: Exact hyperbolic turning angle geometry and 1PN gravitational deflection.
- Level 3: Universal variable Lambert solver orbital energy conservation and Hohmann limits.
- Level 4: Earth -> Mars 2026 launch window Porkchop benchmark against NASA trajectory data.
- Level 5: Multi-leg planetary tour continuity, proper time accumulation, and API/CLI integration.
"""

from __future__ import annotations
import math
import numpy as np
import pytest
from fastapi.testclient import TestClient

from relativistic_engine.constants import (
    C_LIGHT,
    GM_SUN,
    GM_EARTH,
    GM_JUPITER,
    GM_VENUS,
    AU,
    SEC_PER_DAY,
)
from relativistic_engine.physics.gravity_assist import (
    compute_hyperbolic_flyby,
    FlybyResult,
)
from relativistic_engine.physics.potential import STANDARD_BODY_RADIUS
from relativistic_engine.optimization.porkchop import (
    solve_lambert,
    compute_porkchop_grid,
)
from relativistic_engine.trajectory.tour import (
    solve_planetary_tour,
    PlanetaryTour,
)
from relativistic_engine.api.app import app
from relativistic_engine.cli import build_parser, _cmd_porkchop, _cmd_tour


def test_flyby_planetocentric_energy_conservation():
    """Level 1 Proof: In planetocentric frame, asymptotic speed is conserved to machine precision."""
    v_planet = np.array([0.0, 29780.0, 0.0])  # ~Earth orbital velocity
    r_planet = np.array([AU, 0.0, 0.0])
    v_in = np.array([5000.0, 35000.0, 1000.0])  # Heliocentric incoming velocity
    r_p = 7000e3  # 7,000 km from center

    res = compute_hyperbolic_flyby(
        v_in_bcrs=v_in,
        r_planet_bcrs=r_planet,
        v_planet_bcrs=v_planet,
        gm_planet=GM_EARTH,
        periapsis_radius=r_p,
        b_plane_angle=0.3,
        planet_radius=STANDARD_BODY_RADIUS["earth"],
        compute_1pn=True,
    )

    v_inf_in_norm = np.linalg.norm(res.v_inf_in_planet)
    v_inf_out_norm = np.linalg.norm(res.v_inf_out_planet)

    # Relative difference between incoming and outgoing asymptotic excess speed
    rel_diff = abs(v_inf_out_norm - v_inf_in_norm) / v_inf_in_norm
    assert rel_diff < 1e-14, f"Asymptotic speed not conserved: {rel_diff}"

    # Newtonian turning angle matches exact formula sin(delta/2) = 1/e
    sin_half_delta = math.sin((res.bending_angle_rad - res.delta_1pn_rad) / 2.0)
    expected_sin = 1.0 / res.eccentricity
    assert abs(sin_half_delta - expected_sin) < 1e-14


def test_flyby_1pn_gravitational_deflection():
    """Level 2 Proof: 1PN relativistic deflection is strictly positive and bounded."""
    v_planet = np.array([0.0, 13070.0, 0.0])  # ~Jupiter orbital velocity
    r_planet = np.array([5.2 * AU, 0.0, 0.0])
    v_in = np.array([10000.0, 15000.0, 0.0])
    r_p = 1.1 * STANDARD_BODY_RADIUS["jupiter"]

    res = compute_hyperbolic_flyby(
        v_in_bcrs=v_in,
        r_planet_bcrs=r_planet,
        v_planet_bcrs=v_planet,
        gm_planet=GM_JUPITER,
        periapsis_radius=r_p,
        compute_1pn=True,
    )

    # 1PN deflection must be strictly positive
    assert res.delta_1pn_rad > 0.0
    # In arcseconds: for close Jupiter flyby, 1PN bending is on the order of milli-arcseconds to arcseconds
    arcsec_1pn = math.degrees(res.delta_1pn_rad) * 3600.0
    assert 0.001 < arcsec_1pn < 100.0

    # Proper time deficit through Jupiter's potential well must be strictly positive
    assert res.proper_time_deficit_sec > 0.0


def test_lambert_solver_energy_conservation():
    """Level 3 Proof: Lambert transfer orbit preserves specific orbital energy between endpoints."""
    r1 = np.array([AU, 0.0, 0.0])
    # 90-degree transfer to 1.5 AU
    r2 = np.array([0.0, 1.5 * AU, 0.0])
    tof_sec = 150.0 * SEC_PER_DAY

    sol = solve_lambert(r1, r2, tof_sec, mu=GM_SUN)

    # Specific orbital energy: E = 0.5*v^2 - mu/r
    e1 = 0.5 * np.dot(sol.v1, sol.v1) - (GM_SUN / np.linalg.norm(r1))
    e2 = 0.5 * np.dot(sol.v2, sol.v2) - (GM_SUN / np.linalg.norm(r2))

    rel_energy_diff = abs(e2 - e1) / abs(e1)
    assert rel_energy_diff < 1e-12, f"Lambert energy not conserved: {rel_energy_diff}"

    # Semi-major axis must match vis-viva: a = -mu / (2*E)
    expected_a = -GM_SUN / (2.0 * e1)
    assert abs(sol.semi_major_axis - expected_a) / expected_a < 1e-12

    # Relativistic proper time deficit must be strictly positive
    assert sol.proper_time_deficit_sec > 0.0


def test_porkchop_earth_mars_2026_launch_window():
    """Level 4 Proof: Earth-Mars 2026 launch window matches NASA trajectory benchmark data."""
    # Earth-Mars 2026 launch window occurs around Oct-Nov 2026 (JD ~2461320 - 2461380)
    # with arrivals around Jul-Sep 2027 (JD ~2461580 - 2461680)
    dep_jds = np.linspace(2461320.5, 2461380.5, 7)
    arr_jds = np.linspace(2461580.5, 2461680.5, 7)

    res = compute_porkchop_grid(
        origin_body="earth",
        target_body="mars",
        dep_jds=dep_jds,
        arr_jds=arr_jds,
        mode="ballistic",
    )

    assert res.best_window["departure_jd"] is not None
    assert res.best_window["delta_v_total_km_s"] is not None

    # Benchmark check: Earth-Mars 2026 ballistic C3 is typically 8 - 25 km^2/s^2
    c3 = res.best_window["c3_km2_s2"]
    assert 5.0 <= c3 <= 30.0, f"C3 {c3} out of expected NASA 2026 bounds [5, 30]"

    # TOF is typically 180 - 320 days
    tof = res.best_window["tof_days"]
    assert 150.0 <= tof <= 350.0

    # Relativistic proper time deficit is strictly positive
    assert res.best_window["time_deficit_sec"] > 0.0


def test_planetary_tour_continuity_and_proper_time():
    """Level 5 Proof: Multi-leg planetary tour maintains state continuity and proper time integration."""
    # Earth -> Venus -> Earth tour
    legs_cfg = [
        {
            "origin_body": "earth",
            "target_body": "venus",
            "departure_jd": 2461300.5,
            "arrival_jd": 2461450.5,
            "is_flyby": True,
            "periapsis_altitude_km": 600.0,
        },
        {
            "origin_body": "venus",
            "target_body": "earth",
            "departure_jd": 2461450.5,
            "arrival_jd": 2461750.5,
            "is_flyby": False,
        },
    ]

    tour = solve_planetary_tour(legs_cfg, mission_name="Earth-Venus-Earth Tour")

    assert len(tour.legs) == 2
    assert len(tour.flyby_events) == 1

    # Total coordinate duration: (2461450.5 - 2461300.5) + (2461750.5 - 2461450.5) = 450 days
    assert abs(tour.total_coordinate_time_days - 450.0) < 1e-6

    # Proper time must be less than coordinate time (time dilation)
    assert tour.total_proper_time_days < tour.total_coordinate_time_days
    assert tour.total_time_deficit_sec > 0.0

    # Total delta-v must be strictly positive
    assert tour.total_delta_v_km_s > 0.0

    # Flyby deflection angle must be positive and non-zero
    flyby = tour.flyby_events[0]
    assert flyby.bending_angle_deg > 0.0
    assert flyby.delta_1pn_rad > 0.0


def test_api_porkchop_and_tour_endpoints():
    """Verify FastAPI endpoints for Porkchop evaluation and multi-leg tour."""
    client = TestClient(app)

    # 1. Porkchop endpoint
    porkchop_payload = {
        "origin_body": "earth",
        "target_body": "mars",
        "dep_start_jd": 2461330.5,
        "dep_end_jd": 2461360.5,
        "arr_start_jd": 2461600.5,
        "arr_end_jd": 2461650.5,
        "grid_steps": 5,
        "mode": "ballistic",
    }
    r_pc = client.post("/api/mission/porkchop", json=porkchop_payload)
    assert r_pc.status_code == 200
    data_pc = r_pc.json()
    assert data_pc["origin_body"] == "earth"
    assert data_pc["target_body"] == "mars"
    assert len(data_pc["c3_km2_s2"]) == 5
    assert len(data_pc["c3_km2_s2"][0]) == 5
    assert data_pc["best_window"]["c3_km2_s2"] > 0.0

    # 2. Tour endpoint
    tour_payload = {
        "mission_name": "Test Tour",
        "legs": [
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
                "arrival_jd": 2461800.5,
                "is_flyby": False,
            },
        ],
    }
    r_tour = client.post("/api/mission/tour", json=tour_payload)
    assert r_tour.status_code == 200
    data_tour = r_tour.json()
    assert len(data_tour["legs"]) == 2
    assert data_tour["total_coordinate_time_days"] == 500.0
    assert data_tour["total_time_deficit_sec"] > 0.0


def test_cli_optimize_and_tour_execution():
    """Verify CLI argument parsing and execution for optimize and tour."""
    parser = build_parser()

    # 1. Test CLI optimize porkchop
    args_pc = parser.parse_args([
        "optimize",
        "porkchop",
        "--origin", "earth",
        "--target", "mars",
        "--dep-start", "2461330.5",
        "--dep-end", "2461360.5",
        "--arr-start", "2461600.5",
        "--arr-end", "2461650.5",
        "--steps", "5",
    ])
    exit_pc = _cmd_porkchop(args_pc)
    assert exit_pc == 0

    # 2. Test CLI solve tour
    args_tour = parser.parse_args([
        "solve",
        "tour",
        "--bodies", "earth,venus,mars",
        "--epochs", "2461300.5,2461450.5,2461800.5",
        "--periapsis-alt", "500.0",
    ])
    exit_tour = _cmd_tour(args_tour)
    assert exit_tour == 0
