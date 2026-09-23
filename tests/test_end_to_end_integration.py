"""End-to-end pipeline integration test: ephemeris -> trajectory -> navigation arc -> manifest.

Validates that the entire engine pipeline can be composed sequentially without
errors, producing physically plausible outputs and a well-formed JSON-LD
reproducibility manifest.  This test does NOT replace the unit/component tests;
it validates inter-module composition and the API-level reproducibility contract.
"""

from __future__ import annotations

import math
import json
import numpy as np
import pytest

from relativistic_engine.constants import C_LIGHT, AU, G0, SEC_PER_DAY
from relativistic_engine.ephemeris.jpl_loader import load_jpl_ephemeris
from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state
from relativistic_engine.trajectory.rendezvous import (
    RendezvousMode,
    solve_interplanetary_rendezvous,
)
from relativistic_engine.navigation.dsn import (
    DSN_STATIONS,
    compute_shapiro_time_delay,
)
from relativistic_engine.navigation.orbit_determination import (
    ExtendedKalmanFilter,
    TrackingObservation as ODTrackingObservation,
)
from relativistic_engine.api.manifest import generate_manifest
from relativistic_engine.api.app import app


# ---------------------------------------------------------------------------
# Stage 1: Ephemeris sanity
# ---------------------------------------------------------------------------

class TestEphemerisSanity:
    """Verify DE440s ephemeris loads and returns physically plausible states."""

    def test_earth_mars_positions_2030(self):
        spk = load_jpl_ephemeris()
        # 2030-Jan-01 TDB: JD 2462867.5
        jd = 2462867.5
        earth = get_body_barycentric_state("earth", jd, spk=spk)
        mars = get_body_barycentric_state("mars", jd, spk=spk)

        r_earth = np.linalg.norm(earth.position)
        r_mars = np.linalg.norm(mars.position)

        # Earth should be within 0.97-1.02 AU of the Sun
        assert 0.97 * AU < r_earth < 1.02 * AU, (
            f"Earth heliocentric radius implausible: {r_earth / AU:.4f} AU"
        )
        # Mars should be within 1.38-1.67 AU (orbital range)
        assert 1.38 * AU < r_mars < 1.67 * AU, (
            f"Mars heliocentric radius implausible: {r_mars / AU:.4f} AU"
        )


# ---------------------------------------------------------------------------
# Stage 2: Trajectory solve
# ---------------------------------------------------------------------------

class TestTrajectoryPipeline:
    """Earth->Mars 2030 brachistochrone at 1g must produce physically consistent results."""

    @pytest.fixture(scope="class")
    @classmethod
    def rendezvous_solution(cls):
        return solve_interplanetary_rendezvous(
            departure_body="earth",
            target_body="mars",
            departure_epoch_jd=2462867.5,
            accel_magnitude=1.0 * G0,
            mode=RendezvousMode.SOFT_RENDEZVOUS,
        )

    def test_flight_time_plausible(self, rendezvous_solution):
        t = rendezvous_solution.flight_time_days
        # At 1g, Earth-Mars brachistochrone is roughly 3-7 days depending on geometry
        assert 1.0 < t < 20.0, f"Flight time implausible: {t:.2f} days"

    def test_proper_time_less_than_coordinate(self, rendezvous_solution):
        tau = rendezvous_solution.crew_proper_time_days
        t = rendezvous_solution.flight_time_days
        # Traveller's proper time must be strictly less than coordinate time
        assert tau < t, f"tau={tau:.4f} d >= t={t:.4f} d violates SR"

    def test_time_deficit_positive(self, rendezvous_solution):
        deficit = rendezvous_solution.coordinate_time_deficit_seconds
        assert deficit > 0.0, "Time deficit (t - tau) must be positive for relativistic travel"

    def test_miss_distance_below_threshold(self, rendezvous_solution):
        miss_km = rendezvous_solution.position_error_m / 1000.0
        assert miss_km < 1000.0, f"Miss distance too large: {miss_km:.1f} km"


# ---------------------------------------------------------------------------
# Stage 3: DSN Shapiro delay
# ---------------------------------------------------------------------------

class TestShapiroDelay:
    """Shapiro delay at a canonical Earth-Sun-Mars geometry must be non-trivial."""

    def test_typical_solar_conjunction_delay(self):
        # Transmitter (Earth) at 1 AU, receiver (spacecraft) at 1.5 AU from Sun,
        # with the ray passing near the Sun at ~2 R_Sun impact parameter.
        R_SUN = 6.957e8
        AU_M = 1.495978707e11

        # Place both endpoints on opposite sides of the Sun (near-conjunction geometry)
        r_tx = np.array([AU_M, 0.0, 0.0])
        r_rx = np.array([-1.5 * AU_M, 2.0 * R_SUN, 0.0])

        delay_s = compute_shapiro_time_delay(
            r_tx=r_tx,
            r_rx=r_rx,
            gm_body=1.32712440018e20,  # GM_Sun
        )
        delay_us = delay_s * 1e6
        # Historical benchmark: maximum solar conjunction delay ~200-240 us (Moyer 2000)
        assert 50.0 < delay_us < 300.0, (
            f"Shapiro delay outside expected range: {delay_us:.2f} us"
        )


# ---------------------------------------------------------------------------
# Stage 4: EKF navigation arc
# ---------------------------------------------------------------------------

class TestEKFNavigationArc:
    """EKF over a simulated short tracking arc must not diverge."""

    def test_ekf_state_covariance_shrinks(self):
        spk = load_jpl_ephemeris()
        jd0 = 2462867.5
        earth = get_body_barycentric_state("earth", jd0, spk=spk)

        # Spacecraft at Earth + 100 km altitude in +x direction
        r0 = earth.position + np.array([1e5, 0.0, 0.0])
        v0 = earth.velocity + np.array([0.0, 500.0, 0.0])  # small transverse boost
        x0 = np.concatenate([r0, v0])
        P0 = np.diag([1e8, 1e8, 1e8, 1e2, 1e2, 1e2])  # 10 km, 10 m/s prior

        rng = np.random.default_rng(42)
        sigma_r = 2.0   # m
        sigma_v = 5e-4  # m/s

        # Simulate 5 observations over 5 minutes from Goldstone
        ekf = ExtendedKalmanFilter(
            initial_state=x0.copy(),
            initial_covariance=P0.copy(),
            initial_epoch_tdb=0.0,
        )
        obs_list = []
        for k in range(1, 6):
            t_k = float(k * 60.0)  # 60 s spacing
            rho_vec = x0[:3] - DSN_STATIONS["DSS-14"].itrf_pos  # approximate
            rho = np.linalg.norm(rho_vec)
            obs_list.append(
                ODTrackingObservation(
                    epoch_tdb=t_k,
                    station_id="DSS-14",
                    range_m=rho + rng.normal(0.0, sigma_r),
                    range_rate_mps=0.0 + rng.normal(0.0, sigma_v),
                    range_sigma_m=sigma_r,
                    range_rate_sigma_mps=sigma_v,
                )
            )

        results = ekf.process_tracking_arc(obs_list, jd_base=jd0, spk=spk)
        assert len(results) == 5, "Expected 5 filter update steps"

        # Trace of covariance should not grow unboundedly
        tr_P_final = np.trace(results[-1].covariance)
        tr_P_initial = np.trace(P0)
        assert tr_P_final < tr_P_initial, (
            f"EKF covariance trace grew: {tr_P_final:.3e} >= {tr_P_initial:.3e}"
        )


# ---------------------------------------------------------------------------
# Stage 5: Reproducibility manifest
# ---------------------------------------------------------------------------

class TestReproducibilityManifest:
    """The manifest generator must produce a well-formed JSON-LD document
    with all mandatory provenance fields present."""

    def test_mandatory_fields_present(self):
        inputs = {"test": "integration", "epoch_jd": 2462867.5}
        outputs = {"flight_time_days": 4.5, "proper_time_days": 4.499}
        manifest = generate_manifest(inputs=inputs, outputs=outputs, endpoint="/test")

        assert "@context" in manifest
        assert "@type" in manifest
        assert manifest["@type"] == "Dataset"
        assert "engine_version" in manifest
        assert "ephemeris" in manifest
        assert "kernel" in manifest["ephemeris"]
        assert "constants" in manifest
        assert "C_LIGHT_mps" in manifest["constants"]
        assert manifest["constants"]["C_LIGHT_mps"]["value"] == C_LIGHT
        assert "outputs" in manifest
        assert "result_sha256" in manifest["outputs"]
        # SHA-256 must be a 64-char hex string
        sha = manifest["outputs"]["result_sha256"]
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)

    def test_manifest_is_json_serialisable(self):
        manifest = generate_manifest(inputs={"a": 1}, outputs={"b": 2})
        dumped = json.dumps(manifest)
        reloaded = json.loads(dumped)
        assert reloaded["engine_version"] == manifest["engine_version"]

    def test_different_outputs_produce_different_sha256(self):
        m1 = generate_manifest(inputs={}, outputs={"result": 1.0})
        m2 = generate_manifest(inputs={}, outputs={"result": 2.0})
        assert m1["outputs"]["result_sha256"] != m2["outputs"]["result_sha256"]


# ---------------------------------------------------------------------------
# Stage 6: API schema integrity
# ---------------------------------------------------------------------------

class TestAPISchemaIntegrity:
    """Navigation endpoints must appear in the OpenAPI 3.x schema."""

    def test_navigation_endpoints_in_openapi_spec(self):
        spec = app.openapi()
        paths = set(spec.get("paths", {}).keys())
        assert "/api/navigation/shapiro" in paths, (
            "Missing /api/navigation/shapiro in OpenAPI spec"
        )
        assert "/api/navigation/orbit_determination" in paths, (
            "Missing /api/navigation/orbit_determination in OpenAPI spec"
        )
        assert "/api/manifest" in paths, (
            "Missing /api/manifest in OpenAPI spec"
        )

    def test_api_version_is_1_0_0(self):
        spec = app.openapi()
        assert spec["info"]["version"] == "1.0.0", (
            f"Expected API version 1.0.0, got {spec['info']['version']}"
        )
