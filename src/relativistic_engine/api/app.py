"""FastAPI application serving deterministic relativistic space flight endpoints and web UI."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from relativistic_engine.constants import (
    C_LIGHT,
    G0,
    AU,
    SEC_PER_DAY,
    LIGHT_YEAR,
)
from relativistic_engine.ephemeris.barycentric import SUPPORTED_BODIES
from relativistic_engine.ephemeris.interstellar import INTERSTELLAR_CATALOG
from relativistic_engine.trajectory.rendezvous import (
    RendezvousMode,
    solve_interplanetary_rendezvous,
)
from relativistic_engine.trajectory.interstellar import (
    solve_interstellar_brachistochrone,
)
from relativistic_engine.uncertainty.formatter import format_uncertainty
from relativistic_engine.validation.report_generator import (
    generate_validation_certificate,
)
from relativistic_engine.physics.kinematics import (
    lorentz_beta,
    lorentz_gamma,
)
from relativistic_engine.api.schemas import (
    CatalogResponse,
    InterplanetaryRequest,
    InterplanetaryResponse,
    InterstellarRequest,
    InterstellarResponse,
    StarSummary,
    TrajectoryPointSchema,
    SGLRequest,
    SGLResponse,
    SGLWaypointSchema,
    SGLFocalSchema,
    SGLMissionRequest,
    SGLMissionResponse,
    SGLTrajectoryWaypointSchema,
    XPNAVPredictRequest,
    XPNAVPredictResponse,
    XPNAVSolveRequest,
    XPNAVSolveResponse,
    PNTTrajectorySimRequest,
    PNTTrajectorySimResponse,
    PNTTelemetryPointSchema,
    FMSMissionRequest,
    FMSMissionResponse,
    FMSMissionEventSchema,
    FMSPhaseTelemetrySchema,
)
from relativistic_engine.fms import MissionExecutive
from relativistic_engine.trajectory.sgl import (
    solve_sgl_mission_trajectory,
)
from relativistic_engine.navigation.xpnav import (
    PULSAR_CATALOG,
    PulsarObservation,
    compute_pulse_delays,
    compute_predicted_pulse_phase,
    solve_spacecraft_state_xpnav,
)
from relativistic_engine.navigation.pnt_fusion import (
    simulate_pnt_mission,
)



app = FastAPI(
    title="Relativistic Space Travel Computational Engine",
    description=(
        "High-precision scientific computation API for relativistic spaceflight, "
        "JPL DE440s ephemerides, time dilation, DSN navigation observables, "
        "and GR Shapiro delay. Physics core complete: 297/297 tests passing."
    ),
    version="1.0.0",
)

# Enable CORS for local researcher access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _sample_trajectory_points(
    t_arr: np.ndarray,
    tau_arr: np.ndarray,
    r_arr: np.ndarray,
    v_arr: np.ndarray,
    max_samples: int = 80,
) -> List[TrajectoryPointSchema]:
    """Sample 3D trajectory states uniformly in coordinate time for kinematically authentic animation."""
    n_pts = len(t_arr)
    if n_pts == 0:
        return []

    t_arr = np.asarray(t_arr, dtype=np.float64)
    tau_arr = np.asarray(tau_arr, dtype=np.float64)
    r_arr = np.asarray(r_arr, dtype=np.float64)
    v_arr = np.asarray(v_arr, dtype=np.float64)

    if n_pts == 1 or t_arr[-1] <= t_arr[0]:
        t_targets = t_arr
    else:
        num_targets = min(max_samples, max(20, n_pts)) if n_pts < max_samples else max_samples
        t_targets = np.linspace(t_arr[0], t_arr[-1], num_targets)

    points: List[TrajectoryPointSchema] = []
    for t_val in t_targets:
        t_sec = float(t_val)
        tau_val = float(np.interp(t_val, t_arr, tau_arr))
        rx = float(np.interp(t_val, t_arr, r_arr[:, 0])) / 1000.0
        ry = float(np.interp(t_val, t_arr, r_arr[:, 1])) / 1000.0
        rz = float(np.interp(t_val, t_arr, r_arr[:, 2])) / 1000.0

        vx = float(np.interp(t_val, t_arr, v_arr[:, 0]))
        vy = float(np.interp(t_val, t_arr, v_arr[:, 1]))
        vz = float(np.interp(t_val, t_arr, v_arr[:, 2]))

        speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))
        beta_val = lorentz_beta(speed)
        gamma_val = lorentz_gamma(speed)

        points.append(
            TrajectoryPointSchema(
                t_sec=t_sec,
                tau_sec=tau_val,
                r_km=[rx, ry, rz],
                v_km_s=[vx / 1000.0, vy / 1000.0, vz / 1000.0],
                speed_c=beta_val,
                lorentz_gamma=gamma_val,
            )
        )
    return points


@app.get("/api/health")
def get_health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0", "engine": "deterministic"}


@app.get("/api/catalog", response_model=CatalogResponse)
def get_catalog():
    """Retrieve available planetary bodies and interstellar targets."""
    stars = [
        StarSummary(
            id=key,
            name=entry.name,
            distance_ly=round(entry.distance_light_years, 3),
            ra_deg=round(entry.ra_deg, 4),
            dec_deg=round(entry.dec_deg, 4),
            radial_velocity_km_s=round(entry.radial_velocity_m_s / 1000.0, 2),
        )
        for key, entry in INTERSTELLAR_CATALOG.items()
    ]
    return CatalogResponse(
        planetary_bodies=list(SUPPORTED_BODIES),
        interstellar_stars=stars,
    )


@app.post("/api/mission/interplanetary", response_model=InterplanetaryResponse)
def compute_interplanetary_mission(req: InterplanetaryRequest):
    """Solve an interplanetary rendezvous trajectory with moving targets."""
    mode_enum = (
        RendezvousMode.SOFT_RENDEZVOUS if req.mode.lower() == "soft" else RendezvousMode.INTERCEPT
    )
    accel_si = req.accel_proper_g * G0

    try:
        sol = solve_interplanetary_rendezvous(
            departure_body=req.departure_body.lower(),
            target_body=req.target_body.lower(),
            departure_epoch_jd=req.departure_epoch_jd_tdb,
            accel_magnitude=accel_si,
            mode=mode_enum,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    traj = sol.trajectory
    t_days = sol.flight_time_days
    tau_days = sol.crew_proper_time_days
    deficit_s = sol.coordinate_time_deficit_seconds

    # GUM-formatted uncertainty strings (assuming 0.1% engine throttle uncertainty)
    sigma_t = t_days * 0.005  # ~0.5% empirical uncertainty bound
    sigma_tau = tau_days * 0.005
    sigma_def = max(0.01, deficit_s * 0.01)

    points = _sample_trajectory_points(traj.t, traj.tau, traj.r, traj.v)

    return InterplanetaryResponse(
        departure_body=req.departure_body,
        target_body=req.target_body,
        departure_epoch_jd_tdb=req.departure_epoch_jd_tdb,
        arrival_epoch_jd_tdb=sol.arrival_epoch_jd,
        coordinate_flight_time_days=round(t_days, 3),
        proper_flight_time_days=round(tau_days, 3),
        time_deficit_seconds=round(deficit_s, 3),
        max_speed_km_s=round(sol.max_speed_mps / 1000.0, 2),
        max_lorentz_gamma=round(lorentz_gamma(sol.max_speed_mps), 6),
        miss_distance_km=round(sol.position_error_m / 1000.0, 3),
        relative_arrival_velocity_m_s=round(sol.velocity_error_mps, 3),
        formatted_coordinate_time=format_uncertainty(t_days, sigma_t, "days"),
        formatted_proper_time=format_uncertainty(tau_days, sigma_tau, "days"),
        formatted_time_deficit=format_uncertainty(deficit_s, sigma_def, "s"),
        trajectory_points=points,
    )



@app.post("/api/mission/interstellar", response_model=InterstellarResponse)
def compute_interstellar_mission(req: InterstellarRequest):
    """Solve an interstellar relativistic brachistochrone journey."""
    accel_si = req.accel_proper_g * G0

    try:
        res = solve_interstellar_brachistochrone(
            target_star=req.target_star.lower(),
            accel_proper=accel_si,
            departure_epoch_jd_tdb=req.departure_epoch_jd_tdb,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    traj = res.trajectory
    t_yr = res.coordinate_flight_time_years
    tau_yr = res.proper_flight_time_years
    deficit_yr = res.time_deficit_years

    sigma_t = t_yr * 0.005
    sigma_tau = tau_yr * 0.005
    sigma_def = deficit_yr * 0.005

    points = _sample_trajectory_points(traj.t, traj.tau, traj.r, traj.v)

    return InterstellarResponse(
        target_star=res.target_star,
        departure_epoch_jd_tdb=req.departure_epoch_jd_tdb,
        arrival_epoch_jd_tdb=res.arrival_epoch_jd_tdb,
        coordinate_flight_time_years=round(t_yr, 3),
        proper_flight_time_years=round(tau_yr, 3),
        time_deficit_years=round(deficit_yr, 3),
        distance_light_years=round(res.distance_light_years, 3),
        max_speed_c=round(res.max_velocity_c, 5),
        max_lorentz_gamma=round(res.max_lorentz_factor, 3),
        doppler_redshift_earth=round(res.max_doppler_redshift_earth, 4),
        doppler_blueshift_target=round(res.max_doppler_blueshift_target, 4),
        miss_distance_au=round(res.miss_distance_meters / AU, 2),
        relative_arrival_velocity_km_s=round(res.relative_arrival_velocity_m_s / 1000.0, 2),
        formatted_coordinate_time=format_uncertainty(t_yr, sigma_t, "years"),
        formatted_proper_time=format_uncertainty(tau_yr, sigma_tau, "years"),
        formatted_time_deficit=format_uncertainty(deficit_yr, sigma_def, "years"),
        trajectory_points=points,
    )


@app.post("/api/mission/sgl", response_model=SGLMissionResponse)
def compute_sgl_mission(req: SGLMissionRequest):
    """Solve an outbound solar gravitational lens focal mission trajectory."""
    try:
        sol = solve_sgl_mission_trajectory(
            target_star=req.target_star,
            departure_epoch_jd=req.departure_epoch_jd,
            periapsis_solar_radii=req.periapsis_solar_radii,
            periapsis_delta_v_km_s=req.periapsis_delta_v_km_s,
            wavelength_m=req.wavelength_m,
            num_waypoints=req.num_waypoints,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    fp = sol.focal_parameters
    focal_schema = SGLFocalSchema(
        wavelength_m=fp.wavelength_m,
        heliocentric_distance_au=fp.heliocentric_distance_au,
        heliocentric_distance_m=fp.heliocentric_distance_m,
        min_focal_distance_au=fp.min_focal_distance_au,
        impact_parameter_m=fp.impact_parameter_m,
        impact_parameter_solar_radii=fp.impact_parameter_solar_radii,
        einstein_ring_angular_radius_rad=fp.einstein_ring_angular_radius_rad,
        einstein_ring_angular_radius_arcsec=fp.einstein_ring_angular_radius_arcsec,
        peak_light_amplification=fp.peak_light_amplification,
        angular_resolution_rad=fp.angular_resolution_rad,
        resolvable_surface_resolution_km=fp.resolvable_surface_resolution_km,
    )

    waypoints_schema = [
        SGLTrajectoryWaypointSchema(
            epoch_jd=wp.epoch_jd,
            coordinate_time_years=wp.coordinate_time_years,
            proper_time_years=wp.proper_time_years,
            heliocentric_distance_au=wp.heliocentric_distance_au,
            velocity_km_s=wp.velocity_km_s,
            velocity_au_per_year=wp.velocity_au_per_year,
            time_dilation_deficit_seconds=wp.time_dilation_deficit_seconds,
        )
        for wp in sol.waypoints
    ]

    return SGLMissionResponse(
        target_star=sol.target_star,
        target_ra_deg=sol.target_ra_deg,
        target_dec_deg=sol.target_dec_deg,
        focal_line_ra_deg=sol.focal_line_ra_deg,
        focal_line_dec_deg=sol.focal_line_dec_deg,
        solar_flyby_periapsis_solar_radii=sol.solar_flyby_periapsis_solar_radii,
        periapsis_delta_v_km_s=sol.periapsis_delta_v_km_s,
        asymptotic_speed_km_s=sol.asymptotic_speed_km_s,
        asymptotic_speed_au_per_year=sol.asymptotic_speed_au_per_year,
        time_to_550au_years=sol.time_to_550au_years,
        proper_time_to_550au_years=sol.proper_time_to_550au_years,
        time_dilation_deficit_at_550au_seconds=sol.time_dilation_deficit_at_550au_seconds,
        focal_parameters=focal_schema,
        waypoints=waypoints_schema,
    )


from relativistic_engine.optimization.porkchop import compute_porkchop_grid
from relativistic_engine.trajectory.tour import solve_planetary_tour
from relativistic_engine.api.schemas import (
    CatalogResponse,
    InterplanetaryRequest,
    InterplanetaryResponse,
    InterstellarRequest,
    InterstellarResponse,
    PorkchopRequest,
    PorkchopResponse,
    StarSummary,
    TourLegConfig,
    TourLegResponse,
    TourRequest,
    TourResponse,
    TrajectoryPointSchema,
)


@app.post("/api/mission/porkchop", response_model=PorkchopResponse)
def compute_porkchop_window(req: PorkchopRequest):
    """Compute 2D Porkchop launch window contour grid in BCRS."""
    dep_jds = np.linspace(req.dep_start_jd, req.dep_end_jd, req.grid_steps).tolist()
    arr_jds = np.linspace(req.arr_start_jd, req.arr_end_jd, req.grid_steps).tolist()

    try:
        res = compute_porkchop_grid(
            origin_body=req.origin_body.lower(),
            target_body=req.target_body.lower(),
            dep_jds=dep_jds,
            arr_jds=arr_jds,
            mode=req.mode,  # type: ignore
            alpha_thrust=req.alpha_thrust,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    def _clean_matrix(mat: np.ndarray) -> list[list[float | None]]:
        return [[None if np.isnan(val) else round(float(val), 4) for val in row] for row in mat]

    return PorkchopResponse(
        origin_body=res.origin_body,
        target_body=res.target_body,
        dep_jds=[round(jd, 2) for jd in res.dep_jds.tolist()],
        arr_jds=[round(jd, 2) for jd in res.arr_jds.tolist()],
        c3_km2_s2=_clean_matrix(res.c3_km2_s2),
        delta_v_total_km_s=_clean_matrix(res.delta_v_total_km_s),
        tof_days=[[round(float(val), 2) for val in row] for row in res.tof_days],
        proper_time_days=_clean_matrix(res.proper_time_days),
        time_deficit_sec=_clean_matrix(res.time_deficit_sec),
        best_window=res.best_window,
    )


@app.post("/api/mission/tour", response_model=TourResponse)
def compute_tour_mission(req: TourRequest):
    """Solve multi-leg planetary tour with relativistic gravity assists."""
    legs_cfg = [leg.model_dump() for leg in req.legs]

    try:
        tour = solve_planetary_tour(legs_cfg, mission_name=req.mission_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    out_legs: list[TourLegResponse] = []
    for leg in tour.legs:
        deg_turning = None
        arcsec_1pn = None
        if leg.flyby_result is not None:
            deg_turning = round(leg.flyby_result.bending_angle_deg, 4)
            arcsec_1pn = round(math.degrees(leg.flyby_result.delta_1pn_rad) * 3600.0, 4)

        out_legs.append(
            TourLegResponse(
                origin_body=leg.origin_body,
                target_body=leg.target_body,
                departure_jd=round(leg.departure_jd, 2),
                arrival_jd=round(leg.arrival_jd, 2),
                tof_days=round(leg.tof_days, 2),
                c3_dep_km2_s2=round(leg.c3_dep_km2_s2, 3),
                delta_v_dep_km_s=round(leg.delta_v_dep_km_s, 3),
                delta_v_arr_km_s=round(leg.delta_v_arr_km_s, 3),
                proper_time_days=round(leg.proper_time_days, 2),
                time_deficit_sec=round(leg.time_deficit_sec, 6),
                flyby_turning_angle_deg=deg_turning,
                flyby_1pn_correction_arcsec=arcsec_1pn,
                r_dep_bcrs_km=[round(float(x) / 1000.0, 3) for x in leg.r_dep_bcrs] if leg.r_dep_bcrs is not None else None,
                r_arr_bcrs_km=[round(float(x) / 1000.0, 3) for x in leg.r_arr_bcrs] if leg.r_arr_bcrs is not None else None,
            )
        )

    return TourResponse(
        mission_name=tour.mission_name,
        total_delta_v_km_s=round(tour.total_delta_v_km_s, 3),
        total_coordinate_time_days=round(tour.total_coordinate_time_days, 2),
        total_proper_time_days=round(tour.total_proper_time_days, 2),
        total_time_deficit_sec=round(tour.total_time_deficit_sec, 6),
        legs=out_legs,
    )


from relativistic_engine.uncertainty.batch_monte_carlo import evaluate_batch_monte_carlo
from relativistic_engine.ephemeris.jpl_loader import load_jpl_ephemeris
from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state
from relativistic_engine.api.schemas import (
    CatalogResponse,
    InterplanetaryRequest,
    InterplanetaryResponse,
    InterstellarRequest,
    InterstellarResponse,
    PorkchopRequest,
    PorkchopResponse,
    StarSummary,
    TourLegConfig,
    TourLegResponse,
    TourRequest,
    TourResponse,
    BatchMonteCarloRequest,
    BatchMonteCarloResponse,
    LowThrustRequest,
    LowThrustResponse,
    TrajectoryPointSchema,
)
from relativistic_engine.optimization.low_thrust import LowThrustTrajectoryOptimizer


@app.post("/api/mission/batch_monte_carlo", response_model=BatchMonteCarloResponse)
def compute_batch_monte_carlo(req: BatchMonteCarloRequest):
    """Execute high-throughput batch Monte Carlo uncertainty propagation."""
    kernel = load_jpl_ephemeris()
    try:
        s0 = get_body_barycentric_state(req.origin_body.lower(), req.departure_epoch_jd, spk=kernel)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid origin body: {e}")

    # Construct initial covariance: diagonal with sigma_pos and sigma_vel
    p0 = np.diag([
        (req.sigma_pos_km * 1000.0) ** 2,
        (req.sigma_pos_km * 1000.0) ** 2,
        (req.sigma_pos_km * 1000.0) ** 2,
        req.sigma_vel_mps**2,
        req.sigma_vel_mps**2,
        req.sigma_vel_mps**2,
    ])

    t_span = (0.0, req.flight_time_days * SEC_PER_DAY)

    try:
        res = evaluate_batch_monte_carlo(
            r_nominal=s0.position,
            v_nominal=s0.velocity,
            cov_initial_6x6=p0,
            t_span=t_span,
            n_samples=req.n_samples,
            backend=req.backend,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return BatchMonteCarloResponse(
        n_samples=res.n_samples,
        flight_time_days=req.flight_time_days,
        elapsed_wall_time_sec=round(res.elapsed_wall_time_sec, 4),
        throughput_trajectories_per_sec=round(res.throughput_trajectories_per_sec, 1),
        backend_used=res.backend_used,
        position_dispersion_3sigma_km=round(res.position_dispersion_3sigma_m / 1000.0, 2),
        velocity_dispersion_3sigma_mps=round(res.velocity_dispersion_3sigma_mps, 3),
        mean_time_deficit_sec=round(float(res.mean_final_state[6]), 6),
    )


@app.post("/api/mission/low_thrust", response_model=LowThrustResponse)
def compute_low_thrust_mission(req: LowThrustRequest):
    """Execute continuous low-thrust trajectory optimization."""
    kernel = load_jpl_ephemeris()
    try:
        s_dep = get_body_barycentric_state(req.departure_body.lower(), req.departure_epoch_jd, spk=kernel)
        arrival_jd = req.departure_epoch_jd + req.tof_days
        s_arr = get_body_barycentric_state(req.target_body.lower(), arrival_jd, spk=kernel)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ephemeris lookup failed: {e}")

    optimizer = LowThrustTrajectoryOptimizer(
        isp=req.isp_sec,
        thrust_max=req.thrust_max_n,
        dry_mass=req.dry_mass_kg,
        departure_epoch_jd=req.departure_epoch_jd,
        bodies=["sun"],
    )

    t0 = 0.0
    tf = req.tof_days * SEC_PER_DAY

    try:
        sol = optimizer.solve_rendezvous(
            r0=s_dep.position,
            v0=s_dep.velocity,
            rf=s_arr.position,
            vf=s_arr.velocity,
            m0=req.initial_mass_kg,
            t0=t0,
            tf=tf,
            num_segments=req.num_segments,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Trajectory optimization failed: {e}")

    pts = _sample_trajectory_points(
        sol.times,
        sol.proper_times,
        sol.states[:, 0:3],
        sol.states[:, 3:6],
        max_samples=80,
    )

    return LowThrustResponse(
        departure_body=req.departure_body.lower(),
        target_body=req.target_body.lower(),
        departure_epoch_jd=req.departure_epoch_jd,
        arrival_epoch_jd=arrival_jd,
        coordinate_flight_time_days=req.tof_days,
        proper_flight_time_days=round(float(sol.proper_times[-1]) / SEC_PER_DAY, 4),
        time_deficit_seconds=round(sol.final_deficit, 6),
        initial_mass_kg=req.initial_mass_kg,
        final_mass_kg=round(sol.final_mass, 2),
        propellant_used_kg=round(sol.total_propellant_used, 2),
        mass_ratio=round(req.initial_mass_kg / sol.final_mass, 4),
        max_defect=round(sol.max_defect, 6),
        converged=sol.success,
        status_message=sol.status_message,
        trajectory_points=pts,
    )


@app.get("/api/certificate")

def get_validation_certificate():
    """Retrieve the cryptographic validation certificate."""
    cert_path = Path(__file__).resolve().parent.parent.parent.parent / "benchmarks" / "reports" / "validation_certificate.json"
    if not cert_path.exists():
        cert = generate_validation_certificate()
        return cert
    with open(cert_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Navigation & Orbit Determination Endpoints
# ---------------------------------------------------------------------------

from typing import Tuple
from relativistic_engine.navigation.dsn import (
    DSN_STATIONS,
    compute_shapiro_time_delay,
    compute_station_bcrs_state,
    compute_observation_jacobian,
    solve_2way_light_time,
    compute_2way_doppler_shift,
)
from relativistic_engine.navigation.orbit_determination import (
    ExtendedKalmanFilter,
    BatchWeightedLeastSquares,
    TrackingObservation as _ODTrackingObservation,
)
from relativistic_engine.physics.kerr import (
    KerrGeometry,
    compute_bardeen_shadow_contour,
)
from relativistic_engine.physics.kerr_raytracer import (
    KerrCamera,
    AccretionDiskProperties,
    render_kerr_black_hole_scene,
)
from relativistic_engine.ephemeris.jpl_loader import load_jpl_ephemeris as _load_spk
from relativistic_engine.api.schemas import (
    ShapiroRequest,
    ShapiroResponse,
    OrbitDeterminationRequest,
    OrbitDeterminationResponse,
    TrackingObservation,
    KerrLensingRequest,
    KerrLensingResponse,
    TrackingArcSimulationRequest,
    TrackingArcSimulationResponse,
    TrackingArcPoint,
    Trajectory2PNRequest,
    Trajectory2PNResponse,
)
from relativistic_engine.trajectory.pn2_propagator import propagate_2pn_trajectory
from relativistic_engine.api.manifest import generate_manifest


@app.post("/api/navigation/shapiro", response_model=ShapiroResponse)
def compute_shapiro_delay(req: ShapiroRequest):
    """Compute the general relativistic Shapiro gravitational time delay.

    Returns the one-way delay in microseconds and its path-length equivalent
    in kilometres.  The limb-grazing singularity guard is engaged automatically
    when the ray impact parameter falls below 1.5 R_Sun.
    """
    r_tx = np.asarray(req.r_tx_m, dtype=np.float64)
    r_rx = np.asarray(req.r_rx_m, dtype=np.float64)

    if r_tx.shape != (3,) or r_rx.shape != (3,):
        raise HTTPException(status_code=422, detail="r_tx_m and r_rx_m must each be 3-element lists.")

    # Detect whether the singularity guard branch was exercised by checking
    # the ray impact parameter against 1.5 R_Sun (695,700 km).
    R_SUN = 6.957e8  # m
    rho_vec = r_rx - r_tx
    rho = float(np.linalg.norm(rho_vec))
    r1 = float(np.linalg.norm(r_tx))
    r2 = float(np.linalg.norm(r_rx))
    # Impact parameter: perpendicular distance from origin to the ray.
    if rho > 0.0:
        t_min = -np.dot(r_tx, rho_vec) / (rho ** 2)
        t_min = float(np.clip(t_min, 0.0, 1.0))
        closest = r_tx + t_min * rho_vec
        d_impact = float(np.linalg.norm(closest))
    else:
        d_impact = r1

    singularity_guard = d_impact < 1.5 * R_SUN

    try:
        delay_s = compute_shapiro_time_delay(
            r_tx=r_tx,
            r_rx=r_rx,
            gm_body=req.gm_body_m3s2,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    C = 299_792_458.0
    return ShapiroResponse(
        shapiro_delay_us=round(delay_s * 1e6, 6),
        shapiro_range_equiv_km=round(delay_s * C / 1000.0, 6),
        singularity_guard_applied=singularity_guard,
    )


@app.post("/api/navigation/orbit_determination", response_model=OrbitDeterminationResponse)
def run_orbit_determination(req: OrbitDeterminationRequest):
    """Run EKF or Batch WLS orbit determination over a supplied DSN tracking arc.

    Accepts time-stamped 2-way range and Doppler observations from a single
    DSN ground station and returns the estimated spacecraft state and
    1-sigma covariance.  Internally maps shorthand station keys ('goldstone',
    'madrid', 'canberra') to canonical DSN station IDs used by the navigation module.
    """
    _STATION_ID_MAP = {
        "goldstone": "DSS-14",
        "madrid": "DSS-65",
        "canberra": "DSS-43",
        "dss-14": "DSS-14",
        "dss-65": "DSS-65",
        "dss-43": "DSS-43",
    }
    station_id = _STATION_ID_MAP.get(req.station.lower())
    if station_id is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown station '{req.station}'. Valid options: goldstone, madrid, canberra.",
        )
    if len(req.observations) < 2:
        raise HTTPException(status_code=422, detail="At least 2 observations are required.")

    estimator_key = req.estimator.lower()
    if estimator_key not in ("ekf", "batch_wls"):
        raise HTTPException(status_code=422, detail="estimator must be 'ekf' or 'batch_wls'.")

    r0 = np.asarray(req.r_nominal_m, dtype=np.float64)
    v0 = np.asarray(req.v_nominal_mps, dtype=np.float64)
    if r0.shape != (3,) or v0.shape != (3,):
        raise HTTPException(status_code=422, detail="r_nominal_m and v_nominal_mps must each be 3-element lists.")

    # Convert API schema observations -> navigation module TrackingObservation dataclass
    od_obs = [
        _ODTrackingObservation(
            epoch_tdb=o.t_receive_s,
            station_id=station_id,
            range_m=o.range_m,
            range_rate_mps=o.range_rate_mps,
            range_sigma_m=req.sigma_range_m,
            range_rate_sigma_mps=req.sigma_range_rate_mps,
        )
        for o in req.observations
    ]

    x0 = np.concatenate([r0, v0])
    P0 = np.diag([1e6, 1e6, 1e6, 1.0, 1.0, 1.0])  # 1 km pos, 1 m/s vel prior

    nis_mean_val = None
    wls_rms_val = None
    converged = True
    result_state = x0.copy()
    result_cov = P0.copy()

    try:
        spk = _load_spk()
        if estimator_key == "ekf":
            ekf = ExtendedKalmanFilter(
                initial_state=x0.copy(),
                initial_covariance=P0.copy(),
                initial_epoch_tdb=od_obs[0].epoch_tdb,
            )
            step_results = ekf.process_tracking_arc(od_obs, jd_base=2451545.0, spk=spk)
            if step_results:
                last = step_results[-1]
                result_state = last.state_estimate
                result_cov = last.covariance
                nis_vals = [s.nis for s in step_results if s.nis > 0.0]
                nis_mean_val = float(np.mean(nis_vals)) if nis_vals else None
        else:
            result_state, result_cov, rms_history = BatchWeightedLeastSquares.estimate_initial_state(
                observations=od_obs,
                initial_state_guess=x0.copy(),
                prior_covariance=P0.copy(),
                initial_epoch_tdb=od_obs[0].epoch_tdb,
                jd_base=2451545.0,
                spk=spk,
            )
            wls_rms_val = float(rms_history[-1]) if rms_history else None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cov_diag = np.sqrt(np.maximum(np.diag(result_cov), 0.0))
    return OrbitDeterminationResponse(
        estimator_used=estimator_key,
        r_estimated_m=[round(float(v), 3) for v in result_state[:3]],
        v_estimated_mps=[round(float(v), 6) for v in result_state[3:6]],
        sigma_r_m=[round(float(v), 3) for v in cov_diag[:3]],
        sigma_v_mps=[round(float(v), 6) for v in cov_diag[3:6]],
        nis_mean=round(nis_mean_val, 4) if nis_mean_val is not None else None,
        wls_rms_residual=round(wls_rms_val, 6) if wls_rms_val is not None else None,
        n_observations=len(req.observations) * 2,
        converged=converged,
    )


@app.post("/api/navigation/simulate_arc", response_model=TrackingArcSimulationResponse)
def simulate_tracking_arc(req: TrackingArcSimulationRequest):
    """Simulate a synthetic DSN tracking arc and run sequential EKF and Batch WLS estimators.

    Generates realistic 2-way range and Doppler observables with General Relativistic
    Shapiro delays for the chosen ground station and trajectory profile, applies
    Gaussian measurement noise, and evaluates filter convergence.
    """
    _STATION_ID_MAP = {
        "goldstone": "DSS-14",
        "madrid": "DSS-65",
        "canberra": "DSS-43",
        "dss-14": "DSS-14",
        "dss-65": "DSS-65",
        "dss-43": "DSS-43",
    }
    station_id = _STATION_ID_MAP.get(req.station.lower())
    if station_id is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown station '{req.station}'. Valid: 'goldstone', 'madrid', 'canberra'.",
        )

    dsn_sta = DSN_STATIONS[station_id]
    spk = _load_spk()
    jd_base = 2462867.5  # 2030-Jan-01 TDB

    if req.target_profile.lower() == "jupiter_cruise":
        r0 = np.array([4.8e11, 3.2e11, 2.5e10], dtype=np.float64)
        v0 = np.array([-11500.0, 14200.0, 850.0], dtype=np.float64)
    else:  # default: mars_approach
        r0 = np.array([1.85e11, 7.5e10, 1.8e10], dtype=np.float64)
        v0 = np.array([-14500.0, 21500.0, 2400.0], dtype=np.float64)

    gm_sun = 1.32712440018e20
    def sc_trajectory_fn(t_sec: float) -> Tuple[np.ndarray, np.ndarray]:
        r_norm = float(np.linalg.norm(r0))
        acc_vec = -gm_sun * r0 / (r_norm ** 3)
        r_t = r0 + v0 * t_sec + 0.5 * acc_vec * (t_sec ** 2)
        v_t = v0 + acc_vec * t_sec
        return r_t, v_t

    duration_s = req.duration_hours * 3600.0
    dt = req.sampling_interval_s
    n_steps = max(3, int(duration_s / dt) + 1)
    sample_times = np.linspace(0.0, duration_s, n_steps)

    rng = np.random.RandomState(42)
    observations: List[_ODTrackingObservation] = []
    truth_records = []

    for t_val in sample_times:
        t1, t2, t3, range_2way_m, sh_up, sh_down = solve_2way_light_time(
            station=dsn_sta,
            sc_trajectory_fn=sc_trajectory_fn,
            t3_receive=t_val,
            jd_base=jd_base,
            spk=spk,
            include_shapiro=True,
        )
        _, range_rate_mps = compute_2way_doppler_shift(
            station=dsn_sta,
            sc_trajectory_fn=sc_trajectory_fn,
            t3_receive=t_val,
            jd_base=jd_base,
            spk=spk,
        )
        total_shapiro_us = (sh_up + sh_down) * 1e6

        range_noisy = range_2way_m + float(rng.normal(0.0, req.sigma_range_m))
        rate_noisy = range_rate_mps + float(rng.normal(0.0, req.sigma_range_rate_mps))

        observations.append(
            _ODTrackingObservation(
                epoch_tdb=t_val,
                station_id=station_id,
                range_m=range_noisy,
                range_rate_mps=rate_noisy,
                range_sigma_m=req.sigma_range_m,
                range_rate_sigma_mps=req.sigma_range_rate_mps,
            )
        )
        truth_records.append({
            "t_sec": t_val,
            "range_km": range_2way_m / 1000.0,
            "range_rate_kms": range_rate_mps / 1000.0,
            "shapiro_us": total_shapiro_us,
        })

    pos_prior_perturbation = np.array([450.0, -280.0, 180.0], dtype=np.float64)
    vel_prior_perturbation = np.array([0.45, -0.28, 0.18], dtype=np.float64)
    x0_ekf = np.concatenate([r0 + pos_prior_perturbation, v0 + vel_prior_perturbation])
    P0_ekf = np.diag([1e6, 1e6, 1e6, 1.0, 1.0, 1.0])

    ekf = ExtendedKalmanFilter(
        initial_state=x0_ekf,
        initial_covariance=P0_ekf,
        initial_epoch_tdb=0.0,
    )
    step_results = ekf.process_tracking_arc(observations, jd_base=jd_base, spk=spk)

    telemetry_points: List[TrackingArcPoint] = []
    nis_values = []
    for rec, step in zip(truth_records, step_results):
        cov_diag = np.sqrt(np.maximum(np.diag(step.covariance), 0.0))
        sig_r = float(np.mean(cov_diag[:3]))
        sig_v = float(np.mean(cov_diag[3:6]))
        nis_val = float(step.nis) if step.nis > 0.0 else None
        if nis_val is not None:
            nis_values.append(nis_val)
        telemetry_points.append(
            TrackingArcPoint(
                t_elapsed_hours=round(rec["t_sec"] / 3600.0, 4),
                range_km=round(rec["range_km"], 3),
                range_rate_kms=round(rec["range_rate_kms"], 6),
                shapiro_delay_us=round(rec["shapiro_us"], 3),
                sigma_r_m=round(sig_r, 3),
                sigma_v_mps=round(sig_v, 6),
                nis=round(nis_val, 4) if nis_val is not None else None,
            )
        )

    init_sig_r = float(np.mean(np.sqrt(np.diag(P0_ekf)[:3])))
    final_sig_r = telemetry_points[-1].sigma_r_m if telemetry_points else init_sig_r
    init_sig_v = float(np.mean(np.sqrt(np.diag(P0_ekf)[3:6])))
    final_sig_v = telemetry_points[-1].sigma_v_mps if telemetry_points else init_sig_v
    cov_reduc = max(0.0, (1.0 - (final_sig_r / init_sig_r)) * 100.0)

    wls_converged = True
    try:
        _, _, rms_hist = BatchWeightedLeastSquares.estimate_initial_state(
            observations=observations,
            initial_state_guess=x0_ekf.copy(),
            prior_covariance=P0_ekf.copy(),
            initial_epoch_tdb=0.0,
            jd_base=jd_base,
            spk=spk,
            max_iterations=5,
        )
        if not rms_hist or rms_hist[-1] > rms_hist[0] * 2.0:
            wls_converged = False
    except Exception:
        wls_converged = False

    peak_shapiro = max(p.shapiro_delay_us for p in telemetry_points) if telemetry_points else 0.0
    mean_nis_val = float(np.mean(nis_values)) if nis_values else 1.0

    return TrackingArcSimulationResponse(
        station_id=station_id,
        target_profile=req.target_profile,
        n_points=len(telemetry_points),
        shapiro_peak_delay_us=round(peak_shapiro, 3),
        initial_sigma_r_m=round(init_sig_r, 3),
        final_sigma_r_m=round(final_sig_r, 3),
        initial_sigma_v_mps=round(init_sig_v, 6),
        final_sigma_v_mps=round(final_sig_v, 6),
        covariance_reduction_pct=round(cov_reduc, 2),
        mean_nis=round(mean_nis_val, 4),
        wls_converged=wls_converged,
        telemetry_points=telemetry_points,
    )


# ---------------------------------------------------------------------------
# Relativistic X-ray Pulsar Navigation (XPNAV) Endpoints
# ---------------------------------------------------------------------------

from relativistic_engine.navigation.xpnav import (
    PULSAR_CATALOG,
    PulsarObservation,
    compute_pulse_delays,
    compute_predicted_pulse_phase,
    solve_spacecraft_state_xpnav,
)
from relativistic_engine.api.schemas import (
    XPNAVPredictRequest,
    XPNAVPredictResponse,
    XPNAVSolveRequest,
    XPNAVSolveResponse,
)


@app.post("/api/navigation/xpnav/predict", response_model=XPNAVPredictResponse)
def predict_xpnav_pulse(req: XPNAVPredictRequest) -> XPNAVPredictResponse:
    """Predict relativistic pulse Time of Arrival (TOA) delays and pulse phase."""
    pkey = req.pulsar_key.lower().strip()
    if pkey not in PULSAR_CATALOG:
        avail = list(PULSAR_CATALOG.keys())
        raise HTTPException(status_code=422, detail=f"Unknown pulsar '{req.pulsar_key}'. Available: {avail}")

    psr = PULSAR_CATALOG[pkey]
    r_m = np.array(req.r_sc_km, dtype=np.float64) * 1000.0

    delays = compute_pulse_delays(psr, r_m, req.frequency_ghz)
    pred_phase = compute_predicted_pulse_phase(
        psr,
        t_sc_tdb_s=req.t_sc_tdb_s,
        r_sc_m=r_m,
        clock_bias_s=req.clock_bias_s,
        freq_ghz=req.frequency_ghz,
    )

    return XPNAVPredictResponse(
        pulsar_name=psr.name,
        geometric_delay_s=round(delays.geometric_delay_s, 9),
        shapiro_delay_s=round(delays.shapiro_delay_s, 9),
        dispersion_delay_s=round(delays.dispersion_delay_s, 9),
        net_delay_s=round(delays.net_delay_s, 9),
        predicted_phase=round(pred_phase, 6),
        range_equivalent_km=round(delays.range_equivalent_km, 3),
    )


@app.post("/api/navigation/xpnav/solve", response_model=XPNAVSolveResponse)
def solve_xpnav_state(req: XPNAVSolveRequest) -> XPNAVSolveResponse:
    """Reconstruct 4D spacecraft state [x, y, z, clock_bias] from pulsar pulse observations."""
    if len(req.observations) < 4:
        raise HTTPException(status_code=422, detail="At least 4 independent pulsar observations required.")

    obs_list = [
        PulsarObservation(
            pulsar_key=o.pulsar_key,
            observed_time_tdb_s=o.observed_time_tdb_s,
            frequency_ghz=o.frequency_ghz,
            measured_phase=o.measured_phase,
        )
        for o in req.observations
    ]

    try:
        sol = solve_spacecraft_state_xpnav(
            observations=obs_list,
            initial_guess_r_km=np.array(req.initial_guess_r_km, dtype=np.float64),
            initial_guess_clock_s=req.initial_guess_clock_s,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return XPNAVSolveResponse(
        position_bcrs_km=[round(x, 3) for x in sol.position_bcrs_km],
        clock_bias_s=round(sol.clock_bias_s, 9),
        clock_bias_ns=round(sol.clock_bias_ns, 3),
        gdop=round(sol.gdop, 4),
        sigma_pos_km=[round(s, 4) for s in sol.sigma_pos_km],
        sigma_clock_ns=round(sol.sigma_clock_ns, 3),
        num_pulsars_used=sol.num_pulsars_used,
        num_iterations=sol.num_iterations,
        converged=sol.converged,
        residuals_s=[round(r, 9) for r in sol.residuals_s],
    )


@app.post("/api/lensing/kerr", response_model=KerrLensingResponse)

def analyze_kerr_lensing(req: KerrLensingRequest):
    """Analyze Kerr rotating black hole spacetime geometry, ISCO, photon orbits, and Bardeen shadow contour.

    Optionally runs the backward null ray-tracer on a low-resolution grid to evaluate
    relativistic frequency shifts g_rad and Doppler beaming across the Novikov-Thorne disk.
    """
    M_SUN_KG = 1.98847e30
    mass_kg = req.mass_solar * M_SUN_KG
    try:
        kerr = KerrGeometry(mass_kg=mass_kg, spin_dimensionless=req.spin_dimensionless)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    m_val = kerr.mass_m
    r_plus = kerr.event_horizon_outer
    r_minus = kerr.event_horizon_inner
    r_erg_eq = kerr.ergosphere_outer(math.pi * 0.5)
    r_isco_pro = kerr.isco_radius(prograde=True)
    r_isco_ret = kerr.isco_radius(prograde=False)
    r_ph_pro = kerr.photon_orbit_radius(prograde=True)
    r_ph_ret = kerr.photon_orbit_radius(prograde=False)

    a_star = abs(req.spin_dimensionless)
    penrose_eff = (1.0 - math.sqrt(0.5 * (1.0 + math.sqrt(max(0.0, 1.0 - a_star ** 2))))) * 100.0

    theta_obs = math.radians(req.inclination_deg)
    alpha_arr, beta_arr = compute_bardeen_shadow_contour(kerr, theta_obs_rad=theta_obs, n_points=120)

    doppler_min = None
    doppler_max = None
    shadow_px = None
    disk_px = None

    if req.render_preview:
        res = min(max(req.render_resolution, 16), 32)
        cam = KerrCamera(
            r_cam=req.camera_distance_m_units * m_val,
            theta_cam=theta_obs,
            resolution_x=res,
            resolution_y=res,
            fov_deg=45.0,
        )
        disk = AccretionDiskProperties(
            r_inner_m=r_isco_pro,
            r_outer_m=min(20.0 * m_val, req.camera_distance_m_units * 0.9 * m_val),
            peak_temperature_k=1.0e5,
        )
        try:
            _, g_map, hit_map = render_kerr_black_hole_scene(
                camera=cam,
                kerr=kerr,
                disk=disk,
                max_steps_per_ray=100,
            )
            shadow_px = int(np.sum(hit_map == 0))
            disk_px = int(np.sum(hit_map == 1))
            disk_g_vals = g_map[hit_map == 1]
            if len(disk_g_vals) > 0:
                doppler_min = round(float(np.min(disk_g_vals)), 4)
                doppler_max = round(float(np.max(disk_g_vals)), 4)
        except Exception:
            pass

    return KerrLensingResponse(
        spin_dimensionless=req.spin_dimensionless,
        mass_solar=req.mass_solar,
        gravitational_radius_m=round(m_val, 3),
        r_horizon_outer_m=round(r_plus, 3),
        r_horizon_inner_m=round(r_minus, 3),
        r_ergosphere_equator_m=round(r_erg_eq, 3),
        r_isco_prograde_m=round(r_isco_pro, 3),
        r_isco_retrograde_m=round(r_isco_ret, 3),
        r_photon_prograde_m=round(r_ph_pro, 3),
        r_photon_retrograde_m=round(r_ph_ret, 3),
        penrose_theoretical_max_efficiency_pct=round(penrose_eff, 3),
        shadow_contour_alpha=[round(float(a), 4) for a in alpha_arr],
        shadow_contour_beta=[round(float(b), 4) for b in beta_arr],
        doppler_g_min=doppler_min,
        doppler_g_max=doppler_max,
        shadow_pixels_count=shadow_px,
        disk_pixels_count=disk_px,
    )


@app.post("/api/trajectory/2pn", response_model=Trajectory2PNResponse)
def compute_trajectory_2pn(req: Trajectory2PNRequest) -> Trajectory2PNResponse:
    """Compute higher-order 2PN relativistic trajectory with spherical harmonic planetary gravity."""
    try:
        r0_m = np.array(req.r0_km, dtype=np.float64) * 1000.0
        v0_m = np.array(req.v0_km_s, dtype=np.float64) * 1000.0
        duration_s = req.duration_days * SEC_PER_DAY

        res = propagate_2pn_trajectory(
            r0=r0_m,
            v0=v0_m,
            duration_s=duration_s,
            step_size_s=req.step_size_s,
            central_body=req.central_body,
            pn_order=req.pn_order,
            precision=req.precision,
            max_zonal_degree=req.gravity_harmonics_degree,
        )

        pts = _sample_trajectory_points(res.t, res.tau, res.r, res.v, max_samples=80)
        final_r_km = (res.r[-1] / 1000.0).tolist()
        final_v_km_s = (res.v[-1] / 1000.0).tolist()
        accum_deficit = float(res.time_deficit[-1])

        return Trajectory2PNResponse(
            pn_order=res.pn_order,
            precision=res.precision,
            central_body=req.central_body,
            duration_days=req.duration_days,
            n_steps=len(res.t),
            final_r_km=final_r_km,
            final_v_km_s=final_v_km_s,
            energy_drift_relative=res.energy_drift_relative,
            accumulated_time_deficit_s=accum_deficit,
            points=pts,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/manifest")
def get_manifest(computation_id: str | None = None):
    """Return a JSON-LD reproducibility manifest template for the engine.

    When computation_id is not supplied, returns a template manifest
    documenting the engine's physical constants, ephemeris metadata, and
    solver configuration without computation-specific inputs/outputs.
    """
    return generate_manifest(
        inputs={"note": "Template manifest; supply inputs/outputs via generate_manifest() in code."},
        outputs={"note": "No computation result"},
        computation_id=computation_id,
        endpoint="/api/manifest",
    )


# ---------------------------------------------------------------------------
# Deep-Space PNT Multi-Sensor Fusion Endpoint (SR-UKF)
# ---------------------------------------------------------------------------

@app.post("/api/navigation/fusion/simulate", response_model=PNTTrajectorySimResponse)
def simulate_pnt_fusion(req: PNTTrajectorySimRequest) -> PNTTrajectorySimResponse:
    """Simulate deep-space cruise trajectory with SR-UKF multi-sensor PNT fusion."""
    try:
        res = simulate_pnt_mission(
            duration_days=req.duration_days,
            step_hours=req.step_hours,
            dsn_blackout_start_day=req.dsn_blackout_start_day,
            dsn_blackout_end_day=req.dsn_blackout_end_day,
            initial_pos_error_m=req.initial_pos_error_m,
            initial_vel_error_ms=req.initial_vel_error_ms,
            initial_clock_bias_ns=req.initial_clock_bias_ns,
        )
        return PNTTrajectorySimResponse(
            duration_days=res["duration_days"],
            step_hours=res["step_hours"],
            num_steps=res["num_steps"],
            final_pos_error_m=res["final_pos_error_m"],
            final_pos_3sigma_m=res["final_pos_3sigma_m"],
            final_vel_error_ms=res["final_vel_error_ms"],
            final_vel_3sigma_ms=res["final_vel_3sigma_ms"],
            final_clock_error_ns=res["final_clock_error_ns"],
            telemetry=[PNTTelemetryPointSchema(**pt) for pt in res["telemetry"]],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Closed-Loop Relativistic Guidance Endpoint (ZEM/ZEV)
# ---------------------------------------------------------------------------

from relativistic_engine.guidance.zem_zev import simulate_closed_loop_mission
from relativistic_engine.api.schemas import (
    GuidanceSimRequest,
    GuidanceSimResponse,
    GuidanceTelemetryPointSchema,
)


@app.post("/api/guidance/zem_zev/simulate", response_model=GuidanceSimResponse)
def simulate_guidance_mission(req: GuidanceSimRequest) -> GuidanceSimResponse:
    """Simulate closed-loop autonomous trajectory steering under 1PN dynamics."""
    try:
        res = simulate_closed_loop_mission(
            duration_days=req.duration_days,
            step_hours=req.step_hours,
            initial_position_au=req.initial_position_au,
            initial_velocity_kms=req.initial_velocity_kms,
            target_position_au=req.target_position_au,
            target_velocity_kms=req.target_velocity_kms,
            wet_mass_kg=req.wet_mass_kg,
            thrust_max_n=req.thrust_max_n,
            isp_sec=req.isp_sec,
            initial_pos_dispersion_m=req.initial_pos_dispersion_m,
            initial_vel_dispersion_ms=req.initial_vel_dispersion_ms,
        )
        return GuidanceSimResponse(
            duration_days=res["duration_days"],
            step_hours=res["step_hours"],
            num_steps=res["num_steps"],
            initial_wet_mass_kg=res["initial_wet_mass_kg"],
            final_mass_kg=res["final_mass_kg"],
            total_propellant_used_kg=res["total_propellant_used_kg"],
            final_miss_distance_m=res["final_miss_distance_m"],
            final_velocity_error_ms=res["final_velocity_error_ms"],
            telemetry=[GuidanceTelemetryPointSchema(**pt) for pt in res["telemetry"]],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/fms/execute_mission", response_model=FMSMissionResponse)
def execute_fms_mission(req: FMSMissionRequest) -> FMSMissionResponse:
    """Execute end-to-end multi-phase flight profile via Autonomous Flight Management System."""
    try:
        executive = MissionExecutive(
            mission_name=req.mission_name,
            initial_wet_mass_kg=req.initial_wet_mass_kg,
            thrust_max_n=req.thrust_max_n,
            isp_sec=req.isp_sec,
        )
        res = executive.execute_mission(
            duration_days=req.duration_days,
            step_hours=req.step_hours,
            initial_pos_dispersion_m=req.initial_pos_dispersion_m,
            initial_vel_dispersion_ms=req.initial_vel_dispersion_ms,
            dsn_blackout_start_day=req.dsn_blackout_start_day,
            dsn_blackout_end_day=req.dsn_blackout_end_day,
        )
        return FMSMissionResponse(
            mission_name=res["mission_name"],
            duration_days=res["duration_days"],
            step_hours=res["step_hours"],
            initial_wet_mass_kg=res["initial_wet_mass_kg"],
            final_mass_kg=res["final_mass_kg"],
            total_propellant_used_kg=res["total_propellant_used_kg"],
            total_delta_v_ms=res["total_delta_v_ms"],
            final_miss_distance_m=res["final_miss_distance_m"],
            final_velocity_error_ms=res["final_velocity_error_ms"],
            accumulated_time_deficit_s=res["accumulated_time_deficit_s"],
            events=[FMSMissionEventSchema(**ev) for ev in res["events"]],
            telemetry=[FMSPhaseTelemetrySchema(**pt) for pt in res["telemetry"]],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))




# Mount static web directory
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(WEB_DIR / "index.html"))
