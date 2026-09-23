"""Pydantic data schemas for the Relativistic Engine REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class InterplanetaryRequest(BaseModel):
    """Request payload for interplanetary rendezvous trajectory."""

    departure_body: str = Field(default="earth", description="Departure body (e.g. 'earth')")
    target_body: str = Field(default="mars", description="Target body (e.g. 'mars', 'jupiter')")
    departure_epoch_jd_tdb: float = Field(
        default=2462622.5,  # 2030-May-01
        description="Departure epoch in Julian Date (TDB scale)",
    )
    accel_proper_g: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Proper acceleration in standard gravities (g0)",
    )
    mode: str = Field(
        default="soft",
        description="Rendezvous mode: 'soft' (match position and velocity) or 'intercept' (flyby)",
    )


class InterstellarRequest(BaseModel):
    """Request payload for interstellar brachistochrone trajectory."""

    target_star: str = Field(
        default="proxima_centauri",
        description="Target star identifier (e.g. 'proxima_centauri', 'alpha_centauri_a', 'barnards_star')",
    )
    departure_epoch_jd_tdb: float = Field(
        default=2451545.0,  # J2000.0
        description="Departure epoch in Julian Date (TDB scale)",
    )
    accel_proper_g: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Proper acceleration in standard gravities (g0)",
    )


class TrajectoryPointSchema(BaseModel):
    """Downsampled trajectory state for visualization."""

    t_sec: float
    tau_sec: float
    r_km: List[float]
    v_km_s: List[float]
    speed_c: float
    lorentz_gamma: float


class InterplanetaryResponse(BaseModel):
    """Response payload for interplanetary rendezvous."""

    departure_body: str
    target_body: str
    departure_epoch_jd_tdb: float
    arrival_epoch_jd_tdb: float
    coordinate_flight_time_days: float
    proper_flight_time_days: float
    time_deficit_seconds: float
    max_speed_km_s: float
    max_lorentz_gamma: float
    miss_distance_km: float
    relative_arrival_velocity_m_s: float
    formatted_coordinate_time: str
    formatted_proper_time: str
    formatted_time_deficit: str
    trajectory_points: List[TrajectoryPointSchema]


class InterstellarResponse(BaseModel):
    """Response payload for interstellar brachistochrone."""

    target_star: str
    departure_epoch_jd_tdb: float
    arrival_epoch_jd_tdb: float
    coordinate_flight_time_years: float
    proper_flight_time_years: float
    time_deficit_years: float
    distance_light_years: float
    max_speed_c: float
    max_lorentz_gamma: float
    doppler_redshift_earth: float
    doppler_blueshift_target: float
    miss_distance_au: float
    relative_arrival_velocity_km_s: float
    formatted_coordinate_time: str
    formatted_proper_time: str
    formatted_time_deficit: str
    trajectory_points: List[TrajectoryPointSchema]


class StarSummary(BaseModel):
    """Summary of star astrometric data for catalog."""

    id: str
    name: str
    distance_ly: float
    ra_deg: float
    dec_deg: float
    radial_velocity_km_s: float


class CatalogResponse(BaseModel):
    """Available planetary and interstellar targets."""

    planetary_bodies: List[str]
    interstellar_stars: List[StarSummary]


class PorkchopRequest(BaseModel):
    """Request payload for 2D Porkchop launch window evaluation."""

    origin_body: str = Field(default="earth", description="Departure body (e.g. 'earth')")
    target_body: str = Field(default="mars", description="Target body (e.g. 'mars', 'jupiter')")
    dep_start_jd: float = Field(default=2461300.5, description="Start departure JD (TDB)")
    dep_end_jd: float = Field(default=2461400.5, description="End departure JD (TDB)")
    arr_start_jd: float = Field(default=2461500.5, description="Start arrival JD (TDB)")
    arr_end_jd: float = Field(default=2461700.5, description="End arrival JD (TDB)")
    grid_steps: int = Field(default=25, ge=5, le=100, description="Number of grid samples per axis")
    mode: str = Field(default="ballistic", description="Mode: 'ballistic' or 'brachistochrone'")
    alpha_thrust: Optional[float] = Field(default=None, description="Proper acceleration (m/s^2)")


class PorkchopResponse(BaseModel):
    """Response payload containing 2D Porkchop contour matrices."""

    origin_body: str
    target_body: str
    dep_jds: List[float]
    arr_jds: List[float]
    c3_km2_s2: List[List[Optional[float]]]
    delta_v_total_km_s: List[List[Optional[float]]]
    tof_days: List[List[float]]
    proper_time_days: List[List[Optional[float]]]
    time_deficit_sec: List[List[Optional[float]]]
    best_window: Dict[str, Any]


class TourLegConfig(BaseModel):
    """Configuration for a single leg of a planetary tour."""

    origin_body: str
    target_body: str
    departure_jd: float
    arrival_jd: float
    is_flyby: bool = False
    periapsis_altitude_km: float = 500.0
    b_plane_angle: float = 0.0


class TourRequest(BaseModel):
    """Request payload for multi-leg planetary tour."""

    mission_name: str = Field(default="Planetary Tour", description="Descriptive mission name")
    legs: List[TourLegConfig]


class TourLegResponse(BaseModel):
    """Summary of a single completed tour leg."""

    origin_body: str
    target_body: str
    departure_jd: float
    arrival_jd: float
    tof_days: float
    c3_dep_km2_s2: float
    delta_v_dep_km_s: float
    delta_v_arr_km_s: float
    proper_time_days: float
    time_deficit_sec: float
    flyby_turning_angle_deg: Optional[float] = None
    flyby_1pn_correction_arcsec: Optional[float] = None
    r_dep_bcrs_km: Optional[List[float]] = None
    r_arr_bcrs_km: Optional[List[float]] = None


class TourResponse(BaseModel):
    """Response payload for multi-leg planetary tour."""

    mission_name: str
    total_delta_v_km_s: float
    total_coordinate_time_days: float
    total_proper_time_days: float
    total_time_deficit_sec: float
    legs: List[TourLegResponse]


class BatchMonteCarloRequest(BaseModel):
    """Request payload for high-throughput batch Monte Carlo uncertainty evaluation."""

    origin_body: str = Field(default="earth", description="Origin body for nominal state")
    departure_epoch_jd: float = Field(default=2461300.5, description="Departure epoch in Julian Date (TDB)")
    flight_time_days: float = Field(default=30.0, ge=1.0, le=500.0, description="Flight duration in days")
    sigma_pos_km: float = Field(default=100.0, ge=0.01, description="1-sigma initial position uncertainty (km)")
    sigma_vel_mps: float = Field(default=1.0, ge=0.001, description="1-sigma initial velocity uncertainty (m/s)")
    n_samples: int = Field(default=1000, ge=10, le=100000, description="Ensemble sample size N")
    backend: str = Field(default="auto", description="Execution backend: 'auto', 'numpy_cpu', 'torch_cuda'")


class BatchMonteCarloResponse(BaseModel):
    """Response payload for batch Monte Carlo dispersion."""

    n_samples: int
    flight_time_days: float
    elapsed_wall_time_sec: float
    throughput_trajectories_per_sec: float
    backend_used: str
    position_dispersion_3sigma_km: float
    velocity_dispersion_3sigma_mps: float
    mean_time_deficit_sec: float


class LowThrustRequest(BaseModel):
    """Request payload for continuous low-thrust trajectory optimization."""

    departure_body: str = Field(default="earth", description="Departure celestial body")
    target_body: str = Field(default="mars", description="Target celestial body")
    departure_epoch_jd: float = Field(default=2462622.5, description="Departure epoch in Julian Date (TDB)")
    tof_days: float = Field(default=180.0, ge=10.0, le=2000.0, description="Flight duration in days")
    initial_mass_kg: float = Field(default=1500.0, ge=10.0, description="Initial wet mass in kg")
    dry_mass_kg: float = Field(default=500.0, ge=1.0, description="Spacecraft dry mass in kg")
    thrust_max_n: float = Field(default=2.5, ge=0.001, le=1000.0, description="Maximum thrust in Newtons")
    isp_sec: float = Field(default=4500.0, ge=100.0, description="Specific impulse in seconds")
    num_segments: int = Field(default=20, ge=5, le=100, description="Number of collocation segments K")


class LowThrustResponse(BaseModel):
    """Response payload for continuous low-thrust trajectory optimization."""

    departure_body: str
    target_body: str
    departure_epoch_jd: float
    arrival_epoch_jd: float
    coordinate_flight_time_days: float
    proper_flight_time_days: float
    time_deficit_seconds: float
    initial_mass_kg: float
    final_mass_kg: float
    propellant_used_kg: float
    mass_ratio: float
    max_defect: float
    converged: bool
    status_message: str
    trajectory_points: List[TrajectoryPointSchema]


# ---------------------------------------------------------------------------
# Navigation: Shapiro gravitational time delay
# ---------------------------------------------------------------------------

class ShapiroRequest(BaseModel):
    """Request payload for general relativistic Shapiro gravitational time delay.

    All position vectors must be supplied in the Barycentric Celestial Reference
    System (BCRS / ICRF J2000), in SI metres, at the epoch of signal transit.
    """

    r_tx_m: List[float] = Field(
        description="Transmitter BCRS position [m, ICRF J2000].  3-element list [x, y, z].",
    )
    r_rx_m: List[float] = Field(
        description="Receiver BCRS position [m, ICRF J2000].  3-element list [x, y, z].",
    )
    gm_body_m3s2: float = Field(
        default=1.32712440018e20,
        description=(
            "Gravitational parameter GM of the deflecting body [m³ s⁻²]. "
            "Default is GM_Sun (IAU 2015 Resolution B3)."
        ),
    )


class ShapiroResponse(BaseModel):
    """Response payload for Shapiro gravitational time delay."""

    shapiro_delay_us: float = Field(
        description="One-way Shapiro gravitational time delay [microseconds]."
    )
    shapiro_range_equiv_km: float = Field(
        description="Path-length equivalent of the Shapiro delay: c·Δt [km]."
    )
    singularity_guard_applied: bool = Field(
        description=(
            "True when the ray impact parameter fell below 1.5 R_Sun and the "
            "limb-grazing regularisation branch was used."
        )
    )


# ---------------------------------------------------------------------------
# Navigation: Orbit determination (EKF / Batch WLS)
# ---------------------------------------------------------------------------

class TrackingObservation(BaseModel):
    """Single DSN 2-way range + Doppler observation."""

    t_receive_s: float = Field(
        description="Signal receive epoch [s, seconds past J2000 TDB]."
    )
    range_m: float = Field(
        description="2-way range observable ρ = c(t3−t1)/2 [m]."
    )
    range_rate_mps: float = Field(
        description="2-way coherent Doppler range-rate ρ̇ [m s⁻¹]."
    )


class OrbitDeterminationRequest(BaseModel):
    """Request payload for statistical orbit determination from a DSN tracking arc."""

    station: str = Field(
        default="goldstone",
        description="DSN ground station identifier: 'goldstone', 'madrid', or 'canberra'.",
    )
    r_nominal_m: List[float] = Field(
        description=(
            "Nominal spacecraft BCRS position at arc start [m, ICRF J2000]. "
            "3-element list [x, y, z]."
        ),
    )
    v_nominal_mps: List[float] = Field(
        description=(
            "Nominal spacecraft BCRS velocity at arc start [m s⁻¹, ICRF J2000]. "
            "3-element list [vx, vy, vz]."
        ),
    )
    observations: List[TrackingObservation] = Field(
        description="Ordered list of DSN range + Doppler observations across the tracking arc.",
    )
    estimator: str = Field(
        default="ekf",
        description="Statistical estimator to use: 'ekf' (Extended Kalman Filter) or 'batch_wls' (Batch Weighted Least Squares).",
    )
    sigma_range_m: float = Field(
        default=2.0,
        ge=0.001,
        description="1-sigma range measurement noise [m].",
    )
    sigma_range_rate_mps: float = Field(
        default=5e-4,
        ge=1e-6,
        description="1-sigma range-rate (Doppler) measurement noise [m s⁻¹].",
    )


class OrbitDeterminationResponse(BaseModel):
    """Response payload for orbit determination estimator."""

    estimator_used: str = Field(description="Estimator that produced the solution: 'ekf' or 'batch_wls'.")
    r_estimated_m: List[float] = Field(description="Estimated spacecraft BCRS position [m, ICRF J2000].")
    v_estimated_mps: List[float] = Field(description="Estimated spacecraft BCRS velocity [m s⁻¹, ICRF J2000].")
    sigma_r_m: List[float] = Field(description="1-sigma position uncertainty from covariance diagonal [m].")
    sigma_v_mps: List[float] = Field(description="1-sigma velocity uncertainty from covariance diagonal [m s⁻¹].")
    nis_mean: Optional[float] = Field(
        default=None,
        description="Mean Normalised Innovation Squared (EKF only); should be ≈ 1.0 for a consistent filter.",
    )
    wls_rms_residual: Optional[float] = Field(
        default=None,
        description="RMS weighted residual (Batch WLS only); dimensionless.",
    )
    n_observations: int = Field(description="Total number of scalar observations processed.")
    converged: bool = Field(description="True if the estimator converged without numerical issues.")


# ---------------------------------------------------------------------------
# Kerr Rotating Black Hole Spacetime & Gravitational Lensing
# ---------------------------------------------------------------------------

class KerrLensingRequest(BaseModel):
    """Request payload for Kerr rotating black hole spacetime geometry and lensing."""

    spin_dimensionless: float = Field(
        default=0.9,
        ge=-0.998,
        le=0.998,
        description="Dimensionless spin parameter a_* = a/M in [-0.998, 0.998]. Positive=prograde, negative=retrograde.",
    )
    mass_solar: float = Field(
        default=10.0,
        gt=0.0,
        description="Black hole mass in units of solar masses (M_Sun = 1.98847e30 kg).",
    )
    inclination_deg: float = Field(
        default=85.0,
        ge=1.0,
        le=89.0,
        description="Observer inclination angle theta_obs relative to spin axis in degrees.",
    )
    camera_distance_m_units: float = Field(
        default=25.0,
        ge=5.0,
        description="Camera distance from singularity in units of gravitational radii M.",
    )
    render_preview: bool = Field(
        default=False,
        description="When True, evaluates full backward null ray-tracer on a low-res preview grid.",
    )
    render_resolution: int = Field(
        default=24,
        ge=16,
        le=48,
        description="Resolution for synthetic scene ray-tracing preview (width & height in pixels).",
    )


class KerrLensingResponse(BaseModel):
    """Response payload for Kerr spacetime geometry and gravitational lensing."""

    spin_dimensionless: float = Field(description="Dimensionless spin a_*")
    mass_solar: float = Field(description="Mass in solar masses")
    gravitational_radius_m: float = Field(description="Gravitational radius M = G M_BH / c^2 [m].")
    r_horizon_outer_m: float = Field(description="Outer event horizon radius r_+ [m].")
    r_horizon_inner_m: float = Field(description="Inner Cauchy horizon radius r_- [m].")
    r_ergosphere_equator_m: float = Field(description="Outer ergosphere boundary at equator theta=pi/2 [m].")
    r_isco_prograde_m: float = Field(description="Prograde ISCO radius [m].")
    r_isco_retrograde_m: float = Field(description="Retrograde ISCO radius [m].")
    r_photon_prograde_m: float = Field(description="Prograde circular photon orbit radius [m].")
    r_photon_retrograde_m: float = Field(description="Retrograde circular photon orbit radius [m].")
    penrose_theoretical_max_efficiency_pct: float = Field(description="Theoretical maximum Penrose process extraction efficiency [%].")
    shadow_contour_alpha: List[float] = Field(description="Bardeen shadow horizontal impact parameters alpha in units of M.")
    shadow_contour_beta: List[float] = Field(description="Bardeen shadow vertical impact parameters beta in units of M.")
    doppler_g_min: Optional[float] = Field(default=None, description="Minimum observed frequency shift g_rad on accretion disk.")
    doppler_g_max: Optional[float] = Field(default=None, description="Maximum observed frequency shift g_rad on accretion disk.")
    shadow_pixels_count: Optional[int] = Field(default=None, description="Pixels captured by event horizon in synthetic preview.")
    disk_pixels_count: Optional[int] = Field(default=None, description="Pixels intersecting accretion disk in synthetic preview.")


# ---------------------------------------------------------------------------
# Synthetic DSN Tracking Arc Simulation
# ---------------------------------------------------------------------------

class TrackingArcPoint(BaseModel):
    """Single time-series telemetry point along a tracking arc."""

    t_elapsed_hours: float = Field(description="Elapsed time since arc start [hours].")
    range_km: float = Field(description="2-way range observable [km].")
    range_rate_kms: float = Field(description="2-way coherent Doppler range-rate [km s⁻¹].")
    shapiro_delay_us: float = Field(description="Gravitational Shapiro time delay [microseconds].")
    sigma_r_m: float = Field(description="1-sigma formal position uncertainty [m].")
    sigma_v_mps: float = Field(description="1-sigma formal velocity uncertainty [m s⁻¹].")
    nis: Optional[float] = Field(default=None, description="Normalised Innovation Squared (NIS).")


class TrackingArcSimulationRequest(BaseModel):
    """Request payload to simulate a synthetic DSN tracking arc and run OD estimators."""

    station: str = Field(
        default="goldstone",
        description="DSN ground station identifier: 'goldstone' (DSS-14), 'madrid' (DSS-65), or 'canberra' (DSS-43).",
    )
    duration_hours: float = Field(
        default=4.0,
        ge=0.5,
        le=24.0,
        description="Tracking pass duration [hours].",
    )
    sampling_interval_s: float = Field(
        default=600.0,
        ge=60.0,
        le=3600.0,
        description="Observation cadence [seconds between range/Doppler points].",
    )
    sigma_range_m: float = Field(
        default=2.0,
        ge=0.01,
        description="1-sigma white Gaussian noise on 2-way range observables [m].",
    )
    sigma_range_rate_mps: float = Field(
        default=5e-4,
        ge=1e-6,
        description="1-sigma white Gaussian noise on 2-way Doppler observables [m s⁻¹].",
    )
    target_profile: str = Field(
        default="mars_approach",
        description="Nominal spacecraft trajectory profile: 'mars_approach' or 'jupiter_cruise'.",
    )


class TrackingArcSimulationResponse(BaseModel):
    """Response payload containing synthetic tracking arc observables, EKF covariance trace, and WLS summary."""

    station_id: str = Field(description="DSN ground station canonical ID (e.g. DSS-14).")
    target_profile: str = Field(description="Selected trajectory profile.")
    n_points: int = Field(description="Number of telemetry points in tracking arc.")
    shapiro_peak_delay_us: float = Field(description="Peak Shapiro gravitational delay across arc [microseconds].")
    initial_sigma_r_m: float = Field(description="Initial position uncertainty [m].")
    final_sigma_r_m: float = Field(description="Final estimated position uncertainty [m].")
    initial_sigma_v_mps: float = Field(description="Initial velocity uncertainty [m s⁻¹].")
    final_sigma_v_mps: float = Field(description="Final estimated velocity uncertainty [m s⁻¹].")
    covariance_reduction_pct: float = Field(description="Total percentage reduction in covariance trace.")
    mean_nis: float = Field(description="Mean Normalised Innovation Squared.")
    wls_converged: bool = Field(description="True if Batch WLS converged.")
    telemetry_points: List[TrackingArcPoint] = Field(description="Time series of tracking arc points.")


class Trajectory2PNRequest(BaseModel):
    """Request payload for higher-order 2PN relativistic trajectory integration."""

    pn_order: str = Field(
        default="2pn",
        description="Post-Newtonian order: 'newtonian', '1pn', '2pn', or '2.5pn'.",
    )
    precision: str = Field(
        default="float64",
        description="Arithmetic precision mode: 'float64' (IEEE-754) or 'quad' (mpmath binary128 symplectic).",
    )
    r0_km: List[float] = Field(
        default=[149597870.7, 0.0, 0.0],
        description="Initial position vector [x, y, z] in km.",
    )
    v0_km_s: List[float] = Field(
        default=[0.0, 29.78, 0.0],
        description="Initial velocity vector [vx, vy, vz] in km/s.",
    )
    duration_days: float = Field(
        default=5.0,
        ge=0.01,
        le=365.0,
        description="Integration duration in Julian days.",
    )
    step_size_s: float = Field(
        default=3600.0,
        ge=1.0,
        description="Step size in seconds for numerical integration.",
    )
    central_body: str = Field(
        default="Sun",
        description="Central gravitational body: 'Sun', 'Earth', 'Mars', 'Jupiter', or 'Saturn'.",
    )
    gravity_harmonics_degree: int = Field(
        default=2,
        ge=0,
        le=8,
        description="Maximum degree of spherical harmonic zonal gravity field (0 to 8).",
    )


class Trajectory2PNResponse(BaseModel):
    """Response payload for 2PN relativistic trajectory propagation."""

    pn_order: str
    precision: str
    central_body: str
    duration_days: float
    n_steps: int
    final_r_km: List[float]
    final_v_km_s: List[float]
    energy_drift_relative: float
    accumulated_time_deficit_s: float
    points: List[TrajectoryPointSchema]


# ---------------------------------------------------------------------------
# Solar Gravitational Lens (SGL) Relativistic Focus Schemas
# ---------------------------------------------------------------------------

class SGLWaypointSchema(BaseModel):
    """Waypoint along the outbound SGL hyperbolic escape trajectory."""

    epoch_jd: float = Field(description="Epoch in Julian Date (TDB)")
    coordinate_time_years: float = Field(description="Elapsed coordinate time in Julian years")
    proper_time_years: float = Field(description="Elapsed proper time in Julian years")
    heliocentric_distance_au: float = Field(description="Heliocentric distance in Astronomical Units")
    velocity_km_s: float = Field(description="Radial velocity in km/s")
    velocity_au_per_year: float = Field(description="Radial velocity in AU/year")
    time_dilation_deficit_seconds: float = Field(description="Relativistic coordinate time deficit in seconds")


class SGLFocalSchema(BaseModel):
    """Wave-optical and geometric lensing parameters at focal distance."""

    wavelength_m: float = Field(description="Science observation wavelength in meters")
    heliocentric_distance_au: float = Field(description="Focal distance in AU")
    min_focal_distance_au: float = Field(description="Theoretical minimum focal distance in AU (~547.8 AU)")
    impact_parameter_m: float = Field(description="Ray impact parameter at the Sun in meters")
    impact_parameter_solar_radii: float = Field(description="Impact parameter in units of solar radii")
    einstein_ring_angular_radius_arcsec: float = Field(description="Einstein ring angular radius in arcseconds")
    peak_light_amplification: float = Field(description="On-axis wave-optical peak light amplification mu_0")
    angular_resolution_rad: float = Field(description="Diffraction-limited angular resolution in radians")
    resolvable_surface_resolution_km: float = Field(description="Linear spatial resolution on exoplanet surface in km")


class SGLRequest(BaseModel):
    """Request payload for Solar Gravitational Lens (SGL) mission trajectory calculation."""

    target_star: str = Field(
        default="proxima_centauri",
        description="Target star system hosting candidate exoplanet (e.g., 'proxima_centauri', 'alpha_centauri_a').",
    )
    departure_epoch_jd: float = Field(
        default=2462622.5,
        description="Julian Date of solar periapsis burn epoch.",
    )
    periapsis_solar_radii: float = Field(
        default=4.0,
        ge=1.05,
        le=50.0,
        description="Periapsis distance in units of solar radii (R_sun).",
    )
    periapsis_delta_v_km_s: float = Field(
        default=25.0,
        ge=0.1,
        le=200.0,
        description="Impulsive delta-V increment applied at solar periapsis (km/s).",
    )
    wavelength_m: float = Field(
        default=1.0e-6,
        ge=1.0e-9,
        le=1.0e-2,
        description="Imaging wavelength in meters (default 1.0 um, near-IR).",
    )


class SGLResponse(BaseModel):
    """Response payload for Solar Gravitational Lens (SGL) mission trajectory calculation."""

    target_star: str
    target_ra_deg: float
    target_dec_deg: float
    focal_line_ra_deg: float
    focal_line_dec_deg: float
    solar_flyby_periapsis_solar_radii: float
    periapsis_delta_v_km_s: float
    asymptotic_speed_km_s: float
    asymptotic_speed_au_per_year: float
    time_to_550au_years: float
    proper_time_to_550au_years: float
    time_dilation_deficit_at_550au_seconds: float
    focal_parameters: SGLFocalSchema
    waypoints: List[SGLWaypointSchema]


# ---------------------------------------------------------------------------
# X-ray Pulsar Navigation (XPNAV / SEXTANT) Schemas
# ---------------------------------------------------------------------------

class XPNAVPredictRequest(BaseModel):
    """Request payload for predicting pulse Time-of-Arrival and phase."""

    pulsar_key: str = Field(
        default="b1937+21",
        description="Pulsar catalog key: 'b1937+21', 'b1821-24', 'j0437-4715', 'j0218+4232', 'b0531+21'.",
    )
    t_sc_tdb_s: float = Field(
        default=0.0,
        description="Spacecraft receiver time in TDB seconds since J2000.",
    )
    r_sc_km: List[float] = Field(
        default=[149597870.7, 0.0, 0.0],
        description="Spacecraft 3D BCRS position vector [x, y, z] in km.",
    )
    clock_bias_s: float = Field(
        default=0.0,
        description="Spacecraft on-board clock bias in seconds.",
    )
    frequency_ghz: float = Field(
        default=1.0,
        ge=0.01,
        description="Observation radio/X-ray center frequency in GHz.",
    )


class XPNAVPredictResponse(BaseModel):
    """Response payload with decomposed pulse delays and predicted phase."""

    pulsar_name: str
    geometric_delay_s: float
    shapiro_delay_s: float
    dispersion_delay_s: float
    net_delay_s: float
    predicted_phase: float
    range_equivalent_km: float


class PulsarObservationSchema(BaseModel):
    """Single pulsar pulse observation for position determination."""

    pulsar_key: str
    observed_time_tdb_s: float
    frequency_ghz: float = 1.0
    measured_phase: float


class XPNAVSolveRequest(BaseModel):
    """Request payload for 4D spacecraft state reconstruction from pulsar observations."""

    observations: List[PulsarObservationSchema] = Field(
        description="List of at least 4 independent pulsar observations.",
    )
    initial_guess_r_km: List[float] = Field(
        default=[149597870.7, 0.0, 0.0],
        description="Initial estimated position [x, y, z] in km.",
    )
    initial_guess_clock_s: float = Field(
        default=0.0,
        description="Initial estimated clock bias in seconds.",
    )


class XPNAVSolveResponse(BaseModel):
    """Response payload with estimated 3D position, clock bias, and GDOP."""

    position_bcrs_km: List[float]
    clock_bias_s: float
    clock_bias_ns: float
    gdop: float
    sigma_pos_km: List[float]
    sigma_clock_ns: float
    num_pulsars_used: int
    num_iterations: int
    converged: bool
    residuals_s: List[float]


# ---------------------------------------------------------------------------
# Navigation: Relativistic Multi-Sensor PNT Fusion (SR-UKF)
# ---------------------------------------------------------------------------

class PNTTelemetryPointSchema(BaseModel):
    """Single telemetry snapshot from multi-sensor PNT simulation."""

    day: float
    pos_error_m: float
    pos_3sigma_m: float
    vel_error_ms: float
    vel_3sigma_ms: float
    clock_error_ns: float
    clock_3sigma_ns: float
    dsn_active: bool
    in_blackout: bool


class PNTTrajectorySimRequest(BaseModel):
    """Request payload for multi-sensor PNT cruise simulation."""

    duration_days: float = Field(
        default=30.0,
        description="Total duration of the cruise simulation in days.",
    )
    step_hours: float = Field(
        default=6.0,
        description="Filter update step size in hours.",
    )
    dsn_blackout_start_day: Optional[float] = Field(
        default=10.0,
        description="Optional start day of DSN communication blackout.",
    )
    dsn_blackout_end_day: Optional[float] = Field(
        default=20.0,
        description="Optional end day of DSN communication blackout.",
    )
    initial_pos_error_m: float = Field(
        default=5000.0,
        description="A priori initial position error injection [m].",
    )
    initial_vel_error_ms: float = Field(
        default=0.5,
        description="A priori initial velocity error injection [m/s].",
    )
    initial_clock_bias_ns: float = Field(
        default=50.0,
        description="Initial onboard clock bias offset [ns].",
    )


class PNTTrajectorySimResponse(BaseModel):
    """Response payload for multi-sensor PNT cruise simulation."""

    duration_days: float
    step_hours: float
    num_steps: int
    final_pos_error_m: float
    final_pos_3sigma_m: float
    final_vel_error_ms: float
    final_vel_3sigma_ms: float
    final_clock_error_ns: float
    telemetry: List[PNTTelemetryPointSchema]


# ---------------------------------------------------------------------------
# Closed-Loop Autonomous Relativistic Guidance (ZEM/ZEV)
# ---------------------------------------------------------------------------

class GuidanceTelemetryPointSchema(BaseModel):
    """Single time-step telemetry point in closed-loop guidance simulation."""

    day: float
    time_to_go_days: float
    zem_m: float
    zev_ms: float
    thrust_n: float
    thrust_accel_ms2: float
    mass_kg: float
    schiff_precession_arcsec_yr: float
    r_au: List[float]
    v_kms: List[float]


class GuidanceSimRequest(BaseModel):
    """Request payload for closed-loop ZEM/ZEV trajectory simulation."""

    duration_days: float = Field(
        default=30.0,
        description="Total duration of guidance simulation in days.",
    )
    step_hours: float = Field(
        default=6.0,
        description="Guidance update step interval in hours.",
    )
    initial_position_au: List[float] = Field(
        default=[1.0, 0.0, 0.0],
        description="Initial spacecraft 3D barycentric position in AU.",
    )
    initial_velocity_kms: List[float] = Field(
        default=[0.0, 29.78, 0.0],
        description="Initial spacecraft 3D velocity in km/s.",
    )
    target_position_au: List[float] = Field(
        default=[1.1, 0.2, 0.0],
        description="Target intercept coordinates in AU.",
    )
    target_velocity_kms: List[float] = Field(
        default=[-5.0, 27.5, 0.0],
        description="Target intercept velocity in km/s.",
    )
    wet_mass_kg: float = Field(
        default=1500.0,
        description="Initial spacecraft wet mass in kg.",
    )
    thrust_max_n: float = Field(
        default=5.0,
        description="Maximum engine thrust capacity in Newtons.",
    )
    isp_sec: float = Field(
        default=4500.0,
        description="Specific impulse in seconds.",
    )
    initial_pos_dispersion_m: float = Field(
        default=5000.0,
        description="Initial injection position dispersion error in meters.",
    )
    initial_vel_dispersion_ms: float = Field(
        default=0.5,
        description="Initial injection velocity dispersion error in m/s.",
    )


class GuidanceSimResponse(BaseModel):
    """Response payload for closed-loop ZEM/ZEV trajectory simulation."""

    duration_days: float
    step_hours: float
    num_steps: int
    initial_wet_mass_kg: float
    final_mass_kg: float
    total_propellant_used_kg: float
    final_miss_distance_m: float
    final_velocity_error_ms: float
    telemetry: List[GuidanceTelemetryPointSchema]


class SGLFocalSchema(BaseModel):
    """Wave-optical and geometric lensing parameters schema."""

    wavelength_m: float
    heliocentric_distance_au: float
    heliocentric_distance_m: float
    min_focal_distance_au: float
    impact_parameter_m: float
    impact_parameter_solar_radii: float
    einstein_ring_angular_radius_rad: float
    einstein_ring_angular_radius_arcsec: float
    peak_light_amplification: float
    angular_resolution_rad: float
    resolvable_surface_resolution_km: float


class SGLTrajectoryWaypointSchema(BaseModel):
    """State along the outbound hyperbolic SGL trajectory."""

    epoch_jd: float
    coordinate_time_years: float
    proper_time_years: float
    heliocentric_distance_au: float
    velocity_km_s: float
    velocity_au_per_year: float
    time_dilation_deficit_seconds: float


class SGLMissionRequest(BaseModel):
    """Request payload for SGL focal mission solver."""

    target_star: str = Field(default="proxima_centauri", description="Target star system hosting exoplanet")
    departure_epoch_jd: float = Field(default=2462622.5, description="Departure epoch in Julian Date")
    periapsis_solar_radii: float = Field(default=4.0, ge=1.1, description="Solar flyby periapsis in R_sun")
    periapsis_delta_v_km_s: float = Field(default=25.0, ge=0.1, description="Oberth maneuver Delta-V in km/s")
    wavelength_m: float = Field(default=1.0e-6, ge=1e-10, description="Observing wavelength in meters")
    num_waypoints: int = Field(default=50, ge=10, le=200, description="Number of trajectory output waypoints")


class SGLMissionResponse(BaseModel):
    """Response payload for SGL focal mission trajectory and optical profile."""

    target_star: str
    target_ra_deg: float
    target_dec_deg: float
    focal_line_ra_deg: float
    focal_line_dec_deg: float
    solar_flyby_periapsis_solar_radii: float
    periapsis_delta_v_km_s: float
    asymptotic_speed_km_s: float
    asymptotic_speed_au_per_year: float
    time_to_550au_years: float
    proper_time_to_550au_years: float
    time_dilation_deficit_at_550au_seconds: float
    focal_parameters: SGLFocalSchema
    waypoints: List[SGLTrajectoryWaypointSchema]


# Backward compatible aliases
SGLRequest = SGLMissionRequest
SGLResponse = SGLMissionResponse
SGLWaypointSchema = SGLTrajectoryWaypointSchema


class FMSMissionEventSchema(BaseModel):
    """Schema for individual mission event in Sequence of Events."""

    epoch_seconds: float
    epoch_days: float
    phase: str
    event_type: str
    description: str
    delta_v_ms: float
    mass_depleted_kg: float
    details: Dict[str, Any] = Field(default_factory=dict)


class FMSPhaseTelemetrySchema(BaseModel):
    """Time-series telemetry schema for FMS execution."""

    day: float
    phase: str
    in_blackout: bool
    zem_m: float
    zev_ms: float
    thrust_n: float
    thrust_accel_ms2: float
    mass_kg: float
    proper_time_deficit_s: float
    schiff_precession_arcsec_yr: float
    pos_est_error_m: float
    r_au: List[float]
    v_kms: List[float]


class FMSMissionRequest(BaseModel):
    """Request payload for FMS autonomous mission execution."""

    mission_name: str = Field(
        default="Earth-Mars Autonomous Relativistic Transit",
        description="Designation for flight profile.",
    )
    duration_days: float = Field(
        default=180.0,
        ge=5.0,
        le=1000.0,
        description="Total mission duration in days.",
    )
    step_hours: float = Field(
        default=12.0,
        ge=1.0,
        le=48.0,
        description="Simulation integration time step in hours.",
    )
    initial_wet_mass_kg: float = Field(
        default=1500.0,
        ge=50.0,
        description="Initial spacecraft wet mass in kg.",
    )
    thrust_max_n: float = Field(
        default=10.0,
        ge=0.1,
        description="Maximum engine thrust capacity in Newtons.",
    )
    isp_sec: float = Field(
        default=4500.0,
        ge=500.0,
        description="Engine specific impulse in seconds.",
    )
    initial_pos_dispersion_m: float = Field(
        default=5000.0,
        ge=0.0,
        description="Initial injection position dispersion in meters.",
    )
    initial_vel_dispersion_ms: float = Field(
        default=0.5,
        ge=0.0,
        description="Initial injection velocity dispersion in m/s.",
    )
    dsn_blackout_start_day: float = Field(
        default=40.0,
        description="Start day of DSN loss-of-signal blackout.",
    )
    dsn_blackout_end_day: float = Field(
        default=80.0,
        description="End day of DSN loss-of-signal blackout.",
    )


class FMSMissionResponse(BaseModel):
    """Response payload for FMS autonomous mission execution."""

    mission_name: str
    duration_days: float
    step_hours: float
    initial_wet_mass_kg: float
    final_mass_kg: float
    total_propellant_used_kg: float
    total_delta_v_ms: float
    final_miss_distance_m: float
    final_velocity_error_ms: float
    accumulated_time_deficit_s: float
    events: List[FMSMissionEventSchema]
    telemetry: List[FMSPhaseTelemetrySchema]





