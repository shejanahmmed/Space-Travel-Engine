"""Solar Gravitational Lens (SGL) Relativistic Optics & Trajectory Engine.

Models the relativistic and wave-optical imaging characteristics of the Sun's
gravitational monopole field acting as an astronomical lens, and computes
mission trajectory profiles to the solar focal line (z >= 547.8 AU).

Physical & Astrodynamical References:
- Turyshev, S. G., & Toth, V. T. (2017), "Diffraction of light by the gravitational
  field of the Sun and the solar gravitational lens", Phys. Rev. D, 96(2), 024008.
- Turyshev, S. G., et al. (2020), "Direct Multipixel Imaging and Spectroscopy of an
  Exoplanet with a Solar Gravitational Lens Mission", Final Report for NASA NIAC Phase III.
- Einstein, A. (1936), "Lens-like Action of a Star by the Deviation of Light in the
  Gravitational Field", Science, 84(2188), 506-507.
- IAU 2015 Resolution B3 (Nominal Solar Parameters).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional
import numpy as np
from scipy.special import j0

from relativistic_engine.constants import (
    AU,
    C_LIGHT,
    GM_SUN,
    PARSEC,
    RADIUS_SUN,
    SEC_PER_DAY,
    SEC_PER_JULIAN_YEAR,
)
from relativistic_engine.ephemeris.interstellar import (
    INTERSTELLAR_CATALOG,
    get_star_barycentric_state,
)

# Schwarzschild gravitational radius of the Sun: r_g = 2 * GM_sun / c^2 (meters)
R_G_SUN: float = (2.0 * GM_SUN) / (C_LIGHT ** 2)

# Theoretical minimum focal distance for solar grazing rays (meters & AU)
# z_min = R_sun^2 / (2 * r_g) = c^2 * R_sun^2 / (4 * GM_sun)
Z_MIN_SGL_METERS: float = (RADIUS_SUN ** 2) / (2.0 * R_G_SUN)
Z_MIN_SGL_AU: float = Z_MIN_SGL_METERS / AU


@dataclass(frozen=True)
class SGLFocalParameters:
    """Wave-optical and geometric lensing parameters at a specific SGL focal distance."""

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


@dataclass(frozen=True)
class SGLTrajectoryWaypoint:
    """State along the outbound hyperbolic SGL trajectory."""

    epoch_jd: float
    coordinate_time_years: float
    proper_time_years: float
    heliocentric_distance_au: float
    velocity_km_s: float
    velocity_au_per_year: float
    time_dilation_deficit_seconds: float


@dataclass(frozen=True)
class SGLMissionProfile:
    """Complete SGL mission trajectory and optical imaging profile."""

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
    focal_parameters: SGLFocalParameters
    waypoints: List[SGLTrajectoryWaypoint]


def compute_sgl_focal_parameters(
    wavelength_m: float = 1.0e-6,
    heliocentric_distance_au: float = 550.0,
    target_distance_pc: float = 1.30,
) -> SGLFocalParameters:
    """Computes wave-optical and geometric parameters for the Solar Gravitational Lens.

    Parameters
    ----------
    wavelength_m : float
        Observing wavelength in meters (default 1.0 um, optical / near-IR).
    heliocentric_distance_au : float
        Heliocentric distance of the focal spacecraft in Astronomical Units (>= 547.8 AU).
    target_distance_pc : float
        Distance to exoplanet host star in parsecs (default 1.30 pc for Proxima Centauri).

    Returns
    -------
    SGLFocalParameters
        Calculated geometric, wave-optical amplification, and spatial resolution metrics.
    """
    if wavelength_m <= 0.0:
        raise ValueError("Wavelength must be strictly positive.")
    if heliocentric_distance_au < Z_MIN_SGL_AU:
        raise ValueError(
            f"Distance {heliocentric_distance_au:.2f} AU is below the minimum SGL focal "
            f"distance of {Z_MIN_SGL_AU:.2f} AU (solar disk occults the focal line)."
        )

    z_m = heliocentric_distance_au * AU
    target_dist_m = target_distance_pc * PARSEC

    # Ray impact parameter at the Sun: b(z) = sqrt(2 * r_g * z)
    impact_param_m = math.sqrt(2.0 * R_G_SUN * z_m)
    impact_param_rsun = impact_param_m / RADIUS_SUN

    # Einstein ring angular radius at observer distance z: theta_E = sqrt(2 * r_g / z)
    theta_e_rad = math.sqrt(2.0 * R_G_SUN / z_m)
    theta_e_arcsec = math.degrees(theta_e_rad) * 3600.0

    # On-axis peak wave-optical light amplification (Turyshev & Toth 2017 Eq. 47):
    # mu_0 = 4 * pi^2 * r_g / lambda = 8 * pi^2 * GM_sun / (c^2 * lambda)
    mu_0 = (4.0 * (math.pi ** 2) * R_G_SUN) / wavelength_m

    # Diffraction-limited angular resolution: theta_res = 1.22 * lambda / (2 * b)
    theta_res_rad = (1.22 * wavelength_m) / (2.0 * impact_param_m)

    # Linear spatial resolution on exoplanet disk (km)
    res_km = (theta_res_rad * target_dist_m) / 1000.0

    return SGLFocalParameters(
        wavelength_m=wavelength_m,
        heliocentric_distance_au=heliocentric_distance_au,
        heliocentric_distance_m=z_m,
        min_focal_distance_au=Z_MIN_SGL_AU,
        impact_parameter_m=impact_param_m,
        impact_parameter_solar_radii=impact_param_rsun,
        einstein_ring_angular_radius_rad=theta_e_rad,
        einstein_ring_angular_radius_arcsec=theta_e_arcsec,
        peak_light_amplification=mu_0,
        angular_resolution_rad=theta_res_rad,
        resolvable_surface_resolution_km=res_km,
    )


def sgl_point_spread_function(
    rho_m: np.ndarray,
    wavelength_m: float = 1.0e-6,
    heliocentric_distance_au: float = 550.0,
) -> np.ndarray:
    """Computes the on-axis diffraction Point Spread Function (PSF) intensity profile.

    I(rho) / I_0 = mu_0 * [J_0(k * rho * theta_E)]^2
    where k = 2 * pi / lambda and theta_E = sqrt(2 * r_g / z).

    Parameters
    ----------
    rho_m : np.ndarray
        Radial distance from the optical axis in the focal plane (meters).
    wavelength_m : float
        Observing wavelength (meters).
    heliocentric_distance_au : float
        Spacecraft distance along the focal line (AU).

    Returns
    -------
    np.ndarray
        Point spread function intensity amplification profile I(rho)/I_0.
    """
    z_m = heliocentric_distance_au * AU
    k = (2.0 * math.pi) / wavelength_m
    theta_e = math.sqrt(2.0 * R_G_SUN / z_m)
    mu_0 = (4.0 * (math.pi ** 2) * R_G_SUN) / wavelength_m

    arg = k * rho_m * theta_e
    bessel_val = j0(arg)
    return mu_0 * (bessel_val ** 2)


def solve_sgl_mission_trajectory(
    target_star: str = "proxima_centauri",
    departure_epoch_jd: float = 2462622.5,
    periapsis_solar_radii: float = 4.0,
    periapsis_delta_v_km_s: float = 25.0,
    wavelength_m: float = 1.0e-6,
    num_waypoints: int = 50,
) -> SGLMissionProfile:
    """Solves the relativistic trajectory and optical profile for an SGL focal mission.

    The spacecraft executes a close solar gravity-assist flyby with an Oberth maneuver
    burn at periapsis to inject onto a high-speed hyperbolic escape trajectory directed
    along the anti-stellar focal axis (RA + 180 deg, -Dec).

    Parameters
    ----------
    target_star : str
        Target star system hosting the exoplanet (e.g. 'proxima_centauri', 'alpha_centauri_a').
    departure_epoch_jd : float
        Julian Date of solar periapsis burn epoch.
    periapsis_solar_radii : float
        Heliocentric radius of periapsis burn in units of solar radii (R_sun).
    periapsis_delta_v_km_s : float
        Impulsive velocity increment applied at periapsis (km/s).
    wavelength_m : float
        Science imaging wavelength (meters).
    num_waypoints : int
        Number of output trajectory steps between 1 AU and 650 AU.

    Returns
    -------
    SGLMissionProfile
        Complete trajectory solution with coordinate time, proper time, and optical metrics.
    """
    if periapsis_solar_radii < 1.05:
        raise ValueError(
            f"Periapsis {periapsis_solar_radii} R_sun is inside or too close to solar photosphere."
        )
    if periapsis_delta_v_km_s <= 0.0:
        raise ValueError("Periapsis Delta-V must be positive.")

    # Target stellar coordinates from authoritative Gaia DR3 catalog
    clean_name = target_star.lower().strip().replace(" ", "_").replace("'", "")
    if clean_name not in INTERSTELLAR_CATALOG:
        available = list(INTERSTELLAR_CATALOG.keys())
        raise KeyError(f"Unknown target star '{target_star}'. Available: {available}")

    entry = INTERSTELLAR_CATALOG[clean_name]
    target_ra = entry.ra_deg
    target_dec = entry.dec_deg
    target_dist_pc = entry.distance_meters / PARSEC

    # Focal line points directly opposite the target star in the celestial sphere
    focal_ra = (target_ra + 180.0) % 360.0
    focal_dec = -target_dec

    # Periapsis radius in meters
    r_p = periapsis_solar_radii * RADIUS_SUN
    delta_v_p = periapsis_delta_v_km_s * 1000.0

    # Local parabolic escape velocity at periapsis
    v_esc_p = math.sqrt((2.0 * GM_SUN) / r_p)
    v_p = v_esc_p + delta_v_p

    # Asymptotic hyperbolic excess velocity v_infinity (m/s)
    # v_inf^2 = v_p^2 - v_esc_p^2 = 2 * v_esc_p * delta_v_p + delta_v_p^2
    v_inf_sq = (v_p ** 2) - ((2.0 * GM_SUN) / r_p)
    v_inf = math.sqrt(v_inf_sq)

    v_inf_km_s = v_inf / 1000.0
    v_inf_au_yr = (v_inf * SEC_PER_JULIAN_YEAR) / AU

    # Numerical radial integration from periapsis (r_p) to target focal depth (650 AU)
    r_max_m = 650.0 * AU
    radii_m = np.linspace(r_p, r_max_m, num_waypoints)

    waypoints: List[SGLTrajectoryWaypoint] = []
    t_coord_sec = 0.0
    t_proper_sec = 0.0

    r_prev = radii_m[0]
    for i, r in enumerate(radii_m):
        # Local radial speed: v(r) = sqrt(v_inf^2 + 2 * GM_sun / r)
        v_local = math.sqrt(v_inf_sq + (2.0 * GM_SUN) / r)
        v_local_km_s = v_local / 1000.0
        v_local_au_yr = (v_local * SEC_PER_JULIAN_YEAR) / AU

        if i > 0:
            dr = r - r_prev
            r_mid = 0.5 * (r + r_prev)
            v_mid = math.sqrt(v_inf_sq + (2.0 * GM_SUN) / r_mid)

            dt = dr / v_mid
            t_coord_sec += dt

            # 1PN metric proper time differential:
            # dtau = sqrt(1 - (v^2 / c^2) - (2 * GM_sun / (c^2 * r))) * dt
            # Exact rationalized coordinate time deficit rate:
            x_metric = ((v_mid ** 2) / (C_LIGHT ** 2)) + ((2.0 * GM_SUN) / ((C_LIGHT ** 2) * r_mid))
            x_metric = min(x_metric, 0.999999999)
            dtau_rate = math.sqrt(1.0 - x_metric)
            dtau = dtau_rate * dt
            t_proper_sec += dtau

        r_prev = r

        t_coord_yr = t_coord_sec / SEC_PER_JULIAN_YEAR
        t_proper_yr = t_proper_sec / SEC_PER_JULIAN_YEAR
        deficit_sec = t_coord_sec - t_proper_sec

        current_jd = departure_epoch_jd + (t_coord_sec / SEC_PER_DAY)
        r_au = r / AU

        waypoints.append(
            SGLTrajectoryWaypoint(
                epoch_jd=current_jd,
                coordinate_time_years=t_coord_yr,
                proper_time_years=t_proper_yr,
                heliocentric_distance_au=r_au,
                velocity_km_s=v_local_km_s,
                velocity_au_per_year=v_local_au_yr,
                time_dilation_deficit_seconds=deficit_sec,
            )
        )

    # Time to reach 550 AU
    r_550_m = 550.0 * AU
    t_to_550_sec = 0.0
    tau_to_550_sec = 0.0
    num_fine = 1000
    r_fine = np.linspace(r_p, r_550_m, num_fine)
    for j in range(1, num_fine):
        dr = r_fine[j] - r_fine[j - 1]
        r_mid = 0.5 * (r_fine[j] + r_fine[j - 1])
        v_mid = math.sqrt(v_inf_sq + (2.0 * GM_SUN) / r_mid)
        dt = dr / v_mid
        t_to_550_sec += dt

        x_m = ((v_mid ** 2) / (C_LIGHT ** 2)) + ((2.0 * GM_SUN) / ((C_LIGHT ** 2) * r_mid))
        dtau = math.sqrt(1.0 - min(x_m, 0.999999999)) * dt
        tau_to_550_sec += dtau

    t_550_yr = t_to_550_sec / SEC_PER_JULIAN_YEAR
    tau_550_yr = tau_to_550_sec / SEC_PER_JULIAN_YEAR
    deficit_550_sec = t_to_550_sec - tau_to_550_sec

    focal_params = compute_sgl_focal_parameters(
        wavelength_m=wavelength_m,
        heliocentric_distance_au=550.0,
        target_distance_pc=target_dist_pc,
    )

    return SGLMissionProfile(
        target_star=entry.name,
        target_ra_deg=target_ra,
        target_dec_deg=target_dec,
        focal_line_ra_deg=focal_ra,
        focal_line_dec_deg=focal_dec,
        solar_flyby_periapsis_solar_radii=periapsis_solar_radii,
        periapsis_delta_v_km_s=periapsis_delta_v_km_s,
        asymptotic_speed_km_s=v_inf_km_s,
        asymptotic_speed_au_per_year=v_inf_au_yr,
        time_to_550au_years=t_550_yr,
        proper_time_to_550au_years=tau_550_yr,
        time_dilation_deficit_at_550au_seconds=deficit_550_sec,
        focal_parameters=focal_params,
        waypoints=waypoints,
    )
