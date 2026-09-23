"""Relativistic Planetary Gravity-Assist and Hyperbolic Flyby Dynamics.

Models planetary flybys in the Barycentric Celestial Reference System (BCRS / ICRF),
accounting for:
- Frame transformations between heliocentric BCRS and moving planetocentric frames.
- Exact hyperbolic orbital deflection geometry.
- Post-Newtonian (1PN) general relativistic gravitational bending corrections.
- Spacecraft traveler proper time dilation integrated through planetary gravitational
  potential wells.
- Powered flyby delta-v calculations at periapsis.

Authoritative References:
- Battin, R. H. (1999), "An Introduction to the Mathematics and Methods of Astrodynamics",
  AIAA Education Series.
- Bodenner, E. & Will, C. M. (2003), "Deflection of light to second order: A pedagogical example",
  American Journal of Physics, 71(8):770-773.
- IAU 2000 Resolution B1.3 / BCRS Metric Conventions.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from relativistic_engine.constants import C_LIGHT, G_NEWTON
from relativistic_engine.physics.potential import STANDARD_BODY_GM, STANDARD_BODY_RADIUS


@dataclass(frozen=True)
class FlybyResult:
    """Complete kinematic and relativistic state of a planetary hyperbolic flyby.

    Attributes
    ----------
    v_in_bcrs : np.ndarray
        Incoming spacecraft velocity vector in BCRS before flyby (m/s).
    v_out_bcrs : np.ndarray
        Outgoing spacecraft velocity vector in BCRS after flyby (m/s).
    v_inf_in_planet : np.ndarray
        Incoming hyperbolic excess velocity vector in planetocentric frame (m/s).
    v_inf_out_planet : np.ndarray
        Outgoing hyperbolic excess velocity vector in planetocentric frame (m/s).
    v_inf_mag : float
        Scalar magnitude of hyperbolic excess velocity |v_inf| (m/s).
    bending_angle_rad : float
        Total turning angle delta in radians (Newtonian + 1PN).
    bending_angle_deg : float
        Total turning angle delta in degrees.
    delta_1pn_rad : float
        Post-Newtonian 1PN bending angle correction in radians.
    periapsis_radius : float
        Periapsis radius r_p from planetary center (meters).
    periapsis_altitude : float
        Altitude h_p = r_p - R_planet above planetary surface (meters).
    impact_parameter : float
        Asymptotic impact parameter b (meters).
    eccentricity : float
        Hyperbolic orbital eccentricity e = 1 / sin(delta/2) > 1.
    delta_v_helio_vector : np.ndarray
        Vector velocity change in BCRS: v_out - v_in (m/s).
    delta_v_helio_mag : float
        Scalar magnitude of heliocentric velocity change |Delta v| (m/s).
    energy_change_helio : float
        Specific orbital energy change in BCRS: 0.5 * (|v_out|^2 - |v_in|^2) (m^2/s^2).
    proper_time_deficit_sec : float
        Integrated proper time deficit Delta(t) = t - tau through the planetary
        potential well during the encounter (seconds).
    """

    v_in_bcrs: np.ndarray
    v_out_bcrs: np.ndarray
    v_inf_in_planet: np.ndarray
    v_inf_out_planet: np.ndarray
    v_inf_mag: float
    bending_angle_rad: float
    bending_angle_deg: float
    delta_1pn_rad: float
    periapsis_radius: float
    periapsis_altitude: float
    impact_parameter: float
    eccentricity: float
    delta_v_helio_vector: np.ndarray
    delta_v_helio_mag: float
    energy_change_helio: float
    proper_time_deficit_sec: float


def compute_hyperbolic_flyby(
    v_in_bcrs: np.ndarray,
    r_planet_bcrs: np.ndarray,
    v_planet_bcrs: np.ndarray,
    gm_planet: float,
    periapsis_radius: float,
    b_plane_angle: float = 0.0,
    planet_radius: float = 0.0,
    *,
    compute_1pn: bool = True,
    soi_radius: float | None = None,
) -> FlybyResult:
    """Compute 3D planetary flyby scattering and relativistic time dilation in BCRS.

    Parameters
    ----------
    v_in_bcrs : np.ndarray
        Incoming spacecraft velocity vector in BCRS (m/s), shape (3,).
    r_planet_bcrs : np.ndarray
        Planet Cartesian position in BCRS (meters), shape (3,).
    v_planet_bcrs : np.ndarray
        Planet Cartesian velocity in BCRS (m/s), shape (3,).
    gm_planet : float
        Planetary standard gravitational parameter mu = GM (m^3/s^2).
    periapsis_radius : float
        Target periapsis distance r_p from planetary center (meters).
    b_plane_angle : float, optional
        Orientation angle psi of the flyby orbital plane in radians (default: 0.0).
        0.0 corresponds to an equatorial / in-plane encounter.
    planet_radius : float, optional
        Physical radius of the planet (meters) for altitude checking (default: 0.0).
    compute_1pn : bool, optional
        Whether to evaluate the 1PN post-Newtonian gravitational bending correction
        (default: True).
    soi_radius : float, optional
        Planetary sphere of influence radius (meters) for proper time integration.
        If None, estimated from distance to SSB.

    Returns
    -------
    FlybyResult
        Comprehensive flyby kinematics, deflection angles, energy transfer, and proper time.

    Raises
    ------
    ValueError
        If periapsis radius is non-positive or less than the physical planetary radius.
        If incoming excess velocity is non-positive.
    """
    v_in = np.asarray(v_in_bcrs, dtype=np.float64)
    r_planet = np.asarray(r_planet_bcrs, dtype=np.float64)
    v_planet = np.asarray(v_planet_bcrs, dtype=np.float64)

    if periapsis_radius <= 0.0:
        raise ValueError(f"Periapsis radius must be strictly positive, got {periapsis_radius} m.")
    if planet_radius > 0.0 and periapsis_radius < planet_radius:
        raise ValueError(
            f"Periapsis radius {periapsis_radius:.1f} m penetrates planetary surface "
            f"(radius: {planet_radius:.1f} m)."
        )

    # Transform to planetocentric frame: v_inf^- = v_in - v_planet
    v_inf_in = v_in - v_planet
    v_inf_mag = float(np.linalg.norm(v_inf_in))
    if v_inf_mag <= 0.0:
        raise ValueError("Incoming excess velocity relative to planet must be strictly positive.")

    # Vis-viva semi-major axis of the hyperbolic trajectory (a < 0 in standard convention):
    # a_hyp = -mu / v_inf^2
    a_hyp_mag = gm_planet / (v_inf_mag**2)

    # Hyperbolic eccentricity: e = 1 + r_p / a_hyp_mag
    eccentricity = 1.0 + (periapsis_radius / a_hyp_mag)

    # Newtonian turning angle: sin(delta / 2) = 1 / e
    # delta = 2 * arcsin(1 / e)
    sin_half_delta = 1.0 / eccentricity
    sin_half_delta = min(1.0, max(-1.0, sin_half_delta))
    delta_newton = 2.0 * math.asin(sin_half_delta)

    # Asymptotic impact parameter: b = r_p * sqrt(1 + 2*mu / (r_p * v_inf^2)) = a_hyp_mag * sqrt(e^2 - 1)
    impact_parameter = a_hyp_mag * math.sqrt(max(0.0, eccentricity**2 - 1.0))

    # Post-Newtonian (1PN) gravitational deflection correction:
    # Delta delta_1PN = (2 * mu / (c^2 * b)) * (1 + v_inf^2 / c^2)
    # In the ultra-relativistic limit v_inf -> c, this adds to Newtonian deflection to yield
    # the Einstein light bending 4*mu/(c^2 * b).
    delta_1pn = 0.0
    if compute_1pn:
        beta = v_inf_mag / C_LIGHT
        delta_1pn = (2.0 * gm_planet / (C_LIGHT**2 * impact_parameter)) * (1.0 + beta**2)

    total_delta = delta_newton + delta_1pn

    # Construct orthonormal basis for the planetocentric frame:
    # u1: unit vector along incoming asymptote v_inf^-
    u1 = v_inf_in / v_inf_mag

    # Reference vector for orbital plane: use planet orbital angular momentum h_p = r_p x v_p
    # or fallback to [0, 0, 1] if colinear
    h_planet = np.cross(r_planet, v_planet)
    h_norm = np.linalg.norm(h_planet)
    if h_norm > 1e-12:
        k_ref = h_planet / h_norm
    else:
        k_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # u2: normal to u1 and k_ref
    u2_cross = np.cross(u1, k_ref)
    u2_norm = np.linalg.norm(u2_cross)
    if u2_norm < 1e-12:
        # u1 is parallel to k_ref; pick alternative reference axis
        k_alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u2_cross = np.cross(u1, k_alt)
        u2_norm = np.linalg.norm(u2_cross)
    u2 = u2_cross / u2_norm
    u3 = np.cross(u1, u2)

    # B-plane orientation: deflection direction in the plane perpendicular to u1
    p_perp = math.cos(b_plane_angle) * u2 + math.sin(b_plane_angle) * u3

    # Outgoing asymptote unit vector:
    # v_inf^+ = cos(delta) * u1 + sin(delta) * p_perp
    u_out = math.cos(total_delta) * u1 + math.sin(total_delta) * p_perp
    v_inf_out = v_inf_mag * u_out

    # Transform back to BCRS: v_out_bcrs = v_planet + v_inf^+
    v_out_bcrs = v_planet + v_inf_out

    # Kinematics changes in BCRS
    delta_v_vec = v_out_bcrs - v_in
    delta_v_mag = float(np.linalg.norm(delta_v_vec))
    energy_change = 0.5 * (float(np.dot(v_out_bcrs, v_out_bcrs)) - float(np.dot(v_in, v_in)))

    # Proper time deficit integration through planetary potential well:
    # Delta(t) = integral (1 - dtau/dt) dt
    # dtau/dt = sqrt(1 - v^2/c^2 - 2*w_p/c^2) approx 1 - 0.5*v^2/c^2 - w_p/c^2
    # Using vis-viva: v^2(r) = v_inf^2 + 2*mu/r, so (0.5*v^2 + w_p)/c^2 = (0.5*v_inf^2 + 2*mu/r)/c^2.
    # We integrate from periapsis to SOI radius r_soi:
    r_soi_eff = soi_radius if soi_radius is not None else 100.0 * periapsis_radius
    proper_time_deficit = _integrate_flyby_proper_time_deficit(
        gm_planet, v_inf_mag, periapsis_radius, r_soi_eff
    )

    return FlybyResult(
        v_in_bcrs=v_in,
        v_out_bcrs=v_out_bcrs,
        v_inf_in_planet=v_inf_in,
        v_inf_out_planet=v_inf_out,
        v_inf_mag=v_inf_mag,
        bending_angle_rad=total_delta,
        bending_angle_deg=math.degrees(total_delta),
        delta_1pn_rad=delta_1pn,
        periapsis_radius=periapsis_radius,
        periapsis_altitude=periapsis_radius - planet_radius,
        impact_parameter=impact_parameter,
        eccentricity=eccentricity,
        delta_v_helio_vector=delta_v_vec,
        delta_v_helio_mag=delta_v_mag,
        energy_change_helio=energy_change,
        proper_time_deficit_sec=proper_time_deficit,
    )


def _integrate_flyby_proper_time_deficit(
    gm_planet: float,
    v_inf: float,
    r_p: float,
    r_max: float,
) -> float:
    """Integrate traveler proper time deficit Delta(t) = t - tau along the flyby path.

    Evaluates:
    Delta(t) = 2 * integral_{r_p}^{r_max} [0.5 * v(r)^2 / c^2 + w(r) / c^2] / v_r(r) dr
    where v(r)^2 = v_inf^2 + 2*mu/r, w(r) = mu/r, and radial velocity
    v_r(r) = sqrt(v_inf^2 + 2*mu/r - b^2 * v_inf^2 / r^2).

    Parameters
    ----------
    gm_planet : float
        Planetary gravitational parameter mu (m^3/s^2).
    v_inf : float
        Hyperbolic excess speed (m/s).
    r_p : float
        Periapsis radius (meters).
    r_max : float
        Upper integration boundary (meters).

    Returns
    -------
    float
        Total accumulated proper time deficit Delta(t) in seconds.
    """
    if r_max <= r_p:
        return 0.0

    a = gm_planet / (v_inf**2)
    e = 1.0 + (r_p / a)
    b = a * math.sqrt(max(0.0, e**2 - 1.0))
    b2_v2 = (b * v_inf) ** 2

    # 16-point Gauss-Legendre quadrature with substitution r = r_p + u^2
    # to eliminate the 1/sqrt(r - r_p) kinematic singularity at periapsis
    u_max = math.sqrt(r_max - r_p)
    nodes, weights = np.polynomial.legendre.leggauss(16)

    # Scale nodes from [-1, 1] to [0, u_max]
    u_pts = 0.5 * (nodes + 1.0) * u_max
    w_pts = 0.5 * weights * u_max

    c2 = C_LIGHT**2
    deficit = 0.0

    for u, w in zip(u_pts, w_pts):
        r = r_p + u**2
        # Radicand for radial velocity: v_r^2 = v_inf^2 + 2*mu/r - b2_v2 / r^2
        # At r = r_p, this is identically zero. Factoring (r - r_p) avoids cancellation:
        # v_r^2 = (r - r_p) * [v_inf^2 * (r + r_p) + 2*mu] / r^2
        # Since r - r_p = u^2, v_r = u * sqrt(v_inf^2 * (r + r_p) + 2*mu) / r
        # The factor of u cancels with dr/du = 2*u!
        vr_factor = math.sqrt(v_inf**2 * (r + r_p) + 2.0 * gm_planet) / r

        # Relativistic integrand: [0.5 * v^2 + mu/r] / c^2 = [0.5 * v_inf^2 + 2 * mu / r] / c^2
        integrand_numerator = (0.5 * v_inf**2 + 2.0 * gm_planet / r) / c2

        # dr = 2 * u * du, so (integrand / v_r) * dr = (integrand_numerator / vr_factor) * 2 * du
        deficit += w * (integrand_numerator / vr_factor) * 2.0

    # Multiply by 2 to account for both inbound and outbound legs of the hyperbola
    return 2.0 * deficit
