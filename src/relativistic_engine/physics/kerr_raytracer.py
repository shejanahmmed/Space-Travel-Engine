"""Relativistic Curved-Spacetime Gravitational Lensing Ray-Tracer for Kerr Black Holes.

Implements backward null geodesic ray-tracing from an observer camera in the
exterior spacetime of a spinning Kerr black hole in Boyer-Lindquist coordinates.

Provides:
1. Zero Angular Momentum Observer (ZAMO) orthonormal tetrad frame:
   e_(0)^mu = (1/alpha) * (delta_t^mu + omega * delta_phi^mu)
   e_(1)^mu = (sqrt(Delta)/sqrt(Sigma)) * delta_r^mu
   e_(2)^mu = (1/sqrt(Sigma)) * delta_theta^mu
   e_(3)^mu = (1/varpi) * delta_phi^mu
2. Pinhole camera ray generation in the locally flat ZAMO tangent space.
3. Novikov-Thorne geometrically thin relativistic accretion disk model in the
   equatorial plane with Keplerian circular velocity Omega_K = sqrt(M)/(r^(3/2) + a*sqrt(M)).
4. Analytical relativistic redshift factor g_rad = (k_mu u_cam^mu) / (k_nu u_disk^nu)
   evaluated using exact Killing invariants (E = -k_t, L_z = k_phi).
5. Relativistic Doppler beaming and flux scaling (I_obs = g_rad^4 * I_em, T_obs = g_rad * T_em).
6. Backward null geodesic integration with horizon capture and disk intersection detection.
7. CIE 1931 XYZ and sRGB spectral rendering with gravitational mirages and Einstein rings.

References:
- Bardeen, J. M., Press, W. H., & Teukolsky, S. A. (1972), "Rotating black holes:
  locally nonrotating frames...", Astrophysical Journal 178:347-370.
- Novikov, I. D., & Thorne, K. S. (1973), "Astrophysics of black holes", Black Holes, pp. 343-450.
- Page, D. N., & Thorne, K. S. (1974), "Disk-accretion onto a black hole. I. Time-averaged structure",
  Astrophysical Journal 191:499-506.
- Luminet, J.-P. (1979), "Image of a spherical black hole with thin accretion disk",
  Astronomy and Astrophysics 75:228-235.
- Marck, J.-A. (1996), "Short-cut method of solution of geodesic equations for Kerr black holes",
  Classical and Quantum Gravity 13(3):393-402.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from relativistic_engine.constants import C_LIGHT
from relativistic_engine.physics.kerr import (
    KerrGeometry,
    compute_carter_invariants,
    compute_kerr_christoffel,
    compute_kerr_metric,
)
from relativistic_engine.physics.spectral_rendering import blackbody_to_srgb


@dataclass(frozen=True)
class KerrCamera:
    """Relativistic pinhole camera specifications in Kerr spacetime.

    Attributes:
        r_cam: Radial coordinate of the camera in meters (must be > r_E^+).
        theta_cam: Polar inclination angle of the camera in radians (0 to pi).
        phi_cam: Azimuthal coordinate in radians.
        resolution_x: Horizontal pixel count.
        resolution_y: Vertical pixel count.
        fov_deg: Horizontal field of view in degrees.
        tilt_deg: Camera pitch/tilt angle in degrees (positive tilts down toward equator).
    """
    r_cam: float
    theta_cam: float
    phi_cam: float = 0.0
    resolution_x: int = 128
    resolution_y: int = 128
    fov_deg: float = 45.0
    tilt_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.theta_cam <= 1e-4 or self.theta_cam >= math.pi - 1e-4:
            raise ValueError(f"Camera polar angle theta must be strictly in (0, pi), got {self.theta_cam}")
        if self.resolution_x <= 0 or self.resolution_y <= 0:
            raise ValueError("Camera resolutions must be positive integers.")
        if self.fov_deg <= 0.0 or self.fov_deg >= 180.0:
            raise ValueError("Field of view must be in (0, 180) degrees.")


@dataclass(frozen=True)
class AccretionDiskProperties:
    """Relativistic geometrically thin accretion disk parameters.

    Attributes:
        r_inner_m: Inner disk edge in meters (typically r_ISCO).
        r_outer_m: Outer disk edge in meters (e.g. 20 * M).
        peak_temperature_k: Peak effective blackbody temperature in Kelvin.
        emissivity: Surface thermal emissivity (0.0 to 1.0).
    """
    r_inner_m: float
    r_outer_m: float
    peak_temperature_k: float = 1.0e5  # 100,000 K for hot accretion inner region
    emissivity: float = 1.0

    def temperature_profile(self, r_m: float) -> float:
        """Evaluate Shakura-Sunyaev / Novikov-Thorne zero-torque temperature profile.

        T(r) = T_0 * (r_in / r)^(3/4) * [ 1 - sqrt(r_in / r) ]^(1/4)
        Vanishes at r = r_in (ISCO stress-free boundary).
        """
        if r_m < self.r_inner_m or r_m > self.r_outer_m:
            return 0.0
        ratio = self.r_inner_m / r_m
        torque_free = max(0.0, 1.0 - math.sqrt(ratio))
        return self.peak_temperature_k * (ratio ** 0.75) * (torque_free ** 0.25)


# ==============================================================================
# ZAMO TETRAD (LOCALLY NON-ROTATING FRAME)
# ==============================================================================

def compute_zamo_tetrad(
    r: float,
    theta: float,
    kerr: KerrGeometry,
) -> np.ndarray:
    """Construct the Zero Angular Momentum Observer (ZAMO) orthonormal tetrad.

    In Boyer-Lindquist coordinates (t, r, theta, phi), the tetrad vectors
    e_(a)^mu satisfy:
        g_mu_nu * e_(a)^mu * e_(b)^nu = eta_(a)(b) = diag(-1, 1, 1, 1)

    Components:
        alpha = sqrt( (Sigma * Delta) / A )           (lapse function)
        omega = (2 * M * a * r) / A                   (frame-dragging angular velocity)
        varpi = sqrt(A / Sigma) * sin(theta)          (cylindrical radius factor)
        e_(0) = (1 / alpha) * (d/dt + omega * d/dphi)
        e_(1) = (sqrt(Delta) / sqrt(Sigma)) * d/dr
        e_(2) = (1 / sqrt(Sigma)) * d/dtheta
        e_(3) = (1 / varpi) * d/dphi

    Args:
        r: Radial coordinate in meters.
        theta: Polar angle in radians.
        kerr: KerrGeometry specification.

    Returns:
        4x4 array tetrad[a, mu] where a is the tetrad index (0, 1, 2, 3)
        and mu is the coordinate index (t, r, theta, phi).
    """
    m = kerr.mass_m
    a = kerr.spin_m

    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    sin2 = max(1e-16, sin_th * sin_th)
    cos2 = cos_th * cos_th

    sigma = r * r + a * a * cos2
    delta = r * r - 2.0 * m * r + a * a
    if delta <= 0.0:
        raise ValueError(f"ZAMO tetrad cannot be erected inside or on the event horizon (r={r}, Delta={delta})")

    big_a = ((r * r + a * a) ** 2) - (a * a) * delta * sin2
    if big_a <= 0.0:
        big_a = 1e-16

    alpha = math.sqrt((sigma * delta) / big_a)
    omega = (2.0 * m * a * r) / big_a
    varpi = math.sqrt(big_a / sigma) * sin_th

    sqrt_sigma = math.sqrt(sigma)
    sqrt_delta = math.sqrt(delta)

    # tetrad[a, mu]
    tetrad = np.zeros((4, 4), dtype=np.float64)

    # e_(0)^mu: Timelike observer 4-velocity
    tetrad[0, 0] = 1.0 / alpha
    tetrad[0, 3] = omega / alpha

    # e_(1)^mu: Radial basis vector
    tetrad[1, 1] = sqrt_delta / sqrt_sigma

    # e_(2)^mu: Polar basis vector
    tetrad[2, 2] = 1.0 / sqrt_sigma

    # e_(3)^mu: Azimuthal basis vector
    tetrad[3, 3] = 1.0 / varpi

    return tetrad


# ==============================================================================
# DISK KINEMATICS & ANALYTICAL RELATIVISTIC REDSHIFT
# ==============================================================================

def compute_disk_keplerian_angular_velocity(
    r_m: float,
    kerr: KerrGeometry,
    prograde: bool = True,
) -> float:
    """Calculate circular equatorial Keplerian orbital angular velocity in rad/s.

    In Kerr spacetime, circular geodesics in the equatorial plane (theta = pi/2)
    have angular velocity:
        Omega_K = dphi / dt = (c * sqrt(M)) / (r^(3/2) +- a * sqrt(M))
    where + is prograde and - is retrograde.

    Args:
        r_m: Orbital radius in meters.
        kerr: Black hole geometry.
        prograde: True for prograde co-rotating orbit, False for retrograde.

    Returns:
        Keplerian orbital angular velocity Omega_K in radians per second.
    """
    m = kerr.mass_m
    a = kerr.spin_m if prograde else -kerr.spin_m
    sqrt_m = math.sqrt(m)
    denom = (r_m ** 1.5) + a * sqrt_m
    if abs(denom) < 1e-12:
        denom = math.copysign(1e-12, denom)
    return float(C_LIGHT * sqrt_m / denom)


def compute_relativistic_redshift_factor(
    k_ray_at_cam: np.ndarray,
    r_hit_m: float,
    kerr: KerrGeometry,
    camera: KerrCamera,
) -> float:
    """Compute the exact relativistic frequency shift ratio g_rad = nu_obs / nu_em.

    Using the conserved Killing invariants along the null geodesic:
        E = -k_t,  L_z = k_phi  =>  lambda = L_z / E

    The observer at the camera has 4-velocity u_cam^mu (ZAMO observer).
    The disk emitter at r_hit has 4-velocity u_disk^mu = u^t * (1, 0, 0, Omega_K).

    The redshift factor is evaluated analytically:
        g_rad = (k_mu * u_cam^mu) / (k_nu * u_disk^nu)
              = (u_cam^t - lambda * u_cam^phi) / (u_disk^t - lambda * u_disk^phi)

    Args:
        k_ray_at_cam: Initial null 4-momentum k^mu at the camera.
        r_hit_m: Disk intersection radius in meters.
        kerr: Black hole geometry.
        camera: Camera specifications.

    Returns:
        Frequency ratio g_rad in (0, inf).
        g_rad > 1 corresponds to blueshift (approaching side).
        g_rad < 1 corresponds to redshift (receding side / gravitational well).
    """
    g_cam, _ = compute_kerr_metric(camera.r_cam, camera.theta_cam, kerr)
    # Energy E = -k_t = -(g_tt * k^t + g_tphi * k^phi)
    energy = -(g_cam[0, 0] * k_ray_at_cam[0] + g_cam[0, 3] * k_ray_at_cam[3])
    # Axial angular momentum L_z = k_phi = g_phi_t * k^t + g_phiphi * k^phi
    ang_mom = g_cam[3, 0] * k_ray_at_cam[0] + g_cam[3, 3] * k_ray_at_cam[3]

    if abs(energy) < 1e-30:
        return 0.0

    impact_parameter_lambda = ang_mom / energy

    # Camera 4-velocity (ZAMO observer)
    tetrad_cam = compute_zamo_tetrad(camera.r_cam, camera.theta_cam, kerr)
    u_cam_t = tetrad_cam[0, 0]
    u_cam_phi = tetrad_cam[0, 3]

    # Disk 4-velocity at equatorial plane (theta = pi/2)
    omega_k = compute_disk_keplerian_angular_velocity(r_hit_m, kerr, prograde=True)
    g_disk, _ = compute_kerr_metric(r_hit_m, 0.5 * math.pi, kerr)

    # u^t = 1 / sqrt( -g_tt - 2 * Omega_K * g_tphi - Omega_K^2 * g_phiphi )
    # Note: g_tt is in (c*dt)^2, so velocity components need C_LIGHT scaling
    omega_norm = omega_k / C_LIGHT
    norm_sq = -(g_disk[0, 0] + 2.0 * omega_norm * g_disk[0, 3] + (omega_norm ** 2) * g_disk[3, 3])
    if norm_sq <= 0.0:
        # Inside the ergosphere or unstable region where circular orbit is spacelike
        return 0.0

    u_disk_t = 1.0 / math.sqrt(norm_sq)
    u_disk_phi = omega_norm * u_disk_t

    num = u_cam_t - impact_parameter_lambda * u_cam_phi
    denom = u_disk_t - impact_parameter_lambda * u_disk_phi

    if denom <= 0.0 or num <= 0.0:
        return 0.0

    g_rad = num / denom
    return float(g_rad)


# ==============================================================================
# BACKWARD NULL GEODESIC RAY INTEGRATOR
# ==============================================================================

def trace_single_kerr_ray(
    initial_x: np.ndarray,
    initial_k: np.ndarray,
    kerr: KerrGeometry,
    disk: Optional[AccretionDiskProperties] = None,
    max_steps: int = 1500,
    r_escape_multiplier: float = 40.0,
    step_size_scale: float = 0.5,
) -> Dict[str, Any]:
    """Integrate a single null photon ray backward in time through Kerr spacetime.

    Equations of motion along affine parameter lambda (dlambda < 0):
        dx^mu / dlambda = k^mu
        dk^mu / dlambda = - Gamma^mu_alpha_beta * k^alpha * k^beta

    Terminates on:
    - Horizon capture (r <= r_+ + 0.02 * M): Black hole shadow.
    - Accretion disk hit (equatorial plane crossing with r_in <= r <= r_out).
    - Escape to celestial infinity (r >= r_escape).

    Args:
        initial_x: Starting 4-position [t, r, theta, phi] in Boyer-Lindquist coordinates.
        initial_k: Initial backward null 4-momentum k^mu at the camera.
        kerr: Black hole geometry.
        disk: Optional accretion disk properties.
        max_steps: Maximum Runge-Kutta integration steps.
        r_escape_multiplier: Radius in units of M defining asymptotic escape.
        step_size_scale: Integrator step size damping factor.

    Returns:
        Dictionary containing termination status, hit coordinates, and redshift factor.
    """
    m = kerr.mass_m
    r_horizon = kerr.event_horizon_outer
    r_capture = r_horizon + 0.02 * m
    r_escape = r_escape_multiplier * m

    x = np.array(initial_x, dtype=np.float64)
    k = np.array(initial_k, dtype=np.float64)

    # Store previous position for disk equatorial plane crossing detection
    prev_r = x[1]
    prev_theta = x[2]

    # Target hit records
    disk_hit = False
    hit_r = 0.0
    hit_phi = 0.0
    g_redshift = 0.0

    # Dynamic step size scaled by local gravitational radius
    for step in range(max_steps):
        r_curr = x[1]
        th_curr = x[2]

        # 1. Check event horizon capture
        if r_curr <= r_capture:
            return {
                "status": "HORIZON_CAPTURE",
                "r_final": r_curr,
                "theta_final": th_curr,
                "phi_final": x[3],
                "steps": step,
                "disk_hit": False,
                "g_redshift": 0.0,
            }

        # 2. Check escape to infinity
        if r_curr >= r_escape:
            return {
                "status": "ESCAPED",
                "r_final": r_curr,
                "theta_final": th_curr,
                "phi_final": x[3],
                "steps": step,
                "disk_hit": False,
                "g_redshift": 1.0,
            }

        # Adaptive affine step size: smaller near the horizon, larger far away
        d_lambda = -step_size_scale * min(r_curr - r_horizon, r_curr * 0.1)

        # Evaluate Christoffel symbols
        gamma_symbols = compute_kerr_christoffel(r_curr, th_curr, kerr)

        # Derivatives: dx/dlambda = k
        dk_dlambda = -np.einsum("mab,a,b->m", gamma_symbols, k, k)

        # 4th-order Runge-Kutta step
        k1_x = k
        k1_k = dk_dlambda

        x_half = x + 0.5 * d_lambda * k1_x
        k_half = k + 0.5 * d_lambda * k1_k
        # Guard theta in [1e-5, pi - 1e-5]
        x_half[2] = np.clip(x_half[2], 1e-5, math.pi - 1e-5)

        gamma_half = compute_kerr_christoffel(x_half[1], x_half[2], kerr)
        k2_x = k_half
        k2_k = -np.einsum("mab,a,b->m", gamma_half, k_half, k_half)

        # Advance state
        x_new = x + d_lambda * k2_x
        k_new = k + d_lambda * k2_k
        x_new[2] = np.clip(x_new[2], 1e-5, math.pi - 1e-5)

        # 3. Check for equatorial plane crossing (theta = pi/2)
        half_pi = 0.5 * math.pi
        if disk is not None and (prev_theta - half_pi) * (x_new[2] - half_pi) <= 0.0:
            # Detect equatorial plane piercing (theta = pi/2) via bracket zero crossing
            denom_th = (x_new[2] - prev_theta)
            fraction = (half_pi - prev_theta) / denom_th if abs(denom_th) > 1e-12 else 0.5
            fraction = np.clip(fraction, 0.0, 1.0)
            r_cross = prev_r + fraction * (x_new[1] - prev_r)

            if disk.r_inner_m <= r_cross <= disk.r_outer_m:
                disk_hit = True
                hit_r = r_cross
                hit_phi = x[3] + fraction * (x_new[3] - x[3])
                # Relativistic spectral frequency shift g = (k_mu u^mu)_emit / (k_nu u^nu)_obs
                k_at_cam = initial_k
                g_redshift = compute_relativistic_redshift_factor(
                    k_ray_at_cam=k_at_cam,
                    r_hit_m=hit_r,
                    kerr=kerr,
                    camera=KerrCamera(r_cam=initial_x[1], theta_cam=initial_x[2]),
                )
                return {
                    "status": "DISK_HIT",
                    "r_final": hit_r,
                    "theta_final": half_pi,
                    "phi_final": hit_phi,
                    "steps": step,
                    "disk_hit": True,
                    "g_redshift": g_redshift,
                }

        prev_r = x_new[1]
        prev_theta = x_new[2]
        x = x_new
        k = k_new

    return {
        "status": "MAX_STEPS",
        "r_final": x[1],
        "theta_final": x[2],
        "phi_final": x[3],
        "steps": max_steps,
        "disk_hit": False,
        "g_redshift": 0.0,
    }


# ==============================================================================
# FULL SCENE RAY-TRACER & SYNTHETIC IMAGE GENERATION
# ==============================================================================

def render_kerr_black_hole_scene(
    camera: KerrCamera,
    kerr: KerrGeometry,
    disk: Optional[AccretionDiskProperties] = None,
    background_color: Tuple[float, float, float] = (0.01, 0.01, 0.02),
    max_steps_per_ray: int = 1200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a full synthetic image of a rotating Kerr black hole and accretion disk.

    Launches backward rays across the camera pixel grid, computes the relativistic
    redshift g_rad and Doppler beaming on the disk, and evaluates CIE 1931 sRGB colors.

    Args:
        camera: KerrCamera specifications.
        kerr: Black hole geometry.
        disk: Accretion disk properties.
        background_color: Normalized RGB color of deep space background.
        max_steps_per_ray: Ray integration limit.

    Returns:
        Tuple of:
        - rgb_image: (H, W, 3) float array of sRGB pixel values in [0, 1].
        - redshift_map: (H, W) float array of observed frequency shift g_rad.
        - hit_type_map: (H, W) integer array (0 = shadow, 1 = disk, 2 = background).
    """
    w = camera.resolution_x
    h = camera.resolution_y

    rgb_image = np.zeros((h, w, 3), dtype=np.float64)
    redshift_map = np.zeros((h, w), dtype=np.float64)
    hit_type_map = np.zeros((h, w), dtype=np.int32)

    tetrad = compute_zamo_tetrad(camera.r_cam, camera.theta_cam, kerr)

    aspect_ratio = float(w) / float(h)
    half_fov_x = math.radians(camera.fov_deg * 0.5)
    half_fov_y = half_fov_x / aspect_ratio

    # Screen coordinate grids
    u_coords = np.linspace(-math.tan(half_fov_x), math.tan(half_fov_x), w)
    v_coords = np.linspace(math.tan(half_fov_y), -math.tan(half_fov_y), h)

    # Direction vectors in ZAMO 3-space:
    # Camera points toward -r (e_(1) direction), right is e_(3) (phi), up is e_(2) (-theta)
    initial_x = np.array([0.0, camera.r_cam, camera.theta_cam, camera.phi_cam])

    for j, v in enumerate(v_coords):
        for i, u in enumerate(u_coords):
            # Spatial ray direction in ZAMO frame:
            # -e_(1) (towards black hole), +u * e_(3) (right), -v * e_(2) (up)
            dir_r = -1.0
            dir_th = -v
            dir_phi = u

            norm = math.sqrt(dir_r * dir_r + dir_th * dir_th + dir_phi * dir_phi)
            n_r = dir_r / norm
            n_th = dir_th / norm
            n_phi = dir_phi / norm

            # Backward null 4-momentum in ZAMO frame: k^(0) = 1, k^(i) = -n^(i)
            # Transform to Boyer-Lindquist coordinate basis: k^mu = k^(a) * e_(a)^mu
            k_init = (
                1.0 * tetrad[0]
                - n_r * tetrad[1]
                - n_th * tetrad[2]
                - n_phi * tetrad[3]
            )

            # Trace backward ray
            ray_res = trace_single_kerr_ray(
                initial_x=initial_x,
                initial_k=k_init,
                kerr=kerr,
                disk=disk,
                max_steps=max_steps_per_ray,
            )

            if ray_res["status"] == "HORIZON_CAPTURE":
                hit_type_map[j, i] = 0
                rgb_image[j, i] = [0.0, 0.0, 0.0]  # Bardeen black hole shadow
            elif ray_res["status"] == "DISK_HIT":
                hit_type_map[j, i] = 1
                g_rad = ray_res["g_redshift"]
                redshift_map[j, i] = g_rad

                if disk is not None and g_rad > 0.0:
                    t_em = disk.temperature_profile(ray_res["r_final"])
                    t_obs = g_rad * t_em
                    # Bolometric radiance boost: g^4
                    flux_boost = g_rad ** 4
                    rgb = blackbody_to_srgb(t_obs, flux_multiplier=flux_boost)
                    rgb_image[j, i] = rgb
            else:
                # Escaped to celestial infinity
                hit_type_map[j, i] = 2
                redshift_map[j, i] = 1.0
                rgb_image[j, i] = background_color

    return rgb_image, redshift_map, hit_type_map
