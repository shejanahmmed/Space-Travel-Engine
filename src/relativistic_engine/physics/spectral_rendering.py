"""Relativistic Optical Ray-Tracing and Blackbody Spectral Radiance Rendering.

Implements high-fidelity spectral radiance modeling and camera ray-tracing for
accelerating relativistic spacecraft (beta = v/c in [0, 0.9999]).

Provides:
1. Exact Planck spectral radiance with numerical overflow/underflow protection.
2. Peebles-Wilkinson blackbody temperature transformation under relativistic Doppler boost:
   T' = D * T_0 (Peebles & Wilkinson 1968, McKinley & Doherty 1979).
3. CIE 1931 standard 2-degree observer color matching functions (380 nm - 780 nm).
4. Continuous spectrum-to-XYZ tristimulus integration and sRGB electro-optical conversion.
5. Analytical and numerical full-sky integrated flux scaling:
   ratio = gamma * (1 + 1/3 * beta^2).
6. Pinhole camera relativistic starfield ray-caster with Gaussian Point Spread Function (PSF).
7. Relativistic full-sky panoramic equirectangular ray-tracing with optional boosted CMB background.

References:
- CIE (1932), "Commission Internationale de l'Eclairage Proceedings, 1931", Cambridge Univ. Press.
- ITU-R Recommendation BT.709-6 (2015), "Parameter values for the HDTV standards for production
  and international programme exchange".
- McKinley, J. M., & Doherty, P. (1979), "In search of the 'starbow': The appearance
  of the starfield from a relativistic spaceship", American Journal of Physics 47(4), 309-316.
- Misner, C. W., Thorne, K. S., & Wheeler, J. A. (1973), "Gravitation", W. H. Freeman, §2.7.
- Peebles, P. J. E., & Wilkinson, D. T. (1968), "Comment on the anisotropy of the primeval
  fireball", Physical Review 174(5), 2168-2169.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from relativistic_engine.constants import (
    C_LIGHT,
    H_PLANCK,
    K_BOLTZMANN,
    SIGMA_SB,
    WIEN_B,
)
from relativistic_engine.physics.optics import (
    compute_bolometric_flux_scaling,
    compute_doppler_factor,
    inverse_aberration_vector,
    transform_aberration_vector,
)

# ==============================================================================
# CIE 1931 STANDARD 2-DEGREE COLOR MATCHING FUNCTIONS (380 nm to 780 nm, 5 nm step)
# Source: ISO/CIE 11664-1:2019 / CIE Standard Colorimetric Observers
# Wavelengths in nanometers: 380, 385, ..., 780 (81 sample points)
# ==============================================================================

CIE_WAVELENGTHS_NM: np.ndarray = np.arange(380, 785, 5, dtype=np.float64)
CIE_WAVELENGTHS_M: np.ndarray = CIE_WAVELENGTHS_NM * 1e-9

# fmt: off
CIE_X_BAR: np.ndarray = np.array([
    0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023190, 0.043510, 0.077630,
    0.134380, 0.214770, 0.283900, 0.328500, 0.348280, 0.348060, 0.336200, 0.318700,
    0.290800, 0.251100, 0.195360, 0.142100, 0.095640, 0.057950, 0.032010, 0.014700,
    0.004900, 0.002400, 0.009300, 0.029100, 0.063270, 0.109600, 0.165500, 0.225750,
    0.290400, 0.359700, 0.433450, 0.512050, 0.594500, 0.678400, 0.762100, 0.842500,
    0.916300, 0.978600, 1.026300, 1.056700, 1.062200, 1.045600, 1.002600, 0.938400,
    0.854450, 0.751400, 0.642400, 0.541900, 0.447900, 0.360800, 0.283500, 0.218700,
    0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900, 0.022700, 0.015840,
    0.011359, 0.008111, 0.005790, 0.004109, 0.002899, 0.002049, 0.001440, 0.001000,
    0.000690, 0.000476, 0.000332, 0.000235, 0.000166, 0.000117, 0.000083, 0.000059,
    0.000042
], dtype=np.float64)

CIE_Y_BAR: np.ndarray = np.array([
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210, 0.002180,
    0.004000, 0.007300, 0.011600, 0.016840, 0.023000, 0.029800, 0.038000, 0.048000,
    0.060000, 0.073900, 0.090980, 0.112600, 0.139020, 0.169300, 0.208020, 0.258600,
    0.323000, 0.407300, 0.503000, 0.608200, 0.710000, 0.793200, 0.862000, 0.914850,
    0.954000, 0.980300, 0.994950, 1.000000, 0.995000, 0.978600, 0.952000, 0.915400,
    0.870000, 0.816300, 0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200,
    0.381000, 0.321000, 0.265000, 0.217000, 0.175000, 0.138200, 0.107000, 0.081600,
    0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210, 0.005723,
    0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740, 0.000520, 0.000361,
    0.000249, 0.000172, 0.000120, 0.000085, 0.000060, 0.000042, 0.000030, 0.000021,
    0.000015
], dtype=np.float64)

CIE_Z_BAR: np.ndarray = np.array([
    0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200, 0.207400, 0.371300,
    0.645600, 1.039050, 1.385600, 1.622960, 1.747060, 1.782600, 1.772110, 1.744100,
    1.669200, 1.528100, 1.287640, 1.042500, 0.812950, 0.616200, 0.465180, 0.353300,
    0.272000, 0.212300, 0.158200, 0.111700, 0.078250, 0.053800, 0.034600, 0.021010,
    0.011600, 0.005950, 0.002750, 0.001200, 0.000500, 0.000150, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000
], dtype=np.float64)
# fmt: on

# CIE XYZ to linear sRGB transformation matrix (ITU-R BT.709-6, D65 illuminant)
M_XYZ_TO_SRGB: np.ndarray = np.array([
    [+3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, +1.8760108, +0.0415560],
    [+0.0556434, -0.2040259, +1.0572252],
], dtype=np.float64)


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass(frozen=True)
class CelestialStar:
    """Astronomical star definition in the inertial Barycentric frame.

    Attributes:
        direction: 3D unit vector [x, y, z] pointing toward the star from observer.
        temperature_k: Effective stellar surface temperature in Kelvin (rest frame).
        visual_magnitude: Apparent Johnson visual magnitude V_0 (rest frame).
        name: Common name or catalog identifier.
    """
    direction: np.ndarray
    temperature_k: float
    visual_magnitude: float
    name: str = ""

    def __post_init__(self) -> None:
        arr = np.asarray(self.direction, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if norm <= 0.0:
            raise ValueError("Star direction vector cannot be null.")
        if not np.isclose(norm, 1.0, atol=1e-12):
            object.__setattr__(self, "direction", arr / norm)
        else:
            object.__setattr__(self, "direction", arr)


# ==============================================================================
# FUNDAMENTAL PLANCK & RELATIVISTIC THERMODYNAMICS
# ==============================================================================

def planck_spectral_radiance(
    wavelength_m: Union[float, np.ndarray],
    temperature_k: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute exact Planck blackbody spectral radiance B_lambda.

    Evaluates:
        B_lambda(lambda, T) = [2 * h * c^2 / lambda^5] / [exp(h * c / (lambda * k_B * T)) - 1]

    Units:
        W * m^-2 * sr^-1 * m^-1

    Numerical stability:
        Guards against exponential overflow for high frequencies / low temperatures
        (x = h*c / (lambda*k_B*T) > 700 -> B_lambda = 0.0).

    Args:
        wavelength_m: Photon wavelength in meters (must be > 0).
        temperature_k: Absolute temperature in Kelvin (must be > 0).

    Returns:
        Spectral radiance in W/(m^2 * sr * m).
    """
    lam = np.asarray(wavelength_m, dtype=np.float64)
    temp = np.asarray(temperature_k, dtype=np.float64)

    # Broadcast inputs to common shape
    lam_b, temp_b = np.broadcast_arrays(lam, temp)

    # First radiation constant: c_1L = 2 * h * c^2
    c1 = 2.0 * H_PLANCK * (C_LIGHT ** 2)
    # Second radiation constant: c_2 = h * c / k_B
    c2 = (H_PLANCK * C_LIGHT) / K_BOLTZMANN

    # Safe division avoiding zero or negative wavelength/temperature
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = c2 / (lam_b * temp_b)
        
        radiance = np.zeros(lam_b.shape, dtype=np.float64)
        valid_mask = (lam_b > 0.0) & (temp_b > 0.0) & (exponent < 700.0)
        
        if np.any(valid_mask):
            exp_val = np.expm1(exponent[valid_mask])
            inv_lam5 = 1.0 / (lam_b[valid_mask] ** 5)
            radiance[valid_mask] = (c1 * inv_lam5) / exp_val

    if isinstance(wavelength_m, (int, float)) and isinstance(temperature_k, (int, float)):
        return float(radiance.item())
    return radiance


def boosted_blackbody_temperature(
    rest_temp_k: Union[float, np.ndarray],
    doppler_factor: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute effective blackbody temperature under relativistic Doppler shift.

    By the Peebles-Wilkinson theorem (1968), a Planck blackbody radiation field
    Lorentz-boosted by Doppler factor D remains an exact blackbody with an
    effective apparent temperature:
        T' = D * T_0

    Args:
        rest_temp_k: Unperturbed emitter surface temperature in Kelvin.
        doppler_factor: Relativistic Doppler factor D = gamma * (1 + beta . n).

    Returns:
        Apparent boosted temperature T' in Kelvin.
    """
    return np.asarray(rest_temp_k, dtype=np.float64) * np.asarray(doppler_factor, dtype=np.float64)


def wien_peak_wavelength(temperature_k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute peak wavelength of Planck distribution using Wien's displacement law.

    lambda_max = b / T
    where b = 2.897771955... x 10^-3 m * K (CODATA 2018).

    Args:
        temperature_k: Temperature in Kelvin.

    Returns:
        Wavelength of maximum spectral radiance in meters.
    """
    temp = np.asarray(temperature_k, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(temp > 0.0, WIEN_B / temp, 0.0)


# ==============================================================================
# COLORIMETRIC CONVERSION (CIE 1931 & sRGB)
# ==============================================================================

def spectrum_to_cie_xyz(
    wavelengths_m: np.ndarray,
    spectral_radiance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate spectral radiance against CIE 1931 standard color matching functions.

    Evaluates:
        X = integral B_lambda(lambda) * x_bar(lambda) dlambda
        Y = integral B_lambda(lambda) * y_bar(lambda) dlambda
        Z = integral B_lambda(lambda) * z_bar(lambda) dlambda

    Numerical quadrature is performed over the standard visual band (380 - 780 nm)
    using trapezoidal integration.

    Args:
        wavelengths_m: 1D array of sample wavelengths in meters.
        spectral_radiance: 1D or 2D array of spectral radiance values.
            If 2D, shape must be (N_spectra, N_wavelengths).

    Returns:
        Tuple of:
            xyz: CIE XYZ tristimulus values of shape (3,) or (N_spectra, 3).
            xy: Normalized chromaticity coordinates (x, y) = (X/(X+Y+Z), Y/(X+Y+Z)).
    """
    rad = np.atleast_2d(spectral_radiance)
    
    # Interpolate CIE color matching functions onto provided wavelengths if needed
    if not np.array_equal(wavelengths_m, CIE_WAVELENGTHS_M):
        x_bar = np.interp(wavelengths_m, CIE_WAVELENGTHS_M, CIE_X_BAR, left=0.0, right=0.0)
        y_bar = np.interp(wavelengths_m, CIE_WAVELENGTHS_M, CIE_Y_BAR, left=0.0, right=0.0)
        z_bar = np.interp(wavelengths_m, CIE_WAVELENGTHS_M, CIE_Z_BAR, left=0.0, right=0.0)
    else:
        x_bar = CIE_X_BAR
        y_bar = CIE_Y_BAR
        z_bar = CIE_Z_BAR

    # Trapezoidal numerical integration over wavelength in meters
    delta_lam = np.gradient(wavelengths_m)
    
    X = np.sum(rad * (x_bar * delta_lam), axis=1)
    Y = np.sum(rad * (y_bar * delta_lam), axis=1)
    Z = np.sum(rad * (z_bar * delta_lam), axis=1)

    xyz = np.stack([X, Y, Z], axis=-1)
    
    sum_xyz = X + Y + Z
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(sum_xyz > 0.0, X / sum_xyz, 1.0 / 3.0)
        y = np.where(sum_xyz > 0.0, Y / sum_xyz, 1.0 / 3.0)
    xy = np.stack([x, y], axis=-1)

    if spectral_radiance.ndim == 1:
        return xyz[0], xy[0]
    return xyz, xy


def cie_xyz_to_srgb(
    xyz: np.ndarray,
    *,
    exposure: float = 1.0,
    tone_map: bool = True,
    gamma: bool = True,
) -> np.ndarray:
    """Convert CIE XYZ tristimulus array to standard sRGB display values.

    Args:
        xyz: Array with last dimension size 3 containing (X, Y, Z).
        exposure: Multiplicative exposure gain factor.
        tone_map: If True, applies Reinhard luminance tone-mapping (L / (1 + L))
            to avoid hard clipping of high-dynamic-range stellar fluxes.
        gamma: If True, applies the standard ITU-R BT.709 / sRGB piecewise gamma function.

    Returns:
        sRGB array with values clipped in [0.0, 1.0].
    """
    xyz_arr = np.asarray(xyz, dtype=np.float64) * exposure
    
    # Linear sRGB transformation: [R, G, B]^T = M_XYZ_TO_SRGB * [X, Y, Z]^T
    srgb_lin = np.matmul(xyz_arr, M_XYZ_TO_SRGB.T)
    
    if tone_map:
        # Reinhard tone-mapping preserving chromaticity ratios: C / (1 + C)
        srgb_lin = np.where(srgb_lin > 0.0, srgb_lin / (1.0 + srgb_lin), 0.0)
    else:
        srgb_lin = np.clip(srgb_lin, 0.0, None)

    if not gamma:
        return np.clip(srgb_lin, 0.0, 1.0)

    # Standard sRGB piecewise electro-optical transfer function (IEC 61966-2-1)
    low_mask = srgb_lin <= 0.0031308
    srgb = np.empty_like(srgb_lin)
    srgb[low_mask] = 12.92 * srgb_lin[low_mask]
    srgb[~low_mask] = 1.055 * np.power(np.maximum(srgb_lin[~low_mask], 0.0), 1.0 / 2.4) - 0.055

    return np.clip(srgb, 0.0, 1.0)


def blackbody_to_srgb(
    temperature_k: float,
    *,
    flux_multiplier: float = 1.0,
    exposure: float = 1.0,
) -> np.ndarray:
    """Compute apparent sRGB color of a blackbody at given temperature.

    Args:
        temperature_k: Blackbody temperature in Kelvin.
        flux_multiplier: Scale factor for bolometric radiance (e.g. D^4).
        exposure: Exposure scaling.

    Returns:
        1D float array of shape (3,) with sRGB components in [0, 1].
    """
    rad = planck_spectral_radiance(CIE_WAVELENGTHS_M, temperature_k) * flux_multiplier
    xyz, _ = spectrum_to_cie_xyz(CIE_WAVELENGTHS_M, rad)
    return cie_xyz_to_srgb(xyz, exposure=exposure, tone_map=True, gamma=True)


# ==============================================================================
# FULL-SKY PHOTOMETRIC FLUX INVARIANT (McKinley & Doherty 1979)
# ==============================================================================

def analytical_sky_flux_boost_factor(beta: float) -> float:
    """Compute exact analytical ratio of total integrated sky radiation energy density.

    From relativistic electrodynamics (Peebles & Wilkinson 1968, McKinley & Doherty 1979):
        u' / u = (1 / 4*pi) * integral_{4*pi} D^4(n') dOmega'
               = gamma^2 * (1 + (1/3) * beta^2)

    This is the exact relativistic radiation stress-energy transformation T'^00 / T^00
    for an isotropic blackbody radiation field (such as the CMB).

    Args:
        beta: Spacecraft velocity ratio |v|/c in [0, 1).

    Returns:
        Total energy density flux ratio >= 1.0.
    """
    if beta < 0.0 or beta >= 1.0:
        raise ValueError(f"Velocity beta must be in [0, 1), got {beta}")
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    return (gamma ** 2) * (1.0 + (1.0 / 3.0) * (beta ** 2))


def numerical_sky_flux_boost_factor(beta: float, n_theta: int = 10000) -> float:
    """Numerically evaluate the total integrated sky radiation boost on the observer's sky.

    Integrates:
        (1 / 4*pi) * integral_{4*pi} D^4(theta') dOmega'
    Using the solid angle transformation dOmega' = dOmega / D^2, this is equivalent to:
        (1 / 4*pi) * integral_{4*pi} D^2(theta) dOmega
        = (1 / 2) * integral_0^pi D^2(theta) * sin(theta) dtheta

    Args:
        beta: Spacecraft velocity ratio |v|/c in [0, 1).
        n_theta: Number of numerical integration intervals over [0, pi].

    Returns:
        Numerically integrated total flux ratio.
    """
    if beta < 0.0 or beta >= 1.0:
        raise ValueError(f"Velocity beta must be in [0, 1), got {beta}")
    if beta == 0.0:
        return 1.0

    theta = np.linspace(0.0, np.pi, n_theta, dtype=np.float64)
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    # D(theta) = gamma * (1 + beta * cos(theta)) in source coordinates
    d = gamma * (1.0 + beta * np.cos(theta))
    # Integrand in source coordinates: D^4 dOmega' = D^2 dOmega = D^2 sin(theta) dtheta
    integrand = (d ** 2) * np.sin(theta)
    integral = 0.5 * float(np.trapezoid(integrand, theta))
    return integral


# ==============================================================================
# RELATIVISTIC RAY-TRACING & CAMERA PROJECTION
# ==============================================================================

def render_relativistic_starfield_pinhole(
    stars: List[CelestialStar],
    beta_vec: np.ndarray,
    *,
    fov_deg: float = 45.0,
    resolution: Tuple[int, int] = (256, 256),
    camera_rot: Optional[np.ndarray] = None,
    psf_sigma_pixels: float = 1.2,
    exposure: float = 1.0,
) -> np.ndarray:
    """Render the relativistic starfield through a pinhole camera viewport.

    Projects celestial catalog stars through 3D aberration into the moving
    spacecraft camera frame, shifts their spectral blackbody radiance according
    to the Peebles-Wilkinson theorem, applies the D^4 bolometric magnification,
    and convolves with an optical Gaussian Point Spread Function (PSF).

    Args:
        stars: List of CelestialStar objects in inertial BCRS frame.
        beta_vec: Spacecraft 3D velocity vector beta = v/c relative to BCRS.
        fov_deg: Horizontal camera field of view in degrees.
        resolution: Output image resolution (height, width) in pixels.
        camera_rot: 3x3 orthonormal matrix rotating from spacecraft body frame
            to camera optical frame (camera Z = forward optical axis). If None,
            identity matrix is used (ship forward along +Z axis).
        psf_sigma_pixels: Standard deviation of Gaussian star PSF in pixels.
        exposure: Global sensor exposure gain.

    Returns:
        RGB image array of shape (height, width, 3) with float sRGB values in [0, 1].
    """
    height, width = resolution
    image = np.zeros((height, width, 3), dtype=np.float64)
    
    if len(stars) == 0:
        return image

    beta = np.asarray(beta_vec, dtype=np.float64)
    if camera_rot is None:
        r_cam = np.eye(3, dtype=np.float64)
    else:
        r_cam = np.asarray(camera_rot, dtype=np.float64)

    # Focal length in pixels from horizontal FOV
    half_fov_rad = math.radians(fov_deg * 0.5)
    f_pixels = (width * 0.5) / math.tan(half_fov_rad)

    # Transform all catalog star vectors to moving spacecraft frame
    cat_dirs = np.array([s.direction for s in stars], dtype=np.float64)
    app_dirs = transform_aberration_vector(cat_dirs, beta)
    
    # Doppler factors for each star: D = gamma * (1 + beta . n)
    doppler_factors = compute_doppler_factor(cat_dirs, beta)

    # Rotate into camera frame: [x_cam, y_cam, z_cam]
    cam_dirs = np.dot(app_dirs, r_cam.T)

    # Pre-compute boosted RGB color for each star
    for i, star in enumerate(stars):
        z_c = cam_dirs[i, 2]
        # Star is in front of camera lens (positive Z)
        if z_c <= 0.01:
            continue

        # Pinhole perspective projection onto sensor plane
        u = (width * 0.5) + (cam_dirs[i, 0] / z_c) * f_pixels
        v = (height * 0.5) - (cam_dirs[i, 1] / z_c) * f_pixels

        # Cull field sources outside sensor focal plane accounting for PSF support margin
        margin = int(math.ceil(3.5 * psf_sigma_pixels))
        if u < -margin or u >= width + margin or v < -margin or v >= height + margin:
            continue

        # Relativistic Doppler spectral transformation (Peebles-Wilkinson)
        d_val = float(doppler_factors[i])
        t_boosted = star.temperature_k * d_val
        
        # Flux scaling: D^4 bolometric magnification, combined with visual magnitude
        # Pogson formula: unperturbed intensity I0 proportional to 10^(-0.4 * V)
        base_intensity = 10.0 ** (-0.4 * (star.visual_magnitude - 0.0))
        d_flux_scaling = d_val ** 4
        star_flux = base_intensity * d_flux_scaling

        # CIE tristimulus for the boosted star blackbody
        rad = planck_spectral_radiance(CIE_WAVELENGTHS_M, t_boosted)
        xyz, _ = spectrum_to_cie_xyz(CIE_WAVELENGTHS_M, rad)
        
        # Normalize XYZ by Y and scale by total physical star flux
        if xyz[1] > 0.0:
            xyz_scaled = (xyz / xyz[1]) * star_flux
        else:
            xyz_scaled = np.zeros(3, dtype=np.float64)

        # Splat Gaussian PSF onto local pixel grid
        u_min = max(0, int(math.floor(u - margin)))
        u_max = min(width, int(math.ceil(u + margin + 1)))
        v_min = max(0, int(math.floor(v - margin)))
        v_max = min(height, int(math.ceil(v + margin + 1)))

        uu, vv = np.meshgrid(np.arange(u_min, u_max), np.arange(v_min, v_max))
        dist_sq = (uu - u) ** 2 + (vv - v) ** 2
        psf = np.exp(-0.5 * dist_sq / (psf_sigma_pixels ** 2)) / (2.0 * math.pi * (psf_sigma_pixels ** 2))

        for c in range(3):
            image[v_min:v_max, u_min:u_max, c] += psf * xyz_scaled[c]

    # Final conversion from accumulated sensor XYZ to sRGB
    return cie_xyz_to_srgb(image, exposure=exposure, tone_map=True, gamma=True)


def render_relativistic_panorama(
    stars: List[CelestialStar],
    beta_vec: np.ndarray,
    *,
    resolution: Tuple[int, int] = (256, 512),
    exposure: float = 1.0,
    include_cmb: bool = False,
    cmb_temp_k: float = 2.7255,
) -> np.ndarray:
    """Render full-sky 360 x 180 degree relativistic celestial sphere panorama.

    Equirectangular projection mapping:
        longitude phi in [-pi, pi]
        latitude theta in [-pi/2, pi/2]

    For each apparent direction n', evaluates the exact inverse aberration to find
    the inertial origin n = A^-1(n', beta), computes the boosted starfield and
    optional Cosmic Microwave Background (CMB) spectral distortion.

    Args:
        stars: List of CelestialStar objects.
        beta_vec: Spacecraft velocity vector beta = v/c.
        resolution: Output panorama resolution (height, width).
        exposure: Global exposure scaling.
        include_cmb: If True, renders boosted thermal background (e.g. CMB).
        cmb_temp_k: CMB baseline rest temperature (COBE / Planck: 2.7255 K).

    Returns:
        Float sRGB image array of shape (height, width, 3) in [0, 1].
    """
    height, width = resolution
    beta = np.asarray(beta_vec, dtype=np.float64)
    beta_mag = float(np.linalg.norm(beta))

    image_xyz = np.zeros((height, width, 3), dtype=np.float64)

    # Pixel coordinate grid
    phi_grid = np.linspace(-np.pi, np.pi, width, endpoint=False, dtype=np.float64)
    theta_grid = np.linspace(np.pi * 0.5, -np.pi * 0.5, height, dtype=np.float64)
    phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)

    # Unit vector for every apparent pixel in moving spacecraft frame:
    # X = cos(theta) * cos(phi)
    # Y = cos(theta) * sin(phi)
    # Z = sin(theta)  (forward travel along +X or body axes)
    cos_theta = np.cos(theta_mesh)
    app_nx = cos_theta * np.cos(phi_mesh)
    app_ny = cos_theta * np.sin(phi_mesh)
    app_nz = np.sin(theta_mesh)
    
    app_n = np.stack([app_nx, app_ny, app_nz], axis=-1)  # shape (height, width, 3)

    if include_cmb and beta_mag > 0.0:
        # Doppler shift of CMB background at every pixel
        inertial_n = inverse_aberration_vector(app_n.reshape(-1, 3), beta).reshape(height, width, 3)
        d_grid = compute_doppler_factor(inertial_n.reshape(-1, 3), beta).reshape(height, width)
        
        # Boosted temperature T' = D * T_CMB
        t_boosted_grid = cmb_temp_k * d_grid
        
        # When beta > 0.9999, T' enters the optical band (> 1500 K)
        high_mask = t_boosted_grid > 1500.0
        if np.any(high_mask):
            temps_to_eval = t_boosted_grid[high_mask]
            rad = planck_spectral_radiance(
                CIE_WAVELENGTHS_M[None, :],
                temps_to_eval[:, None],
            )
            xyz_vals, _ = spectrum_to_cie_xyz(CIE_WAVELENGTHS_M, rad)
            image_xyz[high_mask] += xyz_vals * 1e-10

    # Project catalog stars onto panorama
    if len(stars) > 0:
        cat_dirs = np.array([s.direction for s in stars], dtype=np.float64)
        app_dirs = transform_aberration_vector(cat_dirs, beta)
        doppler_factors = compute_doppler_factor(cat_dirs, beta)

        # Spherical coordinates of apparent stars
        star_theta = np.arcsin(np.clip(app_dirs[:, 2], -1.0, 1.0))
        star_phi = np.arctan2(app_dirs[:, 1], app_dirs[:, 0])

        # Map to pixel indices
        pix_u = ((star_phi + np.pi) / (2.0 * np.pi) * width) % width
        pix_v = (np.pi * 0.5 - star_theta) / np.pi * (height - 1)

        sigma_pix = 1.0
        for i, star in enumerate(stars):
            u_center = pix_u[i]
            v_center = pix_v[i]
            d_val = float(doppler_factors[i])
            t_boosted = star.temperature_k * d_val
            
            base_intensity = 10.0 ** (-0.4 * star.visual_magnitude)
            star_flux = base_intensity * (d_val ** 4)

            rad = planck_spectral_radiance(CIE_WAVELENGTHS_M, t_boosted)
            xyz, _ = spectrum_to_cie_xyz(CIE_WAVELENGTHS_M, rad)
            if xyz[1] > 0.0:
                xyz_scaled = (xyz / xyz[1]) * star_flux
            else:
                xyz_scaled = np.zeros(3, dtype=np.float64)

            margin = 3
            u_min = int(math.floor(u_center - margin))
            u_max = int(math.ceil(u_center + margin + 1))
            v_min = max(0, int(math.floor(v_center - margin)))
            v_max = min(height, int(math.ceil(v_center + margin + 1)))

            for vv in range(v_min, v_max):
                for uu in range(u_min, u_max):
                    u_wrapped = uu % width
                    d_sq = (uu - u_center) ** 2 + (vv - v_center) ** 2
                    weight = math.exp(-0.5 * d_sq / (sigma_pix ** 2))
                    image_xyz[vv, u_wrapped] += weight * xyz_scaled

    return cie_xyz_to_srgb(image_xyz, exposure=exposure, tone_map=True, gamma=True)
