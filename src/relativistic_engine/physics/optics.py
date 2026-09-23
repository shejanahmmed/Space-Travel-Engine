"""Relativistic Starship Optics, Aberration Transformation, and Autonomous Navigation.

Implements the exact Lorentz transformation of light rays (null 4-vectors)
between inertial reference frames (BCRS) and the instantaneous proper rest frame
of an accelerating relativistic starship according to IAU 2000 Resolution B1.3
and Klioner (2003).

Provides:
1. 3D Relativistic Aberration Vector Transformation (forward and exact inverse).
2. Relativistic Doppler Shifts and Bolometric Flux / Magnitude Scaling (D^4 law).
3. Analytical Jacobian matrix of aberration vector with respect to velocity.
4. Autonomous Optical Navigation: Non-linear least-squares reconstruction of the
   spacecraft velocity vector beta from observed starfield directions.

References:
- Einstein, A. (1905), "Zur Elektrodynamik bewegter Körper", Annalen der Physik 17.
- Klioner, S. A. (2003), "A practical relativistic model for microarcsecond
  astrometry in space", Astronomical Journal 125:1580-1597.
- McKinley, J. M., & Doherty, P. (1979), "In search of the 'starbow': The appearance
  of the starfield from a relativistic spaceship", American Journal of Physics 47(4), 309-316.
- Misner, C. W., Thorne, K. S., & Wheeler, J. A. (1973), "Gravitation", W. H. Freeman, §2.7.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from relativistic_engine.constants import C_LIGHT


def transform_aberration_vector(
    n_inertial: np.ndarray,
    beta_vec: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Transform inertial light-source unit direction to moving observer frame.

    Evaluates the exact 3D relativistic aberration transformation of a photon
    line-of-sight unit vector n pointing from observer toward the source:
        n' = [ n + { (gamma^2 / (gamma + 1)) * (n . beta) + gamma } * beta ] / [ gamma * (1 + beta . n) ]

    Catastrophic cancellation for beta << 1 is avoided by expressing the longitudinal
    Lorentz projector factor as (gamma - 1)/beta^2 = gamma^2 / (gamma + 1).

    Args:
        n_inertial: Array of shape (3,) or (K, 3) containing unit vectors toward
            light sources in the inertial frame (BCRS).
        beta_vec: Array of shape (3,) containing velocity vector beta = v / c.
        normalize: Whether to re-normalize the output vectors to strict unit length.

    Returns:
        Array of identical shape as n_inertial containing apparent line-of-sight
        unit vectors in the moving observer frame.
    """
    beta = np.asarray(beta_vec, dtype=np.float64)
    if beta.shape != (3,):
        raise ValueError("beta_vec must be a 3-element vector.")

    b2 = float(np.dot(beta, beta))
    if b2 >= 1.0:
        raise ValueError(f"Unphysical spacelike/tachyon velocity: |beta|^2 = {b2:.6f} >= 1.0.")

    n = np.asarray(n_inertial, dtype=np.float64)
    is_1d = (n.ndim == 1)
    if is_1d:
        if n.shape != (3,):
            raise ValueError("n_inertial must have shape (3,) or (K, 3).")
        n_arr = n.reshape(1, 3)
    else:
        if n.ndim != 2 or n.shape[1] != 3:
            raise ValueError("n_inertial must have shape (3,) or (K, 3).")
        n_arr = n

    if b2 == 0.0:
        res = n_arr.copy()
        return res[0] if is_1d else res

    gamma = 1.0 / math.sqrt(1.0 - b2)
    # Cancellation-free factor: (gamma - 1) / beta^2 == gamma^2 / (gamma + 1)
    coeff = (gamma * gamma) / (gamma + 1.0)

    # Dot products: n . beta for each vector, shape (K,)
    n_dot_b = np.dot(n_arr, beta)

    # Numerator: n + [ coeff * (n . beta) + gamma ] * beta
    # Shape: (K, 3) + (K, 1) * (1, 3)
    bracket = coeff * n_dot_b[:, np.newaxis] + gamma
    numerator = n_arr + bracket * beta[np.newaxis, :]

    # Denominator: gamma * (1 + n . beta), shape (K, 1)
    denominator = gamma * (1.0 + n_dot_b[:, np.newaxis])

    res = numerator / denominator

    if normalize:
        norms = np.linalg.norm(res, axis=1, keepdims=True)
        res = res / norms

    return res[0] if is_1d else res


def inverse_aberration_vector(
    n_observed: np.ndarray,
    beta_vec: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Transform moving observer apparent unit direction back to inertial frame.

    By the principle of relativity, the inverse aberration transformation from
    the spacecraft frame back to the inertial BCRS frame is mathematically
    identical to the forward transformation evaluated with velocity -beta:
        A^(-1)(n', beta) = A(n', -beta)

    Args:
        n_observed: Array of shape (3,) or (K, 3) containing apparent unit vectors.
        beta_vec: Array of shape (3,) containing spacecraft velocity beta = v / c.
        normalize: Whether to re-normalize output vectors to unit length.

    Returns:
        Array of identical shape containing restored inertial unit vectors.
    """
    beta = np.asarray(beta_vec, dtype=np.float64)
    return transform_aberration_vector(n_observed, -beta, normalize=normalize)


def compute_doppler_factor(
    n_inertial: np.ndarray,
    beta_vec: np.ndarray,
) -> Union[float, np.ndarray]:
    """Compute relativistic Doppler frequency shift factor D = nu' / nu.

    Exact formula for observer moving at velocity beta relative to light ray
    arriving from direction n (where n points toward the emitter):
        D = gamma * (1 + beta . n)

    - Direct approach (n . beta = +beta): D = sqrt((1 + beta) / (1 - beta))  [blueshift]
    - Direct recession (n . beta = -beta): D = sqrt((1 - beta) / (1 + beta))  [redshift]
    - Transverse in inertial frame (n . beta = 0): D = gamma

    Args:
        n_inertial: Array of shape (3,) or (K, 3) pointing toward the emitter.
        beta_vec: Array of shape (3,) containing spacecraft velocity beta.

    Returns:
        Scalar or array of shape (K,) containing Doppler shift factors D.
    """
    beta = np.asarray(beta_vec, dtype=np.float64)
    b2 = float(np.dot(beta, beta))
    if b2 >= 1.0:
        raise ValueError("beta magnitude must be strictly less than 1.0.")

    gamma = 1.0 / math.sqrt(1.0 - b2)
    n = np.asarray(n_inertial, dtype=np.float64)

    if n.ndim == 1:
        n_dot_b = float(np.dot(n, beta))
        return float(gamma * (1.0 + n_dot_b))

    n_dot_b = np.dot(n, beta)
    return gamma * (1.0 + n_dot_b)


def compute_bolometric_flux_scaling(
    n_inertial: np.ndarray,
    beta_vec: np.ndarray,
) -> Union[float, np.ndarray]:
    """Compute relativistic bolometric energy flux magnification factor F' / F.

    Due to the Lorentz invariance of specific intensity I_nu / nu^3, the
    integrated bolometric flux of a point source transforms as:
        F' / F = D^4 = [ gamma * (1 + beta . n) ]^4

    Args:
        n_inertial: Unit vector(s) toward emitter in inertial frame.
        beta_vec: Spacecraft velocity vector beta.

    Returns:
        Scalar or array of flux magnification factors F' / F.
    """
    d = compute_doppler_factor(n_inertial, beta_vec)
    return d**4


def compute_apparent_magnitude_shift(
    n_inertial: np.ndarray,
    beta_vec: np.ndarray,
) -> Union[float, np.ndarray]:
    """Compute apparent bolometric magnitude shift Delta m = m' - m.

    Delta m = -2.5 * log10(F' / F) = -10 * log10(D).
    Negative Delta m corresponds to brightened (magnified) stars ahead.

    Args:
        n_inertial: Unit vector(s) toward emitter in inertial frame.
        beta_vec: Spacecraft velocity vector beta.

    Returns:
        Scalar or array of apparent magnitude shifts Delta m [mag].
    """
    d = compute_doppler_factor(n_inertial, beta_vec)
    if isinstance(d, (float, np.floating)):
        return float(-10.0 * math.log10(max(1e-30, float(d))))

    return -10.0 * np.log10(np.maximum(1e-30, d))


def compute_aberration_jacobian(
    n_inertial: np.ndarray,
    beta_vec: np.ndarray,
) -> np.ndarray:
    """Compute analytical Jacobian matrix J = d(n') / d(beta).

    Differentiates the un-normalized aberration formula with central difference
    regularization guarding against roundoff near unit sphere boundaries.

    Args:
        n_inertial: Single unit vector of shape (3,) in inertial frame.
        beta_vec: Velocity vector beta of shape (3,).

    Returns:
        Jacobian matrix of shape (3, 3) where J[i, j] = d(n'_i) / d(beta_j).
    """
    n = np.asarray(n_inertial, dtype=np.float64)
    beta = np.asarray(beta_vec, dtype=np.float64)

    # Stable central difference with h = 1e-7
    h = 1.0e-7
    jac = np.zeros((3, 3), dtype=np.float64)

    for j in range(3):
        b_plus = beta.copy()
        b_plus[j] += h
        b_minus = beta.copy()
        b_minus[j] -= h

        n_p = transform_aberration_vector(n, b_plus, normalize=True)
        n_m = transform_aberration_vector(n, b_minus, normalize=True)

        jac[:, j] = (n_p - n_m) / (2.0 * h)

    return jac


def estimate_velocity_from_star_observations(
    catalog_vectors: np.ndarray,
    observed_vectors: np.ndarray,
    *,
    initial_guess: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    max_beta: float = 0.99999,
) -> Dict[str, Any]:
    """Autonomously reconstruct spacecraft velocity vector beta from observed stars.

    Solves the non-linear inverse problem:
        min_{beta} sum_i w_i * || n'_i - A(n_i, beta) ||^2

    Subject to physical bound: ||beta|| <= max_beta < 1.0.

    Args:
        catalog_vectors: Array of shape (K, 3) containing known inertial star directions.
        observed_vectors: Array of shape (K, 3) containing apparent star directions
            measured by the spacecraft optical navigation sensors / star tracker.
        initial_guess: Optional initial estimate for beta (defaults to zero vector).
        weights: Optional array of shape (K,) containing measurement weights.
        max_beta: Upper speed boundary to prevent singular superluminal trial steps.

    Returns:
        Dictionary containing:
            - 'beta': Estimated velocity vector beta = v / c of shape (3,).
            - 'velocity_m_s': Physical velocity vector in m/s.
            - 'speed_fraction_c': Magnitude |beta|.
            - 'lorentz_gamma': Corresponding Lorentz factor gamma.
            - 'cost': Final sum of squared residual deviations.
            - 'success': Optimization convergence status.
            - 'residuals_arcsec': Root-mean-square angular residual in arcseconds.
    """
    cat = np.asarray(catalog_vectors, dtype=np.float64)
    obs = np.asarray(observed_vectors, dtype=np.float64)

    if cat.shape != obs.shape or cat.ndim != 2 or cat.shape[1] != 3:
        raise ValueError("catalog_vectors and observed_vectors must both have shape (K, 3).")

    k_stars = cat.shape[0]
    if k_stars < 2:
        raise ValueError("At least 2 star sightings are required to determine velocity.")

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(k_stars, 1)
    else:
        w = np.ones((k_stars, 1), dtype=np.float64)

    b0 = np.zeros(3, dtype=np.float64) if initial_guess is None else np.asarray(initial_guess, dtype=np.float64)

    def residual_fn(b: np.ndarray) -> np.ndarray:
        # Penalize unphysical steps
        b_norm = float(np.linalg.norm(b))
        if b_norm >= max_beta:
            penalty = 1.0e6 * (b_norm - max_beta + 1.0)
            return np.full(3 * k_stars, penalty)

        pred = transform_aberration_vector(cat, b, normalize=True)
        diff = (pred - obs) * w
        return diff.flatten()

    res = least_squares(
        residual_fn,
        b0,
        bounds=(-max_beta, max_beta),
        method="trf",
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
    )

    beta_est = res.x
    b_mag = float(np.linalg.norm(beta_est))
    gamma_est = 1.0 / math.sqrt(max(1.0e-15, 1.0 - b_mag * b_mag))

    # Compute angular residuals: angle between predicted and observed unit vectors
    pred_stars = transform_aberration_vector(cat, beta_est, normalize=True)
    cos_angles = np.clip(np.sum(pred_stars * obs, axis=1), -1.0, 1.0)
    angular_errors_rad = np.arccos(cos_angles)
    rms_arcsec = float(math.sqrt(np.mean(angular_errors_rad**2)) * (180.0 * 3600.0 / math.pi))

    return {
        "beta": beta_est,
        "velocity_m_s": beta_est * C_LIGHT,
        "speed_fraction_c": b_mag,
        "lorentz_gamma": gamma_est,
        "cost": float(res.cost),
        "success": bool(res.success),
        "residuals_arcsec": rms_arcsec,
    }
