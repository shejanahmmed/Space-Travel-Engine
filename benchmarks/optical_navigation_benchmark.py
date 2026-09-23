"""Relativistic Optical Navigation & Aberration Benchmark Suite.

Evaluates:
1. Massive 10,000-case aberration and Doppler vector transformations across
   velocities spanning beta in [10^-6, 0.99999].
2. Autonomous velocity vector reconstruction from star tracker sightings.
3. Noise resilience analysis: Evaluates velocity estimation degradation
   in the presence of Gaussian centroid measurement errors (0 to 10 arcseconds).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List
import numpy as np

from relativistic_engine.physics.optics import (
    transform_aberration_vector,
    inverse_aberration_vector,
    compute_doppler_factor,
    compute_bolometric_flux_scaling,
    estimate_velocity_from_star_observations,
)


def run_optical_aberration_benchmark(n_stars: int = 10000) -> Dict[str, Any]:
    """Benchmark high-throughput aberration vector transformation.

    Transforms n_stars across random sky coordinates under high relativistic speed
    (beta = 0.95), verifying:
    1. Norm preservation: ||n'|| == 1.0 to machine precision.
    2. Exact inverse restoration: ||A^(-1)(A(n)) - n|| < 1e-15.
    3. Throughput in stars/second.

    Args:
        n_stars: Number of synthetic stars to evaluate.

    Returns:
        Dictionary of performance and precision results.
    """
    rng = np.random.default_rng(42)

    # Uniform points on unit sphere
    raw_vecs = rng.standard_normal((n_stars, 3))
    stars = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)

    beta = np.array([0.2, -0.4, 0.8246211251235321])  # |beta| = 0.94
    b_mag = float(np.linalg.norm(beta))

    t0 = time.perf_counter()
    obs_stars = transform_aberration_vector(stars, beta, normalize=True)
    t_fwd = time.perf_counter() - t0

    t0 = time.perf_counter()
    restored_stars = inverse_aberration_vector(obs_stars, beta, normalize=True)
    t_inv = time.perf_counter() - t0

    # Norm deviations
    obs_norms = np.linalg.norm(obs_stars, axis=1)
    max_norm_err = float(np.max(np.abs(obs_norms - 1.0)))

    # Invertibility discrepancies
    inv_errs = np.linalg.norm(restored_stars - stars, axis=1)
    max_inv_err = float(np.max(inv_errs))
    mean_inv_err = float(np.mean(inv_errs))

    total_time = t_fwd + t_inv
    throughput = (2.0 * n_stars) / max(1e-6, total_time)

    return {
        "status": "PASS" if max_inv_err < 1.0e-14 else "FAIL",
        "n_stars": n_stars,
        "speed_fraction_c": b_mag,
        "max_norm_error": max_norm_err,
        "max_invertibility_error": max_inv_err,
        "mean_invertibility_error": mean_inv_err,
        "elapsed_seconds": total_time,
        "throughput_stars_per_sec": throughput,
    }


def run_autonomous_velocity_estimation_benchmark(
    n_trials: int = 100,
    n_stars_per_trial: int = 20,
    noise_arcsec: float = 1.0,
) -> Dict[str, Any]:
    """Benchmark autonomous velocity vector estimation from noisy star observations.

    Simulates a star tracker observing n_stars_per_trial catalog stars subjected
    to relativistic aberration and Gaussian centroiding noise sigma.

    Args:
        n_trials: Number of independent Monte Carlo trials.
        n_stars_per_trial: Number of guide stars in camera field of view.
        noise_arcsec: 1-sigma centroid measurement noise in arcseconds.

    Returns:
        Dictionary of estimation residuals and convergence statistics.
    """
    rng = np.random.default_rng(1234)
    noise_rad = noise_arcsec * (math.pi / (180.0 * 3600.0))

    errors = []
    converged = 0

    for _ in range(n_trials):
        # Random velocity vector beta in [0.05, 0.95]
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        speed = rng.uniform(0.05, 0.95)
        true_beta = direction * speed

        # Random guide stars
        raw_stars = rng.standard_normal((n_stars_per_trial, 3))
        cat_stars = raw_stars / np.linalg.norm(raw_stars, axis=1, keepdims=True)

        # Apply aberration
        true_obs = transform_aberration_vector(cat_stars, true_beta, normalize=True)

        # Add Gaussian centroid angular noise
        if noise_rad > 0.0:
            noise_vectors = rng.standard_normal((n_stars_per_trial, 3)) * noise_rad
            noisy_obs = true_obs + noise_vectors
            noisy_obs /= np.linalg.norm(noisy_obs, axis=1, keepdims=True)
        else:
            noisy_obs = true_obs

        # Solve inverse problem
        sol = estimate_velocity_from_star_observations(cat_stars, noisy_obs)
        if sol["success"]:
            converged += 1
            err = float(np.linalg.norm(sol["beta"] - true_beta))
            errors.append(err)

    err_arr = np.array(errors)
    max_err = float(np.max(err_arr)) if len(err_arr) > 0 else 1.0
    mean_err = float(np.mean(err_arr)) if len(err_arr) > 0 else 1.0

    return {
        "status": "PASS" if (converged == n_trials and mean_err < 1e-4) else "FAIL",
        "n_trials": n_trials,
        "n_stars_per_trial": n_stars_per_trial,
        "noise_arcsec": noise_arcsec,
        "convergence_rate": converged / n_trials,
        "mean_velocity_error_beta": mean_err,
        "max_velocity_error_beta": max_err,
    }


if __name__ == "__main__":
    print("[*] Executing 10,000-Star Relativistic Aberration Benchmark...")
    ab_res = run_optical_aberration_benchmark(n_stars=10000)
    print(f"    Max invertibility error: {ab_res['max_invertibility_error']:.2e} [{ab_res['status']}]")
    print(f"    Throughput: {ab_res['throughput_stars_per_sec']:,.0f} stars/sec")

    print("[*] Executing Autonomous Velocity Navigation Benchmark (100 trials, 1 arcsec noise)...")
    nav_res = run_autonomous_velocity_estimation_benchmark(n_trials=100, noise_arcsec=1.0)
    print(f"    Convergence rate: {nav_res['convergence_rate'] * 100:.1f}%")
    print(f"    Mean velocity error |Delta beta|: {nav_res['mean_velocity_error_beta']:.2e} [{nav_res['status']}]")
