"""High-Throughput Vectorized Monte Carlo Uncertainty Engine.

Executes massive stochastic dispersion sweeps (N = 10^3 to 10^5 samples)
conforming to JCGM 101:2008 (Evaluation of measurement data - Supplement 1
to the "Guide to the expression of uncertainty in measurement" - Propagation
of distributions using a Monte Carlo method).

Couples initial state uncertainty covariance P0 with the vectorized batch
1PN propagator to compute empirical arrival dispersion, covariance matrices,
and confidence intervals at high throughput.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Tuple
import numpy as np

from relativistic_engine.numerical.batch_propagator import (
    propagate_batch_worldlines,
    BatchTrajectoryResult,
)


@dataclass(frozen=True)
class BatchMonteCarloResult:
    """Statistical summary of a high-throughput Monte Carlo dispersion sweep.

    Attributes
    ----------
    nominal_final_state : np.ndarray
        Nominal unperturbed final state [x, y, z, vx, vy, vz, Delta], shape (7,).
    mean_final_state : np.ndarray
        Empirical ensemble mean final state, shape (7,).
    cov_final_state : np.ndarray
        Empirical 7x7 covariance matrix of final states.
    std_final_state : np.ndarray
        Sample standard deviations for each state component, shape (7,).
    position_dispersion_3sigma_m : float
        Scalar 3-sigma position dispersion bound (meters).
    velocity_dispersion_3sigma_mps : float
        Scalar 3-sigma velocity dispersion bound (m/s).
    n_samples : int
        Number of Monte Carlo realizations evaluated.
    elapsed_wall_time_sec : float
        Wall-clock time required for complete ensemble propagation (seconds).
    throughput_trajectories_per_sec : float
        Ensemble integration throughput (trajectories/sec).
    backend_used : str
        Execution backend ('numpy_cpu' or 'torch_cuda').
    """

    nominal_final_state: np.ndarray
    mean_final_state: np.ndarray
    cov_final_state: np.ndarray
    std_final_state: np.ndarray
    position_dispersion_3sigma_m: float
    velocity_dispersion_3sigma_mps: float
    n_samples: int
    elapsed_wall_time_sec: float
    throughput_trajectories_per_sec: float
    backend_used: str


def evaluate_batch_monte_carlo(
    r_nominal: np.ndarray,
    v_nominal: np.ndarray,
    cov_initial_6x6: np.ndarray,
    t_span: Tuple[float, float],
    n_samples: int = 10000,
    h_step_sec: float = 3600.0,
    seed: int = 42,
    backend: str = "auto",
) -> BatchMonteCarloResult:
    """Evaluate massive Monte Carlo ensemble dispersion using vectorized batch execution.

    Parameters
    ----------
    r_nominal : np.ndarray, shape (3,)
        Nominal initial position in BCRS (meters).
    v_nominal : np.ndarray, shape (3,)
        Nominal initial velocity in BCRS (m/s).
    cov_initial_6x6 : np.ndarray, shape (6, 6)
        Initial state covariance matrix for [x, y, z, vx, vy, vz].
    t_span : tuple of (float, float)
        Integration interval (t0, tf) in seconds.
    n_samples : int, optional
        Number of Monte Carlo realizations (default: 10000).
    h_step_sec : float, optional
        Integration step size in seconds (default: 3600.0 s).
    seed : int, optional
        Pseudorandom seed for reproducible sampling (default: 42).
    backend : str, optional
        Execution backend ('auto', 'numpy_cpu', or 'torch_cuda').

    Returns
    -------
    BatchMonteCarloResult
        Statistical dispersion metrics, empirical covariance, and performance benchmarks.
    """
    r0 = np.asarray(r_nominal, dtype=np.float64).flatten()
    v0 = np.asarray(v_nominal, dtype=np.float64).flatten()
    p0 = np.asarray(cov_initial_6x6, dtype=np.float64)

    if p0.shape != (6, 6):
        raise ValueError(f"Initial covariance matrix must be 6x6, got shape {p0.shape}.")

    # Generate N multivariate Gaussian perturbations
    rng = np.random.default_rng(seed=seed)
    mean_zero = np.zeros(6, dtype=np.float64)
    perturbations = rng.multivariate_normal(mean_zero, p0, size=n_samples)

    # State ensemble: shape (N, 3)
    r_batch = r0 + perturbations[:, :3]
    v_batch = v0 + perturbations[:, 3:6]
    tau_batch = np.zeros(n_samples, dtype=np.float64)

    # Record wall-clock start
    t_start = time.perf_counter()

    # Propagate ensemble in lockstep
    batch_res = propagate_batch_worldlines(
        r_initial=r_batch,
        v_initial=v_batch,
        tau_initial=tau_batch,
        t_span=t_span,
        h_step_sec=h_step_sec,
        backend=backend,
    )

    t_end = time.perf_counter()
    elapsed_time = max(1e-6, t_end - t_start)
    throughput = n_samples / elapsed_time

    # Extract final states for all N particles: shape (N, 7)
    # [x, y, z, vx, vy, vz, Delta]
    r_final = batch_res.r[-1]  # (N, 3)
    v_final = batch_res.v[-1]  # (N, 3)
    def_final = batch_res.time_deficit[-1].reshape(n_samples, 1)  # (N, 1)

    y_final = np.hstack([r_final, v_final, def_final])  # (N, 7)

    # Compute empirical statistics (JCGM 101:2008)
    mean_final = np.mean(y_final, axis=0)  # (7,)
    cov_final = np.cov(y_final, rowvar=False)  # (7, 7)
    std_final = np.sqrt(np.diag(cov_final))  # (7,)

    # Nominal propagation (sample 0 without perturbation or dedicated single run)
    nom_res = propagate_batch_worldlines(
        r_initial=r0.reshape(1, 3),
        v_initial=v0.reshape(1, 3),
        tau_initial=np.zeros(1),
        t_span=t_span,
        h_step_sec=h_step_sec,
        backend="numpy_cpu",
    )
    nom_final = np.hstack([
        nom_res.r[-1].flatten(),
        nom_res.v[-1].flatten(),
        nom_res.time_deficit[-1].flatten(),
    ])

    # 3-sigma scalar dispersion bounds
    pos_var = np.trace(cov_final[:3, :3])
    vel_var = np.trace(cov_final[3:6, 3:6])
    pos_3sigma = 3.0 * math.sqrt(max(0.0, pos_var))
    vel_3sigma = 3.0 * math.sqrt(max(0.0, vel_var))

    return BatchMonteCarloResult(
        nominal_final_state=nom_final,
        mean_final_state=mean_final,
        cov_final_state=cov_final,
        std_final_state=std_final,
        position_dispersion_3sigma_m=pos_3sigma,
        velocity_dispersion_3sigma_mps=vel_3sigma,
        n_samples=n_samples,
        elapsed_wall_time_sec=elapsed_time,
        throughput_trajectories_per_sec=throughput,
        backend_used=batch_res.backend_used,
    )
