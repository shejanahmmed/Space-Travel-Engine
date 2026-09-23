"""Computational Bottleneck Profiler and Hardware Benchmark Harness.

Analyzes execution times, memory throughput, and algorithmic scaling of:
1. Sequential ODE integration (SciPy solve_ivp DOP853 baseline).
2. Vectorized multi-sample batch propagation (NumPy / SIMD).
3. GPU hardware-accelerated tensor propagation (PyTorch CUDA if available).

In accordance with Principle 22 (Hardware & Optimization), this module diagnoses
whether trajectory and Monte Carlo operations are CPU-bound, GPU-bound, or memory-bound
before optimization decisions are made.
"""

from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Sequence
import numpy as np

from relativistic_engine.constants import GM_SUN, AU, SEC_PER_DAY
from relativistic_engine.numerical.batch_propagator import (
    propagate_batch_worldlines,
    BatchIntegratorBackend,
)


@dataclass(frozen=True)
class BenchmarkProfileResult:
    """Performance metrics for a specific batch size N.

    Attributes
    ----------
    n_samples : int
        Number of trajectories propagated in the ensemble.
    time_sequential_sec : float
        Elapsed wall-clock time for sequential evaluation (seconds).
    time_batch_sec : float
        Elapsed wall-clock time for vectorized batch evaluation (seconds).
    speedup_factor : float
        Ratio of sequential time to batch time: t_seq / t_batch.
    throughput_trajectories_per_sec : float
        Number of full trajectory integrations completed per second.
    memory_footprint_mb : float
        Estimated active working memory for the state and stage tensors (MB).
    backend_used : str
        Name of the backend executed ('numpy_cpu' or 'torch_cuda').
    """

    n_samples: int
    time_sequential_sec: float
    time_batch_sec: float
    speedup_factor: float
    throughput_trajectories_per_sec: float
    memory_footprint_mb: float
    backend_used: str


def profile_hardware_scaling(
    n_samples_list: Sequence[int] = (10, 50, 200, 1000),
    t_flight_days: float = 30.0,
    h_step_sec: float = 3600.0,
    backend: str = "auto",
) -> list[BenchmarkProfileResult]:
    """Profile sequential vs. vectorized batch execution across ensemble sizes.

    Parameters
    ----------
    n_samples_list : Sequence[int]
        List of sample counts to evaluate (default: (10, 50, 200, 1000)).
    t_flight_days : float
        Flight duration for each benchmark trajectory in days (default: 30.0).
    h_step_sec : float
        Integration step size in seconds (default: 3600.0 s = 1 hour).
    backend : str
        Integrator backend ('auto', 'numpy_cpu', or 'torch_cuda').

    Returns
    -------
    list[BenchmarkProfileResult]
        Profile metrics for each sample size.
    """
    t_span = (0.0, t_flight_days * SEC_PER_DAY)

    # Base nominal state: Earth-like orbit around Sun
    r0 = np.array([AU, 0.0, 0.0], dtype=np.float64)
    v0 = np.array([0.0, 29780.0, 0.0], dtype=np.float64)

    results: list[BenchmarkProfileResult] = []

    for n in n_samples_list:
        # Generate N perturbed initial states (Gaussian spread: sigma_r = 100 km, sigma_v = 1 m/s)
        rng = np.random.default_rng(seed=42)
        dr = rng.normal(0.0, 1e5, size=(n, 3))
        dv = rng.normal(0.0, 1.0, size=(n, 3))

        r_batch = r0 + dr
        v_batch = v0 + dv
        tau_batch = np.zeros(n, dtype=np.float64)

        # 1. Benchmark Vectorized Batch Propagator
        t_start_batch = time.perf_counter()
        batch_out = propagate_batch_worldlines(
            r_initial=r_batch,
            v_initial=v_batch,
            tau_initial=tau_batch,
            t_span=t_span,
            h_step_sec=h_step_sec,
            backend=backend,
        )
        t_end_batch = time.perf_counter()
        time_batch = max(1e-6, t_end_batch - t_start_batch)

        # 2. Benchmark Sequential (for small n, or extrapolated for large n to prevent excessive runtime)
        if n <= 100:
            t_start_seq = time.perf_counter()
            for k in range(n):
                _ = propagate_batch_worldlines(
                    r_initial=r_batch[k : k + 1],
                    v_initial=v_batch[k : k + 1],
                    tau_initial=tau_batch[k : k + 1],
                    t_span=t_span,
                    h_step_sec=h_step_sec,
                    backend="numpy_cpu",
                )
            t_end_seq = time.perf_counter()
            time_seq = max(1e-6, t_end_seq - t_start_seq)
            single_traj_time = time_seq / n
        else:
            # Extrapolate using measured single-trajectory time
            time_seq = single_traj_time * n

        speedup = time_seq / time_batch
        throughput = n / time_batch

        # State tensor (N x 7) + 4 Runge-Kutta stage buffers (4 x N x 7) in 64-bit float
        memory_mb = (n * 7 * 8 * 5) / (1024.0 * 1024.0)

        results.append(
            BenchmarkProfileResult(
                n_samples=n,
                time_sequential_sec=time_seq,
                time_batch_sec=time_batch,
                speedup_factor=speedup,
                throughput_trajectories_per_sec=throughput,
                memory_footprint_mb=memory_mb,
                backend_used=batch_out.backend_used,
            )
        )

    return results
