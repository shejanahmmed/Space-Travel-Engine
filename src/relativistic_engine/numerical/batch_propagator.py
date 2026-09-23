"""High-Throughput Vectorized Batch Propagator for Relativistic Worldlines.

Propagates an ensemble of N spacecraft trajectories simultaneously in BCRS,
coupling:
- Vectorized 1PN post-Newtonian gravitational acceleration.
- Vectorized metric proper time and stabilized coordinate time deficit integration.
- Dual-backend execution: Multi-threaded NumPy CPU and optional PyTorch CUDA GPU.
- High-order Runge-Kutta integration in IEEE 754 float64 double precision.

Authoritative References:
- Soffel, M., et al. (2003), "The IAU 2000 Resolutions for Astrometry, Celestial Mechanics
  and Metrology in the Relativistic Framework", The Astronomical Journal, 126:2687-2706.
- Dormand, J. R., & Prince, P. J. (1980), "A family of embedded Runge-Kutta formulae",
  Journal of Computational and Applied Mathematics, 6(1):19-26.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence, Tuple
import numpy as np

from relativistic_engine.constants import C_LIGHT, GM_SUN


class BatchIntegratorBackend(str, Enum):
    """Available execution backends for batch propagation."""

    AUTO = "auto"
    NUMPY_CPU = "numpy_cpu"
    TORCH_CUDA = "torch_cuda"


@dataclass(frozen=True)
class BatchTrajectoryResult:
    """Ensemble trajectory output for N particles.

    Attributes
    ----------
    t : np.ndarray
        1D array of coordinate time steps (seconds), shape (M,).
    r : np.ndarray
        Position tensor in meters, shape (M, N, 3).
    v : np.ndarray
        Velocity tensor in m/s, shape (M, N, 3).
    tau : np.ndarray
        Proper time tensor in seconds, shape (M, N).
    time_deficit : np.ndarray
        Time deficit tensor Delta(t) = t - tau in seconds, shape (M, N).
    n_samples : int
        Number of trajectories in the ensemble.
    backend_used : str
        The actual backend executed ('numpy_cpu' or 'torch_cuda').
    """

    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    tau: np.ndarray
    time_deficit: np.ndarray
    n_samples: int
    backend_used: str


def _rhs_numpy(
    r: np.ndarray,
    v: np.ndarray,
    mu: float = GM_SUN,
    c2: float = C_LIGHT**2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute vectorized equations of motion in NumPy (float64).

    Parameters
    ----------
    r : np.ndarray, shape (N, 3)
        Positions of N particles in meters.
    v : np.ndarray, shape (N, 3)
        Velocities of N particles in m/s.
    mu : float
        Central gravitational parameter (m^3/s^2).
    c2 : float
        c^2 in m^2/s^2.

    Returns
    -------
    dr_dt : np.ndarray, shape (N, 3)
    dv_dt : np.ndarray, shape (N, 3)
    d_delta_dt : np.ndarray, shape (N, 1)
    """
    # Magnitudes via vector reduction along axis 1
    r_sq = np.sum(r**2, axis=1, keepdims=True)
    r_norm = np.sqrt(r_sq)
    v_sq = np.sum(v**2, axis=1, keepdims=True)
    r_dot_v = np.sum(r * v, axis=1, keepdims=True)

    # 1. Newtonian gravitational acceleration
    a_newton = -(mu / (r_norm * r_sq)) * r

    # 2. 1PN Post-Newtonian acceleration (EIH metric)
    factor_1pn = mu / (c2 * r_norm * r_sq)
    term_pos = (4.0 * mu / r_norm - v_sq) * r
    term_vel = (4.0 * r_dot_v) * v
    a_1pn = factor_1pn * (term_pos + term_vel)

    a_total = a_newton + a_1pn

    # 3. Stabilized time deficit rate: d(Delta)/dt
    w = mu / r_norm
    X = (2.0 * w / c2) * (1.0 + v_sq / c2) + (v_sq / c2)
    # Clip X < 1.0 to prevent domain errors
    X_safe = np.minimum(X, 1.0 - 1e-14)
    d_delta_dt = X_safe / (1.0 + np.sqrt(np.maximum(1e-15, 1.0 - X_safe)))

    return v, a_total, d_delta_dt


def propagate_batch_worldlines(
    r_initial: np.ndarray,
    v_initial: np.ndarray,
    tau_initial: np.ndarray | None = None,
    t_span: Tuple[float, float] = (0.0, 86400.0),
    h_step_sec: float = 3600.0,
    mu: float = GM_SUN,
    backend: str = "auto",
) -> BatchTrajectoryResult:
    """Propagate N relativistic trajectories simultaneously in BCRS.

    Parameters
    ----------
    r_initial : np.ndarray, shape (N, 3)
        Initial Cartesian positions in meters.
    v_initial : np.ndarray, shape (N, 3)
        Initial Cartesian velocities in m/s.
    tau_initial : np.ndarray, optional, shape (N,)
        Initial proper times in seconds (default: zeros).
    t_span : tuple of (float, float)
        Integration time interval (t0, tf) in seconds.
    h_step_sec : float
        Integration step size in seconds (default: 3600.0 s).
    mu : float, optional
        Central gravitational parameter (default: GM_SUN).
    backend : str, optional
        Execution backend ('auto', 'numpy_cpu', or 'torch_cuda').

    Returns
    -------
    BatchTrajectoryResult
        Ensemble worldlines and proper times.
    """
    r0 = np.asarray(r_initial, dtype=np.float64)
    v0 = np.asarray(v_initial, dtype=np.float64)

    if r0.ndim == 1:
        r0 = r0.reshape(1, 3)
    if v0.ndim == 1:
        v0 = v0.reshape(1, 3)

    n_samples = r0.shape[0]
    tau0 = (
        np.asarray(tau_initial, dtype=np.float64).reshape(n_samples, 1)
        if tau_initial is not None
        else np.zeros((n_samples, 1), dtype=np.float64)
    )

    t0, tf = float(t_span[0]), float(t_span[1])
    total_time = tf - t0
    n_steps = max(1, int(math.ceil(abs(total_time) / h_step_sec)))
    h = total_time / n_steps

    # Determine backend: check for PyTorch CUDA
    use_torch_cuda = False
    if backend in (BatchIntegratorBackend.AUTO, BatchIntegratorBackend.TORCH_CUDA):
        try:
            import torch
            if torch.cuda.is_available():
                use_torch_cuda = True
        except ImportError:
            pass

    if use_torch_cuda:
        import torch
        actual_backend = "torch_cuda"
        device = torch.device("cuda")

        r_tensor = torch.tensor(r0, dtype=torch.float64, device=device)
        v_tensor = torch.tensor(v0, dtype=torch.float64, device=device)
        delta_tensor = torch.zeros((n_samples, 1), dtype=torch.float64, device=device)

        c2 = C_LIGHT**2
        mu_val = float(mu)
        h_val = float(h)

        # Pre-allocate history buffers on CPU to avoid GPU VRAM exhaustion for large M
        t_hist = np.linspace(t0, tf, n_steps + 1, dtype=np.float64)
        r_hist = np.zeros((n_steps + 1, n_samples, 3), dtype=np.float64)
        v_hist = np.zeros((n_steps + 1, n_samples, 3), dtype=np.float64)
        tau_hist = np.zeros((n_steps + 1, n_samples), dtype=np.float64)
        def_hist = np.zeros((n_steps + 1, n_samples), dtype=np.float64)

        r_hist[0] = r0
        v_hist[0] = v0
        tau_hist[0] = tau0.flatten()
        def_hist[0] = 0.0

        def _rhs_torch(r_t: torch.Tensor, v_t: torch.Tensor):
            r_sq_t = torch.sum(r_t**2, dim=1, keepdim=True)
            r_norm_t = torch.sqrt(r_sq_t)
            v_sq_t = torch.sum(v_t**2, dim=1, keepdim=True)
            r_dot_v_t = torch.sum(r_t * v_t, dim=1, keepdim=True)

            a_newton_t = -(mu_val / (r_norm_t * r_sq_t)) * r_t
            factor_1pn_t = mu_val / (c2 * r_norm_t * r_sq_t)
            term_pos_t = (4.0 * mu_val / r_norm_t - v_sq_t) * r_t
            term_vel_t = (4.0 * r_dot_v_t) * v_t
            a_1pn_t = factor_1pn_t * (term_pos_t + term_vel_t)

            w_t = mu_val / r_norm_t
            X_t = (2.0 * w_t / c2) * (1.0 + v_sq_t / c2) + (v_sq_t / c2)
            X_safe_t = torch.clamp(X_t, max=1.0 - 1e-14)
            d_delta_t = X_safe_t / (1.0 + torch.sqrt(torch.clamp(1.0 - X_safe_t, min=1e-15)))

            return v_t, a_newton_t + a_1pn_t, d_delta_t

        for step in range(1, n_steps + 1):
            # 4th-order Runge-Kutta on GPU
            k1_r, k1_v, k1_d = _rhs_torch(r_tensor, v_tensor)
            k2_r, k2_v, k2_d = _rhs_torch(
                r_tensor + 0.5 * h_val * k1_r, v_tensor + 0.5 * h_val * k1_v
            )
            k3_r, k3_v, k3_d = _rhs_torch(
                r_tensor + 0.5 * h_val * k2_r, v_tensor + 0.5 * h_val * k2_v
            )
            k4_r, k4_v, k4_d = _rhs_torch(
                r_tensor + h_val * k3_r, v_tensor + h_val * k3_v
            )

            r_tensor = r_tensor + (h_val / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
            v_tensor = v_tensor + (h_val / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
            delta_tensor = delta_tensor + (h_val / 6.0) * (
                k1_d + 2.0 * k2_d + 2.0 * k3_d + k4_d
            )

            r_hist[step] = r_tensor.cpu().numpy()
            v_hist[step] = v_tensor.cpu().numpy()
            cur_def = delta_tensor.cpu().numpy().flatten()
            def_hist[step] = cur_def
            tau_hist[step] = (t0 + step * h_val) - cur_def

    else:
        # Pure NumPy Vectorized CPU Backend
        actual_backend = "numpy_cpu"
        c2 = C_LIGHT**2

        t_hist = np.linspace(t0, tf, n_steps + 1, dtype=np.float64)
        r_hist = np.zeros((n_steps + 1, n_samples, 3), dtype=np.float64)
        v_hist = np.zeros((n_steps + 1, n_samples, 3), dtype=np.float64)
        tau_hist = np.zeros((n_steps + 1, n_samples), dtype=np.float64)
        def_hist = np.zeros((n_steps + 1, n_samples), dtype=np.float64)

        r_curr = np.copy(r0)
        v_curr = np.copy(v0)
        delta_curr = np.zeros((n_samples, 1), dtype=np.float64)

        r_hist[0] = r_curr
        v_hist[0] = v_curr
        tau_hist[0] = tau0.flatten()
        def_hist[0] = 0.0

        for step in range(1, n_steps + 1):
            # Vectorized RK4 integration step across all N particles
            k1_r, k1_v, k1_d = _rhs_numpy(r_curr, v_curr, mu=mu, c2=c2)
            k2_r, k2_v, k2_d = _rhs_numpy(
                r_curr + 0.5 * h * k1_r, v_curr + 0.5 * h * k1_v, mu=mu, c2=c2
            )
            k3_r, k3_v, k3_d = _rhs_numpy(
                r_curr + 0.5 * h * k2_r, v_curr + 0.5 * h * k2_v, mu=mu, c2=c2
            )
            k4_r, k4_v, k4_d = _rhs_numpy(
                r_curr + h * k3_r, v_curr + h * k3_v, mu=mu, c2=c2
            )

            r_curr += (h / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
            v_curr += (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
            delta_curr += (h / 6.0) * (k1_d + 2.0 * k2_d + 2.0 * k3_d + k4_d)

            r_hist[step] = r_curr
            v_hist[step] = v_curr
            cur_def = delta_curr.flatten()
            def_hist[step] = cur_def
            tau_hist[step] = (t0 + step * h) - cur_def

    return BatchTrajectoryResult(
        t=t_hist,
        r=r_hist,
        v=v_hist,
        tau=tau_hist,
        time_deficit=def_hist,
        n_samples=n_samples,
        backend_used=actual_backend,
    )
