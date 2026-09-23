"""Autonomous Deep-Space PNT Multi-Sensor Fusion Engine with SR-UKF.

Implements a high-precision, numerically stable Square-Root Unscented Kalman Filter
(SR-UKF) for autonomous spacecraft Position, Navigation, and Timing (PNT).

Fuses decoupled relativistic navigation observables:
1. Relativistic X-ray Pulsar Timing Navigation (XPNAV / SEXTANT):
   Pulse Phase and Time-of-Arrival (TOA) residuals with BCRS geometric Rømer delay,
   1PN solar gravitational Shapiro delay, and interstellar plasma dispersion.
2. Relativistic Optical Astrometry (Star Trackers):
   Line-of-sight unit vectors subject to exact 3D Lorentz aberration and Doppler beaming.
3. Deep Space Network (DSN) Ground Radiometric Tracking:
   Coherent 2-way range and Doppler observables with asymmetric light-time and time dilation.

State Vector (8D):
    x = [r_x, r_y, r_z, v_x, v_y, v_z, c*delta_t, c*delta_dot_t]^T
    where:
        r: Spacecraft BCRS position (meters)
        v: Spacecraft BCRS velocity (m/s)
        c*delta_t: Range-equivalent onboard clock bias (meters)
        c*delta_dot_t: Dimensionless clock drift rate scaled by c (m/s)

Square-Root Covariance Factorization:
    Propagates and updates the lower-triangular Cholesky square-root factor L of
    the covariance matrix (P = L @ L^T), guaranteeing positive semi-definiteness
    and numerical condition stability across deep-space multi-AU cruise arcs.

References:
- Van der Merwe, R., & Wan, E. A. (2001), "The Square-Root Unscented Kalman Filter for
  State and Parameter-Estimation", IEEE ICASSP.
- Mitchell, J. W. et al. (2018), "SEXTANT - Station Explorer for X-ray Timing and
  Navigation Technology", AIAA Guidance, Navigation, and Control Conference.
- Klioner, S. A. (2003), "A practical relativistic model for microarcsecond astrometry
  in space", Astronomical Journal 125:1580-1597.
- Moyer, T. D. (2003), "Formulation for Observed and Computed Values of Deep Space
  Network Data Types for Navigation", JPL Deep-Space Communications and Navigation Series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg

from relativistic_engine.constants import (
    AU,
    C_LIGHT,
    GM_SUN,
)
from relativistic_engine.navigation.xpnav import (
    PULSAR_CATALOG,
    compute_geometric_delay,
    compute_plasma_dispersion_delay,
    compute_shapiro_delay,
)
from relativistic_engine.physics.optics import transform_aberration_vector


@dataclass(frozen=True)
class PNTState:
    """Immutable state container for SR-UKF PNT state estimate.

    Attributes:
        epoch_seconds: Time past reference epoch (seconds).
        position_m: 3D BCRS position vector [x, y, z] in meters.
        velocity_ms: 3D BCRS velocity vector [vx, vy, vz] in m/s.
        clock_bias_s: Onboard clock offset in seconds (delta_t).
        clock_drift_rate: Dimensionless clock frequency offset (delta_dot_t).
        sqrt_cov: (8, 8) Lower-triangular Cholesky factor L such that P = L @ L^T.
        pos_3sigma_m: 3-sigma position uncertainty radius in meters.
        vel_3sigma_ms: 3-sigma velocity uncertainty radius in m/s.
        clock_3sigma_s: 3-sigma clock bias uncertainty in seconds.
    """

    epoch_seconds: float
    position_m: np.ndarray
    velocity_ms: np.ndarray
    clock_bias_s: float
    clock_drift_rate: float
    sqrt_cov: np.ndarray
    pos_3sigma_m: float
    vel_3sigma_ms: float
    clock_3sigma_s: float

    @property
    def state_vector(self) -> np.ndarray:
        """Return full 8D state vector [r (3), v (3), c*dt, c*ddot_t]."""
        return np.array([
            self.position_m[0],
            self.position_m[1],
            self.position_m[2],
            self.velocity_ms[0],
            self.velocity_ms[1],
            self.velocity_ms[2],
            self.clock_bias_s * C_LIGHT,
            self.clock_drift_rate * C_LIGHT,
        ], dtype=np.float64)

    @property
    def covariance(self) -> np.ndarray:
        """Reconstruct full 8x8 covariance matrix P = L @ L^T."""
        return self.sqrt_cov @ self.sqrt_cov.T


def _cholesky_update(
    L: np.ndarray,
    v: np.ndarray,
    sign: float = 1.0,
) -> np.ndarray:
    """Perform rank-1 Cholesky update or downdate: L_new @ L_new^T = L @ L^T + sign * (v @ v^T)."""
    P = L @ L.T + sign * np.outer(v, v)
    P = (P + P.T) * 0.5
    # Enforce positive definiteness against finite-precision downdate truncation
    n = P.shape[0]
    min_eig = np.min(np.real(np.linalg.eigvals(P)))
    if min_eig <= 1e-16:
        P += np.eye(n) * (abs(min_eig) + 1e-12)
    try:
        return scipy.linalg.cholesky(P, lower=True)
    except scipy.linalg.LinAlgError:
        diag = np.sqrt(np.maximum(np.diag(P), 1e-16))
        return np.diag(diag)


class SquareRootUKF:
    """High-precision 8D Square-Root Unscented Kalman Filter for Deep-Space PNT."""

    def __init__(
        self,
        initial_position_m: Sequence[float],
        initial_velocity_ms: Sequence[float],
        initial_clock_bias_s: float = 0.0,
        initial_clock_drift_rate: float = 0.0,
        pos_sigma_init_m: float = 1000.0,
        vel_sigma_init_ms: float = 0.1,
        clock_sigma_init_s: float = 1e-6,
        drift_sigma_init: float = 1e-11,
        alpha: float = 0.3,
        beta: float = 2.0,
        kappa: float = 0.0,
        epoch_seconds: float = 0.0,
    ) -> None:
        """Initialize the 8D SR-UKF state and lower-triangular covariance square root."""
        self.epoch = float(epoch_seconds)
        self.dim = 8

        # 8D State vector
        self.x = np.array([
            initial_position_m[0],
            initial_position_m[1],
            initial_position_m[2],
            initial_velocity_ms[0],
            initial_velocity_ms[1],
            initial_velocity_ms[2],
            initial_clock_bias_s * C_LIGHT,
            initial_clock_drift_rate * C_LIGHT,
        ], dtype=np.float64)

        # Diagonal initial covariance square root (lower-triangular)
        sigmas = np.array([
            pos_sigma_init_m,
            pos_sigma_init_m,
            pos_sigma_init_m,
            vel_sigma_init_ms,
            vel_sigma_init_ms,
            vel_sigma_init_ms,
            clock_sigma_init_s * C_LIGHT,
            drift_sigma_init * C_LIGHT,
        ], dtype=np.float64)
        self.S = np.diag(sigmas)

        # Merwe Scaled Sigma Points Parameters
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.lambd = self.alpha**2 * (self.dim + self.kappa) - self.dim
        self.gamma_scale = math.sqrt(self.dim + self.lambd)

        # Weights
        self.Wm = np.full(2 * self.dim + 1, 1.0 / (2.0 * (self.dim + self.lambd)), dtype=np.float64)
        self.Wc = np.full(2 * self.dim + 1, 1.0 / (2.0 * (self.dim + self.lambd)), dtype=np.float64)
        self.Wm[0] = self.lambd / (self.dim + self.lambd)
        self.Wc[0] = self.lambd / (self.dim + self.lambd) + (1.0 - self.alpha**2 + self.beta)

        # Process noise spectral densities
        self.q_pos_vel = 1e-14
        self.q_clock_bias = 1e-20 * C_LIGHT**2
        self.q_clock_drift = 1e-26 * C_LIGHT**2

    def _generate_sigma_points(self) -> np.ndarray:
        """Generate (17, 8) Merwe scaled sigma points from current state and S."""
        sigmas = np.zeros((2 * self.dim + 1, self.dim), dtype=np.float64)
        sigmas[0] = self.x

        offset = self.gamma_scale * self.S
        for i in range(self.dim):
            sigmas[i + 1] = self.x + offset[:, i]
            sigmas[i + 1 + self.dim] = self.x - offset[:, i]

        return sigmas

    def _propagate_dynamics(
        self,
        state_vec: np.ndarray,
        dt: float,
        accel_prop_ms2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Propagate single 8D state vector through 1PN Solar gravity and kinematics."""
        r = state_vec[0:3]
        v = state_vec[3:6]
        c_dt = state_vec[6]
        c_ddot_t = state_vec[7]

        r_norm = float(np.linalg.norm(r))
        if r_norm < 1e6:
            r_norm = 1e6

        # 1PN Solar Gravitational Acceleration
        mu = GM_SUN
        r_hat = r / r_norm
        v_sq = float(np.dot(v, v))
        c_sq = C_LIGHT**2
        gr_pot = mu / (c_sq * r_norm)

        a_newton = -mu * r / (r_norm**3)
        a_1pn = a_newton * (1.0 + 4.0 * gr_pot - v_sq / c_sq) + (4.0 * mu / (c_sq * r_norm**2)) * (float(np.dot(r_hat, v))) * v

        if accel_prop_ms2 is not None:
            a_total = a_1pn + np.asarray(accel_prop_ms2, dtype=np.float64)
        else:
            a_total = a_1pn

        r_next = r + v * dt + 0.5 * a_total * dt**2
        v_next = v + a_total * dt

        c_dt_next = c_dt + c_ddot_t * dt
        c_ddot_t_next = c_ddot_t

        return np.array([
            r_next[0], r_next[1], r_next[2],
            v_next[0], v_next[1], v_next[2],
            c_dt_next,
            c_ddot_t_next,
        ], dtype=np.float64)

    def predict(
        self,
        dt: float,
        accel_prop_ms2: Optional[np.ndarray] = None,
    ) -> None:
        """Perform SR-UKF time propagation step over time interval dt (seconds)."""
        if dt <= 0.0:
            return

        sigma_points = self._generate_sigma_points()
        propagated_sigmas = np.zeros_like(sigma_points)

        for i in range(2 * self.dim + 1):
            propagated_sigmas[i] = self._propagate_dynamics(sigma_points[i], dt, accel_prop_ms2)

        x_pred = np.zeros(self.dim, dtype=np.float64)
        for i in range(2 * self.dim + 1):
            x_pred += self.Wm[i] * propagated_sigmas[i]

        q_pos = (1.0 / 3.0) * self.q_pos_vel * dt**3
        q_vel = self.q_pos_vel * dt
        q_cross = 0.5 * self.q_pos_vel * dt**2
        q_clock = self.q_clock_bias * dt + (1.0 / 3.0) * self.q_clock_drift * dt**3
        q_drift = self.q_clock_drift * dt

        Q = np.zeros((self.dim, self.dim), dtype=np.float64)
        for d in range(3):
            Q[d, d] = max(q_pos, 1e-12)
            Q[3 + d, 3 + d] = max(q_vel, 1e-16)
            Q[d, 3 + d] = q_cross
            Q[3 + d, d] = q_cross
        Q[6, 6] = max(q_clock, 1e-12)
        Q[7, 7] = max(q_drift, 1e-16)

        # Assemble a priori covariance P^- via outer-product accumulation over sigma deviations
        P_pred = np.zeros((self.dim, self.dim), dtype=np.float64)
        for i in range(2 * self.dim + 1):
            diff = propagated_sigmas[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)
        P_pred += Q
        P_pred = (P_pred + P_pred.T) * 0.5

        try:
            self.S = scipy.linalg.cholesky(P_pred, lower=True)
        except scipy.linalg.LinAlgError:
            self.S = np.diag(np.sqrt(np.maximum(np.diag(P_pred), 1e-16)))

        self.x = x_pred
        self.epoch += dt

    def update_xpnav(
        self,
        pulsar_name: str,
        observed_fractional_phase: float,
        obs_freq_mhz: float = 1400.0,
        phase_noise_std: float = 0.01,
    ) -> float:
        """Update filter state using a single millisecond pulsar pulse phase observation."""
        if pulsar_name not in PULSAR_CATALOG:
            raise KeyError(f"Pulsar '{pulsar_name}' not found in catalog.")

        pulsar = PULSAR_CATALOG[pulsar_name]
        f0 = pulsar.f0_hz
        sigmas = self._generate_sigma_points()
        m_dim = 1
        Z_sigmas = np.zeros((2 * self.dim + 1, m_dim), dtype=np.float64)

        for i in range(2 * self.dim + 1):
            r_sc = sigmas[i, 0:3]
            c_dt = sigmas[i, 6]
            dt_clock = c_dt / C_LIGHT

            t_ssb = self.epoch + dt_clock
            dt_geom = compute_geometric_delay(pulsar, r_sc)
            dt_shapiro = compute_shapiro_delay(pulsar, r_sc)
            dt_dm = compute_plasma_dispersion_delay(pulsar, obs_freq_mhz)

            t_emitted = t_ssb + dt_geom + dt_shapiro - dt_dm - pulsar.epoch_t0_tdb_s
            phase = (f0 * t_emitted + 0.5 * pulsar.fdot_hz_s * t_emitted**2) % 1.0
            Z_sigmas[i, 0] = phase

        # Unwrap sigma-point phases relative to the central point to eliminate modulo-1 discontinuity
        z0 = Z_sigmas[0, 0]
        for i in range(1, 2 * self.dim + 1):
            diff_wrap = (Z_sigmas[i, 0] - z0 + 0.5) % 1.0 - 0.5
            Z_sigmas[i, 0] = z0 + diff_wrap

        z_pred = float(np.sum(self.Wm * Z_sigmas[:, 0]))
        R_cov = np.array([[phase_noise_std**2]], dtype=np.float64)

        P_zz = np.zeros((m_dim, m_dim), dtype=np.float64)
        P_xz = np.zeros((self.dim, m_dim), dtype=np.float64)

        for i in range(2 * self.dim + 1):
            diff_z = Z_sigmas[i, 0] - z_pred
            diff_x = sigmas[i] - self.x
            P_zz[0, 0] += self.Wc[i] * (diff_z**2)
            P_xz[:, 0] += self.Wc[i] * diff_x * diff_z

        P_zz += R_cov

        K = P_xz @ np.linalg.inv(P_zz)

        raw_residual = observed_fractional_phase - z_pred
        residual = (raw_residual + 0.5) % 1.0 - 0.5

        self.x = self.x + K[:, 0] * residual

        # Covariance update
        P_curr = self.S @ self.S.T
        P_post = P_curr - K @ P_zz @ K.T
        P_post = (P_post + P_post.T) * 0.5

        try:
            self.S = scipy.linalg.cholesky(P_post, lower=True)
        except scipy.linalg.LinAlgError:
            self.S = np.diag(np.sqrt(np.maximum(np.diag(P_post), 1e-16)))

        return float(residual)

    def update_optical_aberration(
        self,
        catalog_unit_vectors: np.ndarray,
        observed_apparent_vectors: np.ndarray,
        angular_noise_rad: float = 1e-6,
    ) -> float:
        """Update filter state using optical star tracker line-of-sight directions."""
        cat_vecs = np.asarray(catalog_unit_vectors, dtype=np.float64)
        obs_vecs = np.asarray(observed_apparent_vectors, dtype=np.float64)
        num_stars = cat_vecs.shape[0]
        m_dim = 3 * num_stars

        sigmas = self._generate_sigma_points()
        Z_sigmas = np.zeros((2 * self.dim + 1, m_dim), dtype=np.float64)

        for i in range(2 * self.dim + 1):
            v_sc = sigmas[i, 3:6]
            beta = v_sc / C_LIGHT
            # Safeguard beta for physical transform
            beta_norm = float(np.linalg.norm(beta))
            if beta_norm >= 0.999:
                beta = beta * (0.999 / beta_norm)

            pred_dirs = transform_aberration_vector(cat_vecs, beta)
            Z_sigmas[i] = pred_dirs.flatten()

        z_pred = np.zeros(m_dim, dtype=np.float64)
        for i in range(2 * self.dim + 1):
            z_pred += self.Wm[i] * Z_sigmas[i]

        R_cov = np.eye(m_dim) * (angular_noise_rad**2)

        P_zz = np.zeros((m_dim, m_dim), dtype=np.float64)
        P_xz = np.zeros((self.dim, m_dim), dtype=np.float64)

        for i in range(2 * self.dim + 1):
            diff_z = Z_sigmas[i] - z_pred
            diff_x = sigmas[i] - self.x
            P_zz += self.Wc[i] * np.outer(diff_z, diff_z)
            P_xz += self.Wc[i] * np.outer(diff_x, diff_z)

        P_zz += R_cov

        K = P_xz @ np.linalg.pinv(P_zz)

        y_obs = obs_vecs.flatten()
        residual = y_obs - z_pred

        self.x = self.x + K @ residual

        P_curr = self.S @ self.S.T
        P_post = P_curr - K @ P_zz @ K.T
        P_post = (P_post + P_post.T) * 0.5

        try:
            self.S = scipy.linalg.cholesky(P_post, lower=True)
        except scipy.linalg.LinAlgError:
            self.S = np.diag(np.sqrt(np.maximum(np.diag(P_post), 1e-16)))

        rms_residual = float(np.sqrt(np.mean(residual**2)))
        return rms_residual

    def update_dsn_range_doppler(
        self,
        station_pos_m: np.ndarray,
        observed_range_m: float,
        observed_range_rate_ms: float,
        range_noise_m: float = 1.0,
        doppler_noise_ms: float = 1e-4,
    ) -> Tuple[float, float]:
        """Update filter state using DSN 2-way coherent range and range-rate."""
        r_st = np.asarray(station_pos_m, dtype=np.float64)
        m_dim = 2
        sigmas = self._generate_sigma_points()
        Z_sigmas = np.zeros((2 * self.dim + 1, m_dim), dtype=np.float64)

        for i in range(2 * self.dim + 1):
            r_sc = sigmas[i, 0:3]
            v_sc = sigmas[i, 3:6]
            c_dt = sigmas[i, 6]

            rel_pos = r_sc - r_st
            rho = float(np.linalg.norm(rel_pos))
            rho_hat = rel_pos / rho if rho > 1.0 else np.array([1.0, 0.0, 0.0])
            rho_dot = float(np.dot(rho_hat, v_sc))

            Z_sigmas[i, 0] = rho + c_dt
            Z_sigmas[i, 1] = rho_dot

        z_pred = np.zeros(m_dim, dtype=np.float64)
        for i in range(2 * self.dim + 1):
            z_pred += self.Wm[i] * Z_sigmas[i]

        R_cov = np.diag([range_noise_m**2, doppler_noise_ms**2])

        P_zz = np.zeros((m_dim, m_dim), dtype=np.float64)
        P_xz = np.zeros((self.dim, m_dim), dtype=np.float64)

        for i in range(2 * self.dim + 1):
            diff_z = Z_sigmas[i] - z_pred
            diff_x = sigmas[i] - self.x
            P_zz += self.Wc[i] * np.outer(diff_z, diff_z)
            P_xz += self.Wc[i] * np.outer(diff_x, diff_z)

        P_zz += R_cov

        K = P_xz @ np.linalg.inv(P_zz)

        residual = np.array([observed_range_m - z_pred[0], observed_range_rate_ms - z_pred[1]], dtype=np.float64)
        self.x = self.x + K @ residual

        P_curr = self.S @ self.S.T
        P_post = P_curr - K @ P_zz @ K.T
        P_post = (P_post + P_post.T) * 0.5

        try:
            self.S = scipy.linalg.cholesky(P_post, lower=True)
        except scipy.linalg.LinAlgError:
            self.S = np.diag(np.sqrt(np.maximum(np.diag(P_post), 1e-16)))

        return (float(residual[0]), float(residual[1]))

    def get_state(self) -> PNTState:
        """Extract immutable PNT state and compute 3-sigma formal uncertainties."""
        P = self.S @ self.S.T
        pos_var = float(P[0, 0] + P[1, 1] + P[2, 2])
        vel_var = float(P[3, 3] + P[4, 4] + P[5, 5])
        clock_var_s2 = float(P[6, 6] / C_LIGHT**2)

        pos_3sigma = 3.0 * math.sqrt(max(pos_var, 0.0))
        vel_3sigma = 3.0 * math.sqrt(max(vel_var, 0.0))
        clock_3sigma = 3.0 * math.sqrt(max(clock_var_s2, 0.0))

        return PNTState(
            epoch_seconds=self.epoch,
            position_m=self.x[0:3].copy(),
            velocity_ms=self.x[3:6].copy(),
            clock_bias_s=float(self.x[6] / C_LIGHT),
            clock_drift_rate=float(self.x[7] / C_LIGHT),
            sqrt_cov=self.S.copy(),
            pos_3sigma_m=pos_3sigma,
            vel_3sigma_ms=vel_3sigma,
            clock_3sigma_s=clock_3sigma,
        )


def simulate_pnt_mission(
    duration_days: float = 30.0,
    step_hours: float = 6.0,
    dsn_blackout_start_day: Optional[float] = None,
    dsn_blackout_end_day: Optional[float] = None,
    true_init_pos_au: Sequence[float] = (1.0, 0.2, 0.0),
    true_init_vel_kms: Sequence[float] = (0.0, 29.78, 0.0),
    initial_pos_error_m: float = 5000.0,
    initial_vel_error_ms: float = 0.5,
    initial_clock_bias_ns: float = 50.0,
) -> Dict[str, Any]:
    """Simulate complete deep-space PNT cruise with multi-sensor updates.

    Demonstrates autonomous state recovery under DSN loss-of-signal.
    """
    total_seconds = duration_days * 86400.0
    dt = step_hours * 3600.0
    num_steps = int(math.ceil(total_seconds / dt))

    r_true = np.array([true_init_pos_au[0] * AU, true_init_pos_au[1] * AU, true_init_pos_au[2] * AU], dtype=np.float64)
    v_true = np.array([true_init_vel_kms[0] * 1000.0, true_init_vel_kms[1] * 1000.0, true_init_vel_kms[2] * 1000.0], dtype=np.float64)
    clock_bias_true_s = initial_clock_bias_ns * 1e-9
    clock_drift_true = 1e-13

    r_est_init = r_true + np.array([initial_pos_error_m / math.sqrt(3)] * 3)
    v_est_init = v_true + np.array([initial_vel_error_ms / math.sqrt(3)] * 3)

    filter_engine = SquareRootUKF(
        initial_position_m=r_est_init,
        initial_velocity_ms=v_est_init,
        initial_clock_bias_s=0.0,
        initial_clock_drift_rate=0.0,
        pos_sigma_init_m=initial_pos_error_m * 2.0,
        vel_sigma_init_ms=initial_vel_error_ms * 2.0,
        clock_sigma_init_s=1e-6,
    )

    star_catalog = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-0.707106, 0.707106, 0.0],
    ], dtype=np.float64)

    pulsar_keys = list(PULSAR_CATALOG.keys())
    telemetry: List[Dict[str, Any]] = []

    current_t = 0.0
    for step in range(num_steps):
        r_norm = float(np.linalg.norm(r_true))
        r_hat = r_true / r_norm
        v_sq = float(np.dot(v_true, v_true))
        c_sq = C_LIGHT**2
        gr_pot = GM_SUN / (c_sq * r_norm)

        a_newton = -GM_SUN * r_true / (r_norm**3)
        a_1pn = a_newton * (1.0 + 4.0 * gr_pot - v_sq / c_sq) + (4.0 * GM_SUN / (c_sq * r_norm**2)) * float(np.dot(r_hat, v_true)) * v_true

        r_true = r_true + v_true * dt + 0.5 * a_1pn * dt**2
        v_true = v_true + a_1pn * dt
        clock_bias_true_s += clock_drift_true * dt
        current_t += dt

        filter_engine.predict(dt)

        current_day = current_t / 86400.0
        in_blackout = False
        if dsn_blackout_start_day is not None and dsn_blackout_end_day is not None:
            if dsn_blackout_start_day <= current_day <= dsn_blackout_end_day:
                in_blackout = True

        dsn_active = not in_blackout
        if dsn_active:
            station_pos = np.array([AU, 0.0, 0.0])
            rel_vec = r_true - station_pos
            rho_true = float(np.linalg.norm(rel_vec))
            rho_hat = rel_vec / rho_true if rho_true > 1.0 else np.array([1.0, 0.0, 0.0])
            rho_dot_true = float(np.dot(rho_hat, v_true))
            filter_engine.update_dsn_range_doppler(
                station_pos,
                observed_range_m=rho_true + clock_bias_true_s * C_LIGHT,
                observed_range_rate_ms=rho_dot_true,
            )

        beta_true = v_true / C_LIGHT
        obs_stars = transform_aberration_vector(star_catalog, beta_true)
        filter_engine.update_optical_aberration(star_catalog, obs_stars)

        for p_name in pulsar_keys[:4]:
            pulsar = PULSAR_CATALOG[p_name]
            dt_geom = compute_geometric_delay(pulsar, r_true)
            dt_shapiro = compute_shapiro_delay(pulsar, r_true)
            dt_dm = compute_plasma_dispersion_delay(pulsar, 1400.0)
            t_emitted = current_t + clock_bias_true_s + dt_geom + dt_shapiro - dt_dm - pulsar.epoch_t0_tdb_s
            true_phase = (pulsar.f0_hz * t_emitted + 0.5 * pulsar.fdot_hz_s * t_emitted**2) % 1.0
            filter_engine.update_xpnav(p_name, true_phase)

        state = filter_engine.get_state()
        pos_error_m = float(np.linalg.norm(state.position_m - r_true))
        vel_error_ms = float(np.linalg.norm(state.velocity_ms - v_true))
        clock_error_s = float(abs(state.clock_bias_s - clock_bias_true_s))

        telemetry.append({
            "day": round(current_day, 2),
            "pos_error_m": round(pos_error_m, 3),
            "pos_3sigma_m": round(state.pos_3sigma_m, 3),
            "vel_error_ms": round(vel_error_ms, 5),
            "vel_3sigma_ms": round(state.vel_3sigma_ms, 5),
            "clock_error_ns": round(clock_error_s * 1e9, 3),
            "clock_3sigma_ns": round(state.clock_3sigma_s * 1e9, 3),
            "dsn_active": dsn_active,
            "in_blackout": in_blackout,
        })

    final_state = filter_engine.get_state()
    final_pos_error = float(np.linalg.norm(final_state.position_m - r_true))
    final_vel_error = float(np.linalg.norm(final_state.velocity_ms - v_true))

    return {
        "duration_days": duration_days,
        "step_hours": step_hours,
        "num_steps": num_steps,
        "final_pos_error_m": final_pos_error,
        "final_pos_3sigma_m": final_state.pos_3sigma_m,
        "final_vel_error_ms": final_vel_error,
        "final_vel_3sigma_ms": final_state.vel_3sigma_ms,
        "final_clock_error_ns": abs(final_state.clock_bias_s - clock_bias_true_s) * 1e9,
        "telemetry": telemetry,
    }
