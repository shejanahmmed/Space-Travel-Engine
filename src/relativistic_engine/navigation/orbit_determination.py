"""Statistical orbit determination and state estimation from DSN tracking telemetry.

Implements sequential Extended Kalman Filter (EKF) with Joseph-stabilized covariance
propagation and Batch Weighted Least Squares (Batch WLS) differential correction.

Authoritative References:
- Tapley, B. D., Schutz, B. E., & Born, G. H. (2004), "Statistical Orbit Determination",
  Elsevier Academic Press.
- Bierman, G. J. (1977), "Factorization Methods for Discrete Sequential Estimation",
  Academic Press.
- Montenbruck, O., & Gill, E. (2000), "Satellite Orbits: Models, Methods and Applications", Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import math
import numpy as np
from jplephem.spk import SPK

from relativistic_engine.constants import (
    C_LIGHT,
    SEC_PER_DAY,
)
from relativistic_engine.physics.potential import (
    solar_system_gravitational_acceleration,
)
from relativistic_engine.uncertainty.variational import evaluate_gravity_gradient
from relativistic_engine.navigation.dsn import (
    DSNStation,
    DSN_STATIONS,
    J2000_JD,
    compute_station_bcrs_state,
    compute_observation_jacobian,
)


@dataclass(frozen=True)
class TrackingObservation:
    """DSN radio tracking observation point.

    Attributes:
        epoch_tdb: Ground station reception epoch (seconds from base Julian Date).
        station_id: Canonical tracking station ID (e.g. 'DSS-14').
        range_m: Observed 2-way range in meters (or None if Doppler-only).
        range_rate_mps: Observed 2-way range-rate in m/s (or None if range-only).
        range_sigma_m: Standard 1-sigma uncertainty of range measurement (meters).
        range_rate_sigma_mps: Standard 1-sigma uncertainty of range-rate measurement (m/s).
    """

    epoch_tdb: float
    station_id: str
    range_m: Optional[float] = None
    range_rate_mps: Optional[float] = None
    range_sigma_m: float = 2.0
    range_rate_sigma_mps: float = 0.0005


@dataclass
class FilterStepResult:
    """Extended Kalman Filter estimation step result.

    Attributes:
        epoch_tdb: Epoch of estimation in seconds from base Julian Date.
        state_estimate: 6D estimated state vector [x, y, z, vx, vy, vz] (m, m/s).
        covariance: 6x6 state error covariance matrix P.
        innovation: Vector of measurement residuals (y - h(x)).
        nis: Normalized Innovation Squared (y^T S^-1 y).
    """

    epoch_tdb: float
    state_estimate: np.ndarray
    covariance: np.ndarray
    innovation: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    nis: float = 0.0


def compute_discrete_process_noise(
    dt_sec: float,
    spectral_density_mps2_per_sqrt_hz: float = 1e-9,
) -> np.ndarray:
    """Compute 6x6 discrete process noise covariance matrix Q(dt).

    Models unmodeled accelerations (e.g. outgassing, minor non-gravitational perturbations)
    as continuous white noise with power spectral density q_a = sigma_a^2.

    Formula (Tapley et al., 2004, Eq. 4.6.14):
        Q_rr = (1/3) * q_a * dt^3 * I_3
        Q_rv = (1/2) * q_a * dt^2 * I_3
        Q_vv = q_a * dt * I_3
    """
    dt = float(dt_sec)
    q_a = float(spectral_density_mps2_per_sqrt_hz ** 2)
    q = np.zeros((6, 6), dtype=np.float64)

    i3 = np.eye(3, dtype=np.float64)
    q[0:3, 0:3] = (1.0 / 3.0) * q_a * (dt ** 3) * i3
    q[0:3, 3:6] = (1.0 / 2.0) * q_a * (dt ** 2) * i3
    q[3:6, 0:3] = q[0:3, 3:6]
    q[3:6, 3:6] = q_a * dt * i3

    return q


def rk4_step(
    state: np.ndarray,
    t: float,
    dt: float,
    jd_base: float,
    spk: Optional[SPK] = None,
) -> np.ndarray:
    """Advance 6D state by dt using 4th-order Runge-Kutta in BCRS frame."""

    def deriv(curr_t: float, curr_s: np.ndarray) -> np.ndarray:
        r = curr_s[0:3]
        v = curr_s[3:6]
        jd_epoch = jd_base + (curr_t / SEC_PER_DAY)
        a = solar_system_gravitational_acceleration(
            r, v, jd_epoch, spk=spk, include_1pn=True
        )
        return np.concatenate([v, a])

    k1 = deriv(t, state)
    k2 = deriv(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = deriv(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = deriv(t + dt, state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def compute_stm_step(
    state: np.ndarray,
    t: float,
    dt: float,
    jd_base: float,
    spk: Optional[SPK] = None,
) -> np.ndarray:
    """Compute local 6x6 State Transition Matrix Phi(t + dt, t).

    Evaluates the Newtonian tidal gravity gradient G = da/dr and forms:
        F = [0, I; G, 0]
        Phi(dt) approx I + F*dt + 0.5*(F*dt)^2
    """
    jd_epoch = jd_base + (t / SEC_PER_DAY)
    g_tensor = evaluate_gravity_gradient(state[0:3], jd_epoch, spk=spk)

    f_mat = np.zeros((6, 6), dtype=np.float64)
    f_mat[0:3, 3:6] = np.eye(3, dtype=np.float64)
    f_mat[3:6, 0:3] = g_tensor

    f_dt = f_mat * dt
    phi = np.eye(6, dtype=np.float64) + f_dt + 0.5 * (f_dt @ f_dt)
    return phi


class ExtendedKalmanFilter:
    """Extended Kalman Filter for sequential spacecraft orbit determination."""

    def __init__(
        self,
        initial_state: np.ndarray | Sequence[float],
        initial_covariance: np.ndarray,
        initial_epoch_tdb: float = 0.0,
        process_noise_psd: float = 1e-9,
    ) -> None:
        """Initialize EKF.

        Args:
            initial_state: 6D initial state vector [x, y, z, vx, vy, vz].
            initial_covariance: 6x6 initial state covariance matrix P0.
            initial_epoch_tdb: Initial epoch in seconds from base JD.
            process_noise_psd: Acceleration noise spectral density (m/s^2 / sqrt(Hz)).
        """
        self.state = np.asarray(initial_state, dtype=np.float64).copy()
        self.covariance = np.asarray(initial_covariance, dtype=np.float64).copy()
        self.epoch_tdb = float(initial_epoch_tdb)
        self.process_noise_psd = float(process_noise_psd)

    def predict(
        self,
        target_epoch_tdb: float,
        jd_base: float,
        spk: Optional[SPK] = None,
        max_step_sec: float = 60.0,
    ) -> None:
        """Propagate state and covariance forward to target_epoch_tdb."""
        total_dt = target_epoch_tdb - self.epoch_tdb
        if total_dt <= 0.0:
            return

        n_steps = max(1, int(math.ceil(total_dt / max_step_sec)))
        dt = total_dt / n_steps

        curr_t = self.epoch_tdb
        curr_state = self.state
        curr_cov = self.covariance

        for _ in range(n_steps):
            phi = compute_stm_step(curr_state, curr_t, dt, jd_base, spk)
            curr_state = rk4_step(curr_state, curr_t, dt, jd_base, spk)
            q = compute_discrete_process_noise(dt, self.process_noise_psd)
            curr_cov = phi @ curr_cov @ phi.T + q
            # Enforce symmetry
            curr_cov = 0.5 * (curr_cov + curr_cov.T)
            curr_t += dt

        self.state = curr_state
        self.covariance = curr_cov
        self.epoch_tdb = target_epoch_tdb

    def update(
        self,
        observation: TrackingObservation,
        jd_base: float,
        spk: Optional[SPK] = None,
    ) -> FilterStepResult:
        """Perform measurement update step using a DSN tracking observation.

        Implements Joseph-stabilized covariance update to guarantee positive-definiteness:
            P_k|k = (I - K*H) * P_k|k-1 * (I - K*H)^T + K * R * K^T
        """
        station = DSN_STATIONS.get(observation.station_id)
        if station is None:
            raise ValueError(f"Unknown DSN station ID: {observation.station_id}")

        r_sta, v_sta = compute_station_bcrs_state(
            station, jd_base, observation.epoch_tdb, spk
        )

        r_sc = self.state[0:3]
        v_sc = self.state[3:6]
        rho_vec = r_sc - r_sta
        rho = float(np.linalg.norm(rho_vec))
        v_rel = v_sc - v_sta
        rho_dot = float(np.dot(rho_vec, v_rel) / rho)

        # Build active measurements y, predictions h(x), covariance R, and sensitivity H
        meas_list = []
        pred_list = []
        var_list = []
        h_rows = []

        h_full = compute_observation_jacobian(r_sta, v_sta, r_sc, v_sc)

        if observation.range_m is not None:
            meas_list.append(observation.range_m)
            pred_list.append(rho)
            var_list.append(observation.range_sigma_m ** 2)
            h_rows.append(h_full[0, :])

        if observation.range_rate_mps is not None:
            meas_list.append(observation.range_rate_mps)
            pred_list.append(rho_dot)
            var_list.append(observation.range_rate_sigma_mps ** 2)
            h_rows.append(h_full[1, :])

        if not meas_list:
            return FilterStepResult(
                epoch_tdb=self.epoch_tdb,
                state_estimate=self.state.copy(),
                covariance=self.covariance.copy(),
            )

        y = np.array(meas_list, dtype=np.float64)
        h_x = np.array(pred_list, dtype=np.float64)
        r_mat = np.diag(var_list)
        h_mat = np.vstack(h_rows)

        # Innovation
        innovation = y - h_x

        # Innovation covariance S = H * P * H^T + R
        s_mat = h_mat @ self.covariance @ h_mat.T + r_mat

        # Kalman gain K = P * H^T * S^-1
        kalman_gain = self.covariance @ h_mat.T @ np.linalg.inv(s_mat)

        # State update
        self.state = self.state + kalman_gain @ innovation

        # Joseph-stabilized covariance update
        eye = np.eye(6, dtype=np.float64)
        i_minus_kh = eye - kalman_gain @ h_mat
        self.covariance = (
            i_minus_kh @ self.covariance @ i_minus_kh.T
            + kalman_gain @ r_mat @ kalman_gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

        # Normalized Innovation Squared (NIS)
        nis = float(innovation.T @ np.linalg.inv(s_mat) @ innovation)

        return FilterStepResult(
            epoch_tdb=self.epoch_tdb,
            state_estimate=self.state.copy(),
            covariance=self.covariance.copy(),
            innovation=innovation,
            nis=nis,
        )

    def process_tracking_arc(
        self,
        observations: Sequence[TrackingObservation],
        jd_base: float,
        spk: Optional[SPK] = None,
    ) -> List[FilterStepResult]:
        """Filter an entire time-sorted arc of tracking observations."""
        results: List[FilterStepResult] = []
        for obs in observations:
            if obs.epoch_tdb > self.epoch_tdb:
                self.predict(obs.epoch_tdb, jd_base, spk)
            step_res = self.update(obs, jd_base, spk)
            results.append(step_res)
        return results


class BatchWeightedLeastSquares:
    """Batch Weighted Least Squares differential corrector for orbit determination."""

    @staticmethod
    def estimate_initial_state(
        observations: Sequence[TrackingObservation],
        initial_state_guess: np.ndarray | Sequence[float],
        prior_covariance: Optional[np.ndarray] = None,
        initial_epoch_tdb: float = 0.0,
        jd_base: float = J2000_JD,
        spk: Optional[SPK] = None,
        max_iterations: int = 10,
        convergence_tol_m: float = 1e-2,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Estimate optimal initial state x(t0) using batch normal equations.

        Iteratively solves the normal equations:
            Lambda * Delta x0 = N
            Lambda = P0^-1 + sum_k [ H_tilde_k^T * R_k^-1 * H_tilde_k ]
            N      = sum_k [ H_tilde_k^T * R_k^-1 * (y_k - h_k) ]
            where H_tilde_k = H_k * Phi(t_k, t_0).

        Args:
            observations: Sequence of DSN tracking observations.
            initial_state_guess: 6D initial state guess at t0 (m, m/s).
            prior_covariance: Optional 6x6 prior state covariance matrix P0.
            initial_epoch_tdb: Epoch corresponding to initial_state_guess (seconds from jd_base).
            jd_base: Base Julian Date.
            spk: Optional pre-loaded JPL SPK kernel.
            max_iterations: Maximum differential correction iterations.
            convergence_tol_m: Position convergence threshold in meters.

        Returns:
            (estimated_state_0, covariance_0, rms_residual_history)
        """
        x0_prior = np.asarray(initial_state_guess, dtype=np.float64).copy()
        curr_x0 = x0_prior.copy()
        rms_history: List[float] = []

        if prior_covariance is not None:
            p0_inv = np.linalg.inv(prior_covariance)
        else:
            p0_inv = np.zeros((6, 6), dtype=np.float64)

        t0 = float(initial_epoch_tdb)

        for _ in range(max_iterations):
            lambda_mat = p0_inv.copy()
            n_vec = p0_inv @ (x0_prior - curr_x0)
            residuals_all = []

            # Propagate reference trajectory and composite STM from t0 to each obs
            curr_state = curr_x0.copy()
            curr_t = t0
            phi_total = np.eye(6, dtype=np.float64)

            for obs in observations:
                dt = obs.epoch_tdb - curr_t
                if dt > 0.0:
                    phi_step = compute_stm_step(curr_state, curr_t, dt, jd_base, spk)
                    curr_state = rk4_step(curr_state, curr_t, dt, jd_base, spk)
                    phi_total = phi_step @ phi_total
                    curr_t = obs.epoch_tdb

                station = DSN_STATIONS[obs.station_id]
                r_sta, v_sta = compute_station_bcrs_state(
                    station, jd_base, obs.epoch_tdb, spk
                )

                r_sc = curr_state[0:3]
                v_sc = curr_state[3:6]
                rho_vec = r_sc - r_sta
                rho = float(np.linalg.norm(rho_vec))
                v_rel = v_sc - v_sta
                rho_dot = float(np.dot(rho_vec, v_rel) / rho)

                h_full = compute_observation_jacobian(r_sta, v_sta, r_sc, v_sc)

                meas_list = []
                pred_list = []
                weight_list = []
                h_rows = []

                if obs.range_m is not None:
                    meas_list.append(obs.range_m)
                    pred_list.append(rho)
                    weight_list.append(1.0 / (obs.range_sigma_m ** 2))
                    h_rows.append(h_full[0, :])

                if obs.range_rate_mps is not None:
                    meas_list.append(obs.range_rate_mps)
                    pred_list.append(rho_dot)
                    weight_list.append(1.0 / (obs.range_rate_sigma_mps ** 2))
                    h_rows.append(h_full[1, :])

                if not meas_list:
                    continue

                y_k = np.array(meas_list, dtype=np.float64)
                h_k = np.array(pred_list, dtype=np.float64)
                res_k = y_k - h_k
                residuals_all.extend(res_k.tolist())

                w_k = np.diag(weight_list)
                h_tilde_k = np.vstack(h_rows) @ phi_total

                lambda_mat += h_tilde_k.T @ w_k @ h_tilde_k
                n_vec += h_tilde_k.T @ w_k @ res_k

            rms = float(np.sqrt(np.mean(np.array(residuals_all) ** 2))) if residuals_all else 0.0
            rms_history.append(rms)

            # Diagonal preconditioning to eliminate ill-conditioning from mixed units (m and m/s)
            diag_l = np.diag(lambda_mat)
            scale = np.where(diag_l > 0.0, np.sqrt(diag_l), 1.0)
            d_mat = np.diag(1.0 / scale)
            lambda_scaled = d_mat @ lambda_mat @ d_mat
            n_scaled = d_mat @ n_vec

            dx_scaled = np.linalg.solve(lambda_scaled, n_scaled)
            delta_x0 = d_mat @ dx_scaled
            cov_0 = d_mat @ np.linalg.inv(lambda_scaled) @ d_mat

            curr_x0 = curr_x0 + delta_x0
            pos_corr_norm = float(np.linalg.norm(delta_x0[0:3]))

            if pos_corr_norm < convergence_tol_m:
                break

        cov_0 = 0.5 * (cov_0 + cov_0.T)
        return curr_x0, cov_0, rms_history
