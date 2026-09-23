"""High-Precision Performance Benchmark for DSN Relativistic Observables & Orbit Determination.

Measures:
1. Ground station topocentric BCRS state evaluation throughput (evals/sec).
2. General Relativistic Shapiro gravitational time delay throughput (evals/sec).
3. 2-way iterative light-time root solver throughput (solutions/sec).
4. Extended Kalman Filter (EKF) sequential prediction-correction cycles (cycles/sec).
5. Batch Weighted Least Squares (Batch WLS) normal equation iteration throughput.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from relativistic_engine.constants import AU, GM_SUN
from relativistic_engine.navigation.dsn import (
    DSS_14,
    DSS_65,
    DSS_43,
    compute_station_bcrs_state,
    compute_shapiro_time_delay,
    solve_2way_light_time,
    compute_observation_jacobian,
)
from relativistic_engine.navigation.orbit_determination import (
    TrackingObservation,
    ExtendedKalmanFilter,
    BatchWeightedLeastSquares,
    rk4_step,
)


def run_dsn_od_benchmark() -> dict:
    print("=" * 80)
    print("MILESTONE 10: DSN RELATIVISTIC OBSERVABLES & ORBIT DETERMINATION BENCHMARK")
    print("=" * 80)

    # 1. Ground Station State Evaluation Throughput
    n_station_evals = 10000
    t0 = time.perf_counter()
    stations = [DSS_14, DSS_65, DSS_43]
    for i in range(n_station_evals):
        compute_station_bcrs_state(stations[i % 3], 2451545.0, float(i * 10.0))
    t_station = time.perf_counter() - t0
    station_rate = n_station_evals / t_station
    print(f"1. DSN Station BCRS States: {station_rate:,.0f} evals/sec ({t_station:.4f} s for {n_station_evals:,} evals)")

    # 2. Shapiro Time Delay Throughput
    n_shapiro_evals = 50000
    r1_arr = np.array([1.0 * AU, 7.0e8, 0.0])
    r2_arr = np.array([-1.5 * AU, 7.0e8, 0.0])
    t0 = time.perf_counter()
    for _ in range(n_shapiro_evals):
        compute_shapiro_time_delay(r1_arr, r2_arr, GM_SUN)
    t_shapiro = time.perf_counter() - t0
    shapiro_rate = n_shapiro_evals / t_shapiro
    print(f"2. Shapiro Gravitational Delay: {shapiro_rate:,.0f} evals/sec ({t_shapiro:.4f} s for {n_shapiro_evals:,} evals)")

    # 3. 2-Way Light-Time Solution Throughput
    n_light_time = 1000
    fixed_pos = np.array([1.4 * AU, 2.0e10, -5.0e9])
    fixed_vel = np.array([12000.0, 24000.0, -1000.0])

    def sc_traj(t: float):
        return fixed_pos + fixed_vel * (t / 1000.0), fixed_vel

    t0 = time.perf_counter()
    for i in range(n_light_time):
        solve_2way_light_time(DSS_14, sc_traj, float(1000.0 + i * 10.0), 2451545.0)
    t_light_time = time.perf_counter() - t0
    light_time_rate = n_light_time / t_light_time
    print(f"3. 2-Way Relativistic Light-Time: {light_time_rate:,.0f} solutions/sec ({t_light_time:.4f} s for {n_light_time:,} solves)")

    # 4. Sequential EKF Cycle Throughput
    n_ekf_steps = 100
    x0 = np.array([1.1 * AU, 2.0e10, 1.0e9, -4000.0, 28000.0, 1000.0])
    p0 = np.diag([1e8, 1e8, 1e8, 1.0, 1.0, 1.0])
    ekf = ExtendedKalmanFilter(x0, p0, 0.0, 1e-11)

    obs_list = []
    curr_t = 0.0
    for i in range(n_ekf_steps):
        t_obs = curr_t + 100.0
        curr_t = t_obs
        obs_list.append(
            TrackingObservation(
                epoch_tdb=t_obs,
                station_id="DSS-14",
                range_m=1.8e11 + i * 1e5,
                range_rate_mps=25000.0,
            )
        )

    t0 = time.perf_counter()
    ekf.process_tracking_arc(obs_list, 2451545.0)
    t_ekf = time.perf_counter() - t0
    ekf_rate = n_ekf_steps / t_ekf
    print(f"4. Sequential EKF Filtering: {ekf_rate:,.1f} cycles/sec ({t_ekf:.4f} s for {n_ekf_steps} updates)")

    # 5. Batch WLS Estimation
    guess_x0 = x0 + np.array([500.0, -400.0, 300.0, 0.1, -0.05, 0.02])
    t0 = time.perf_counter()
    est_x0, cov_0, rms_hist = BatchWeightedLeastSquares.estimate_initial_state(
        observations=obs_list[:15],
        initial_state_guess=guess_x0,
        prior_covariance=p0,
        initial_epoch_tdb=0.0,
        jd_base=2451545.0,
        max_iterations=4,
    )
    t_wls = time.perf_counter() - t0
    print(f"5. Batch WLS Normal Equations: {t_wls:.4f} s for 4 iterations (RMS: {rms_hist[0]:.2f} m -> {rms_hist[-1]:.2f} m)")
    print("=" * 80)

    return {
        "station_rate_hz": station_rate,
        "shapiro_rate_hz": shapiro_rate,
        "light_time_rate_hz": light_time_rate,
        "ekf_rate_hz": ekf_rate,
        "wls_time_sec": t_wls,
    }


if __name__ == "__main__":
    run_dsn_od_benchmark()
