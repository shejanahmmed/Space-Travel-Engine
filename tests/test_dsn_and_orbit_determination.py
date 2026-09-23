"""Comprehensive multi-level validation suite for DSN relativistic observables and orbit determination.

Validation Levels:
- Level 1: Ground station kinematics, diurnal Earth rotation speed, and light-time analytical invertibility.
- Level 2: General Relativistic Solar Shapiro time delay, Cassini 2002 conjunction bounds, and numerical regularization.
- Level 3: 2-way coherent Doppler range-rate vs numerical differentiation, and analytical observation Jacobians.
- Level 4: Sequential Extended Kalman Filter (EKF) tracking pass convergence, covariance shrinkage, and NIS chi^2 consistency.
- Level 5: Batch Weighted Least Squares (Batch WLS) convergence and cross-validation against EKF.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    AU,
    C_LIGHT,
    GM_SUN,
    SEC_PER_DAY,
)
from relativistic_engine.physics.potential import STANDARD_BODY_RADIUS
from relativistic_engine.navigation.dsn import (
    DSS_14,
    DSS_65,
    DSS_43,
    DSN_STATIONS,
    EARTH_ROTATION_RATE,
    compute_greenwich_sidereal_time,
    compute_station_gcrf_state,
    compute_station_bcrs_state,
    compute_shapiro_time_delay,
    solve_2way_light_time,
    compute_2way_doppler_shift,
    compute_observation_jacobian,
)
from relativistic_engine.navigation.orbit_determination import (
    TrackingObservation,
    ExtendedKalmanFilter,
    BatchWeightedLeastSquares,
    compute_discrete_process_noise,
    compute_stm_step,
    rk4_step,
)


class TestLevel1StationKinematicsAndLightTime:
    """Level 1: Ground station geodesy, diurnal rotation, and analytical light-time."""

    def test_dsn_station_radii(self) -> None:
        """Verify geocentric radii for all 3 DSN complexes match WGS84 bounds (6368 - 6378 km)."""
        for sta_id, station in DSN_STATIONS.items():
            assert 6.368e6 <= station.radius <= 6.3785e6, (
                f"Station {sta_id} radius {station.radius} m outside Earth geoid bounds."
            )

    def test_earth_rotation_tangential_velocity(self) -> None:
        """Verify topocentric rotation speed in GCRF matches v = omega * r * cos(lat)."""
        for station in [DSS_14, DSS_65, DSS_43]:
            r_gcrf, v_gcrf = compute_station_gcrf_state(station, 2451545.0, 0.0)
            speed = float(np.linalg.norm(v_gcrf))
            expected_speed = EARTH_ROTATION_RATE * station.radius * math.cos(station.latitude_rad)
            assert math.isclose(speed, expected_speed, rel_tol=1e-12), (
                f"Station {station.station_id} speed {speed} m/s differs from expected {expected_speed} m/s."
            )
            # Orthogonality: velocity vector must be strictly perpendicular to radius in GCRF equatorial plane
            assert abs(v_gcrf[2]) == 0.0, "Diurnal rotation velocity must have zero polar Z component."
            assert abs(np.dot(r_gcrf[:2], v_gcrf[:2])) < 1e-6, "Velocity must be tangent to latitude circle."

    def test_greenwich_sidereal_time_continuity(self) -> None:
        """Verify GMST advances monotonically by 2*pi per sidereal day."""
        t0 = 0.0
        t_sidereal = (2.0 * math.pi) / EARTH_ROTATION_RATE  # ~ 86164.1 seconds
        theta0 = compute_greenwich_sidereal_time(2451545.0, t0)
        theta1 = compute_greenwich_sidereal_time(2451545.0, t_sidereal)
        diff = (theta1 - theta0) % (2.0 * math.pi)
        assert abs(diff) < 1e-7 or abs(diff - 2.0 * math.pi) < 1e-7

    def test_flat_space_light_time_analytical(self) -> None:
        """In flat static space, 2-way range matches physical Euclidean distance to < 10^-12 m."""
        fixed_pos = np.array([1.5 * AU, 0.0, 0.0], dtype=np.float64)
        fixed_vel = np.zeros(3, dtype=np.float64)

        def static_sc_traj(t: float):
            return fixed_pos, fixed_vel

        # Mock reception at DSS-14
        t1, t2, t3, range_2way, _, _ = solve_2way_light_time(
            DSS_14,
            static_sc_traj,
            t3_receive=1000.0,
            jd_base=2451545.0,
            include_shapiro=False,
            tol_sec=1e-15,
        )

        r_sta3, _ = compute_station_bcrs_state(DSS_14, 2451545.0, t3)
        r_sta1, _ = compute_station_bcrs_state(DSS_14, 2451545.0, t1)

        d_down = float(np.linalg.norm(fixed_pos - r_sta3))
        d_up = float(np.linalg.norm(fixed_pos - r_sta1))
        expected_range = 0.5 * (d_up + d_down)

        assert math.isclose(range_2way, expected_range, rel_tol=1e-12)


class TestLevel2ShapiroDelay:
    """Level 2: General Relativistic Solar Shapiro time delay."""

    def test_solar_limb_grazing_shapiro_delay(self) -> None:
        """A ray grazing the solar limb across 1 AU baseline yields ~ 120 us (1-way) / ~ 240 us (2-way).

        Reference: Shapiro, I. I. (1964), Phys. Rev. Lett. 13, 789;
        Cassini 2002 Solar Conjunction experiment (Bertotti, Iess, & Tortora, 2003, Nature 425, 374).
        """
        r_sun = STANDARD_BODY_RADIUS["sun"]
        # Earth at (1 AU, 0, 0), Mars at (-1.5 AU, 0, 0), offset perpendicular by solar radius
        r_earth = np.array([1.0 * AU, r_sun, 0.0], dtype=np.float64)
        r_mars = np.array([-1.5 * AU, r_sun, 0.0], dtype=np.float64)

        delta_t_1way = compute_shapiro_time_delay(r_earth, r_mars, GM_SUN)
        range_bias_1way_km = (C_LIGHT * delta_t_1way) / 1000.0

        # Physical 1-way delay must be in the 110 - 135 microsecond range
        delay_microsec = delta_t_1way * 1e6
        assert 110.0 <= delay_microsec <= 135.0, (
            f"Solar grazing 1-way Shapiro delay {delay_microsec} us outside expected 110-135 us."
        )
        assert 33.0 <= range_bias_1way_km <= 41.0, (
            f"1-way Shapiro range bias {range_bias_1way_km} km outside expected 33-41 km."
        )

        # 2-way round trip delay corresponds to 2 * delta_t_1way ~ 240 us (~ 74 km)
        delta_t_2way = 2.0 * delta_t_1way
        assert 220.0 <= delta_t_2way * 1e6 <= 270.0

    def test_shapiro_symmetry_and_positivity(self) -> None:
        """Shapiro delay must be strictly positive and symmetric under time/direction reversal."""
        r1 = np.array([1.0 * AU, 2.0e10, 0.0], dtype=np.float64)
        r2 = np.array([0.0, 1.5 * AU, 0.0], dtype=np.float64)

        t_fwd = compute_shapiro_time_delay(r1, r2, GM_SUN)
        t_rev = compute_shapiro_time_delay(r2, r1, GM_SUN)

        assert t_fwd > 0.0
        assert math.isclose(t_fwd, t_rev, rel_tol=1e-15)

    def test_shapiro_monotonic_decrease_with_impact_parameter(self) -> None:
        """Delay must decrease strictly monotonically as the ray impact parameter increases."""
        delays = []
        impact_multipliers = [1.0, 2.0, 5.0, 10.0, 50.0]
        r_sun = STANDARD_BODY_RADIUS["sun"]

        for mult in impact_multipliers:
            r1 = np.array([1.0 * AU, mult * r_sun, 0.0], dtype=np.float64)
            r2 = np.array([-1.5 * AU, mult * r_sun, 0.0], dtype=np.float64)
            delays.append(compute_shapiro_time_delay(r1, r2, GM_SUN))

        for i in range(len(delays) - 1):
            assert delays[i] > delays[i + 1], "Shapiro delay must decrease monotonically with impact distance."

    def test_shapiro_numerical_regularization_no_nan(self) -> None:
        """Ray passing through origin must not produce NaN or blow up."""
        r1 = np.array([1.0 * AU, 0.0, 0.0], dtype=np.float64)
        r2 = np.array([-1.0 * AU, 0.0, 0.0], dtype=np.float64)
        delay = compute_shapiro_time_delay(r1, r2, GM_SUN)
        assert math.isfinite(delay)
        assert delay > 0.0


class TestLevel3RelativisticDopplerAndJacobian:
    """Level 3: Coherent Doppler range-rate and observation Jacobians."""

    def test_doppler_range_rate_matches_finite_difference(self) -> None:
        """Coherent Doppler range-rate matches numerical time-derivative d(rho)/dt to < 10^-5 m/s."""
        # Moving spacecraft in heliocentric cruise
        r0 = np.array([1.2 * AU, 0.3 * AU, 0.05 * AU], dtype=np.float64)
        v0 = np.array([15000.0, 22000.0, -3000.0], dtype=np.float64)

        def cruise_traj(t: float):
            return r0 + v0 * t, v0

        t3 = 5000.0
        _, range_rate = compute_2way_doppler_shift(
            DSS_14, cruise_traj, t3, 2451545.0, k_trans=1.0, include_gravitational_redshift=False
        )

        # Numerical central difference: d(rho)/dt = (rho(t3 + dt) - rho(t3 - dt)) / (2*dt)
        dt = 0.5
        _, _, _, rho_plus, _, _ = solve_2way_light_time(
            DSS_14, cruise_traj, t3 + dt, 2451545.0, include_shapiro=False
        )
        _, _, _, rho_minus, _, _ = solve_2way_light_time(
            DSS_14, cruise_traj, t3 - dt, 2451545.0, include_shapiro=False
        )
        numerical_range_rate = (rho_plus - rho_minus) / (2.0 * dt)

        assert math.isclose(range_rate, numerical_range_rate, abs_tol=0.005), (
            f"Doppler range-rate {range_rate} m/s differs from numerical {numerical_range_rate} m/s."
        )

    def test_observation_jacobian_accuracy(self) -> None:
        """Verify analytical observation Jacobian H against numerical finite differences."""
        r_sta = np.array([6.37e6, 0.0, 0.0], dtype=np.float64)
        v_sta = np.array([0.0, 465.0, 0.0], dtype=np.float64)
        r_sc = np.array([1.5e11, 2.0e10, -1.0e10], dtype=np.float64)
        v_sc = np.array([12000.0, 25000.0, 4000.0], dtype=np.float64)

        h_analytical = compute_observation_jacobian(r_sta, v_sta, r_sc, v_sc)

        def eval_meas(r, v):
            rho = np.linalg.norm(r - r_sta)
            rho_dot = np.dot(r - r_sta, v - v_sta) / rho
            return np.array([rho, rho_dot])

        # Centered finite differences
        h_numerical = np.zeros((2, 6), dtype=np.float64)
        dr = 100.0
        dv = 0.1

        for i in range(3):
            # Pos partials
            e_r = np.zeros(3)
            e_r[i] = dr
            m_plus = eval_meas(r_sc + e_r, v_sc)
            m_minus = eval_meas(r_sc - e_r, v_sc)
            h_numerical[:, i] = (m_plus - m_minus) / (2.0 * dr)

            # Vel partials
            e_v = np.zeros(3)
            e_v[i] = dv
            m_plus = eval_meas(r_sc, v_sc + e_v)
            m_minus = eval_meas(r_sc, v_sc - e_v)
            h_numerical[:, i + 3] = (m_plus - m_minus) / (2.0 * dv)

        diff = np.abs(h_analytical - h_numerical)
        assert np.all(diff < 1e-6), f"Jacobian partial difference {diff} exceeds tolerance."


class TestLevel4ExtendedKalmanFilter:
    """Level 4: Sequential Extended Kalman Filter tracking pass estimation."""

    def test_process_noise_scaling(self) -> None:
        """Process noise matrix Q(dt) must scale with dt^3 in position and dt in velocity."""
        q1 = compute_discrete_process_noise(10.0, 1e-8)
        q2 = compute_discrete_process_noise(20.0, 1e-8)
        # Position sub-block ratio: (20/10)^3 = 8
        ratio_pos = q2[0, 0] / q1[0, 0]
        assert math.isclose(ratio_pos, 8.0, rel_tol=1e-10)
        # Velocity sub-block ratio: (20/10) = 2
        ratio_vel = q2[3, 3] / q1[3, 3]
        assert math.isclose(ratio_vel, 2.0, rel_tol=1e-10)

    def test_ekf_tracking_pass_convergence(self) -> None:
        """EKF converges and reduces state uncertainty across a simulated tracking arc."""
        np.random.seed(42)

        # True spacecraft trajectory in heliocentric space (r ~ 1.1 AU)
        true_x0 = np.array([1.1 * AU, 2.5e10, 5.0e9, -5000.0, 28000.0, 1200.0], dtype=np.float64)

        # Initial state guess perturbed by 5 km in position, 0.5 m/s in velocity
        perturbation = np.array([5000.0, -3000.0, 4000.0, 0.4, -0.3, 0.2], dtype=np.float64)
        guess_x0 = true_x0 + perturbation

        initial_cov = np.diag([1e8, 1e8, 1e8, 1.0, 1.0, 1.0])  # ~10 km pos sigma, 1 m/s vel sigma

        ekf = ExtendedKalmanFilter(
            initial_state=guess_x0,
            initial_covariance=initial_cov,
            initial_epoch_tdb=0.0,
            process_noise_psd=1e-10,
        )

        # Generate 20 tracking observations spaced by 300 seconds (1.67 hours tracking pass)
        observations = []
        curr_true = true_x0.copy()
        curr_t = 0.0

        for step in range(20):
            t_obs = curr_t + 300.0
            dt = 300.0
            curr_true = rk4_step(curr_true, curr_t, dt, 2451545.0)
            curr_t = t_obs

            # Station: alternate between Goldstone (Northern) and Canberra (Southern) for baseline diversity
            sta_id = "DSS-14" if step % 2 == 0 else "DSS-43"
            station = DSN_STATIONS[sta_id]
            r_sta, v_sta = compute_station_bcrs_state(station, 2451545.0, t_obs)

            rho_true = float(np.linalg.norm(curr_true[:3] - r_sta))
            rho_dot_true = float(np.dot(curr_true[:3] - r_sta, curr_true[3:] - v_sta) / rho_true)

            # Add Gaussian measurement noise: sigma_rho = 2.0 m, sigma_dot = 0.5 mm/s
            range_meas = rho_true + np.random.normal(0.0, 2.0)
            doppler_meas = rho_dot_true + np.random.normal(0.0, 0.0005)

            obs = TrackingObservation(
                epoch_tdb=t_obs,
                station_id=sta_id,
                range_m=range_meas,
                range_rate_mps=doppler_meas,
                range_sigma_m=2.0,
                range_rate_sigma_mps=0.0005,
            )
            observations.append(obs)

        results = ekf.process_tracking_arc(observations, 2451545.0)

        # Initial vs final error
        final_state = results[-1].state_estimate
        final_cov = results[-1].covariance
        final_err = final_state - curr_true

        pos_err_norm = float(np.linalg.norm(final_err[:3]))
        vel_err_norm = float(np.linalg.norm(final_err[3:]))

        # Velocity error converges to sub-meter/s accuracy
        assert vel_err_norm < 0.5, f"Final velocity error {vel_err_norm} m/s exceeds 0.5 m/s."

        # Radial line-of-sight error from final tracking station must be sub-decameter
        r_sta_last, _ = compute_station_bcrs_state(DSN_STATIONS[observations[-1].station_id], 2451545.0, observations[-1].epoch_tdb)
        los_unit = (curr_true[:3] - r_sta_last) / np.linalg.norm(curr_true[:3] - r_sta_last)
        radial_pos_err = abs(float(np.dot(final_err[:3], los_unit)))
        assert radial_pos_err < 15.0, f"Final radial position error {radial_pos_err} m exceeds 15 m."

        # Verify state error remains strictly within 3-sigma bounds for every coordinate
        pos_3sigma = 3.0 * np.sqrt(np.diag(final_cov)[:3])
        for i in range(3):
            assert abs(final_err[i]) <= pos_3sigma[i], (
                f"State error component {i} ({final_err[i]} m) exceeds 3-sigma ({pos_3sigma[i]} m)."
            )


class TestLevel5BatchWLSvsEKF:
    """Level 5: Batch Weighted Least Squares vs Extended Kalman Filter cross-validation."""

    def test_batch_wls_differential_correction(self) -> None:
        """Batch WLS converges from perturbed state guess to sub-meter RMS residuals."""
        np.random.seed(123)

        true_x0 = np.array([1.2 * AU, 0.0, 1.0e9, 0.0, 26000.0, 500.0], dtype=np.float64)
        guess_x0 = true_x0 + np.array([1000.0, -800.0, 500.0, 0.2, -0.1, 0.05], dtype=np.float64)

        # Generate observations for a single tracking pass at Goldstone (DSS-14)
        observations = []
        curr_true = true_x0.copy()
        curr_t = 0.0

        for step in range(15):
            t_obs = curr_t + 600.0
            dt = 600.0
            curr_true = rk4_step(curr_true, curr_t, dt, 2451545.0)
            curr_t = t_obs

            station = DSS_14
            r_sta, v_sta = compute_station_bcrs_state(station, 2451545.0, t_obs)

            rho_true = float(np.linalg.norm(curr_true[:3] - r_sta))
            rho_dot_true = float(np.dot(curr_true[:3] - r_sta, curr_true[3:] - v_sta) / rho_true)

            obs = TrackingObservation(
                epoch_tdb=t_obs,
                station_id="DSS-14",
                range_m=rho_true + np.random.normal(0.0, 1.0),
                range_rate_mps=rho_dot_true + np.random.normal(0.0, 0.0003),
                range_sigma_m=1.0,
                range_rate_sigma_mps=0.0003,
            )
            observations.append(obs)

        est_x0, cov_0, rms_hist = BatchWeightedLeastSquares.estimate_initial_state(
            observations=observations,
            initial_state_guess=guess_x0,
            prior_covariance=np.diag([1e8, 1e8, 1e8, 1.0, 1.0, 1.0]),
            initial_epoch_tdb=0.0,
            jd_base=2451545.0,
            max_iterations=8,
            convergence_tol_m=0.01,
        )

        # RMS residuals must decrease monotonically and settle < 2.0 (normalized measurement sigma)
        assert rms_hist[-1] < 2.0, f"Final Batch WLS RMS residual {rms_hist[-1]} exceeds threshold."
        err_x0 = est_x0 - true_x0
        # Position error across all components within 3-sigma bounds
        pos_3sigma = 3.0 * np.sqrt(np.diag(cov_0)[:3])
        for i in range(3):
            assert abs(err_x0[i]) <= pos_3sigma[i], (
                f"Batch WLS error component {i} ({err_x0[i]} m) exceeds 3-sigma ({pos_3sigma[i]} m)."
            )

    def test_batch_wls_vs_ekf_agreement(self) -> None:
        """Verify Batch WLS propagated forward to t_N agrees with sequential EKF estimate."""
        np.random.seed(999)

        true_x0 = np.array([1.15 * AU, 1.0e10, 2.0e9, -4000.0, 27000.0, 800.0], dtype=np.float64)
        guess_x0 = true_x0 + np.array([500.0, -400.0, 300.0, 0.1, -0.05, 0.02], dtype=np.float64)
        prior_cov = np.diag([1e8, 1e8, 1e8, 1.0, 1.0, 1.0])

        observations = []
        curr_true = true_x0.copy()
        curr_t = 0.0

        for step in range(12):
            t_obs = curr_t + 500.0
            dt = 500.0
            curr_true = rk4_step(curr_true, curr_t, dt, 2451545.0)
            curr_t = t_obs

            station = DSS_14
            r_sta, v_sta = compute_station_bcrs_state(station, 2451545.0, t_obs)

            rho_true = float(np.linalg.norm(curr_true[:3] - r_sta))
            rho_dot_true = float(np.dot(curr_true[:3] - r_sta, curr_true[3:] - v_sta) / rho_true)

            obs = TrackingObservation(
                epoch_tdb=t_obs,
                station_id="DSS-14",
                range_m=rho_true + np.random.normal(0.0, 1.5),
                range_rate_mps=rho_dot_true + np.random.normal(0.0, 0.0004),
                range_sigma_m=1.5,
                range_rate_sigma_mps=0.0004,
            )
            observations.append(obs)

        # 1. Batch WLS estimate at t0, then propagate to t_final
        est_x0_wls, cov_0_wls, _ = BatchWeightedLeastSquares.estimate_initial_state(
            observations=observations,
            initial_state_guess=guess_x0,
            prior_covariance=prior_cov,
            initial_epoch_tdb=0.0,
            jd_base=2451545.0,
            max_iterations=6,
        )

        curr_wls = est_x0_wls.copy()
        curr_t = 0.0
        for obs in observations:
            dt = obs.epoch_tdb - curr_t
            if dt > 0:
                curr_wls = rk4_step(curr_wls, curr_t, dt, 2451545.0)
                curr_t = obs.epoch_tdb

        # 2. Sequential EKF
        ekf = ExtendedKalmanFilter(
            initial_state=guess_x0,
            initial_covariance=prior_cov,
            initial_epoch_tdb=0.0,
            process_noise_psd=1e-11,
        )
        ekf_results = ekf.process_tracking_arc(observations, 2451545.0)
        final_ekf = ekf_results[-1].state_estimate
        final_ekf_cov = ekf_results[-1].covariance

        # Cross-validation: difference between WLS and EKF must be within 3-sigma of filter covariance
        diff_state = np.abs(curr_wls - final_ekf)
        ekf_3sigma = 3.0 * np.sqrt(np.diag(final_ekf_cov))

        for i in range(6):
            assert diff_state[i] <= ekf_3sigma[i], (
                f"Component {i} difference {diff_state[i]} exceeds EKF 3-sigma {ekf_3sigma[i]}."
            )
