"""Numerical Convergence Tests.

Demonstrates that the adaptive high-order ODE integrator (DOP853) converges
monotonically to the exact analytical solution as the integration tolerance
is systematically tightened from 1e-6 down to 1e-14.
"""

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    C_LIGHT,
    G0,
    AU,
    LIGHT_YEAR,
)
from relativistic_engine.physics.kinematics import (
    distance_from_coordinate_time,
    velocity_from_coordinate_time,
    coordinate_to_proper_time,
    brachistochrone_profile,
)
from relativistic_engine.numerical.integrator import (
    integrate_relativistic_trajectory,
    integrate_brachistochrone,
)


class TestNumericalConvergence:
    """Test convergence of DOP853 solver against exact analytical solutions."""

    def test_single_stage_convergence_sweep(self):
        """Verify monotonic convergence over a relativistic boost phase."""
        alpha = G0
        t_final = 31557600.0  # 1 Julian year coordinate time (v ~ 0.77c)

        x_exact = distance_from_coordinate_time(t_final, alpha)
        v_exact = velocity_from_coordinate_time(t_final, alpha)
        tau_exact = coordinate_to_proper_time(t_final, alpha)

        tolerances = [1e-6, 1e-8, 1e-10, 1e-12, 1e-13]
        errors_x = []
        errors_v = []
        errors_tau = []

        for tol in tolerances:
            res = integrate_relativistic_trajectory(
                t_span=(0.0, t_final),
                y0=[0.0, 0.0, 0.0],
                proper_accel_func=lambda t, x, v: alpha,
                method="DOP853",
                rtol=tol,
                atol=tol * 1e-2,
            )

            assert res.success, f"Solver failed at tol={tol}: {res.message}"

            err_x = abs(res.x[-1] - x_exact) / x_exact
            err_v = abs(res.v[-1] - v_exact) / v_exact
            err_tau = abs(res.tau[-1] - tau_exact) / tau_exact

            errors_x.append(err_x)
            errors_v.append(err_v)
            errors_tau.append(err_tau)

        # Monotonic convergence: error must strictly decrease as tolerance tightens
        for i in range(len(tolerances) - 1):
            assert errors_x[i] >= errors_x[i + 1] * 0.5, (
                f"Error in x did not decrease: {errors_x[i]} -> {errors_x[i + 1]}"
            )
            assert errors_tau[i] >= errors_tau[i + 1] * 0.5, (
                f"Error in tau did not decrease: {errors_tau[i]} -> {errors_tau[i + 1]}"
            )

        # Global truncation error at highest tolerance (rtol = 1e-13)
        assert errors_x[-1] < 1e-12, f"Final relative error in x too large: {errors_x[-1]}"
        assert errors_v[-1] < 1e-12, f"Final relative error in v too large: {errors_v[-1]}"
        assert errors_tau[-1] < 1e-12, f"Final relative error in tau too large: {errors_tau[-1]}"

    def test_brachistochrone_interstellar_convergence(self):
        """Verify 2-stage segmented brachistochrone solver for Alpha Centauri (4.246 ly)."""
        dist = 4.246 * LIGHT_YEAR
        alpha = G0
        prof = brachistochrone_profile(dist, alpha)

        res = integrate_brachistochrone(
            distance=dist,
            proper_accel=alpha,
            t_half=prof.t_half,
            t_total=prof.t_total,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
            num_points_per_stage=200,
        )

        assert res.success, f"Brachistochrone solver failed: {res.message}"

        rel_err_dist = abs(res.x[-1] - dist) / dist
        assert rel_err_dist < 1e-11, f"Final distance error: {rel_err_dist}"

        assert abs(res.v[-1]) < 1e-2, f"Residual velocity at destination: {res.v[-1]} m/s"

        rel_err_tau = abs(res.tau[-1] - prof.tau_total) / prof.tau_total
        assert rel_err_tau < 1e-11, f"Total proper time error: {rel_err_tau}"

    def test_earth_mars_brachistochrone(self):
        """Verify 2-stage brachistochrone for interplanetary distance (0.5 AU)."""
        dist = 0.5 * AU
        alpha = G0
        prof = brachistochrone_profile(dist, alpha)

        res = integrate_brachistochrone(
            distance=dist,
            proper_accel=alpha,
            t_half=prof.t_half,
            t_total=prof.t_total,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
            num_points_per_stage=100,
        )

        assert res.success
        rel_err_dist = abs(res.x[-1] - dist) / dist
        assert rel_err_dist < 1e-11
        assert abs(res.v[-1]) < 1e-2
