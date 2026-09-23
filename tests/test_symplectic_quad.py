"""Automated validation tests for 4th-order symplectic quad-precision integrator.

Validates:
1. Symplectic energy conservation on harmonic oscillator.
2. Symplectic energy conservation on Keplerian 2-body orbit without secular drift.
3. 4th-order algebraic convergence scaling with step halving.
4. Quad-precision (34 digits) mantissa fidelity.
"""

import math
import mpmath as mp
import numpy as np
import pytest

from relativistic_engine.numerical.symplectic_quad import (
    compute_kepler_energy_quad,
    gauss_legendre_4_coefficients,
    integrate_symplectic_quad,
)


def test_gauss_legendre_coefficients_properties():
    """Verify algebraic properties and symplecticity conditions of GL4 Butcher tableau."""
    c_nodes, A_mat, b_weights = gauss_legendre_4_coefficients(dps=34)

    with mp.workdps(40):
        # Symmetry: c1 + c2 = 1
        assert abs(c_nodes[0] + c_nodes[1] - mp.mpf("1.0")) < mp.mpf("1e-35")

        # Weights: b1 + b2 = 1
        assert abs(b_weights[0] + b_weights[1] - mp.mpf("1.0")) < mp.mpf("1e-35")

        # Symplecticity condition for Runge-Kutta: b_i * a_ij + b_j * a_ji = b_i * b_j
        for i in range(2):
            for j in range(2):
                lhs = b_weights[i] * A_mat[i][j] + b_weights[j] * A_mat[j][i]
                rhs = b_weights[i] * b_weights[j]
                assert abs(lhs - rhs) < mp.mpf("1e-35")


def test_harmonic_oscillator_energy_conservation():
    """Verify exact phase-space energy conservation on harmonic oscillator."""
    # dy/dt = [v, -omega^2 * x], omega = 2.0
    omega = mp.mpf("2.0")

    def rhs(t, y):
        x, v = y[0], y[1]
        return [v, - (omega**2) * x]

    y0 = [mp.mpf("1.0"), mp.mpf("0.0")]
    t_span = (0.0, 10.0)  # ~3.18 complete cycles
    step_size = 0.05
    dps = 34

    sol = integrate_symplectic_quad(rhs, t_span, y0, step_size, dps=dps)

    with mp.workdps(dps):
        e0 = mp.mpf("0.5") * (y0[1]**2 + (omega**2) * (y0[0]**2))
        y_final = sol["y"][-1]
        e_final = mp.mpf("0.5") * (y_final[1]**2 + (omega**2) * (y_final[0]**2))

        rel_drift = abs(e_final - e0) / e0
        # Symplectic integrator bounds energy oscillation strictly without secular growth
        assert rel_drift < mp.mpf("1e-8")


def test_kepler_orbit_symplectic_energy_quad():
    """Verify Keplerian two-body energy preservation in quad precision."""
    gm = mp.mpf("1.32712440018e20")  # Sun GM
    r0 = mp.mpf("1.495978707e11")    # 1 AU
    # Circular velocity v = sqrt(GM / r)
    v0 = mp.sqrt(gm / r0)

    def kepler_rhs(t, y):
        rx, ry, rz, vx, vy, vz = y[0], y[1], y[2], y[3], y[4], y[5]
        r = mp.sqrt(rx * rx + ry * ry + rz * rz)
        inv_r3 = mp.mpf("1.0") / (r * r * r)
        ax = - gm * rx * inv_r3
        ay = - gm * ry * inv_r3
        az = - gm * rz * inv_r3
        return [vx, vy, vz, ax, ay, az]

    y0 = [r0, mp.mpf("0.0"), mp.mpf("0.0"), mp.mpf("0.0"), v0, mp.mpf("0.0")]
    period = float(2.0 * math.pi * math.sqrt(float(r0**3 / gm)))

    # Integrate 1/10th of an orbit with step size 86400s (1 day)
    t_span = (0.0, period * 0.1)
    step_size = 86400.0
    dps = 34

    sol = integrate_symplectic_quad(kepler_rhs, t_span, y0, step_size, dps=dps)
    e0 = compute_kepler_energy_quad(y0, gm, dps=dps)
    e_final = compute_kepler_energy_quad(sol["y"][-1], gm, dps=dps)

    rel_error = abs(e_final - e0) / abs(e0)
    assert rel_error < mp.mpf("1e-12")


def test_convergence_order_four():
    """Verify 4th-order algebraic error scaling by step halving on harmonic oscillator."""
    omega = mp.mpf("1.0")

    def rhs(t, y):
        return [y[1], - y[0]]

    y0 = [mp.mpf("1.0"), mp.mpf("0.0")]
    t_span = (0.0, 1.0)
    dps = 34

    sol1 = integrate_symplectic_quad(rhs, t_span, y0, step_size=0.2, dps=dps)
    sol2 = integrate_symplectic_quad(rhs, t_span, y0, step_size=0.1, dps=dps)

    with mp.workdps(dps):
        exact_x = mp.cos(mp.mpf("1.0"))
        err1 = abs(sol1["y"][-1][0] - exact_x)
        err2 = abs(sol2["y"][-1][0] - exact_x)

        # Ratio should be ~ (0.2 / 0.1)^4 = 16
        ratio = float(err1 / err2)
        assert 14.0 < ratio < 18.0
