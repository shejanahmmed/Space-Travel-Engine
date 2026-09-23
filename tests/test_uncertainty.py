"""Verification and Benchmark Suite for Uncertainty Propagation & GUM Reporting.

Validation Levels:
- Level 1: GUM significant figure formatting and scientific rounding rules.
- Level 2: Gravity gradient tensor invariants (trace = 0 Laplace equation, symmetry).
- Level 3: Flat-space analytical State Transition Matrix (STM) match.
- Level 4: STM covariance mapping vs independent Monte Carlo ensemble.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    C_LIGHT,
    G0,
    AU,
    SEC_PER_DAY,
    GM_SUN,
)
from relativistic_engine.uncertainty.formatter import (
    format_uncertainty,
    UncertainQuantity,
)
from relativistic_engine.uncertainty.variational import (
    evaluate_gravity_gradient,
    propagate_with_stm,
)
from relativistic_engine.uncertainty.monte_carlo import run_monte_carlo_ensemble
from relativistic_engine.trajectory.profiles import TwoStageThrustProfile


# ==============================================================================
# LEVEL 1: GUM SIGNIFICANT FIGURE FORMATTER BENCHMARKS
# ==============================================================================


def test_level1_gum_formatting_standard_cases():
    """Level 1: Verify GUM-compliant rounding for standard physical quantities."""
    # 1. Decimal rounding (2 significant figures for uncertainty: 0.0231 -> 0.023, value -> 4.498)
    formatted = format_uncertainty(4.498064, 0.0231, "days")
    assert formatted == "(4.498 ± 0.023) days"

    # 2. Integer rounding
    formatted_m = format_uncertainty(123456.7, 45.2, "m")
    assert formatted_m == "(123457 ± 45) m"

    # 3. Scientific notation for high velocity
    formatted_v = format_uncertainty(1.905335e6, 2500.0, "m/s")
    assert "x 10^6 m/s" in formatted_v
    assert "1.9053" in formatted_v

    # 4. Exact quantity
    formatted_exact = format_uncertainty(299792458.0, 0.0, "m/s")
    assert "exact" in formatted_exact

    # 5. Negative uncertainty raises ValueError
    with pytest.raises(ValueError):
        format_uncertainty(10.0, -1.0)


def test_level1_uncertain_quantity_dataclass():
    """Level 1: Verify UncertainQuantity dataclass properties and string conversion."""
    q = UncertainQuantity(value=4.498064, uncertainty=0.0231, unit="days")
    assert str(q) == "(4.498 ± 0.023) days"
    assert np.isclose(q.relative_uncertainty, 0.0231 / 4.498064, rtol=1e-6)


# ==============================================================================
# LEVEL 2: GRAVITY GRADIENT TENSOR INVARIANTS
# ==============================================================================


def test_level2_gravity_gradient_trace_and_symmetry():
    """Level 2: Vacuum gravity gradient tensor must be traceless (Laplace eq) and symmetric."""
    # Test at 1 AU from Sun at J2000
    r_test = np.array([AU, 0.0, 0.0])
    jd_j2000 = 2451545.0

    g_mat = evaluate_gravity_gradient(r_test, jd_j2000, bodies=["sun"])

    # 1. Symmetry: G_jk = G_kj
    assert np.allclose(g_mat, g_mat.T, atol=1e-25), "Gravity gradient tensor is not symmetric"

    # 2. Laplace equation: Tr(G) = d^2w/dx^2 + d^2w/dy^2 + d^2w/dz^2 = 0 in vacuum
    trace_val = float(np.trace(g_mat))
    diag_norm = float(np.linalg.norm(np.diag(g_mat)))

    # Tr(G) / ||diag(G)|| must be near machine zero (< 1e-14)
    rel_trace = abs(trace_val) / diag_norm
    assert rel_trace < 1e-14, f"Laplace equation violated: relative trace = {rel_trace:.2e}"


# ==============================================================================
# LEVEL 3: FLAT-SPACE ANALYTICAL STM BENCHMARK
# ==============================================================================


def test_level3_flat_space_analytical_stm():
    """Level 3: In zero-g ballistic flight, STM must match analytical [I, t*I; 0, I]."""
    r0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([1000.0, 2000.0, -500.0])
    t_flight = 10000.0  # 10,000 s

    res = propagate_with_stm(
        r0=r0,
        v0=v0,
        t_span=(0.0, t_flight),
        thrust_func=None,
        epoch_jd_tdb=None,  # Pure flat space
        rtol=1e-10,
        atol=1e-11,
    )

    phi = res.stm

    # Position-position block: d r(t) / d r0 = I_3x3
    phi_rr = phi[0:3, 0:3]
    assert np.allclose(phi_rr, np.eye(3), rtol=1e-8, atol=1e-8)

    # Position-velocity block: d r(t) / d v0 = t * I_3x3
    phi_rv = phi[0:3, 3:6]
    expected_phi_rv = t_flight * np.eye(3)
    assert np.allclose(phi_rv, expected_phi_rv, rtol=1e-8, atol=1e-8)

    # Velocity-position block: d v(t) / d r0 = 0_3x3
    phi_vr = phi[3:6, 0:3]
    assert np.allclose(phi_vr, np.zeros((3, 3)), atol=1e-8)

    # Velocity-velocity block: d v(t) / d v0 = I_3x3
    phi_vv = phi[3:6, 3:6]
    assert np.allclose(phi_vv, np.eye(3), rtol=1e-8, atol=1e-8)


# ==============================================================================
# LEVEL 4: STM COVARIANCE MAPPING VS MONTE CARLO ENSEMBLE
# ==============================================================================


def test_level4_stm_covariance_vs_monte_carlo():
    """Level 4: STM covariance mapping matches empirical Monte Carlo standard deviations."""
    # Cruise trajectory with 1g thrust for 1 day
    r0 = np.array([AU, 0.0, 0.0])
    v0 = np.array([0.0, 30000.0, 0.0])
    t_flight = 86400.0  # 1 day
    alpha = G0

    # Constant 1g thrust along +x
    profile = TwoStageThrustProfile(
        t1=t_flight,
        t2=0.0,
        a1=np.array([alpha, 0.0, 0.0]),
        a2=np.zeros(3),
    )

    # Initial uncertainties
    sigma_r0 = 5000.0  # 5 km
    sigma_v0 = 1.0     # 1 m/s

    # 1. STM Propagation
    stm_res = propagate_with_stm(
        r0=r0,
        v0=v0,
        t_span=(0.0, t_flight),
        thrust_func=profile,
        epoch_jd_tdb=None,  # Flat space for clean comparison
        rtol=1e-8,
        atol=1e-9,
    )

    # Initial 7x7 covariance P0
    p0 = np.zeros((7, 7), dtype=np.float64)
    p0[0:3, 0:3] = (sigma_r0 ** 2) * np.eye(3)
    p0[3:6, 3:6] = (sigma_v0 ** 2) * np.eye(3)

    p_arrival = stm_res.propagate_covariance(p0)
    sigma_r_stm = np.sqrt(np.diag(p_arrival[0:3, 0:3]))
    sigma_v_stm = np.sqrt(np.diag(p_arrival[3:6, 3:6]))

    # 2. Monte Carlo Ensemble (N = 100 samples)
    mc_res = run_monte_carlo_ensemble(
        r0=r0,
        v0=v0,
        t_span=(0.0, t_flight),
        thrust_profile=profile,
        sigma_r0=sigma_r0,
        sigma_v0=sigma_v0,
        num_samples=100,
        seed=123,
        epoch_jd_tdb=None,
        rtol=1e-7,
        atol=1e-8,
    )

    # 3. Compare standard deviations
    # For N = 100, standard error of the sample std is ~ 1/sqrt(2N) ~ 7%
    rel_err_r = np.abs(mc_res.std_position - sigma_r_stm) / sigma_r_stm
    rel_err_v = np.abs(mc_res.std_velocity - sigma_v_stm) / sigma_v_stm

    # Check that all components match within 20% (3 standard errors of N=100)
    assert np.all(rel_err_r < 0.20), f"Position std discrepancy too large: {rel_err_r}"
    assert np.all(rel_err_v < 0.20), f"Velocity std discrepancy too large: {rel_err_v}"
