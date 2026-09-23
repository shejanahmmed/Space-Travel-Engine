"""Level 1 Validation: Exact Analytical Relativistic Kinematics Tests.

Validates the analytical physics module against:
1. Exact mathematical invertibility identities (f(f^-1(y)) == y).
2. Asymptotic limits (Newtonian v << c and ultra-relativistic v -> c).
3. Independent cross-formula consistency (x(t(tau)) vs x(tau)).
4. High-precision arbitrary precision benchmark using mpmath (50 decimal digits).
"""

import math
import mpmath
import pytest

from relativistic_engine.constants import (
    C_LIGHT,
    G0,
    AU,
    LIGHT_YEAR,
)
from relativistic_engine.physics.kinematics import (
    lorentz_beta,
    lorentz_gamma,
    lorentz_gamma_minus_one,
    proper_to_coordinate_time,
    coordinate_to_proper_time,
    proper_time_deficit,
    distance_from_coordinate_time,
    distance_from_proper_time,
    velocity_from_coordinate_time,
    velocity_from_proper_time,
    coordinate_time_from_distance,
    proper_time_from_distance,
    brachistochrone_profile,
    evaluate_brachistochrone_state,
)


class TestLorentzFactors:
    """Tests for Lorentz gamma and beta transformations."""

    def test_lorentz_zero_velocity(self):
        assert lorentz_beta(0.0) == 0.0
        assert lorentz_gamma(0.0) == 1.0
        assert lorentz_gamma_minus_one(0.0) == 0.0

    def test_lorentz_sub_relativistic(self):
        # 30 km/s (typical Earth orbital speed)
        v = 30_000.0
        beta = lorentz_beta(v)
        gamma = lorentz_gamma(v)
        gamma_minus_one = lorentz_gamma_minus_one(v)

        # Expected gamma - 1 ~ 0.5 * beta^2 + 0.375 * beta^4
        expected_excess = 0.5 * (beta**2) + 0.375 * (beta**4)
        assert math.isclose(gamma_minus_one, expected_excess, rel_tol=1e-12)
        # Verify gamma is consistent with 1 + gamma_minus_one to machine precision
        assert math.isclose(gamma, 1.0 + gamma_minus_one, rel_tol=1e-15)

    def test_lorentz_high_relativistic(self):
        # v = 0.8660254037844386 c => beta^2 = 0.75 => gamma = 2.0
        v = 0.5 * math.sqrt(3.0) * C_LIGHT
        assert math.isclose(lorentz_gamma(v), 2.0, rel_tol=1e-14)

    def test_lorentz_velocity_exceeding_c_raises(self):
        with pytest.raises(ValueError):
            lorentz_gamma(C_LIGHT)
        with pytest.raises(ValueError):
            lorentz_gamma(C_LIGHT * 1.01)
        with pytest.raises(ValueError):
            lorentz_gamma(-C_LIGHT)


class TestInvertibilityIdentities:
    """Verify that forward and backward transformations invert to machine precision."""

    @pytest.mark.parametrize("alpha", [0.1, G0, 100.0])
    @pytest.mark.parametrize("tau", [1.0, 3600.0, 86400.0, 31557600.0, 1e8])
    def test_time_inversion(self, alpha: float, tau: float):
        t = proper_to_coordinate_time(tau, alpha)
        tau_recovered = coordinate_to_proper_time(t, alpha)
        rel_err = abs(tau_recovered - tau) / tau
        assert rel_err < 1e-14, f"Time inversion failed for tau={tau}, rel_err={rel_err}"

    @pytest.mark.parametrize("alpha", [G0, 20.0])
    @pytest.mark.parametrize("distance", [1e6, AU, 4.246 * LIGHT_YEAR])
    def test_distance_time_inversion(self, alpha: float, distance: float):
        t = coordinate_time_from_distance(distance, alpha)
        x_recovered = distance_from_coordinate_time(t, alpha)
        rel_err = abs(x_recovered - distance) / distance
        assert rel_err < 1e-14, f"Distance-time inversion failed for D={distance}, rel_err={rel_err}"

    @pytest.mark.parametrize("alpha", [G0, 20.0])
    @pytest.mark.parametrize("distance", [1e6, AU, 4.246 * LIGHT_YEAR])
    def test_distance_proper_time_inversion(self, alpha: float, distance: float):
        tau = proper_time_from_distance(distance, alpha)
        x_recovered = distance_from_proper_time(tau, alpha)
        rel_err = abs(x_recovered - distance) / distance
        assert rel_err < 1e-14, f"Distance-tau inversion failed for D={distance}, rel_err={rel_err}"


class TestCrossFormulaConsistency:
    """Verify that alternative mathematical formulations agree to machine precision."""

    @pytest.mark.parametrize("tau", [100.0, 86400.0, 1e7, 1e8])
    def test_distance_consistency(self, tau: float):
        alpha = G0
        t = proper_to_coordinate_time(tau, alpha)
        x_from_t = distance_from_coordinate_time(t, alpha)
        x_from_tau = distance_from_proper_time(tau, alpha)
        rel_err = abs(x_from_t - x_from_tau) / x_from_tau
        assert rel_err < 1e-14, f"x(t) vs x(tau) mismatch: {rel_err}"

    @pytest.mark.parametrize("tau", [100.0, 86400.0, 1e7, 1e8])
    def test_velocity_consistency(self, tau: float):
        alpha = G0
        t = proper_to_coordinate_time(tau, alpha)
        v_from_t = velocity_from_coordinate_time(t, alpha)
        v_from_tau = velocity_from_proper_time(tau, alpha)
        rel_err = abs(v_from_t - v_from_tau) / v_from_tau
        assert rel_err < 1e-14, f"v(t) vs v(tau) mismatch: {rel_err}"


class TestPhysicalLimits:
    """Verify classical Newtonian limits and ultra-relativistic limits."""

    def test_newtonian_limit(self):
        # Short time and small acceleration: t = 100 s, alpha = 1 m/s^2
        t = 100.0
        alpha = 1.0
        v_exact = velocity_from_coordinate_time(t, alpha)
        x_exact = distance_from_coordinate_time(t, alpha)
        tau_exact = coordinate_to_proper_time(t, alpha)

        v_newton = alpha * t  # 100 m/s
        x_newton = 0.5 * alpha * (t**2)  # 5000 m

        # Relative difference should scale as ~ (v/c)^2 ~ 1e-13
        assert math.isclose(v_exact, v_newton, rel_tol=1e-12)
        assert math.isclose(x_exact, x_newton, rel_tol=1e-12)

        # First relativistic correction to proper time: t - tau ~ alpha^2 * t^3 / (6 c^2)
        dt_expected = (alpha**2 * t**3) / (6.0 * C_LIGHT**2)
        dt_deficit = proper_time_deficit(t, alpha)
        assert math.isclose(dt_deficit, dt_expected, rel_tol=1e-10)

        # At longer times (e.g. t = 1e6 s, ~11.5 days), direct subtraction is stable:
        t_long = 1e6
        tau_long = coordinate_to_proper_time(t_long, alpha)
        dt_long_deficit = proper_time_deficit(t_long, alpha)
        assert math.isclose(t_long - tau_long, dt_long_deficit, rel_tol=1e-10)

    def test_ultra_relativistic_limit(self):
        # High acceleration over 10 years coordinate time
        alpha = 10.0 * G0
        t = 10.0 * 31557600.0
        v = velocity_from_coordinate_time(t, alpha)
        # Velocity must approach c from below
        assert 0.0 < v < C_LIGHT
        assert (C_LIGHT - v) / C_LIGHT < 1e-3


class TestBrachistochroneProfile:
    """Test the complete 2-stage turnaround trajectory."""

    def test_earth_mars_scale(self):
        # Earth to Mars at opposition ~ 0.5 AU
        dist = 0.5 * AU
        alpha = G0
        prof = brachistochrone_profile(dist, alpha)

        # Profile sanity checks
        assert prof.t_total == 2.0 * prof.t_half
        assert prof.tau_total == 2.0 * prof.tau_half
        assert prof.tau_total < prof.t_total
        assert 0.0 < prof.beta_peak < 1.0

        # Boundary checks at t = 0, t = t_half, t = t_total
        s_start = evaluate_brachistochrone_state(dist, alpha, 0.0)
        assert s_start.x == 0.0
        assert s_start.v == 0.0
        assert s_start.tau == 0.0

        s_mid = evaluate_brachistochrone_state(dist, alpha, prof.t_half)
        assert math.isclose(s_mid.x, 0.5 * dist, rel_tol=1e-14)
        assert math.isclose(s_mid.v, prof.v_peak, rel_tol=1e-14)
        assert math.isclose(s_mid.tau, prof.tau_half, rel_tol=1e-14)

        s_end = evaluate_brachistochrone_state(dist, alpha, prof.t_total)
        assert math.isclose(s_end.x, dist, rel_tol=1e-14)
        assert math.isclose(s_end.v, 0.0, abs_tol=1e-6)
        assert math.isclose(s_end.tau, prof.tau_total, rel_tol=1e-14)

    def test_interstellar_scale_alpha_centauri(self):
        # Alpha Centauri ~ 4.246 ly
        dist = 4.246 * LIGHT_YEAR
        alpha = G0
        prof = brachistochrone_profile(dist, alpha)

        # For 4.246 ly at 1g brachistochrone:
        # Coordinate time is ~ 5.8-6.0 years, proper time is ~ 3.5-3.6 years
        t_years = prof.t_total / 31557600.0
        tau_years = prof.tau_total / 31557600.0

        assert 5.5 < t_years < 6.5
        assert 3.2 < tau_years < 4.0
        assert prof.beta_peak > 0.9  # Relativistic cruise


class TestArbitraryPrecisionMpmath:
    """Verify IEEE 754 float64 accuracy against mpmath (50 decimal digits)."""

    def test_mpmath_verification(self):
        mpmath.mp.dps = 50
        c_mp = mpmath.mpf(str(C_LIGHT))
        alpha_mp = mpmath.mpf(str(G0))

        # Alpha Centauri distance in mpmath
        dist_mp = mpmath.mpf("4.246") * mpmath.mpf(str(LIGHT_YEAR))
        d_half_mp = dist_mp / 2

        # Exact coordinate time to midpoint in mpmath:
        # t = sqrt(2 * d_half / alpha + (d_half / c)^2)
        t_half_mp = mpmath.sqrt((2 * d_half_mp / alpha_mp) + (d_half_mp / c_mp) ** 2)

        # Exact proper time to midpoint in mpmath:
        # tau = (c / alpha) * asinh(alpha * t_half / c)
        theta_mp = (alpha_mp * t_half_mp) / c_mp
        tau_half_mp = (c_mp / alpha_mp) * mpmath.asinh(theta_mp)

        # Compare with float64 implementation
        prof_float = brachistochrone_profile(float(dist_mp), G0)

        rel_err_t = abs(prof_float.t_half - float(t_half_mp)) / float(t_half_mp)
        rel_err_tau = abs(prof_float.tau_half - float(tau_half_mp)) / float(tau_half_mp)

        # Must agree to near machine epsilon (< 2e-15)
        assert rel_err_t < 2e-15, f"t_half relative error vs mpmath: {rel_err_t}"
        assert rel_err_tau < 2e-15, f"tau_half relative error vs mpmath: {rel_err_tau}"
