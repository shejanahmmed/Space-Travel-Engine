"""Exact Analytical Relativistic Kinematics in Flat Spacetime.

This module provides closed-form, mathematically exact solutions for 1D relativistic
motion under constant proper acceleration (hyperbolic motion) in Minkowski spacetime.

Reference Frame & Coordinates:
- Coordinate Frame: Inertial observer (e.g., Solar System Barycenter or Earth at rest).
  Time: t (coordinate time, seconds)
  Position: x (coordinate distance, meters)
  Velocity: v = dx/dt (coordinate velocity, m/s)
- Spacecraft Rest Frame:
  Time: tau (proper time, seconds)
  Acceleration: alpha (proper acceleration, m/s^2)

Signatures & Conventions:
- Spacetime metric: Minkowski eta_mu_nu = diag(-c^2, 1, 1, 1)
- Line element: c^2 dtau^2 = c^2 dt^2 - dx^2 = dt^2 (c^2 - v^2)
- Proper time differential: dtau = dt / gamma = sqrt(1 - v^2/c^2) dt
- Coordinate acceleration: a(t) = dv/dt = alpha / gamma^3 = alpha * (1 - v^2/c^2)^(3/2)

Authoritative References:
- Misner, Thorne, & Wheeler (1973), "Gravitation", Chapter 6: Accelerated Observers.
- Rindler, W. (2006), "Relativity: Special, General, and Cosmological", Oxford University Press.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Tuple

from relativistic_engine.constants import C_LIGHT


def lorentz_beta(velocity: float, c: float = C_LIGHT) -> float:
    """Compute the normalized velocity beta = v / c.

    Parameters
    ----------
    velocity : float
        Coordinate velocity v in m/s.
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    float
        Dimensionless velocity ratio beta in (-1, 1).

    Raises
    ------
    ValueError
        If |velocity| >= c.
    """
    if abs(velocity) >= c:
        raise ValueError(
            f"Velocity magnitude |v| = {abs(velocity):.6e} m/s cannot equal or exceed c = {c:.6e} m/s."
        )
    return velocity / c


def lorentz_gamma(velocity: float, c: float = C_LIGHT) -> float:
    """Compute the relativistic Lorentz factor gamma = 1 / sqrt(1 - v^2 / c^2).

    Parameters
    ----------
    velocity : float
        Coordinate velocity v in m/s.
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    float
        Lorentz factor gamma >= 1.0.

    Raises
    ------
    ValueError
        If |velocity| >= c.
    """
    beta = lorentz_beta(velocity, c)
    # Factored form (1 - beta)(1 + beta) mitigates loss of significance as beta -> 1.0
    return 1.0 / math.sqrt((1.0 - beta) * (1.0 + beta))


def lorentz_gamma_minus_one(velocity: float, c: float = C_LIGHT) -> float:
    """Compute gamma - 1 with full precision, avoiding cancellation when v << c.

    Algebraic identity:
        gamma - 1 = beta^2 / (sqrt(1 - beta^2) * (1 + sqrt(1 - beta^2)))

    Parameters
    ----------
    velocity : float
        Coordinate velocity v in m/s.
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    float
        gamma - 1 (>= 0.0).
    """
    beta = lorentz_beta(velocity, c)
    if beta == 0.0:
        return 0.0
    beta_sq = beta * beta
    sqrt_factor = math.sqrt((1.0 - beta) * (1.0 + beta))
    return beta_sq / (sqrt_factor * (1.0 + sqrt_factor))


def proper_time_deficit(t: float, alpha: float, c: float = C_LIGHT) -> float:
    """Compute coordinate time minus proper time (t - tau) stably across all scales.

    When alpha * t / c << 1, standard subtraction (t - tau) suffers catastrophic
    floating point cancellation. This function uses a high-order Taylor expansion
    for small arguments:
        y - asinh(y) = y^3/6 - 3*y^5/40 + 5*y^7/112 - 35*y^9/1152 + ...

    Parameters
    ----------
    t : float
        Coordinate time in seconds (t >= 0).
    alpha : float
        Constant proper acceleration in m/s^2 (alpha > 0).
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    float
        Time deficit (t - tau) in seconds (>= 0.0).
    """
    if t < 0.0:
        raise ValueError(f"Coordinate time t must be non-negative, got {t}.")
    if alpha <= 0.0:
        raise ValueError(f"Proper acceleration alpha must be strictly positive, got {alpha}.")

    y = (alpha * t) / c
    if y < 1e-3:
        # Truncated expansion of y - asinh(y) around y = 0 preserves 53-bit precision when y < 1e-3
        y2 = y * y
        series = (y**3) * (
            (1.0 / 6.0)
            - (3.0 / 40.0) * y2
            + (5.0 / 112.0) * (y2**2)
            - (35.0 / 1152.0) * (y2**3)
        )
        return (c / alpha) * series
    else:
        tau = (c / alpha) * math.asinh(y)
        return t - tau


def proper_to_coordinate_time(tau: float, alpha: float, c: float = C_LIGHT) -> float:
    """Compute coordinate time t from traveler proper time tau under constant proper acceleration.

    Formula:
        t(tau) = (c / alpha) * sinh(alpha * tau / c)

    Parameters
    ----------
    tau : float
        Traveler proper time in seconds (tau >= 0).
    alpha : float
        Constant proper acceleration in m/s^2 (alpha > 0).
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    float
        Coordinate time t in seconds.
    """
    if tau < 0.0:
        raise ValueError(f"Proper time tau must be non-negative, got {tau}.")
    if alpha <= 0.0:
        raise ValueError(f"Proper acceleration alpha must be strictly positive, got {alpha}.")

    theta = (alpha * tau) / c
    return (c / alpha) * math.sinh(theta)


def coordinate_to_proper_time(t: float, alpha: float, c: float = C_LIGHT) -> float:
    """Compute traveler proper time tau from coordinate time t under constant proper acceleration.

    Formula:
        tau(t) = (c / alpha) * asinh(alpha * t / c)

    Parameters
    ----------
    t : float
        Coordinate time in seconds (t >= 0).
    alpha : float
        Constant proper acceleration in m/s^2 (alpha > 0).
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    float
        Traveler proper time tau in seconds.
    """
    if t < 0.0:
        raise ValueError(f"Coordinate time t must be non-negative, got {t}.")
    if alpha <= 0.0:
        raise ValueError(f"Proper acceleration alpha must be strictly positive, got {alpha}.")

    theta = (alpha * t) / c
    return (c / alpha) * math.asinh(theta)


def distance_from_coordinate_time(t: float, alpha: float, c: float = C_LIGHT) -> float:
    """Compute coordinate distance x traversed from rest at t = 0 under constant proper acceleration.

    Formula:
        x(t) = (c^2 / alpha) * (sqrt(1 + (alpha * t / c)^2) - 1)

    Numerically stabilized for small (alpha * t / c) to avoid catastrophic cancellation:
        x(t) = (c^2 / alpha) * [ y^2 / (sqrt(1 + y^2) + 1) ] where y = alpha * t / c.

    Parameters
    ----------
    t : float
        Coordinate time in seconds (t >= 0).
    alpha : float
        Constant proper acceleration in m/s^2 (alpha > 0).
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    float
        Traversed coordinate distance in meters.
    """
    if t < 0.0:
        raise ValueError(f"Coordinate time t must be non-negative, got {t}.")
    if alpha <= 0.0:
        raise ValueError(f"Proper acceleration alpha must be strictly positive, got {alpha}.")

    y = (alpha * t) / c
    # Rationalized form y^2 / (hypot(1, y) + 1) avoids catastrophic cancellation for y << 1
    hypot_minus_one = (y * y) / (math.hypot(1.0, y) + 1.0)
    return (c * c / alpha) * hypot_minus_one


def distance_from_proper_time(tau: float, alpha: float, c: float = C_LIGHT) -> float:
    """Compute coordinate distance x traversed from rest under constant proper acceleration.

    Formula:
        x(tau) = (c^2 / alpha) * (cosh(alpha * tau / c) - 1)

    Numerically stabilized for small (alpha * tau / c):
        cosh(z) - 1 == 2 * sinh^2(z / 2)

    Parameters
    ----------
    tau : float
        Traveler proper time in seconds (tau >= 0).
    alpha : float
        Constant proper acceleration in m/s^2 (alpha > 0).
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    float
        Traversed coordinate distance in meters.
    """
    if tau < 0.0:
        raise ValueError(f"Proper time tau must be non-negative, got {tau}.")
    if alpha <= 0.0:
        raise ValueError(f"Proper acceleration alpha must be strictly positive, got {alpha}.")

    z = (alpha * tau) / c
    # Half-angle identity cosh(z) - 1 = 2*sinh^2(z/2) eliminates cancellation for small z
    s = math.sinh(0.5 * z)
    return (c * c / alpha) * (2.0 * s * s)


def velocity_from_coordinate_time(t: float, alpha: float, c: float = C_LIGHT) -> float:
    """Compute coordinate velocity v at coordinate time t under constant proper acceleration.

    Formula:
        v(t) = (alpha * t) / sqrt(1 + (alpha * t / c)^2)

    Parameters
    ----------
    t : float
        Coordinate time in seconds (t >= 0).
    alpha : float
        Constant proper acceleration in m/s^2 (alpha > 0).
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    float
        Coordinate velocity v in m/s (0 <= v < c).
    """
    if t < 0.0:
        raise ValueError(f"Coordinate time t must be non-negative, got {t}.")
    if alpha <= 0.0:
        raise ValueError(f"Proper acceleration alpha must be strictly positive, got {alpha}.")

    y = (alpha * t) / c
    return (alpha * t) / math.hypot(1.0, y)


def velocity_from_proper_time(tau: float, alpha: float, c: float = C_LIGHT) -> float:
    """Compute coordinate velocity v at proper time tau under constant proper acceleration.

    Formula:
        v(tau) = c * tanh(alpha * tau / c)

    Parameters
    ----------
    tau : float
        Traveler proper time in seconds (tau >= 0).
    alpha : float
        Constant proper acceleration in m/s^2 (alpha > 0).
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    float
        Coordinate velocity v in m/s (0 <= v < c).
    """
    if tau < 0.0:
        raise ValueError(f"Proper time tau must be non-negative, got {tau}.")
    if alpha <= 0.0:
        raise ValueError(f"Proper acceleration alpha must be strictly positive, got {alpha}.")

    z = (alpha * tau) / c
    return c * math.tanh(z)


def coordinate_time_from_distance(distance: float, alpha: float, c: float = C_LIGHT) -> float:
    """Compute coordinate time t required to traverse coordinate distance under constant proper acceleration.

    Formula:
        t(x) = sqrt( 2 * x / alpha + (x / c)^2 )

    Note that as c -> inf, t -> sqrt(2 * x / alpha), which recovers the exact Newtonian limit.

    Parameters
    ----------
    distance : float
        Coordinate distance in meters (distance >= 0).
    alpha : float
        Constant proper acceleration in m/s^2 (alpha > 0).
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    float
        Coordinate time t in seconds.
    """
    if distance < 0.0:
        raise ValueError(f"Distance must be non-negative, got {distance}.")
    if alpha <= 0.0:
        raise ValueError(f"Proper acceleration alpha must be strictly positive, got {alpha}.")

    term_newton = (2.0 * distance) / alpha
    term_rel = (distance / c) ** 2
    return math.sqrt(term_newton + term_rel)


def proper_time_from_distance(distance: float, alpha: float, c: float = C_LIGHT) -> float:
    """Compute traveler proper time tau required to traverse coordinate distance under constant proper acceleration.

    Formula:
        tau(x) = (c / alpha) * acosh(1 + alpha * x / c^2)

    Numerically stabilized for small u = alpha * x / c^2 using:
        acosh(1 + u) == asinh(sqrt(2 * u + u^2))

    Parameters
    ----------
    distance : float
        Coordinate distance in meters (distance >= 0).
    alpha : float
        Constant proper acceleration in m/s^2 (alpha > 0).
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    float
        Traveler proper time tau in seconds.
    """
    if distance < 0.0:
        raise ValueError(f"Distance must be non-negative, got {distance}.")
    if alpha <= 0.0:
        raise ValueError(f"Proper acceleration alpha must be strictly positive, got {alpha}.")

    u = (alpha * distance) / (c * c)
    # Logarithmic identity acosh(1 + u) = asinh(sqrt(2u + u^2)) avoids loss of significance near u = 0
    arg = math.sqrt(2.0 * u + u * u)
    return (c / alpha) * math.asinh(arg)


@dataclass(frozen=True)
class BrachistochroneProfile:
    """Exact kinematic summary of a 1D relativistic brachistochrone trajectory.

    A brachistochrone trajectory consists of continuous proper acceleration +alpha
    from rest to the midpoint D / 2, followed immediately by continuous proper
    deceleration -alpha from D / 2 to D, arriving at rest (v = 0).

    Attributes
    ----------
    distance : float
        Total journey coordinate distance in meters.
    proper_accel : float
        Proper acceleration magnitude alpha in m/s^2.
    t_half : float
        Coordinate time to midpoint in seconds.
    t_total : float
        Total journey coordinate time in seconds (2 * t_half).
    tau_half : float
        Proper time to midpoint in seconds.
    tau_total : float
        Total journey traveler proper time in seconds (2 * tau_half).
    v_peak : float
        Peak coordinate velocity attained at midpoint in m/s.
    beta_peak : float
        Peak normalized velocity v_peak / c.
    gamma_peak : float
        Peak Lorentz factor at midpoint.
    """

    distance: float
    proper_accel: float
    t_half: float
    t_total: float
    tau_half: float
    tau_total: float
    v_peak: float
    beta_peak: float
    gamma_peak: float


@dataclass(frozen=True)
class TrajectoryState:
    """Exact instantaneous physical state along a trajectory at coordinate time t.

    Attributes
    ----------
    t : float
        Coordinate time in seconds.
    x : float
        Coordinate position in meters.
    v : float
        Coordinate velocity in m/s.
    tau : float
        Traveler proper time in seconds.
    gamma : float
        Lorentz factor gamma.
    alpha_coord : float
        Coordinate acceleration dv/dt in m/s^2.
    """

    t: float
    x: float
    v: float
    tau: float
    gamma: float
    alpha_coord: float


def brachistochrone_profile(
    distance: float, proper_accel: float, c: float = C_LIGHT
) -> BrachistochroneProfile:
    """Compute the exact relativistic brachistochrone profile for distance D.

    Parameters
    ----------
    distance : float
        Total coordinate distance in meters (D > 0).
    proper_accel : float
        Proper acceleration magnitude in m/s^2 (alpha > 0).
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    BrachistochroneProfile
        Complete analytical summary of the journey.
    """
    if distance <= 0.0:
        raise ValueError(f"Journey distance must be strictly positive, got {distance}.")
    if proper_accel <= 0.0:
        raise ValueError(f"Proper acceleration must be strictly positive, got {proper_accel}.")

    d_half = 0.5 * distance
    t_half = coordinate_time_from_distance(d_half, proper_accel, c)
    t_total = 2.0 * t_half

    tau_half = proper_time_from_distance(d_half, proper_accel, c)
    tau_total = 2.0 * tau_half

    v_peak = velocity_from_coordinate_time(t_half, proper_accel, c)
    beta_peak = v_peak / c
    gamma_peak = 1.0 + (proper_accel * d_half) / (c * c)

    return BrachistochroneProfile(
        distance=distance,
        proper_accel=proper_accel,
        t_half=t_half,
        t_total=t_total,
        tau_half=tau_half,
        tau_total=tau_total,
        v_peak=v_peak,
        beta_peak=beta_peak,
        gamma_peak=gamma_peak,
    )


def evaluate_brachistochrone_state(
    distance: float, proper_accel: float, t: float, c: float = C_LIGHT
) -> TrajectoryState:
    """Compute the exact instantaneous physical state at coordinate time t.

    Parameters
    ----------
    distance : float
        Total coordinate distance in meters (D > 0).
    proper_accel : float
        Proper acceleration magnitude in m/s^2 (alpha > 0).
    t : float
        Coordinate time in seconds (0 <= t <= t_total).
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    TrajectoryState
        Instantaneous state: (t, x, v, tau, gamma, alpha_coord).
    """
    profile = brachistochrone_profile(distance, proper_accel, c)

    if t < 0.0 or t > profile.t_total:
        raise ValueError(
            f"Coordinate time t = {t} must be in range [0, {profile.t_total}]."
        )

    t_half = profile.t_half

    if t <= t_half:
        x = distance_from_coordinate_time(t, proper_accel, c)
        v = velocity_from_coordinate_time(t, proper_accel, c)
        tau = coordinate_to_proper_time(t, proper_accel, c)
        gamma = lorentz_gamma(v, c)
        alpha_coord = proper_accel / (gamma**3)
    else:
        # Deceleration to rest is isomorphic to time-reversed acceleration from rest
        delta_t_from_end = profile.t_total - t
        x_from_end = distance_from_coordinate_time(delta_t_from_end, proper_accel, c)
        x = distance - x_from_end
        v = velocity_from_coordinate_time(delta_t_from_end, proper_accel, c)
        tau_from_end = coordinate_to_proper_time(delta_t_from_end, proper_accel, c)
        tau = profile.tau_total - tau_from_end
        gamma = lorentz_gamma(v, c)
        alpha_coord = -proper_accel / (gamma**3)

    return TrajectoryState(
        t=t,
        x=x,
        v=v,
        tau=tau,
        gamma=gamma,
        alpha_coord=alpha_coord,
    )
