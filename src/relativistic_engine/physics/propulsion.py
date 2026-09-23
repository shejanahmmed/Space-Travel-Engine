"""Relativistic rocket dynamics, variable-mass mechanics, and exhaust thermodynamics.

Implements exact analytical formulations for relativistic rocket propulsion:
- Ackeret (1946) relativistic rocket equation and its inverse.
- Ideal photon rocket mechanics (Sanger 1953).
- Relativistic exhaust kinetic power and mass-flow coupling.
- Rest-frame thrust to BCRS coordinate acceleration mapping.

Authoritative References:
- Ackeret, J. (1946), "Zur Theorie der Raketen", Helvetica Physica Acta, 19, 103-112.
- Sanger, E. (1953), "Zur Theorie der Photonenraketen", Ingenieur-Archiv, 21(3), 213-226.
- Misner, C. W., Thorne, K. S., & Wheeler, J. A. (1973), "Gravitation", W. H. Freeman, §6.2.
"""

from __future__ import annotations

from typing import Optional, Tuple
import math
import numpy as np

from relativistic_engine.constants import C_LIGHT, G0
from relativistic_engine.physics.dynamics import proper_to_coordinate_acceleration


def _resolve_exhaust_velocity(isp: Optional[float], ve: Optional[float]) -> float:
    """Resolve effective exhaust velocity from Isp or direct velocity input.

    Parameters
    ----------
    isp : float, optional
        Specific impulse in seconds (relative to standard g0 = 9.80665 m/s^2).
    ve : float, optional
        Effective exhaust velocity in m/s.

    Returns
    -------
    float
        Effective exhaust velocity in m/s.
    """
    if ve is not None and isp is not None:
        raise ValueError("Specify either 'isp' or 've', not both.")
    if ve is not None:
        if ve <= 0.0 or ve > C_LIGHT:
            raise ValueError(f"Exhaust velocity must satisfy 0 < ve <= c, got {ve:.6e} m/s")
        return float(ve)
    if isp is not None:
        if isp <= 0.0:
            raise ValueError(f"Specific impulse must be positive, got {isp:.6e} s")
        v = float(isp) * G0
        if v > C_LIGHT:
            raise ValueError(f"Implied exhaust velocity {v:.6e} m/s exceeds speed of light")
        return v
    raise ValueError("Must specify either 'isp' or 've'.")


def relativistic_rocket_velocity(
    m0: float,
    mf: float,
    isp: Optional[float] = None,
    ve: Optional[float] = None,
) -> float:
    """Compute terminal coordinate velocity from the Ackeret (1946) relativistic rocket equation.

    For a rocket expelling propellant at constant exhaust velocity v_e = beta_e * c
    in its instantaneous rest frame, the coordinate velocity reached in flat spacetime is:

        beta = tanh( beta_e * ln(m0 / mf) ) = tanh(theta)

    where theta = beta_e * ln(m0 / mf) is the trajectory rapidity.

    Parameters
    ----------
    m0 : float
        Initial wet mass in kilograms (m0 > 0).
    mf : float
        Final dry mass in kilograms (0 < mf <= m0).
    isp : float, optional
        Specific impulse in seconds.
    ve : float, optional
        Effective exhaust velocity in m/s.

    Returns
    -------
    float
        Terminal coordinate velocity in m/s.
    """
    if m0 <= 0.0 or mf <= 0.0:
        raise ValueError(f"Masses must be positive: m0={m0}, mf={mf}")
    if mf > m0:
        raise ValueError(f"Final mass mf ({mf}) cannot exceed initial mass m0 ({m0})")

    v_exhaust = _resolve_exhaust_velocity(isp=isp, ve=ve)
    beta_e = v_exhaust / C_LIGHT
    mass_ratio = m0 / mf

    rapidity = beta_e * math.log(mass_ratio)
    # math.tanh is numerically stable across all non-negative rapidity arguments
    beta = math.tanh(rapidity)
    return beta * C_LIGHT


def relativistic_mass_ratio(
    beta: float,
    isp: Optional[float] = None,
    ve: Optional[float] = None,
) -> float:
    """Compute required propellant mass ratio m0 / mf from target velocity beta = v / c.

    Inverts Ackeret's equation:
        mu = m0 / mf = exp( artanh(beta) / beta_e ) = ( (1 + beta) / (1 - beta) )^(1 / (2 * beta_e))

    Parameters
    ----------
    beta : float
        Target velocity fraction v / c (0 <= beta < 1).
    isp : float, optional
        Specific impulse in seconds.
    ve : float, optional
        Effective exhaust velocity in m/s.

    Returns
    -------
    float
        Mass ratio m0 / mf (>= 1.0).
    """
    if beta < 0.0 or beta >= 1.0:
        raise ValueError(f"Velocity fraction beta must satisfy 0 <= beta < 1, got {beta}")

    if beta == 0.0:
        return 1.0

    v_exhaust = _resolve_exhaust_velocity(isp=isp, ve=ve)
    beta_e = v_exhaust / C_LIGHT

    # artanh(beta) evaluated with log1p to guard against catastrophic cancellation as beta -> 1
    # artanh(beta) = 0.5 * ln((1 + beta) / (1 - beta))
    rapidity = 0.5 * (math.log1p(beta) - math.log1p(-beta))
    return math.exp(rapidity / beta_e)


def photon_rocket_velocity(m0: float, mf: float) -> float:
    """Compute terminal velocity for an ideal photon rocket (beta_e = 1).

    Sanger (1953) exact solution:
        beta = (mu^2 - 1) / (mu^2 + 1)
        v = c * (mu^2 - 1) / (mu^2 + 1)

    where mu = m0 / mf.

    Parameters
    ----------
    m0 : float
        Initial wet mass in kilograms.
    mf : float
        Final dry mass in kilograms.

    Returns
    -------
    float
        Terminal coordinate velocity in m/s.
    """
    if m0 <= 0.0 or mf <= 0.0:
        raise ValueError(f"Masses must be positive: m0={m0}, mf={mf}")
    if mf > m0:
        raise ValueError(f"Final mass mf ({mf}) cannot exceed initial mass m0 ({m0})")

    mu = m0 / mf
    mu_sq = mu * mu
    beta = (mu_sq - 1.0) / (mu_sq + 1.0)
    return beta * C_LIGHT


def photon_rocket_mass_ratio(beta: float) -> float:
    """Compute required mass ratio m0 / mf for an ideal photon rocket (beta_e = 1).

    Exact relation:
        mu = sqrt((1 + beta) / (1 - beta)) = gamma * (1 + beta)

    Parameters
    ----------
    beta : float
        Target velocity fraction v / c (0 <= beta < 1).

    Returns
    -------
    float
        Mass ratio m0 / mf.
    """
    if beta < 0.0 or beta >= 1.0:
        raise ValueError(f"Velocity fraction beta must satisfy 0 <= beta < 1, got {beta}")

    return math.sqrt((1.0 + beta) / (1.0 - beta))


def relativistic_jet_power(
    thrust: float,
    isp: Optional[float] = None,
    ve: Optional[float] = None,
) -> float:
    """Compute relativistic kinetic beam power of the propellant exhaust.

    In special relativity, the proper mass-loss rate is dm/dtau = T / v_e.
    The kinetic power carried by the exhaust beam is:
        P_jet = (gamma_e - 1) * (dm/dtau) * c^2 = (gamma_e - 1) * (T * c^2 / v_e)

    For small exhaust velocities (beta_e << 1), this reduces smoothly to the
    Newtonian expression P_jet -> 0.5 * T * v_e without catastrophic loss of significance.

    Parameters
    ----------
    thrust : float
        Proper thrust magnitude in Newtons (T >= 0).
    isp : float, optional
        Specific impulse in seconds.
    ve : float, optional
        Effective exhaust velocity in m/s.

    Returns
    -------
    float
        Exhaust beam kinetic power in Watts.
    """
    if thrust < 0.0:
        raise ValueError(f"Thrust must be non-negative, got {thrust}")
    if thrust == 0.0:
        return 0.0

    v_exhaust = _resolve_exhaust_velocity(isp=isp, ve=ve)
    beta_e = v_exhaust / C_LIGHT

    if beta_e >= 1.0:
        # Ideal photon rocket limit: all power is radiant beam power P = T * c
        return thrust * C_LIGHT

    # Stabilized evaluation of (gamma_e - 1) / beta_e to avoid cancellation as beta_e -> 0:
    # gamma_e - 1 = beta_e^2 / (sqrt(1 - beta_e^2) * (1 + sqrt(1 - beta_e^2)))
    # (gamma_e - 1) / beta_e = beta_e / (sqrt(1 - beta_e^2) * (1 + sqrt(1 - beta_e^2)))
    radicand = 1.0 - beta_e * beta_e
    gamma_inv = math.sqrt(radicand)
    ratio = beta_e / (gamma_inv * (1.0 + gamma_inv))

    return ratio * thrust * C_LIGHT


def mass_flow_rate(
    thrust: float,
    isp: Optional[float] = None,
    ve: Optional[float] = None,
    dtau_dt: float = 1.0,
) -> Tuple[float, float]:
    """Compute propellant mass-loss rates with respect to proper time and coordinate time.

    Parameters
    ----------
    thrust : float
        Proper thrust magnitude in Newtons.
    isp : float, optional
        Specific impulse in seconds.
    ve : float, optional
        Effective exhaust velocity in m/s.
    dtau_dt : float, optional
        Metric rate of proper time to coordinate time (dtau / dt). Defaults to 1.0.

    Returns
    -------
    tuple of (dm_dtau, dm_dt)
        dm_dtau : Rate of change of rest mass per proper second (kg/s, <= 0).
        dm_dt   : Rate of change of rest mass per coordinate second (kg/s, <= 0).
    """
    if thrust < 0.0:
        raise ValueError(f"Thrust must be non-negative, got {thrust}")
    if thrust == 0.0:
        return 0.0, 0.0

    v_exhaust = _resolve_exhaust_velocity(isp=isp, ve=ve)
    dm_dtau = -thrust / v_exhaust
    dm_dt = dm_dtau * dtau_dt
    return dm_dtau, dm_dt


def thrust_to_coordinate_acceleration(
    thrust_vector: np.ndarray | list[float],
    mass: float,
    v_bcrs: np.ndarray | list[float],
) -> np.ndarray:
    """Transform proper thrust vector into coordinate 3-acceleration in BCRS.

    Parameters
    ----------
    thrust_vector : np.ndarray
        Thrust force vector [Tx, Ty, Tz] in Newtons in spacecraft rest frame.
    mass : float
        Instantaneous spacecraft rest mass in kg (mass > 0).
    v_bcrs : np.ndarray
        Spacecraft coordinate velocity [vx, vy, vz] in m/s in BCRS.

    Returns
    -------
    np.ndarray
        Coordinate acceleration dv/dt in m/s^2.
    """
    if mass <= 0.0:
        raise ValueError(f"Spacecraft mass must be positive, got {mass}")

    t_vec = np.asarray(thrust_vector, dtype=np.float64)
    alpha_proper = t_vec / mass
    return proper_to_coordinate_acceleration(v_bcrs, alpha_proper)
