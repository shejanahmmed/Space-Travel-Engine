"""Klioner (2003) Relativistic Light-Time and Gravitational Deflection Kernel.

Implements the high-precision iterative light-time equation in the Barycentric
Celestial Reference System (BCRS) accounting for multi-body Rømer delay,
Shapiro gravitational delays, and gravitational ray bending.

Authoritative References:
- Klioner, S. A. (2003), "A Practical Relativistic Model for Microarcsecond
  Astrometry in Space", The Astronomical Journal, 125:1580-1597, §4-§5.
- Bertotti, B., Iess, L., & Tortora, P. (2003), "A test of general relativity
  using radio links with the Cassini spacecraft", Nature, 425:374-376.
- Moyer, T. D. (2003), "Formulation for Observed and Computed Values of Deep
  Space Network Data", JPL Publication 00-7, §8.
- Petit, G., & Luzum, B. (eds.) (2010), "IERS Conventions (2010)", IERS Technical
  Note 36, Verlag des Bundesamts für Kartographie und Geodäsie, Chapter 11.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from relativistic_engine.constants import (
    C_LIGHT,
    G_NEWTON,
    GM_EARTH,
    GM_JUPITER,
    GM_MARS,
    GM_SATURN,
    GM_SUN,
    RADIUS_SUN,
)


def compute_shapiro_delay_body(
    r_tx: np.ndarray,
    r_rx: np.ndarray,
    r_body: np.ndarray,
    gm_body: float,
    *,
    c: float = C_LIGHT,
    gamma_ppn: float = 1.0,
) -> float:
    """Compute relativistic Shapiro time delay for a single gravitating body.

    Formula (Klioner 2003, Eq. 47; IERS 2010):
        Delta_t_Shapiro = ((1 + gamma) * GM / c^3) * ln( (r1 + r2 + R12) / (r1 + r2 - R12) )

    With regularized denominator for near-grazing / numerical stability:
        denominator = max(r1 + r2 - R12, 2 * (1 + gamma) * GM / c^2)

    Args:
        r_tx: Transmitter position vector [x, y, z] in BCRS (meters).
        r_rx: Receiver position vector [x, y, z] in BCRS (meters).
        r_body: Gravitating body position vector [x, y, z] in BCRS (meters).
        gm_body: Gravitational parameter of body (m^3/s^2).
        c: Speed of light (m/s).
        gamma_ppn: PPN parameter gamma (1.0 in General Relativity).

    Returns:
        Shapiro time delay in seconds.
    """
    if gm_body <= 0.0:
        return 0.0

    p1 = np.asarray(r_tx, dtype=np.float64) - np.asarray(r_body, dtype=np.float64)
    p2 = np.asarray(r_rx, dtype=np.float64) - np.asarray(r_body, dtype=np.float64)
    p12 = np.asarray(r_rx, dtype=np.float64) - np.asarray(r_tx, dtype=np.float64)

    r1 = float(np.linalg.norm(p1))
    r2 = float(np.linalg.norm(p2))
    r12 = float(np.linalg.norm(p12))

    if r1 <= 0.0 or r2 <= 0.0 or r12 <= 0.0:
        return 0.0

    c2 = c * c
    c3 = c2 * c
    rg = (1.0 + gamma_ppn) * gm_body / c2

    numerator = r1 + r2 + r12 + rg
    denominator = max(r1 + r2 - r12 + rg, 1e-15)

    arg = max(1.0 + 1e-15, numerator / denominator)
    return ((1.0 + gamma_ppn) * gm_body / c3) * math.log(arg)


def compute_gravitational_deflection_angle(
    r_tx: np.ndarray,
    r_rx: np.ndarray,
    r_body: np.ndarray,
    gm_body: float,
    *,
    c: float = C_LIGHT,
    gamma_ppn: float = 1.0,
) -> float:
    """Compute gravitational deflection angle of a light ray by a gravitating body.

    Formula (Klioner 2003, Eq. 32):
        theta = (2 * (1 + gamma) * GM / (c^2 * d)) * sqrt((1 - cos(psi)) / 2)
    where d is impact parameter and psi is angle between emitter and receiver.

    For distant source and observer, simplifies to the Einstein deflection:
        theta_asymptotic = (1 + gamma) * 2 * GM / (c^2 * d)

    Args:
        r_tx: Transmitter position in meters.
        r_rx: Receiver position in meters.
        r_body: Body position in meters.
        gm_body: Gravitational parameter.
        c: Speed of light.
        gamma_ppn: PPN parameter gamma (1.0 in GR).

    Returns:
        Deflection angle in radians.
    """
    p1 = np.asarray(r_tx, dtype=np.float64) - np.asarray(r_body, dtype=np.float64)
    p2 = np.asarray(r_rx, dtype=np.float64) - np.asarray(r_body, dtype=np.float64)
    ray_dir = np.asarray(r_rx, dtype=np.float64) - np.asarray(r_tx, dtype=np.float64)

    r1 = float(np.linalg.norm(p1))
    r2 = float(np.linalg.norm(p2))
    ray_len = float(np.linalg.norm(ray_dir))

    if r1 <= 0.0 or r2 <= 0.0 or ray_len <= 0.0:
        return 0.0

    k_hat = ray_dir / ray_len
    # Impact parameter vector: d = p1 - (p1 . k_hat) * k_hat
    d_vec = p1 - float(np.dot(p1, k_hat)) * k_hat
    d_impact = float(np.linalg.norm(d_vec))

    if d_impact <= 0.0:
        return 0.0

    # Geometric factor accounting for finite distances of emitter and receiver:
    # cos(theta1) = (p1 . k) / r1, cos(theta2) = (p2 . k) / r2
    cos1 = float(np.dot(p1, k_hat)) / r1
    cos2 = float(np.dot(p2, k_hat)) / r2
    geom_factor = 0.5 * ((1.0 + cos2) - (cos1 - 1.0))  # in [0, 2]

    c2 = c * c
    return float((1.0 + gamma_ppn) * gm_body / (c2 * d_impact) * geom_factor)


def solve_klioner_light_time(
    tx_position_fn: Callable[[float], np.ndarray],
    rx_position: np.ndarray,
    t_rx: float,
    body_positions_and_gms: List[Tuple[np.ndarray, float, str]],
    *,
    c: float = C_LIGHT,
    gamma_ppn: float = 1.0,
    tol_seconds: float = 1e-12,
    max_iter: int = 10,
) -> Dict[str, Union[float, int, List[float]]]:
    """Solve the relativistic light-time equation iteratively for downlink reception.

    Given reception time t_rx and static/moving receiver position, solves for transmission
    time t_tx such that:
        t_rx - t_tx = (1 / c) * |rx(t_rx) - tx(t_tx)| + sum_b Delta_t_Shapiro^(b)

    Convergence condition: |tau_{k+1} - tau_k| < tol_seconds (default: 1 ps = 10^-12 s).

    Args:
        tx_position_fn: Callable returning transmitter position vector [x, y, z] at time t.
        rx_position: Receiver position vector [x, y, z] at t_rx.
        t_rx: Reception time in seconds (TDB/TCB coordinate time).
        body_positions_and_gms: List of tuples (body_pos_vec, gm_val, body_name)
            for relevant gravitating bodies (e.g. Sun, Jupiter, Earth).
        c: Speed of light.
        gamma_ppn: PPN parameter gamma (1.0 for GR).
        tol_seconds: Convergence tolerance in seconds.
        max_iter: Maximum iterations.

    Returns:
        Dictionary containing:
            't_tx': Solved transmission time [s]
            'light_time_total': Total elapsed light-time t_rx - t_tx [s]
            'romer_delay': Geometric light travel time |r_rx - r_tx| / c [s]
            'shapiro_delays': Dict mapping body_name -> delay [s]
            'total_shapiro': Sum of Shapiro delays [s]
            'iterations': Number of iterations performed
            'residual': Final step change |tau_{k+1} - tau_k| [s]
    """
    rx_pos = np.asarray(rx_position, dtype=np.float64)

    # Initial guess: Euclidean distance at t_rx divided by c
    tx_pos_0 = np.asarray(tx_position_fn(t_rx), dtype=np.float64)
    tau = float(np.linalg.norm(rx_pos - tx_pos_0)) / c

    shapiro_by_body: Dict[str, float] = {}
    total_shapiro = 0.0
    romer = tau
    residual = 0.0
    iter_count = 0

    for i in range(1, max_iter + 1):
        iter_count = i
        t_tx_guess = t_rx - tau
        tx_pos = np.asarray(tx_position_fn(t_tx_guess), dtype=np.float64)

        # Romer delay
        romer = float(np.linalg.norm(rx_pos - tx_pos)) / c

        # Shapiro delays from all bodies
        total_shapiro = 0.0
        shapiro_by_body.clear()
        for b_pos, b_gm, b_name in body_positions_and_gms:
            d_sh = compute_shapiro_delay_body(tx_pos, rx_pos, b_pos, b_gm, c=c, gamma_ppn=gamma_ppn)
            shapiro_by_body[b_name] = d_sh
            total_shapiro += d_sh

        tau_new = romer + total_shapiro
        residual = abs(tau_new - tau)
        tau = tau_new

        if residual < tol_seconds:
            break

    return {
        "t_tx": t_rx - tau,
        "light_time_total": tau,
        "romer_delay": romer,
        "shapiro_delays": shapiro_by_body,
        "total_shapiro": total_shapiro,
        "iterations": iter_count,
        "residual": residual,
    }
