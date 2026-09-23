"""Einstein-Infeld-Hoffmann (EIH) 2PN and 2.5PN Relativistic Dynamics.

Implements higher-order post-Newtonian equations of motion up to 2PN (order 1/c^4)
and 2.5PN gravitational radiation reaction (order 1/c^5) in harmonic coordinates
(BCRS-compatible).

Authoritative References:
- Blanchet, L., & Iyer, B. R. (1989), "Post-Newtonian generation of gravitational
  waves in the center of mass frame", Classical and Quantum Gravity, 6(8), L87-L92.
- Will, C. M., & Wiseman, A. G. (1996), "Gravitational radiation from compact
  binaries: An approach to the post-Newtonian expansion based on the direct
  integration of the relaxed Einstein equations", Phys. Rev. D, 54, 4813.
- Damour, T., & Deruelle, N. (1985), "General relativistic celestial mechanics of
  binary systems. I. The post-Newtonian motion", Ann. Inst. Henri Poincaré, 43, 107-132.
- Blanchet, L. (2014), "Gravitational Radiation from Post-Newtonian Sources and
  Inspirals of Compact Binaries", Living Rev. Relativity, 17, 2, Section 3.
- Peters, P. C. (1964), "Gravitational Radiation and the Motion of Two Point
  Masses", Phys. Rev., 136, B1224.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union
import numpy as np

from relativistic_engine.constants import (
    C_LIGHT,
    G_NEWTON,
)


def compute_relative_2pn_acceleration(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    m1: float,
    m2: float,
    *,
    c: float = C_LIGHT,
    g: float = G_NEWTON,
    include_1pn: bool = True,
    include_2pn: bool = True,
    include_25pn: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute relative 2-body acceleration up to 2PN and 2.5PN order in harmonic coordinates.

    Calculates the relative coordinate acceleration a = d^2 r / dt^2 of body 1 relative
    to body 2 in the center-of-mass frame:
        a = a_N + a_1PN + a_2PN + a_2.5PN

    Formulations based on Blanchet & Iyer (1989) and Will & Wiseman (1996):
    Let:
        M = m1 + m2
        eta = m1 * m2 / M^2
        r = |r_vec|, n = r_vec / r
        v = |v_vec|
        rdot = (r_vec . v_vec) / r
        gamma = G * M / r

    Newtonian:
        a_N = - (G * M / r^2) * n

    1PN:
        a_1PN = - (G * M / (c^2 * r^2)) * n * [
            -(1 + 3*eta)*v^2 + (3/2)*eta*rdot^2 + 2*(2 + eta)*(G*M/r)
        ] + (G * M / (c^2 * r^2)) * rdot * v_vec * (4 - 2*eta)

    2PN:
        a_2PN = - (G * M / (c^4 * r^2)) * n * [
            (3/4)*(1 - 5*eta + 5*eta^2)*v^4
            - (15/8)*eta*(1 - 3*eta)*v^2*rdot^2
            + (35/16)*eta*(1 - 3*eta)*rdot^4
            + (1/2)*(13 - 41*eta + 4*eta^2)*(G*M/r)*v^2
            - (25 - 8*eta + 4*eta^2)*(G*M/r)^2
            - (3/2)*(3 - 15*eta - 2*eta^2)*(G*M/r)*rdot^2
        ] + (G * M / (c^4 * r^2)) * rdot * v_vec * [
            (1/2)*(15 + 4*eta - 52*eta^2)*v^2
            - (3/2)*(3 + 2*eta - 12*eta^2)*rdot^2
            + (1/2)*(41 + 8*eta)*(G*M/r)
        ]

    2.5PN (dissipative radiation reaction):
        a_2.5PN = (8/5) * (G^2 * M^2 * eta / (c^5 * r^3)) * [
            rdot * (3*v^2 + (17/3)*(G*M/r)) * r_vec - (v^2 + 3*(G*M/r)) * v_vec
        ]

    Args:
        r_vec: Relative position vector r = r1 - r2 in meters, shape (3,).
        v_vec: Relative velocity vector v = v1 - v2 in m/s, shape (3,).
        m1: Mass of body 1 in kg.
        m2: Mass of body 2 in kg.
        c: Speed of light in m/s.
        g: Gravitational constant in m^3 / (kg s^2).
        include_1pn: Whether to include 1PN corrections.
        include_2pn: Whether to include 2PN corrections.
        include_25pn: Whether to include 2.5PN radiation reaction.

    Returns:
        Dictionary mapping component keys ('newtonian', '1pn', '2pn', '25pn', 'total')
        to np.ndarray acceleration vectors of shape (3,) in m/s^2.
    """
    r = np.asarray(r_vec, dtype=np.float64)
    v = np.asarray(v_vec, dtype=np.float64)
    r_norm = float(np.linalg.norm(r))

    if r_norm <= 0.0 or m1 <= 0.0 or m2 <= 0.0:
        zero = np.zeros(3, dtype=np.float64)
        return {
            "newtonian": zero,
            "1pn": zero,
            "2pn": zero,
            "25pn": zero,
            "total": zero,
        }

    total_m = m1 + m2
    eta = (m1 * m2) / (total_m * total_m)
    gm = g * total_m
    n_hat = r / r_norm

    v_sq = float(np.dot(v, v))
    rdot = float(np.dot(r, v)) / r_norm
    rdot_sq = rdot * rdot
    inv_r = 1.0 / r_norm
    inv_r2 = inv_r * inv_r
    gm_over_r = gm * inv_r

    # Newtonian acceleration
    a_newt = - (gm * inv_r2) * n_hat

    # 1PN acceleration
    a_1pn = np.zeros(3, dtype=np.float64)
    if include_1pn and c > 0.0:
        c2 = c * c
        factor_1pn = gm / (c2 * (r_norm**2))
        radial_coeff_1pn = (
            - (1.0 + 3.0 * eta) * v_sq
            + 1.5 * eta * rdot_sq
            + 2.0 * (2.0 + eta) * gm_over_r
        )
        tangential_coeff_1pn = rdot * (4.0 - 2.0 * eta)
        a_1pn = - factor_1pn * radial_coeff_1pn * n_hat + factor_1pn * tangential_coeff_1pn * v

    # 2PN acceleration
    a_2pn = np.zeros(3, dtype=np.float64)
    if include_2pn and c > 0.0:
        c4 = c**4
        factor_2pn = gm / (c4 * (r_norm**2))

        # Radial 2PN terms
        t1 = 0.75 * (1.0 - 5.0 * eta + 5.0 * (eta**2)) * (v_sq**2)
        t2 = - 1.875 * eta * (1.0 - 3.0 * eta) * v_sq * rdot_sq
        t3 = (35.0 / 16.0) * eta * (1.0 - 3.0 * eta) * (rdot_sq**2)
        t4 = 0.5 * (13.0 - 41.0 * eta + 4.0 * (eta**2)) * gm_over_r * v_sq
        t5 = - (25.0 - 8.0 * eta + 4.0 * (eta**2)) * (gm_over_r**2)
        t6 = - 1.5 * (3.0 - 15.0 * eta - 2.0 * (eta**2)) * gm_over_r * rdot_sq
        radial_coeff_2pn = t1 + t2 + t3 + t4 + t5 + t6

        # Tangential 2PN terms
        u1 = 0.5 * (15.0 + 4.0 * eta - 52.0 * (eta**2)) * v_sq
        u2 = - 1.5 * (3.0 + 2.0 * eta - 12.0 * (eta**2)) * rdot_sq
        u3 = 0.5 * (41.0 + 8.0 * eta) * gm_over_r
        tangential_coeff_2pn = rdot * (u1 + u2 + u3)

        a_2pn = - factor_2pn * radial_coeff_2pn * n_hat + factor_2pn * tangential_coeff_2pn * v

    # 2.5PN dissipative radiation reaction
    a_25pn = np.zeros(3, dtype=np.float64)
    if include_25pn and c > 0.0:
        c5 = c**5
        pref = (1.6 * (gm**2) * eta) / (c5 * (r_norm**3))
        term_r = rdot * (3.0 * v_sq + (17.0 / 3.0) * gm_over_r)
        term_v = v_sq + 3.0 * gm_over_r
        a_25pn = pref * (term_r * r - term_v * v)

    a_total = a_newt + a_1pn + a_2pn + a_25pn
    return {
        "newtonian": a_newt,
        "1pn": a_1pn,
        "2pn": a_2pn,
        "25pn": a_25pn,
        "total": a_total,
    }


def compute_eih_2pn_nbody_accelerations(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm_values: np.ndarray,
    masses: Optional[np.ndarray] = None,
    *,
    order: str = "2pn",
    c: float = C_LIGHT,
    g: float = G_NEWTON,
) -> Dict[str, np.ndarray]:
    """Compute N-body relativistic accelerations up to 2PN / 2.5PN order.

    For general N bodies, evaluates the 1PN EIH multi-body tensor sum and
    couples the 2PN two-body interaction terms for each interacting pair
    (Blanchet & Iyer 1989; Will 2014).

    Args:
        positions: Array of shape (N, 3) containing Cartesian positions [m].
        velocities: Array of shape (N, 3) containing Cartesian velocities [m/s].
        gm_values: Array of shape (N,) containing gravitational parameters G*M_i [m^3/s^2].
        masses: Optional array of shape (N,) containing body masses in kg.
            If None, inferred from gm_values / G_NEWTON.
        order: Approximation order: 'newtonian', '1pn', '2pn', or '2.5pn'.
        c: Speed of light [m/s].
        g: Gravitational constant [m^3/(kg s^2)].

    Returns:
        Dictionary containing:
            'total': Shape (N, 3) total acceleration [m/s^2]
            'newtonian': Shape (N, 3) Newtonian acceleration [m/s^2]
            '1pn': Shape (N, 3) 1PN EIH acceleration [m/s^2]
            '2pn': Shape (N, 3) 2PN acceleration [m/s^2]
            '25pn': Shape (N, 3) 2.5PN radiation acceleration [m/s^2]
    """
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    gms = np.asarray(gm_values, dtype=np.float64)

    n_bodies = pos.shape[0]
    if masses is None:
        m_arr = gms / g
    else:
        m_arr = np.asarray(masses, dtype=np.float64)

    order_norm = order.lower().strip()
    include_1pn = order_norm in ("1pn", "2pn", "2.5pn", "25pn")
    include_2pn = order_norm in ("2pn", "2.5pn", "25pn")
    include_25pn = order_norm in ("2.5pn", "25pn")

    a_newt = np.zeros((n_bodies, 3), dtype=np.float64)
    a_1pn = np.zeros((n_bodies, 3), dtype=np.float64)
    a_2pn = np.zeros((n_bodies, 3), dtype=np.float64)
    a_25pn = np.zeros((n_bodies, 3), dtype=np.float64)

    # Relative displacement and distance matrices
    dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # dr[A, B] = pos[A] - pos[B]
    r_dist = np.linalg.norm(dr, axis=2)
    np.fill_diagonal(r_dist, np.inf)
    n_vec = dr / r_dist[:, :, np.newaxis]
    inv_r2 = 1.0 / (r_dist**2)

    # Standard Newtonian N-body acceleration
    for a in range(n_bodies):
        for b in range(n_bodies):
            if a == b or gms[b] == 0.0:
                continue
            a_newt[a] -= (gms[b] * inv_r2[a, b]) * n_vec[a, b]

    if not include_1pn:
        return {
            "newtonian": a_newt,
            "1pn": a_1pn,
            "2pn": a_2pn,
            "25pn": a_25pn,
            "total": a_newt,
        }

    # 1PN Einstein-Infeld-Hoffmann standard multi-body evaluation (IAU 2000 Res B1.3)
    c2 = c * c
    v_sq = np.sum(vel**2, axis=1)  # shape (N,)
    inv_r = 1.0 / r_dist

    # Precompute gravitational potentials at each body: U_A = sum_{C != A} GM_C / r_AC
    potentials = np.zeros(n_bodies, dtype=np.float64)
    for a in range(n_bodies):
        potentials[a] = np.sum(gms * inv_r[a, :])

    for a in range(n_bodies):
        for b in range(n_bodies):
            if a == b or gms[b] == 0.0:
                continue

            n_ab = n_vec[a, b]  # unit vector pointing from b to a
            r_ab = r_dist[a, b]
            v_a = vel[a]
            v_b = vel[b]
            v_rel = v_a - v_b

            u_a = potentials[a]
            u_b = potentials[b]

            n_dot_vb = float(np.dot(n_ab, v_b))
            n_dot_va = float(np.dot(n_ab, v_a))
            va_dot_vb = float(np.dot(v_a, v_b))

            # Bracket 1
            bracket1 = (
                4.0 * u_a
                + u_b
                - v_sq[a]
                - 2.0 * v_sq[b]
                + 4.0 * va_dot_vb
                + 1.5 * (n_dot_vb**2)
            )

            # Bracket 2
            bracket2 = float(np.dot(n_ab, 4.0 * v_a - 3.0 * v_b))

            term_direct = (gms[b] / (c2 * (r_ab**2))) * (bracket1 * n_ab + bracket2 * v_rel)
            a_1pn[a] += term_direct

            # Cross-body acceleration: sum_C (GM_C / r_BC^2) * n_BC
            cross_acc = np.zeros(3, dtype=np.float64)
            for c_idx in range(n_bodies):
                if c_idx == b or gms[c_idx] == 0.0:
                    continue
                n_bc = n_vec[b, c_idx]
                cross_acc += (gms[c_idx] * inv_r2[b, c_idx]) * n_bc

            a_1pn[a] += (3.5 * gms[b] / (c2 * r_ab)) * cross_acc

    # 2PN pairwise terms
    if include_2pn:
        for a in range(n_bodies):
            for b in range(n_bodies):
                if a == b or gms[b] == 0.0 or m_arr[a] <= 0.0 or m_arr[b] <= 0.0:
                    continue
                pair_res = compute_relative_2pn_acceleration(
                    dr[a, b],
                    vel[a] - vel[b],
                    m_arr[a],
                    m_arr[b],
                    c=c,
                    g=g,
                    include_1pn=False,
                    include_2pn=True,
                    include_25pn=include_25pn,
                )
                # In center of mass, acceleration on body A from body B:
                # a_A = (m_B / (m_A + m_B)) * a_rel
                m_frac_b = m_arr[b] / (m_arr[a] + m_arr[b])
                a_2pn[a] += m_frac_b * pair_res["2pn"]
                if include_25pn:
                    a_25pn[a] += m_frac_b * pair_res["25pn"]

    a_total = a_newt + a_1pn + a_2pn + a_25pn
    return {
        "newtonian": a_newt,
        "1pn": a_1pn,
        "2pn": a_2pn,
        "25pn": a_25pn,
        "total": a_total,
    }


def circular_orbit_frequency_2pn(
    m1: float,
    m2: float,
    r_m: float,
    *,
    c: float = C_LIGHT,
    g: float = G_NEWTON,
) -> float:
    """Compute the 2PN orbital angular frequency omega for a circular binary orbit.

    Uses the exact Blanchet & Iyer (1989) Eq. (3.18) invariant relation:
        omega^2 = (G*M / r^3) * [
            1 - (3 - eta) * gamma_pn + (6 + 41/4 * eta + eta^2) * gamma_pn^2
        ]
    where gamma_pn = G*M / (c^2 * r), M = m1 + m2, eta = m1*m2 / M^2.

    Args:
        m1: Mass of body 1 [kg].
        m2: Mass of body 2 [kg].
        r_m: Orbital separation radius [m].
        c: Speed of light [m/s].
        g: Gravitational constant [m^3 / (kg s^2)].

    Returns:
        Orbital frequency omega in rad/s.
    """
    total_m = m1 + m2
    eta = (m1 * m2) / (total_m * total_m)
    gm = g * total_m
    gamma_pn = gm / (c * c * r_m)

    omega_sq = (gm / (r_m**3)) * (
        1.0
        - (3.0 - eta) * gamma_pn
        + (6.0 + 10.25 * eta + (eta**2)) * (gamma_pn**2)
    )
    return math.sqrt(max(0.0, omega_sq))
