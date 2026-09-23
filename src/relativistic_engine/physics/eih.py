"""Einstein-Infeld-Hoffmann (EIH) Multi-Body 1PN Relativistic Dynamics.

Implements the complete self-consistent N-body post-Newtonian equations of
motion in the Barycentric Celestial Reference System (BCRS / harmonic gauge)
according to IAU 2000 Resolution B1.3 and Will (2014).

References:
- Will, C. M. (2014), "The Confrontation between General Relativity and
  Experiment", Living Reviews in Relativity, 17(1), Eq. (7.1)-(7.2).
- Soffel, M., Klioner, S. A., Petit, G., et al. (2003), "The IAU 2000 Resolutions
  for Astrometry, Celestial Mechanics, and Metrology in the Relativistic
  Framework", Astronomical Journal 126:2687-2706, Eq. (3.11).
- Moyer, T. D. (2003), "Formulation for Observed and Computed Values of Deep
  Space Network Data", JPL Publication 00-7, Eq. (4-1).
- Landau, L. D., & Lifshitz, E. M. (1975), "The Classical Theory of Fields",
  Course of Theoretical Physics Vol. 2, Pergamon Press, §106.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from relativistic_engine.constants import (
    C_LIGHT,
    G_NEWTON,
    GM_SUN,
    SEC_PER_JULIAN_YEAR,
)


def compute_eih_nbody_accelerations(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm_values: np.ndarray,
    *,
    include_1pn: bool = True,
    c: float = C_LIGHT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute N-body gravitational accelerations (Newtonian + 1PN EIH).

    Evaluates the complete coupled N-body post-Newtonian acceleration in harmonic
    coordinates:
        a_A = a_A^(Newt) + a_A^(1PN)

    Where:
        a_A^(Newt) = sum_{B != A} (GM_B / r_AB^3) * (r_B - r_A)
        a_A^(1PN)  = (1 / c^2) * sum_{B != A} (GM_B / r_AB^2) * [
            n_AB * { 4*sum_{C != A} (GM_C / r_AC) + sum_{C != B} (GM_C / r_BC)
                     - v_A^2 - 2*v_B^2 + 4*(v_A . v_B) + (3/2)*(n_AB . v_B)^2 }
            + { n_AB . (4*v_A - 3*v_B) } * (v_A - v_B)
        ] + (7 / (2 * c^2)) * sum_{B != A} (GM_B / r_AB) * sum_{C != B} (GM_C / r_BC^2) * n_BC

    with n_AB = (r_A - r_B) / r_AB, n_BC = (r_B - r_C) / r_BC.

    Args:
        positions: Array of shape (N, 3) containing Cartesian positions [m].
        velocities: Array of shape (N, 3) containing Cartesian velocities [m/s].
        gm_values: Array of shape (N,) containing gravitational parameters G*M_i [m^3/s^2].
        include_1pn: Whether to evaluate post-Newtonian relativistic corrections.
        c: Speed of light [m/s].

    Returns:
        Tuple of (a_total, a_newtonian, a_1pn), each of shape (N, 3) in [m/s^2].
    """
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    gms = np.asarray(gm_values, dtype=np.float64)

    n_bodies = pos.shape[0]
    if pos.shape != (n_bodies, 3) or vel.shape != (n_bodies, 3) or gms.shape != (n_bodies,):
        raise ValueError("Incompatible input shapes for N-body EIH dynamics.")

    a_newt = np.zeros((n_bodies, 3), dtype=np.float64)
    a_1pn = np.zeros((n_bodies, 3), dtype=np.float64)

    # Pre-compute relative displacement vectors r_AB = r_A - r_B and distances r_AB
    dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # shape (N, N, 3)
    r_dist = np.linalg.norm(dr, axis=2)  # shape (N, N)

    # Regularization guard along diagonal to avoid division by zero
    np.fill_diagonal(r_dist, np.inf)

    # Unit vectors n_AB = (r_A - r_B) / r_AB
    n_vec = dr / r_dist[:, :, np.newaxis]  # shape (N, N, 3)

    # Newtonian acceleration: a_A = sum_{B != A} GM_B * (r_B - r_A) / r_AB^3 = - sum_{B != A} (GM_B / r_AB^2) * n_AB
    inv_r2 = 1.0 / (r_dist**2)
    inv_r = 1.0 / r_dist

    # Newtonian sum
    for a in range(n_bodies):
        for b in range(n_bodies):
            if a == b or gms[b] == 0.0:
                continue
            a_newt[a] -= (gms[b] * inv_r2[a, b]) * n_vec[a, b]

    if not include_1pn:
        return a_newt.copy(), a_newt.copy(), a_1pn

    c2 = c * c
    inv_c2 = 1.0 / c2

    # Precompute Newtonian potentials at each body: U_A = sum_{C != A} GM_C / r_AC
    potential_at_body = np.zeros(n_bodies, dtype=np.float64)
    for a in range(n_bodies):
        potential_at_body[a] = np.sum([gms[c_idx] * inv_r[a, c_idx] for c_idx in range(n_bodies) if c_idx != a])

    v2 = np.sum(vel**2, axis=1)  # shape (N,)

    # 1PN acceleration evaluation
    for a in range(n_bodies):
        acc_1pn_a = np.zeros(3, dtype=np.float64)
        v_a = vel[a]
        v_a_sq = v2[a]

        for b in range(n_bodies):
            if a == b or gms[b] == 0.0:
                continue

            gm_b = gms[b]
            r_ab_inv2 = inv_r2[a, b]
            r_ab_inv = inv_r[a, b]
            n_ab = n_vec[a, b]
            v_b = vel[b]
            v_b_sq = v2[b]

            v_a_dot_v_b = float(np.dot(v_a, v_b))
            n_ab_dot_v_b = float(np.dot(n_ab, v_b))
            n_ab_dot_v_a = float(np.dot(n_ab, v_a))

            # Potential term: 4 * U_A + U_B
            # where U_A = sum_{C != A} GM_C / r_AC
            # and U_B = sum_{C != B} GM_C / r_BC
            u_term = 4.0 * potential_at_body[a] + potential_at_body[b]

            bracket_n = (
                u_term
                - v_a_sq
                - 2.0 * v_b_sq
                + 4.0 * v_a_dot_v_b
                + 1.5 * (n_ab_dot_v_b**2)
            )

            # Term 1: (GM_B / r_AB^2) * bracket_n * n_AB
            term1 = (gm_b * r_ab_inv2 * bracket_n) * n_ab

            # Term 2: (GM_B / r_AB^2) * [ n_AB . (4 v_A - 3 v_B) ] * (v_A - v_B)
            proj_v = 4.0 * n_ab_dot_v_a - 3.0 * n_ab_dot_v_b
            term2 = (gm_b * r_ab_inv2 * proj_v) * (v_a - v_b)

            # Term 3: (7 / 2) * (GM_B / r_AB) * sum_{C != B} (GM_C / r_BC^2) * n_BC
            sum_c = np.zeros(3, dtype=np.float64)
            for c_idx in range(n_bodies):
                if c_idx == b or gms[c_idx] == 0.0:
                    continue
                sum_c += (gms[c_idx] * inv_r2[b, c_idx]) * n_vec[b, c_idx]

            term3 = (3.5 * gm_b * r_ab_inv) * sum_c

            acc_1pn_a += term1 + term2 + term3

        a_1pn[a] = inv_c2 * acc_1pn_a

    a_total = a_newt + a_1pn
    return a_total, a_newt, a_1pn


def compute_eih_conserved_quantities(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm_values: np.ndarray,
    *,
    masses_kg: Optional[np.ndarray] = None,
    c: float = C_LIGHT,
) -> Dict[str, Any]:
    """Compute exact conserved invariants for N-body EIH system.

    Calculates:
    1. Total 1PN Relativistic Energy:
       E = E_Newt + E_1PN
       E_Newt = (1/2)*sum m_A v_A^2 - (1/2)*sum_{A != B} (G m_A m_B / r_AB)
       E_1PN  = (3/(8 c^2))*sum m_A v_A^4
                + (1/(2 c^2))*sum_{A != B} (G m_A m_B / r_AB) * [
                    (3/2)*v_A^2 - (7/2)*(v_A . v_B) - (1/2)*(n_AB . v_A)*(n_AB . v_B)
                ] + (1/(2 c^2))*sum_A sum_{B != A} sum_{C != A} (G^2 m_A m_B m_C / (r_AB * r_AC))

    2. Total 1PN Relativistic Linear Momentum:
       P = sum_A m_A v_A * [ 1 + (1/(2 c^2))*v_A^2 - (1/(2 c^2))*sum_{B != A} (G m_B / r_AB) ]
           - (1/(2 c^2))*sum_A sum_{B != A} (G m_A m_B / r_AB) * (n_AB . v_B) * n_AB

    3. Total 1PN Relativistic Angular Momentum:
       J = sum_A r_A x P_A^(local)
           - (1/(2 c^2))*sum_A sum_{B != A} (G m_A m_B / r_AB) * (n_AB . v_B) * (r_A x n_AB)

    Args:
        positions: Array of shape (N, 3) containing Cartesian positions [m].
        velocities: Array of shape (N, 3) containing Cartesian velocities [m/s].
        gm_values: Array of shape (N,) containing gravitational parameters G*M_i [m^3/s^2].
        masses_kg: Optional array of true inertial masses [kg]. If None, inferred via GM / G_NEWTON.
        c: Speed of light [m/s].

    Returns:
        Dictionary containing energy, linear momentum, and angular momentum totals.
    """
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    gms = np.asarray(gm_values, dtype=np.float64)
    n_bodies = pos.shape[0]

    if masses_kg is None:
        m_kg = gms / G_NEWTON
    else:
        m_kg = np.asarray(masses_kg, dtype=np.float64)

    dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r_dist = np.linalg.norm(dr, axis=2)
    np.fill_diagonal(r_dist, np.inf)
    inv_r = 1.0 / r_dist
    n_vec = dr / r_dist[:, :, np.newaxis]

    v2 = np.sum(vel**2, axis=1)
    c2 = c * c
    inv_c2 = 1.0 / c2

    # 1. Newtonian Energy
    e_kin_newt = 0.5 * float(np.sum(m_kg * v2))
    e_pot_newt = 0.0
    for a in range(n_bodies):
        for b in range(a + 1, n_bodies):
            if m_kg[a] > 0.0 and m_kg[b] > 0.0:
                e_pot_newt -= G_NEWTON * m_kg[a] * m_kg[b] * inv_r[a, b]

    e_newt = e_kin_newt + e_pot_newt

    # 2. 1PN Energy Corrections
    e_kin_1pn = (3.0 / (8.0 * c2)) * float(np.sum(m_kg * (v2**2)))

    e_pot_v_1pn = 0.0
    for a in range(n_bodies):
        if m_kg[a] == 0.0:
            continue
        for b in range(n_bodies):
            if a == b or m_kg[b] == 0.0:
                continue
            gm_pair = G_NEWTON * m_kg[a] * m_kg[b] * inv_r[a, b]
            v_a_dot_v_b = float(np.dot(vel[a], vel[b]))
            n_dot_va = float(np.dot(n_vec[a, b], vel[a]))
            n_dot_vb = float(np.dot(n_vec[a, b], vel[b]))

            term = 1.5 * v2[a] - 3.5 * v_a_dot_v_b - 0.5 * n_dot_va * n_dot_vb
            e_pot_v_1pn += (0.5 * inv_c2) * gm_pair * term

    e_pot_pot_1pn = 0.0
    for a in range(n_bodies):
        if m_kg[a] == 0.0:
            continue
        for b in range(n_bodies):
            if a == b or m_kg[b] == 0.0:
                continue
            for c_idx in range(n_bodies):
                if c_idx == a or m_kg[c_idx] == 0.0:
                    continue
                term3 = (G_NEWTON**2) * m_kg[a] * m_kg[b] * m_kg[c_idx] * inv_r[a, b] * inv_r[a, c_idx]
                e_pot_pot_1pn += (0.5 * inv_c2) * term3

    e_1pn = e_kin_1pn + e_pot_v_1pn + e_pot_pot_1pn
    e_total = e_newt + e_1pn

    # 3. 1PN Relativistic Linear Momentum
    p_total = np.zeros(3, dtype=np.float64)
    for a in range(n_bodies):
        if m_kg[a] == 0.0:
            continue
        u_a = np.sum([gms[b] * inv_r[a, b] for b in range(n_bodies) if b != a])
        coeff = 1.0 + (0.5 * inv_c2) * v2[a] - (0.5 * inv_c2) * u_a
        p_total += m_kg[a] * coeff * vel[a]

        for b in range(n_bodies):
            if a == b or m_kg[b] == 0.0:
                continue
            n_dot_vb = float(np.dot(n_vec[a, b], vel[b]))
            p_total -= (0.5 * inv_c2) * (G_NEWTON * m_kg[a] * m_kg[b] * inv_r[a, b]) * n_dot_vb * n_vec[a, b]

    # 4. 1PN Relativistic Angular Momentum
    j_total = np.zeros(3, dtype=np.float64)
    for a in range(n_bodies):
        if m_kg[a] == 0.0:
            continue
        u_a = np.sum([gms[b] * inv_r[a, b] for b in range(n_bodies) if b != a])
        coeff = 1.0 + (0.5 * inv_c2) * v2[a] - (0.5 * inv_c2) * u_a
        j_total += np.cross(pos[a], m_kg[a] * coeff * vel[a])

        for b in range(n_bodies):
            if a == b or m_kg[b] == 0.0:
                continue
            n_dot_vb = float(np.dot(n_vec[a, b], vel[b]))
            r_cross_n = np.cross(pos[a], n_vec[a, b])
            j_total -= (0.5 * inv_c2) * (G_NEWTON * m_kg[a] * m_kg[b] * inv_r[a, b]) * n_dot_vb * r_cross_n

    return {
        "energy_total": float(e_total),
        "energy_newtonian": float(e_newt),
        "energy_1pn": float(e_1pn),
        "momentum_vector": p_total,
        "momentum_magnitude": float(np.linalg.norm(p_total)),
        "angular_momentum_vector": j_total,
        "angular_momentum_magnitude": float(np.linalg.norm(j_total)),
    }


def compute_mercury_perihelion_advance_rate(
    semi_major_axis_m: float,
    eccentricity: float,
    gm_central: float = GM_SUN,
    *,
    c: float = C_LIGHT,
) -> Dict[str, float]:
    """Compute Einstein's analytical secular relativistic perihelion advance rate.

    Standard Einstein (1915) formula:
        Delta_varpi = 6 * pi * GM / (c^2 * a * (1 - e^2))   [radians / revolution]
        dot_varpi   = 3 * (GM)^(3/2) / (c^2 * a^(5/2) * (1 - e^2))  [rad / s]

    Args:
        semi_major_axis_m: Semi-major axis in meters.
        eccentricity: Orbital eccentricity e (0 <= e < 1).
        gm_central: Gravitational parameter of central body [m^3/s^2].
        c: Speed of light [m/s].

    Returns:
        Dictionary containing advance per orbit [rad], rate in [rad/s],
        and secular rate in [arcseconds / century].
    """
    if eccentricity >= 1.0 or eccentricity < 0.0:
        raise ValueError("Eccentricity must be in [0, 1) for bound orbit.")
    if semi_major_axis_m <= 0.0:
        raise ValueError("Semi-major axis must be positive.")

    c2 = c * c
    one_minus_e2 = 1.0 - eccentricity * eccentricity
    d_varpi_rad_per_rev = (6.0 * math.pi * gm_central) / (c2 * semi_major_axis_m * one_minus_e2)

    # Orbital period T = 2 * pi * sqrt(a^3 / GM)
    orbital_period_s = 2.0 * math.pi * math.sqrt((semi_major_axis_m**3) / gm_central)
    rate_rad_per_s = d_varpi_rad_per_rev / orbital_period_s

    # Conversion to arcseconds per Julian century (36,525 ephemeris days)
    # 1 Julian century = 100 * JULIAN_YEAR = 3,155,760,000 seconds
    # 1 radian = 180 * 3600 / pi arcseconds
    rad_to_arcsec = (180.0 * 3600.0) / math.pi
    century_seconds = 100.0 * SEC_PER_JULIAN_YEAR
    rate_arcsec_per_century = rate_rad_per_s * century_seconds * rad_to_arcsec

    return {
        "advance_per_orbit_rad": float(d_varpi_rad_per_rev),
        "orbital_period_s": float(orbital_period_s),
        "rate_rad_per_s": float(rate_rad_per_s),
        "rate_arcsec_per_century": float(rate_arcsec_per_century),
    }


def propagate_eih_nbody_system(
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    gm_values: np.ndarray,
    t_span: Tuple[float, float],
    *,
    include_1pn: bool = True,
    rtol: float = 1e-11,
    atol: float = 1e-13,
    max_step: float = 86400.0,
    dense_output: bool = False,
) -> Dict[str, Any]:
    """Numerically propagate N-body system under EIH post-Newtonian dynamics.

    Uses high-order adaptive Runge-Kutta DOP853 integration over the 6N state
    vector [r_1, ..., r_N, v_1, ..., v_N].

    Args:
        initial_positions: Array of shape (N, 3) in meters.
        initial_velocities: Array of shape (N, 3) in m/s.
        gm_values: Array of shape (N,) in m^3/s^2.
        t_span: Tuple (t_start, t_end) in coordinate seconds.
        include_1pn: Whether to evaluate 1PN EIH accelerations.
        rtol: Relative integration tolerance.
        atol: Absolute integration tolerance.
        max_step: Maximum integrator step size in seconds.
        dense_output: Whether to generate continuous polynomial interpolation.

    Returns:
        Dictionary containing solution times, position trajectories, velocity
        trajectories, and solver convergence statistics.
    """
    from scipy.integrate import solve_ivp

    pos_0 = np.asarray(initial_positions, dtype=np.float64)
    vel_0 = np.asarray(initial_velocities, dtype=np.float64)
    gms = np.asarray(gm_values, dtype=np.float64)
    n_bodies = pos_0.shape[0]

    y0 = np.concatenate([pos_0.flatten(), vel_0.flatten()])

    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        p = y[: 3 * n_bodies].reshape((n_bodies, 3))
        v = y[3 * n_bodies :].reshape((n_bodies, 3))

        a_tot, _, _ = compute_eih_nbody_accelerations(p, v, gms, include_1pn=include_1pn)
        return np.concatenate([v.flatten(), a_tot.flatten()])

    sol = solve_ivp(
        ode_rhs,
        t_span,
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=dense_output,
    )

    if not sol.success:
        raise RuntimeError(f"EIH N-body propagation failed: {sol.message}")

    n_steps = len(sol.t)
    sol_pos = sol.y[: 3 * n_bodies, :].T.reshape((n_steps, n_bodies, 3))
    sol_vel = sol.y[3 * n_bodies :, :].T.reshape((n_steps, n_bodies, 3))

    return {
        "success": bool(sol.success),
        "t": sol.t,
        "positions": sol_pos,
        "velocities": sol_vel,
        "n_steps": n_steps,
        "n_fev": sol.nfev,
        "sol": sol.sol if dense_output else None,
    }
