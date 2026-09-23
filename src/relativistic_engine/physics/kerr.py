"""Kerr Metric Rotating Black Hole Geodesics, Carter Invariants, and Gravitational Lensing.

Implements exact strong-field General Relativity in the spacetime of a rotating,
uncharged black hole using Boyer-Lindquist coordinates (t, r, theta, phi).

Provides:
1. Exact Kerr metric tensor g_mu_nu and contravariant inverse g^mu_nu.
2. Exact Christoffel symbols Gamma^mu_alpha_beta evaluated analytically from
   exact metric derivatives.
3. Conserved first integrals of motion:
   - Energy E
   - Axial angular momentum L_z
   - Rest-mass norm g_mu_nu u^mu u^nu = -epsilon * c^2
   - Carter constant Q (Carter 1968)
4. Characteristic radii:
   - Outer/inner event horizons r_pm
   - Outer/inner ergosphere boundaries r_E(theta)
   - Innermost Stable Circular Orbit r_ISCO(a_*) (Bardeen, Press & Teukolsky 1972)
   - Circular photon orbit radii r_ph(a_*)
5. Adaptive 8-state DOP853 geodesic propagator for timelike (spacecraft) and
   null (photon) worldlines with Carter invariant auditing.
6. Bardeen (1973) celestial black hole shadow boundary contour.
7. Penrose process kinematics and energy extraction in the ergosphere.

References:
- Bardeen, J. M. (1973), "Timelike and null geodesics in the Kerr metric",
  Black Holes (Les Astres Occlus), Gordon & Breach, pp. 215-239.
- Bardeen, J. M., Press, W. H., & Teukolsky, S. A. (1972), "Rotating black holes:
  locally nonrotating frames, energy extraction, and scalar synchrotron radiation",
  Astrophysical Journal 178:347-370.
- Carter, B. (1968), "Global structure of the Kerr family of gravitational fields",
  Physical Review 174(5):1559-1571.
- Misner, C. W., Thorne, K. S., & Wheeler, J. A. (1973), "Gravitation", W. H. Freeman, §33.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp

from relativistic_engine.constants import C_LIGHT, G_NEWTON


# ==============================================================================
# KERR GEOMETRY DEFINITION
# ==============================================================================

@dataclass(frozen=True)
class KerrGeometry:
    """Kerr rotating black hole geometric parameters.

    Attributes:
        mass_kg: Black hole physical mass in kilograms.
        spin_dimensionless: Dimensionless spin parameter a_* = a / M in (-1, 1).
            Positive: prograde spin relative to +z axis.
            Negative: retrograde spin.
    """
    mass_kg: float
    spin_dimensionless: float = 0.0

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError(f"Black hole mass must be positive, got {self.mass_kg}")
        if abs(self.spin_dimensionless) >= 1.0:
            raise ValueError(
                f"Dimensionless spin |a_*| must be strictly less than 1.0 for sub-extremal Kerr, "
                f"got {self.spin_dimensionless}"
            )

    @property
    def mass_m(self) -> float:
        """Gravitational radius M = G * M_BH / c^2 in meters."""
        return (G_NEWTON * self.mass_kg) / (C_LIGHT ** 2)

    @property
    def spin_m(self) -> float:
        """Kerr spin parameter a = a_* * M in meters."""
        return self.spin_dimensionless * self.mass_m

    @property
    def event_horizon_outer(self) -> float:
        """Outer event horizon radius r_+ = M + sqrt(M^2 - a^2) in meters."""
        m = self.mass_m
        a = self.spin_m
        return m + math.sqrt(m * m - a * a)

    @property
    def event_horizon_inner(self) -> float:
        """Inner Cauchy horizon radius r_- = M - sqrt(M^2 - a^2) in meters."""
        m = self.mass_m
        a = self.spin_m
        return m - math.sqrt(m * m - a * a)

    def ergosphere_outer(self, theta_rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Outer ergosphere radius r_E^+(theta) = M + sqrt(M^2 - a^2 * cos^2(theta))."""
        m = self.mass_m
        a = self.spin_m
        cos_th = np.cos(theta_rad)
        return m + np.sqrt(np.maximum(0.0, m * m - (a * a) * (cos_th ** 2)))

    def ergosphere_inner(self, theta_rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Inner ergosphere radius r_E^-(theta) = M - sqrt(M^2 - a^2 * cos^2(theta))."""
        m = self.mass_m
        a = self.spin_m
        cos_th = np.cos(theta_rad)
        return m - np.sqrt(np.maximum(0.0, m * m - (a * a) * (cos_th ** 2)))

    def isco_radius(self, prograde: bool = True) -> float:
        """Innermost Stable Circular Orbit (ISCO) radius in meters.

        Evaluates the exact Bardeen, Press & Teukolsky (1972) formula:
            Z_1 = 1 + (1 - a_*^2)^(1/3) * [ (1 + a_*)^(1/3) + (1 - a_*)^(1/3) ]
            Z_2 = sqrt(3 * a_*^2 + Z_1^2)
            r_ISCO = M * [ 3 + Z_2 -+ sqrt((3 - Z_1) * (3 + Z_1 + 2 * Z_2)) ]
        where - is prograde and + is retrograde.
        """
        m = self.mass_m
        a_star = self.spin_dimensionless
        if not prograde:
            a_star = -a_star

        # Handle Schwarzschild limit
        if abs(a_star) < 1e-12:
            return 6.0 * m

        term1 = (1.0 - a_star * a_star) ** (1.0 / 3.0)
        term2 = (1.0 + a_star) ** (1.0 / 3.0) + (1.0 - a_star) ** (1.0 / 3.0)
        z1 = 1.0 + term1 * term2
        z2 = math.sqrt(3.0 * (a_star ** 2) + z1 * z1)

        inner = (3.0 - z1) * (3.0 + z1 + 2.0 * z2)
        sqrt_term = math.sqrt(max(0.0, inner))

        sign = -1.0 if a_star >= 0.0 else 1.0
        return m * (3.0 + z2 + sign * sqrt_term)

    def photon_orbit_radius(self, prograde: bool = True) -> float:
        """Equatorial circular photon orbit radius r_ph in meters.

        r_ph = 2 * M * [ 1 + cos( (2/3) * arccos(-+ a_*) ) ]
        where - is prograde and + is retrograde.
        """
        m = self.mass_m
        a_star = self.spin_dimensionless if prograde else -self.spin_dimensionless
        if abs(a_star) < 1e-12:
            return 3.0 * m
        arg = -a_star
        return 2.0 * m * (1.0 + math.cos((2.0 / 3.0) * math.acos(arg)))


# ==============================================================================
# EXACT METRIC TENSOR AND CHRISTOFFEL SYMBOLS
# ==============================================================================

def compute_kerr_metric(
    r: float,
    theta: float,
    kerr: KerrGeometry,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute covariant g_mu_nu and contravariant g^mu_nu metric tensors in Boyer-Lindquist coordinates.

    Coordinates: x^mu = (ct, r, theta, phi), signature (- + + +).

    Returns:
        Tuple of:
            g_cov: 4x4 covariant metric tensor matrix.
            g_con: 4x4 contravariant inverse metric tensor matrix.
    """
    m = kerr.mass_m
    a = kerr.spin_m
    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    sin2 = sin_th * sin_th
    cos2 = cos_th * cos_th

    sigma = r * r + a * a * cos2
    delta = r * r - 2.0 * m * r + a * a

    if abs(delta) < 1e-30:
        delta = math.copysign(1e-30, delta)

    # Covariant metric components
    g_tt = -(1.0 - (2.0 * m * r) / sigma)
    g_tphi = -(2.0 * m * a * r * sin2) / sigma
    g_rr = sigma / delta
    g_thth = sigma
    g_phiphi = (r * r + a * a + (2.0 * m * (a ** 2) * r * sin2) / sigma) * sin2

    g_cov = np.zeros((4, 4), dtype=np.float64)
    g_cov[0, 0] = g_tt
    g_cov[0, 3] = g_tphi
    g_cov[3, 0] = g_tphi
    g_cov[1, 1] = g_rr
    g_cov[2, 2] = g_thth
    g_cov[3, 3] = g_phiphi

    # Contravariant inverse metric components
    xi = ((r * r + a * a) ** 2) - (a * a) * delta * sin2
    denom = sigma * delta

    g_con = np.zeros((4, 4), dtype=np.float64)
    g_con[0, 0] = -xi / denom
    g_con[0, 3] = -(2.0 * m * a * r) / denom
    g_con[3, 0] = g_con[0, 3]
    g_con[1, 1] = delta / sigma
    g_con[2, 2] = 1.0 / sigma
    g_con[3, 3] = (delta - (a * a) * sin2) / (denom * sin2) if sin2 > 1e-15 else 0.0

    return g_cov, g_con


def compute_kerr_christoffel(
    r: float,
    theta: float,
    kerr: KerrGeometry,
) -> np.ndarray:
    """Compute exact Christoffel symbols Gamma^mu_alpha_beta for the Kerr metric.

    Evaluates:
        Gamma^mu_alpha_beta = (1/2) * g^mu_sigma * [ d_alpha g_beta_sigma + d_beta g_alpha_sigma - d_sigma g_alpha_beta ]
    from exact analytical spatial derivatives (d/dr, d/dtheta) of the metric.

    Coordinates: x^mu = (ct, r, theta, phi).

    Returns:
        4x4x4 array where gamma[mu, alpha, beta] = Gamma^mu_alpha_beta.
    """
    m = kerr.mass_m
    a = kerr.spin_m
    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    sin2 = sin_th * sin_th
    cos2 = cos_th * cos_th

    sigma = r * r + a * a * cos2
    delta = r * r - 2.0 * m * r + a * a
    sigma2 = sigma * sigma

    _, g_con = compute_kerr_metric(r, theta, kerr)

    # Partial derivatives with respect to r (index 1) and theta (index 2)
    # Metric does not depend on ct (index 0) or phi (index 3) due to stationarity and axisymmetry
    dg = np.zeros((4, 4, 4), dtype=np.float64)  # dg[coord, mu, nu] = d(g_mu_nu) / d(x^coord)

    # 1. Derivatives with respect to r
    d_sigma_dr = 2.0 * r
    d_delta_dr = 2.0 * r - 2.0 * m

    # g_tt = -(1 - 2*M*r / sigma)
    dg[1, 0, 0] = -(2.0 * m * (r * r - a * a * cos2)) / sigma2

    # g_tphi = -(2*M*a*r*sin^2(theta)) / sigma
    dg[1, 0, 3] = (2.0 * m * a * sin2 * (r * r - a * a * cos2)) / sigma2
    dg[1, 3, 0] = dg[1, 0, 3]

    # g_rr = sigma / delta
    dg[1, 1, 1] = (d_sigma_dr * delta - sigma * d_delta_dr) / (delta * delta)

    # g_thth = sigma
    dg[1, 2, 2] = d_sigma_dr

    # g_phiphi = (r^2 + a^2 + 2*M*a^2*r*sin2 / sigma) * sin2
    dg[1, 3, 3] = (2.0 * r + (2.0 * m * (a ** 2) * sin2 * (a * a * cos2 - r * r)) / sigma2) * sin2

    # 2. Derivatives with respect to theta
    d_sigma_dth = -2.0 * a * a * sin_th * cos_th

    # g_tt = -(1 - 2*M*r / sigma)
    dg[2, 0, 0] = (4.0 * m * (a ** 2) * r * sin_th * cos_th) / sigma2

    # g_tphi = -(2*M*a*r*sin^2(theta)) / sigma
    dg[2, 0, 3] = -(4.0 * m * a * r * sin_th * cos_th * (r * r + a * a)) / sigma2
    dg[2, 3, 0] = dg[2, 0, 3]

    # g_rr = sigma / delta
    dg[2, 1, 1] = d_sigma_dth / delta

    # g_thth = sigma
    dg[2, 2, 2] = d_sigma_dth

    # g_phiphi
    dg[2, 3, 3] = 2.0 * sin_th * cos_th * (r * r + a * a + (2.0 * m * (a ** 2) * r * sin2) / sigma) + \
                  sin2 * ((4.0 * m * (a ** 2) * r * sin_th * cos_th * (r * r + a * a)) / sigma2)

    # Construct Christoffel symbols: Gamma^mu_alpha_beta = (1/2) * g^mu_sig * (dg[alpha, beta, sig] + dg[beta, alpha, sig] - dg[sig, alpha, beta])
    gamma = np.zeros((4, 4, 4), dtype=np.float64)
    for mu in range(4):
        for alpha in range(4):
            for beta in range(4):
                val = 0.0
                for sig in range(4):
                    if g_con[mu, sig] != 0.0:
                        term = dg[alpha, beta, sig] + dg[beta, alpha, sig] - dg[sig, alpha, beta]
                        val += g_con[mu, sig] * term
                gamma[mu, alpha, beta] = 0.5 * val

    return gamma


# ==============================================================================
# CARTER INVARIANTS & INTEGRALS OF MOTION
# ==============================================================================

def compute_carter_invariants(
    pos_bl: np.ndarray,
    four_vel: np.ndarray,
    kerr: KerrGeometry,
    *,
    is_null: bool = False,
) -> Dict[str, float]:
    """Compute the exact conserved integrals of motion for a Kerr geodesic.

    Conserved quantities:
    1. rest_mass_norm: g_mu_nu u^mu u^nu (-1 for timelike, 0 for null).
    2. energy_per_mass: E = -u_t.
    3. angular_momentum_z: L_z = u_phi.
    4. carter_constant: Q = u_theta^2 + cos^2(theta) * [ a^2 * (epsilon - E^2) + L_z^2 / sin^2(theta) ].

    Args:
        pos_bl: Boyer-Lindquist position [ct, r, theta, phi].
        four_vel: Boyer-Lindquist 4-velocity [u^0, u^1, u^2, u^3] = dx^mu / dtau.
        kerr: KerrGeometry black hole.
        is_null: If True, particle is a null photon (epsilon = 0).

    Returns:
        Dictionary containing E, L_z, Q, and metric_norm.
    """
    r = float(pos_bl[1])
    th = float(pos_bl[2])
    g_cov, _ = compute_kerr_metric(r, th, kerr)

    # Covariant 4-momentum u_mu = g_mu_nu u^nu
    u_cov = np.dot(g_cov, four_vel)

    # Specific energy E = -u_0
    e_val = float(-u_cov[0])
    # Axial angular momentum L_z = u_3
    lz_val = float(u_cov[3])
    # Rest mass norm epsilon = -g_mu_nu u^mu u^nu
    norm = float(np.dot(u_cov, four_vel))
    epsilon = 0.0 if is_null else 1.0

    a = kerr.spin_m
    sin_th = math.sin(th)
    cos_th = math.cos(th)
    sin2 = max(1e-15, sin_th * sin_th)
    cos2 = cos_th * cos_th

    # Carter constant Q
    # Q = u_theta^2 + cos^2(theta) * [ a^2 * (epsilon - E^2) + L_z^2 / sin^2(theta) ]
    u_theta = float(u_cov[2])
    q_val = (u_theta ** 2) + cos2 * (a * a * (epsilon - e_val * e_val) + (lz_val ** 2) / sin2)

    return {
        "energy": e_val,
        "angular_momentum_z": lz_val,
        "carter_constant": q_val,
        "metric_norm": norm,
    }


# ==============================================================================
# DOP853 GEODESIC PROPAGATOR
# ==============================================================================

def propagate_kerr_geodesic(
    initial_pos_bl: np.ndarray,
    initial_four_vel: np.ndarray,
    affine_span: Tuple[float, float],
    kerr: KerrGeometry,
    *,
    is_null: bool = False,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_steps: int = 100000,
) -> Dict[str, Any]:
    """Integrate timelike or null geodesic equations of motion in Kerr spacetime using DOP853.

    State vector: Y = [ct, r, theta, phi, u^0, u^1, u^2, u^3] (8 states).

    Equations of motion:
        d(x^mu) / dtau = u^mu
        d(u^mu) / dtau = -Gamma^mu_alpha_beta * u^alpha * u^beta

    Terminates automatically if the trajectory crosses the outer event horizon r_+
    or escapes beyond 2000 M.

    Args:
        initial_pos_bl: Initial coordinates [ct, r, theta, phi].
        initial_four_vel: Initial 4-velocity [u^0, u^1, u^2, u^3].
        affine_span: (tau_start, tau_end) integration range.
        kerr: KerrGeometry.
        is_null: If True, integrates null geodesic (photon).
        rtol: Relative integration tolerance.
        atol: Absolute integration tolerance.
        max_steps: Maximum internal solver steps.

    Returns:
        Dictionary with:
            tau: 1D array of proper time / affine parameter samples.
            pos: (N, 4) array of coordinates [ct, r, theta, phi].
            vel: (N, 4) array of 4-velocity components.
            invariants: Initial and final Carter constants and drift.
            horizon_crossed: Boolean flag indicating if event horizon was crossed.
    """
    r_horizon = kerr.event_horizon_outer
    r_cutoff = r_horizon * 1.0001  # Safety guard just above event horizon

    y0 = np.concatenate([initial_pos_bl, initial_four_vel]).astype(np.float64)

    def rhs(tau: float, y: np.ndarray) -> np.ndarray:
        r = float(y[1])
        th = float(y[2])

        # If inside or near horizon, freeze evolution to prevent coordinate singularity blowup
        if r <= r_cutoff:
            return np.zeros(8, dtype=np.float64)

        u = y[4:8]
        gamma = compute_kerr_christoffel(r, th, kerr)

        # Acceleration: du^mu / dtau = -Gamma^mu_ab u^a u^b
        du = -np.einsum("mab,a,b->m", gamma, u, u)

        return np.concatenate([u, du])

    def horizon_event(tau: float, y: np.ndarray) -> float:
        return float(y[1] - r_cutoff)

    horizon_event.terminal = True
    horizon_event.direction = -1.0

    def escape_event(tau: float, y: np.ndarray) -> float:
        return float(y[1] - 2000.0 * kerr.mass_m)

    escape_event.terminal = True
    escape_event.direction = 1.0

    sol = solve_ivp(
        rhs,
        affine_span,
        y0,
        method="DOP853",
        events=[horizon_event, escape_event],
        rtol=rtol,
        atol=atol,
        max_step=(affine_span[1] - affine_span[0]) / 100.0,
    )

    trajectory_pos = sol.y[0:4, :].T
    trajectory_vel = sol.y[4:8, :].T

    # Audit Carter invariant drift between initial and final steps
    inv_init = compute_carter_invariants(trajectory_pos[0], trajectory_vel[0], kerr, is_null=is_null)
    inv_final = compute_carter_invariants(trajectory_pos[-1], trajectory_vel[-1], kerr, is_null=is_null)

    e_init = inv_init["energy"]
    e_final = inv_final["energy"]
    delta_e = abs(e_final - e_init) / max(1e-12, abs(e_init))

    q_init = inv_init["carter_constant"]
    q_final = inv_final["carter_constant"]
    delta_q = abs(q_final - q_init) / max(1e-12, abs(q_init))

    return {
        "tau": sol.t,
        "pos": trajectory_pos,
        "vel": trajectory_vel,
        "success": sol.success,
        "horizon_crossed": bool(len(sol.t_events[0]) > 0),
        "escaped": bool(len(sol.t_events[1]) > 0),
        "initial_invariants": inv_init,
        "final_invariants": inv_final,
        "energy_drift_fraction": delta_e,
        "carter_drift_fraction": delta_q,
    }


# ==============================================================================
# BARDEEN BLACK HOLE SHADOW
# ==============================================================================

def compute_bardeen_shadow_contour(
    kerr: KerrGeometry,
    theta_obs_rad: float = math.pi * 0.5,
    n_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the celestial shadow contour (alpha, beta) on the observer's screen.

    Evaluates the exact Bardeen (1973) parametric curve of unstable spherical
    photon orbits for an observer at infinity with inclination angle theta_obs:
        alpha(r) = -xi(r) / sin(theta_obs)
        beta(r) = +- sqrt( eta(r) + a^2 * cos^2(theta_obs) - xi(r)^2 * cot^2(theta_obs) )

    Special cases:
        - a = 0 (Schwarzschild): alpha^2 + beta^2 = 27 * M^2 (pure circular shadow, radius 3*sqrt(3)*M).
        - a > 0: Asymmetric D-shaped shadow flattened on prograde side by frame dragging.

    Args:
        kerr: KerrGeometry.
        theta_obs_rad: Observer inclination angle relative to spin axis in radians.
        n_points: Number of boundary contour samples.

    Returns:
        Tuple of:
            alpha: 1D array of horizontal celestial impact parameters in units of M.
            beta: 1D array of vertical celestial impact parameters in units of M.
    """
    m = kerr.mass_m
    a = kerr.spin_m
    sin_obs = math.sin(theta_obs_rad)
    cos_obs = math.cos(theta_obs_rad)
    sin_obs = max(1e-6, sin_obs)  # Guard against exact polar division by zero

    # For a = 0, shadow is an exact circle of radius 3*sqrt(3)*M
    if abs(a) < 1e-12:
        r_shadow = math.sqrt(27.0)
        phi_angles = np.linspace(0.0, 2.0 * math.pi, n_points)
        alpha_circ = r_shadow * np.cos(phi_angles)
        beta_circ = r_shadow * np.sin(phi_angles)
        return alpha_circ, beta_circ

    # Unstable photon orbits exist between r_ph_prograde and r_ph_retrograde
    r1 = kerr.photon_orbit_radius(prograde=True)
    r2 = kerr.photon_orbit_radius(prograde=False)

    # Sample r across the unstable photon orbit range (slightly inside boundary to avoid division by zero)
    r_vals = np.linspace(r1 * 1.0001, r2 * 0.9999, n_points // 2)

    delta_vals = r_vals * r_vals - 2.0 * m * r_vals + a * a
    r_minus_m = r_vals - m

    # Impact parameter ratios xi = L_z / E and eta = Q / E^2
    xi = (m * (r_vals * r_vals - a * a) - r_vals * delta_vals) / (a * r_minus_m)
    eta = (r_vals ** 3) * (4.0 * m * delta_vals - r_vals * (r_minus_m ** 2)) / ((a ** 2) * (r_minus_m ** 2))

    # Observer celestial coordinates (alpha, beta) normalized to M
    alpha_top = -xi / sin_obs / m
    cot_obs = cos_obs / sin_obs
    beta_sq = (eta + (a * a) * (cos_obs ** 2) - (xi ** 2) * (cot_obs ** 2)) / (m * m)

    # Only include real points where beta^2 >= 0
    valid_mask = beta_sq >= 0.0
    alpha_valid = alpha_top[valid_mask]
    beta_valid = np.sqrt(beta_sq[valid_mask])

    # Symmetrically complete the upper and lower halves of the contour
    alpha_full = np.concatenate([alpha_valid, alpha_valid[::-1]])
    beta_full = np.concatenate([beta_valid, -beta_valid[::-1]])

    return alpha_full, beta_full


# ==============================================================================
# PENROSE PROCESS KINEMATICS
# ==============================================================================

def compute_penrose_energy_gain(
    e_incoming: float,
    e_absorbed_negative: float,
) -> Dict[str, float]:
    """Compute relativistic energy extraction via the Penrose process in the ergosphere.

    In the ergosphere (r_+ < r < r_E), g_tt > 0, allowing timelike particles to have
    negative conserved energy E < 0 relative to an observer at infinity.

    By 4-momentum conservation, if particle 0 splits into particle 1 (negative energy,
    falls through event horizon) and particle 2 (escaping to infinity):
        p_0 = p_1 + p_2  ==>  E_2 = E_0 - E_1 = E_0 + |E_1| > E_0

    Theoretical maximum efficiency for an extremal Kerr black hole (a = M) is:
        eta_max = (sqrt(2) - 1) / 2 approx 20.7%

    Args:
        e_incoming: Conserved specific energy of incoming particle E_0 > 0.
        e_absorbed_negative: Conserved specific energy of captured particle E_1 < 0.

    Returns:
        Dictionary with escaping energy E_2 and efficiency eta.
    """
    if e_incoming <= 0.0:
        raise ValueError(f"Incoming particle energy must be positive, got {e_incoming}")
    if e_absorbed_negative >= 0.0:
        raise ValueError(f"Absorbed particle must have negative energy E_1 < 0, got {e_absorbed_negative}")

    e_escaping = e_incoming - e_absorbed_negative
    efficiency = (e_escaping - e_incoming) / e_incoming

    return {
        "energy_incoming": e_incoming,
        "energy_absorbed": e_absorbed_negative,
        "energy_escaping": e_escaping,
        "extraction_efficiency": efficiency,
    }
