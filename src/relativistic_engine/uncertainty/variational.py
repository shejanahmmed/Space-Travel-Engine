"""Variational state transition matrix (STM) and covariance propagation.

Computes the 7x7 State Transition Matrix Phi(t, t0) = dy(t)/dy(t0) along 3D relativistic
trajectories by integrating the coupled variational equations of motion:
    dy/dt = f(t, y, alpha)
    dPhi/dt = A(t) * Phi,   Phi(t0, t0) = I_7x7
    dS/dt   = A(t) * S + df/dalpha, S(t0) = 0_7x1

Authoritative References:
- Battin, R. H. (1999), "An Introduction to the Mathematics and Methods of Astrodynamics", AIAA.
- Montenbruck, O., & Gill, E. (2000), "Satellite Orbits: Models, Methods and Applications", Springer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import math
import numpy as np
from scipy.integrate import solve_ivp
from jplephem.spk import SPK

from relativistic_engine.constants import C_LIGHT, SEC_PER_DAY
from relativistic_engine.physics.dynamics import proper_to_coordinate_acceleration
from relativistic_engine.physics.metrics import coordinate_time_deficit_rate
from relativistic_engine.physics.potential import (
    STANDARD_BODY_GM,
    STANDARD_BODY_RADIUS,
    solar_system_potential,
    solar_system_gravitational_acceleration,
)
from relativistic_engine.ephemeris.barycentric import get_body_barycentric_state
from relativistic_engine.numerical.trajectory import (
    TrajectoryResult3D,
    ThrustFunction,
)


def evaluate_gravity_gradient(
    r_bcrs: np.ndarray | Sequence[float],
    jd_tdb: float,
    spk: Optional[SPK] = None,
    *,
    jd_fraction: float = 0.0,
    bodies: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Compute the 3x3 Newtonian tidal gravity gradient tensor G = da_grav / dr.

    Outside body surface (|r - r_i| >= R_i):
        G_jk = sum_i [ - (GM_i / d_i^3) * delta_jk + 3 * (GM_i / d_i^5) * d_j * d_k ]
    Inside body surface (|r - r_i| < R_i, Gauss's law):
        G_jk = sum_i [ - (GM_i / R_i^3) * delta_jk ]

    Args:
        r_bcrs: Position vector [x, y, z] in meters.
        jd_tdb: Epoch in Julian Date (TDB scale).
        spk: Optional pre-loaded SPK kernel.
        jd_fraction: Optional fractional day component.
        bodies: Sequence of bodies to include (default: all standard bodies).

    Returns:
        3x3 symmetric gravity gradient tensor in s^-2.
    """
    r = np.asarray(r_bcrs, dtype=np.float64)
    target_bodies = bodies if bodies is not None else STANDARD_BODY_GM.keys()

    g_tensor = np.zeros((3, 3), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)

    for name in target_bodies:
        gm = STANDARD_BODY_GM[name]
        radius = STANDARD_BODY_RADIUS.get(name, 1.0)
        body_state = get_body_barycentric_state(name, jd_tdb, spk=spk, jd_fraction=jd_fraction)
        dr = r - body_state.position
        dist = float(np.linalg.norm(dr))

        if dist >= radius:
            inv_dist3 = 1.0 / (dist * dist * dist)
            inv_dist5 = inv_dist3 / (dist * dist)
            g_tensor += -gm * inv_dist3 * eye + 3.0 * gm * inv_dist5 * np.outer(dr, dr)
        else:
            # Constant tidal gradient inside uniform sphere
            g_tensor += - (gm / (radius * radius * radius)) * eye

    return g_tensor


def evaluate_dynamics_jacobian(
    r: np.ndarray,
    v: np.ndarray,
    t: float,
    thrust_func: Optional[ThrustFunction],
    epoch_jd_tdb: Optional[float],
    spk: Optional[SPK],
    gravitational_bodies: Optional[Sequence[str]],
    include_1pn: bool,
) -> np.ndarray:
    """Evaluate the 7x7 Jacobian matrix A(t) = df/dy of the relativistic equations of motion.

    y = [r, v, Delta] where r in R^3, v in R^3, Delta in R.

    Returns:
        7x7 Jacobian matrix A(t) in SI units.
    """
    a_mat = np.zeros((7, 7), dtype=np.float64)

    # 1. dr/dt = v -> df_r / dv = I_3x3
    a_mat[0:3, 3:6] = np.eye(3, dtype=np.float64)

    # 2. df_v / dr: Gravity gradient tensor
    if epoch_jd_tdb is not None:
        jd_frac = t / SEC_PER_DAY
        g_grav = evaluate_gravity_gradient(
            r,
            epoch_jd_tdb,
            spk=spk,
            jd_fraction=jd_frac,
            bodies=gravitational_bodies,
        )
        a_mat[3:6, 0:3] = g_grav

    # 3. df_v / dv: Velocity derivative of coordinate thrust acceleration
    if thrust_func is not None:
        a_prop = np.asarray(thrust_func(t, r, v), dtype=np.float64)
        h_v = 1.0e-3  # 1 mm/s finite difference step
        for k in range(3):
            v_plus = v.copy()
            v_minus = v.copy()
            v_plus[k] += h_v
            v_minus[k] -= h_v
            a_c_plus = proper_to_coordinate_acceleration(v_plus, a_prop)
            a_c_minus = proper_to_coordinate_acceleration(v_minus, a_prop)
            a_mat[3:6, 3 + k] = (a_c_plus - a_c_minus) / (2.0 * h_v)

    # 4. df_Delta / dr and df_Delta / dv: Time deficit rate derivatives
    if epoch_jd_tdb is not None:
        jd_frac = t / SEC_PER_DAY
        w_pot = solar_system_potential(
            r,
            epoch_jd_tdb,
            spk=spk,
            jd_fraction=jd_frac,
            bodies=gravitational_bodies,
        )
    else:
        w_pot = 0.0

    # Sensitivity with respect to position via potential gradient: dw/dr = -a_newton
    if epoch_jd_tdb is not None:
        h_w = 1.0  # 1 m^2/s^2 potential perturbation
        rate_w_plus = coordinate_time_deficit_rate(v, w_pot + h_w)
        rate_w_minus = coordinate_time_deficit_rate(v, max(0.0, w_pot - h_w))
        d_rate_dw = (rate_w_plus - rate_w_minus) / (2.0 * h_w)

        # Grad(w) is -a_grav
        a_grav = solar_system_gravitational_acceleration(
            r,
            v,
            epoch_jd_tdb,
            spk=spk,
            jd_fraction=jd_frac,
            bodies=gravitational_bodies,
            include_1pn=False,
        )
        grad_w = -a_grav
        a_mat[6, 0:3] = d_rate_dw * grad_w

    # Sensitivity with respect to velocity
    h_v = 1.0e-3
    for k in range(3):
        v_plus = v.copy()
        v_minus = v.copy()
        v_plus[k] += h_v
        v_minus[k] -= h_v
        rate_v_plus = coordinate_time_deficit_rate(v_plus, w_pot)
        rate_v_minus = coordinate_time_deficit_rate(v_minus, w_pot)
        a_mat[6, 3 + k] = (rate_v_plus - rate_v_minus) / (2.0 * h_v)

    return a_mat


@dataclass(frozen=True)
class STMPropagationResult:
    """Result of worldline propagation with State Transition Matrix and sensitivities.

    Attributes
    ----------
    trajectory : TrajectoryResult3D
        Base worldline trajectory solution.
    stm : np.ndarray
        7x7 State Transition Matrix Phi(t_end, t0) = dy(t_end) / dy(t0).
    sensitivity_alpha : np.ndarray
        7x1 sensitivity vector S(t_end) = dy(t_end) / dalpha.
    """

    trajectory: TrajectoryResult3D
    stm: np.ndarray
    sensitivity_alpha: np.ndarray

    def propagate_covariance(
        self,
        p0: np.ndarray,
        sigma_alpha: float = 0.0,
    ) -> np.ndarray:
        """Map initial state covariance P0 (7x7) and thrust variance to arrival covariance.

        P_arrival = Phi * P0 * Phi^T + S * sigma_alpha^2 * S^T

        Args:
            p0: 7x7 initial state covariance matrix.
            sigma_alpha: Standard uncertainty in proper acceleration magnitude (m/s^2).

        Returns:
            7x7 mapped covariance matrix at t_end.
        """
        p0_arr = np.asarray(p0, dtype=np.float64)
        p_mapped = self.stm @ p0_arr @ self.stm.T

        if sigma_alpha > 0.0:
            s_vec = self.sensitivity_alpha.reshape((7, 1))
            p_mapped += (sigma_alpha * sigma_alpha) * (s_vec @ s_vec.T)

        return p_mapped


def propagate_with_stm(
    r0: np.ndarray | Sequence[float],
    v0: np.ndarray | Sequence[float],
    t_span: tuple[float, float],
    *,
    thrust_func: Optional[ThrustFunction] = None,
    accel_magnitude: Optional[float] = None,
    epoch_jd_tdb: Optional[float] = None,
    spk: Optional[SPK] = None,
    gravitational_bodies: Optional[Sequence[str]] = ("sun",),
    include_1pn: bool = True,
    rtol: float = 1e-8,
    atol: float = 1e-9,
) -> STMPropagationResult:
    """Integrate 3D relativistic worldline coupled with 7x7 STM and parameter sensitivity.

    Total state size: 7 (state) + 49 (STM elements) + 7 (sensitivity) = 63 variables.

    Parameters
    ----------
    r0 : Sequence[float]
        Initial position [x, y, z] in meters.
    v0 : Sequence[float]
        Initial velocity [vx, vy, vz] in m/s.
    t_span : tuple[float, float]
        Integration time interval (t_start, t_end) in coordinate seconds.
    thrust_func : Optional[ThrustFunction]
        Thrust profile returning proper acceleration vector.
    accel_magnitude : Optional[float]
        Nominal proper acceleration magnitude alpha (for df/dalpha sensitivity).
    epoch_jd_tdb : Optional[float]
        Julian Date at t = 0.
    spk : Optional[SPK]
        NASA/JPL SPK kernel.
    gravitational_bodies : Optional[Sequence[str]]
        Bodies to include in gravity.
    include_1pn : bool
        Whether to evaluate 1PN solar acceleration.
    rtol : float
        Relative integration tolerance.
    atol : float
        Absolute integration tolerance.

    Returns
    -------
    STMPropagationResult
        Solution containing trajectory, 7x7 STM, and 7-element sensitivity vector.
    """
    r_init = np.asarray(r0, dtype=np.float64)
    v_init = np.asarray(v0, dtype=np.float64)

    # Initial state vector y0: [rx, ry, rz, vx, vy, vz, delta]
    y0 = np.concatenate([r_init, v_init, [0.0]])

    # Initial STM: Phi(t0, t0) = I_7x7 flattened to 49 elements
    phi0 = np.eye(7, dtype=np.float64).reshape(-1)

    # Initial sensitivity: S(t0) = 0_7x1
    s0 = np.zeros(7, dtype=np.float64)

    # Combined state vector of length 63
    state0 = np.concatenate([y0, phi0, s0])

    def variational_rhs(t: float, state: np.ndarray) -> np.ndarray:
        y = state[0:7]
        phi = state[7:56].reshape((7, 7))
        s = state[56:63]

        r = y[0:3]
        v = y[3:6]

        # 1. Evaluate base dynamics f(t, y)
        if epoch_jd_tdb is not None:
            jd_frac = t / SEC_PER_DAY
            w_pot = solar_system_potential(
                r,
                epoch_jd_tdb,
                spk=spk,
                jd_fraction=jd_frac,
                bodies=gravitational_bodies,
            )
            a_grav = solar_system_gravitational_acceleration(
                r,
                v,
                epoch_jd_tdb,
                spk=spk,
                jd_fraction=jd_frac,
                bodies=gravitational_bodies,
                include_1pn=include_1pn,
            )
        else:
            w_pot = 0.0
            a_grav = np.zeros(3, dtype=np.float64)

        if thrust_func is not None:
            a_prop = np.asarray(thrust_func(t, r, v), dtype=np.float64)
            a_thrust = proper_to_coordinate_acceleration(v, a_prop)
        else:
            a_prop = np.zeros(3, dtype=np.float64)
            a_thrust = np.zeros(3, dtype=np.float64)

        dv_dt = a_grav + a_thrust
        d_delta_dt = coordinate_time_deficit_rate(v, w_pot)
        dy_dt = np.concatenate([v, dv_dt, [d_delta_dt]])

        # 2. Evaluate dynamics Jacobian A(t) = df/dy
        a_mat = evaluate_dynamics_jacobian(
            r=r,
            v=v,
            t=t,
            thrust_func=thrust_func,
            epoch_jd_tdb=epoch_jd_tdb,
            spk=spk,
            gravitational_bodies=gravitational_bodies,
            include_1pn=include_1pn,
        )

        # 3. Variational derivative: dPhi/dt = A * Phi
        dphi_dt = (a_mat @ phi).reshape(-1)

        # 4. Sensitivity derivative: dS/dt = A * S + df/dalpha
        df_dalpha = np.zeros(7, dtype=np.float64)
        if accel_magnitude is not None and accel_magnitude > 0.0 and thrust_func is not None:
            # df/dalpha = a_thrust / alpha
            df_dalpha[3:6] = a_thrust / accel_magnitude

        ds_dt = a_mat @ s + df_dalpha

        return np.concatenate([dy_dt, dphi_dt, ds_dt])

    sol = solve_ivp(
        variational_rhs,
        t_span,
        state0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )

    t_arr = sol.t
    r_arr = sol.y[0:3, :].T
    v_arr = sol.y[3:6, :].T
    delta_arr = sol.y[6, :]
    tau_arr = t_arr - delta_arr

    traj = TrajectoryResult3D(
        t=t_arr,
        r=r_arr,
        v=v_arr,
        time_deficit=delta_arr,
        tau=tau_arr,
        status=sol.status,
        message=sol.message,
        nfev=sol.nfev,
    )

    final_phi = sol.y[7:56, -1].reshape((7, 7))
    final_s = sol.y[56:63, -1]

    return STMPropagationResult(
        trajectory=traj,
        stm=final_phi,
        sensitivity_alpha=final_s,
    )
