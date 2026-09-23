"""High-Order Adaptive Relativistic Trajectory Integrator.

This module formulates and solves the differential equations of motion for a
relativistic rocket in 1D flat spacetime using high-order adaptive Runge-Kutta
methods (specifically 8th-order Dormand-Prince, DOP853).

Mathematical Formulation:
State Vector:
    y(t) = [x(t), v(t), tau(t)]^T

Equations of Motion:
    dx/dt   = v(t)
    dv/dt   = alpha_proper(t, x, v) * (1 - v^2/c^2)^(3/2)
    dtau/dt = sqrt(1 - v^2/c^2)

Numerical Considerations:
- Discontinuous thrust (e.g., turnaround from +alpha to -alpha) degrades high-order
  integrator convergence if integrated across the discontinuity in a single span.
  Therefore, multi-stage journeys are integrated in smooth segments, restarting the
  integrator at each event boundary with the exact terminal state of the preceding stage.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np
from scipy.integrate import solve_ivp

from relativistic_engine.constants import C_LIGHT


@dataclass(frozen=True)
class IntegrationResult:
    """Container for relativistic numerical integration results.

    Attributes
    ----------
    t : np.ndarray
        Array of coordinate time evaluation points (seconds).
    x : np.ndarray
        Array of coordinate positions (meters).
    v : np.ndarray
        Array of coordinate velocities (m/s).
    tau : np.ndarray
        Array of traveler proper times (seconds).
    gamma : np.ndarray
        Array of Lorentz factors gamma.
    success : bool
        Whether the numerical integration terminated successfully.
    n_evals : int
        Number of right-hand-side evaluations performed by the solver.
    message : str
        Termination status message from the underlying solver.
    """

    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    tau: np.ndarray
    gamma: np.ndarray
    success: bool
    n_evals: int
    message: str


class RelativisticRocketODE:
    """Right-hand side callable for 1D relativistic rocket equations of motion."""

    def __init__(
        self,
        proper_accel_func: Callable[[float, float, float], float],
        c: float = C_LIGHT,
    ) -> None:
        """Initialize the relativistic ODE system.

        Parameters
        ----------
        proper_accel_func : Callable[[float, float, float], float]
            Function alpha(t, x, v) returning the proper acceleration in m/s^2.
        c : float, optional
            Speed of light in m/s (default: C_LIGHT).
        """
        self.proper_accel_func = proper_accel_func
        self.c = c
        self.c_sq = c * c

    def __call__(self, t: float, y: Sequence[float]) -> list[float]:
        """Evaluate the derivative vector dy/dt.

        Parameters
        ----------
        t : float
            Coordinate time in seconds.
        y : Sequence[float]
            Current state: [x, v, tau].

        Returns
        -------
        list[float]
            Derivatives: [dx/dt, dv/dt, dtau/dt].
        """
        x, v, _tau = y[0], y[1], y[2]

        # Numerical guard preventing trial step overshoot into spacelike domain (|v| >= c)
        v_clamped = max(min(v, self.c - 1e-12), -self.c + 1e-12)
        beta_sq = (v_clamped * v_clamped) / self.c_sq
        one_minus_beta_sq = max(1.0 - beta_sq, 0.0)

        # Inverse Lorentz factor: inv_gamma = 1 / gamma = sqrt(1 - beta^2)
        inv_gamma = np.sqrt(one_minus_beta_sq)

        # Coordinate 3-acceleration from proper acceleration: a(t) = alpha / gamma^3 = alpha * (1 - beta^2)^(3/2)
        alpha_proper = self.proper_accel_func(t, x, v_clamped)
        a_coord = alpha_proper * (one_minus_beta_sq * inv_gamma)

        # Proper time differential along worldline: dtau/dt = 1 / gamma
        dtau_dt = inv_gamma

        return [v_clamped, a_coord, dtau_dt]


def integrate_relativistic_trajectory(
    t_span: tuple[float, float],
    y0: Sequence[float],
    proper_accel_func: Callable[[float, float, float], float],
    method: str = "DOP853",
    rtol: float = 1e-12,
    atol: float = 1e-14,
    t_eval: Optional[np.ndarray] = None,
    c: float = C_LIGHT,
) -> IntegrationResult:
    """Numerically integrate a relativistic trajectory over a specified time span.

    Parameters
    ----------
    t_span : tuple[float, float]
        Integration interval (t_start, t_end) in seconds.
    y0 : Sequence[float]
        Initial state vector [x0, v0, tau0].
    proper_accel_func : Callable[[float, float, float], float]
        Callable alpha(t, x, v) providing proper acceleration in m/s^2.
    method : str, optional
        Integration solver method (default: 'DOP853' - 8th-order Dormand-Prince).
    rtol : float, optional
        Relative tolerance (default: 1e-12).
    atol : float, optional
        Absolute tolerance (default: 1e-14).
    t_eval : Optional[np.ndarray], optional
        Times at which to store the computed solution.
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    IntegrationResult
        Full integration results with time series and diagnostics.
    """
    ode = RelativisticRocketODE(proper_accel_func, c=c)

    sol = solve_ivp(
        fun=ode,
        t_span=t_span,
        y0=y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    t_arr = sol.t
    x_arr = sol.y[0]
    v_arr = sol.y[1]
    tau_arr = sol.y[2]

    # Compute Lorentz gamma array safely
    beta_arr = np.clip(np.abs(v_arr) / c, 0.0, 1.0 - 1e-15)
    gamma_arr = 1.0 / np.sqrt((1.0 - beta_arr) * (1.0 + beta_arr))

    return IntegrationResult(
        t=t_arr,
        x=x_arr,
        v=v_arr,
        tau=tau_arr,
        gamma=gamma_arr,
        success=sol.success,
        n_evals=sol.nfev,
        message=sol.message,
    )


def integrate_brachistochrone(
    distance: float,
    proper_accel: float,
    t_half: float,
    t_total: float,
    method: str = "DOP853",
    rtol: float = 1e-12,
    atol: float = 1e-14,
    num_points_per_stage: int = 100,
    c: float = C_LIGHT,
) -> IntegrationResult:
    """Integrate a 2-stage brachistochrone trajectory (boost + deceleration).

    Uses segmented integration across the turnaround point to preserve high-order
    accuracy without discontinuity chatter.

    Parameters
    ----------
    distance : float
        Total journey coordinate distance in meters.
    proper_accel : float
        Constant proper acceleration magnitude alpha in m/s^2.
    t_half : float
        Coordinate time of the turnaround event in seconds.
    t_total : float
        Total coordinate journey time in seconds.
    method : str, optional
        ODE solver method (default: 'DOP853').
    rtol : float, optional
        Relative tolerance (default: 1e-12).
    atol : float, optional
        Absolute tolerance (default: 1e-14).
    num_points_per_stage : int, optional
        Number of output time points per stage (default: 100).
    c : float, optional
        Speed of light in m/s (default: C_LIGHT).

    Returns
    -------
    IntegrationResult
        Combined trajectory across both boost and deceleration phases.
    """
    # Segmented integration across turnaround avoids order reduction from acceleration discontinuity
    t_eval_1 = np.linspace(0.0, t_half, num_points_per_stage)
    sol1 = integrate_relativistic_trajectory(
        t_span=(0.0, t_half),
        y0=[0.0, 0.0, 0.0],
        proper_accel_func=lambda t, x, v: proper_accel,
        method=method,
        rtol=rtol,
        atol=atol,
        t_eval=t_eval_1,
        c=c,
    )

    if not sol1.success:
        return sol1

    y_mid = [sol1.x[-1], sol1.v[-1], sol1.tau[-1]]

    t_eval_2 = np.linspace(t_half, t_total, num_points_per_stage)
    sol2 = integrate_relativistic_trajectory(
        t_span=(t_half, t_total),
        y0=y_mid,
        proper_accel_func=lambda t, x, v: -proper_accel,
        method=method,
        rtol=rtol,
        atol=atol,
        t_eval=t_eval_2,
        c=c,
    )

    if not sol2.success:
        return sol2

    t_combined = np.concatenate([sol1.t, sol2.t[1:]])
    x_combined = np.concatenate([sol1.x, sol2.x[1:]])
    v_combined = np.concatenate([sol1.v, sol2.v[1:]])
    tau_combined = np.concatenate([sol1.tau, sol2.tau[1:]])
    gamma_combined = np.concatenate([sol1.gamma, sol2.gamma[1:]])

    return IntegrationResult(
        t=t_combined,
        x=x_combined,
        v=v_combined,
        tau=tau_combined,
        gamma=gamma_combined,
        success=True,
        n_evals=sol1.n_evals + sol2.n_evals,
        message="2-stage brachistochrone integration successful.",
    )
